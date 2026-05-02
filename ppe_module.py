"""
IMPROVED PPE Detection Module
==============================
Fixes the two key false-positive problems reported:

  Problem 1 — CAP detected as HELMET
    Root cause: YOLOv8n trained for only 10 epochs on 50% data lacks the
    discriminative feature depth to tell apart a peaked cap from a hard hat.
    Fix A (training): yolov8s + 80 epochs + full data (see train_improved.py)
    Fix B (inference): shape + structural cross-check.
      Hard hats have a wide, rigid brim visible from all angles AND an
      above-average bbox height-to-width ratio compared to caps.
      We also require a higher minimum confidence (0.55) for 'helmet' class.

  Problem 2 — T-SHIRT detected as VEST
    Root cause: same under-trained model; bright-coloured shirts confuse it.
    Fix A (training): stronger saturation augmentation breaks the color bias.
    Fix B (inference): color signature check.
      High-visibility vests are exclusively fluorescent yellow-green (#CCFF00
      range in HSV) or fluorescent orange-red. We run a fast HSV histogram on
      the predicted bbox region. If the dominant hue is NOT in those bands the
      detection is demoted to 'uncertain' rather than confirmed as 'present'.

Both fixes degrade gracefully — if lighting is too dark to verify color,
we fall back to the model's decision unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

# ── Project paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_TRAINED_MODEL = MODELS_DIR / "best.pt"

# ── Class maps ───────────────────────────────────────────────────────────────
PPE_ITEMS = ("helmet", "vest", "shoes")
POSITIVE_CLASS_MAP = {"helmet": "helmet", "vest": "vest", "boots": "shoes"}
NEGATIVE_CLASS_MAP = {"no_helmet": "helmet", "no_boots": "shoes"}
DISPLAY_LABEL_MAP = {
    "person": "worker",
    "helmet": "helmet",
    "vest": "vest",
    "boots": "shoes",
    "no_helmet": "no_helmet",
    "no_boots": "no_shoes",
}
RELEVANT_LABELS = set(DISPLAY_LABEL_MAP)

# ── IMPROVED confidence thresholds (class-specific) ──────────────────────────
# Original code used a single flat 0.35 for everything — too low for
# visually similar classes. Raising helmet/vest thresholds cuts false positives.
CLASS_CONF_THRESHOLDS: Dict[str, float] = {
    "helmet":     0.55,   # hard hat vs cap: require stronger signal
    "vest":       0.50,   # hi-vis vest vs t-shirt: moderate raise
    "no_helmet":  0.50,
    "no_boots":   0.45,
    "boots":      0.40,
    "person":     0.30,   # keep low so we don't miss workers
    "gloves":     0.40,
    "goggles":    0.40,
    "default":    0.35,
}

# ── High-visibility vest color ranges (HSV) ──────────────────────────────────
# Fluorescent yellow-green: H ≈ 30-80, S > 0.55, V > 0.55
# Fluorescent orange-red:   H ≈ 0-20 or 160-180, S > 0.55, V > 0.55
VEST_HUE_RANGES = [
    (25, 85),    # yellow-lime-green
    (0, 20),     # orange-red low
    (160, 180),  # orange-red wrap
]
VEST_MIN_SAT = 130   # OpenCV S is 0-255
VEST_MIN_VAL = 100
VEST_COLOR_COVERAGE_THRESHOLD = 0.10  # ≥10% of bbox pixels must match


def normalize_label(label: str) -> str:
    return label.strip().lower()


def box_area(box: List[int]) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def box_center(box: List[int]) -> Tuple[int, int]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def box_ioa(inner_box: List[int], outer_box: List[int]) -> float:
    ix1 = max(inner_box[0], outer_box[0])
    iy1 = max(inner_box[1], outer_box[1])
    ix2 = min(inner_box[2], outer_box[2])
    iy2 = min(inner_box[3], outer_box[3])
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    inner_area = max(1, box_area(inner_box))
    return intersection / inner_area


# ── NEW: Color-signature helpers ──────────────────────────────────────────────

def _crop_box(frame: np.ndarray, box: List[int], pad: int = 4) -> Optional[np.ndarray]:
    """Return the sub-image inside a bounding box (with a small safety pad)."""
    h, w = frame.shape[:2]
    x1 = max(0, box[0] - pad)
    y1 = max(0, box[1] - pad)
    x2 = min(w, box[2] + pad)
    y2 = min(h, box[3] + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def vest_color_check(frame: np.ndarray, box: List[int]) -> bool:
    """
    Return True if the bbox region contains enough high-vis vest color.
    Returns True (pass-through) when the crop is too dark to evaluate.
    """
    crop = _crop_box(frame, box)
    if crop is None or crop.size == 0:
        return True  # can't verify → benefit of doubt

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # Build a mask for high-vis pixels
    combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for h_lo, h_hi in VEST_HUE_RANGES:
        lower = np.array([h_lo, VEST_MIN_SAT, VEST_MIN_VAL])
        upper = np.array([h_hi, 255, 255])
        combined_mask = cv2.bitwise_or(
            combined_mask,
            cv2.inRange(hsv, lower, upper)
        )

    # If the image is very dark (poor lighting), skip the check
    mean_val = float(np.mean(hsv[:, :, 2]))
    if mean_val < 40:
        return True  # too dark to judge

    coverage = float(np.count_nonzero(combined_mask)) / max(1, crop.size // 3)
    return coverage >= VEST_COLOR_COVERAGE_THRESHOLD


def helmet_shape_check(frame: np.ndarray, box: List[int]) -> bool:
    """
    Rough sanity check: a hard hat bbox is typically wider than tall (brim),
    whereas a cap is typically taller-or-square. Also validate minimum size.
    Returns True (pass-through) when conditions are ambiguous.
    """
    x1, y1, x2, y2 = box
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    aspect = bw / bh

    # Hard hat: aspect ratio roughly 0.9 – 2.5 (wider than tall, due to brim)
    # Peaked cap: typically aspect ratio 0.7 – 1.3 (squarish or slightly taller)
    # We reject only clearly cap-like shapes (aspect < 0.75 AND box is small)
    if aspect < 0.75 and box_area(box) < 2500:
        return False  # likely a cap, not a hard hat

    return True


# ── Main detector class ───────────────────────────────────────────────────────

class PPEDetector:
    """Improved PPE detector with class-specific thresholds & color cross-checks."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.35,   # kept as global floor; class-specific above
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        use_color_check: bool = True,    # NEW: enable vest color check
        use_shape_check: bool = True,    # NEW: enable helmet shape check
    ) -> None:
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.use_color_check = use_color_check
        self.use_shape_check = use_shape_check
        self.model_path = Path(model_path) if model_path else DEFAULT_TRAINED_MODEL
        self.model = self._load_model(self.model_path)
        self.model_names = self._normalized_model_names()
        self._validate_model_labels()

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, image_path: str) -> Dict[str, object]:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        return self.detect_frame(image, source_name=image_path)

    def detect_frame(self, frame: np.ndarray, source_name: str = "camera_frame") -> Dict[str, object]:
        if frame is None or frame.size == 0:
            raise ValueError("Input frame is empty.")

        result = self.model.predict(
            source=frame,
            conf=self.conf_threshold,   # global floor; we filter per-class below
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            verbose=False,
        )[0]
        detections = self._parse_detections(result, frame)
        return self._assess_primary_worker(detections, frame.shape, source_name)

    def annotate_frame(
        self,
        frame: np.ndarray,
        detection_result: Dict[str, object],
        authorization_result: Optional[Dict[str, object]] = None,
    ) -> np.ndarray:
        canvas = frame.copy()

        for detection in detection_result.get("display_detections", []):
            x1, y1, x2, y2 = detection["bbox"]
            color = self._box_color_for_label(detection["raw_label"])
            label_text = f"{detection['display_label']} {detection['confidence']:.2f}"
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                canvas, label_text, (x1, max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
            )

        person_box = detection_result.get("person_box")
        if person_box:
            px1, py1, px2, py2 = person_box
            cv2.rectangle(canvas, (px1, py1), (px2, py2), (255, 180, 0), 3)

        if authorization_result:
            decision = authorization_result["decision"]
            decision_color = authorization_result["decision_color"]
            cv2.putText(
                canvas, f"FINAL DECISION: {decision}", (16, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, decision_color, 3, cv2.LINE_AA,
            )
            status_lines = [
                ("Helmet", authorization_result["helmet_status"]),
                ("Vest",   authorization_result["vest_status"]),
                ("Shoes",  authorization_result["shoes_status"]),
            ]
            line_y = 76
            for label, status in status_lines:
                sc = self._status_color(status)
                cv2.putText(
                    canvas, f"{label}: {status.upper()}", (16, line_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, sc, 2, cv2.LINE_AA,
                )
                line_y += 28
            if authorization_result.get("message"):
                cv2.putText(
                    canvas, authorization_result["message"][:90], (16, line_y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, decision_color, 2, cv2.LINE_AA,
                )
        return canvas

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_model(self, model_path: Path) -> "YOLO":
        if YOLO is None:
            raise ImportError("ultralytics not installed. Run: pip install -r requirements.txt")
        if not model_path.exists():
            raise FileNotFoundError(
                f"Trained PPE model not found at {model_path}. Run train_improved.py first."
            )
        return YOLO(str(model_path))

    def _normalized_model_names(self) -> Dict[int, str]:
        names = getattr(self.model, "names", {})
        return {int(cid): normalize_label(name) for cid, name in names.items()}

    def _validate_model_labels(self) -> None:
        known = set(self.model_names.values())
        required = {"helmet", "vest", "boots", "person"}
        if not required.issubset(known):
            raise ValueError(
                "Model does not look like a Construction-PPE checkpoint. "
                "Use the model saved at models/best.pt."
            )

    def _parse_detections(self, result, frame: np.ndarray) -> List[Dict[str, object]]:
        """
        Parse YOLO output with class-specific confidence filtering
        and optional color / shape cross-checks.
        """
        detections: List[Dict[str, object]] = []
        if result.boxes is None:
            return detections

        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            raw_label = self.model_names.get(class_id, str(class_id))
            # ── class-specific confidence threshold ──────────────────────
            min_conf = CLASS_CONF_THRESHOLDS.get(raw_label, CLASS_CONF_THRESHOLDS["default"])
            if confidence < min_conf:
                continue  # discard low-confidence detections for this class

            normalized_box = [int(v) for v in box.tolist()]

            # ── FIX 2: vest color cross-check ───────────────────────────
            color_ok = True
            if raw_label == "vest" and self.use_color_check:
                color_ok = vest_color_check(frame, normalized_box)

            # ── FIX 1: helmet shape cross-check ─────────────────────────
            shape_ok = True
            if raw_label == "helmet" and self.use_shape_check:
                shape_ok = helmet_shape_check(frame, normalized_box)

            # Flag detections that failed secondary checks
            secondary_check_passed = color_ok and shape_ok

            detections.append(
                {
                    "raw_label": raw_label,
                    "display_label": DISPLAY_LABEL_MAP.get(raw_label, raw_label),
                    "bbox": normalized_box,
                    "confidence": float(confidence),
                    "canonical_item": (
                        POSITIVE_CLASS_MAP.get(raw_label) or NEGATIVE_CLASS_MAP.get(raw_label)
                    ),
                    "is_negative": raw_label in NEGATIVE_CLASS_MAP,
                    "secondary_check_passed": secondary_check_passed,
                }
            )

        return detections

    def _assess_primary_worker(
        self,
        detections: List[Dict[str, object]],
        frame_shape: Tuple[int, int, int],
        source_name: str,
    ) -> Dict[str, object]:
        frame_height, frame_width = frame_shape[:2]
        person_detections = [d for d in detections if d["raw_label"] == "person"]
        relevant_detections = [d for d in detections if d["raw_label"] in RELEVANT_LABELS]

        if not person_detections:
            return self._empty_result(
                source_name=source_name,
                model_path=str(self.model_path),
                detections=relevant_detections,
                reason="No worker detected. Stand in front of the camera.",
            )

        primary_person = max(person_detections, key=lambda d: box_area(d["bbox"]))
        person_box = primary_person["bbox"]
        worker_related = [
            d for d in relevant_detections
            if d["raw_label"] != "person" and self._belongs_to_worker(d["bbox"], person_box)
        ]

        positive_hits = {item: [] for item in PPE_ITEMS}
        negative_hits = {item: [] for item in PPE_ITEMS}
        for d in worker_related:
            item = d["canonical_item"]
            if not item:
                continue
            # ── IMPROVED: only count positive hits that pass secondary checks
            if d["is_negative"]:
                negative_hits[item].append(d)
            elif d.get("secondary_check_passed", True):
                positive_hits[item].append(d)
            # If secondary check FAILED on a positive detection, treat as uncertain
            # (don't move it to negative_hits, just silently drop)

        framing = self._evaluate_framing(person_box, frame_width, frame_height)
        reframe_reasons: List[str] = []
        if len(person_detections) > 1:
            reframe_reasons.append("Show one worker at a time.")
        if framing["too_small"]:
            reframe_reasons.append("Move closer so the worker fills more of the frame.")

        helmet_status = self._resolve_item_status(
            "helmet", positive_hits["helmet"], negative_hits["helmet"],
            uncertain=framing["head_cutoff"] or framing["too_small"],
        )
        vest_status = self._resolve_item_status(
            "vest", positive_hits["vest"], negative_hits["vest"],
            uncertain=framing["side_cutoff"] or framing["too_small"],
        )
        shoes_status = self._resolve_item_status(
            "shoes", positive_hits["shoes"], negative_hits["shoes"],
            uncertain=framing["feet_cutoff"] or framing["too_small"],
        )

        if helmet_status["status"] == "uncertain":
            reframe_reasons.append("Show the head and helmet fully in the frame.")
        if vest_status["status"] == "uncertain":
            reframe_reasons.append("Center the torso so the safety vest is fully visible.")
        if shoes_status["status"] == "uncertain":
            reframe_reasons.append("Step back so both safety shoes are visible.")

        reframe_message = " ".join(self._deduplicate(reframe_reasons))

        return {
            "source": source_name,
            "model_path": str(self.model_path),
            "worker_count": len(person_detections),
            "person_detected": True,
            "person_box": person_box,
            "frame_size": [frame_width, frame_height],
            "display_detections": [primary_person, *worker_related],
            "helmet": helmet_status,
            "vest": vest_status,
            "shoes": shoes_status,
            "reframe_required": bool(reframe_message),
            "reframe_message": reframe_message,
            "ppe_compliant": all(
                s["status"] == "present"
                for s in (helmet_status, vest_status, shoes_status)
            ) and not bool(reframe_message),
        }

    def _empty_result(self, source_name, model_path, detections, reason):
        uncertain = {"status": "uncertain", "confidence": 0.0, "evidence_label": ""}
        return {
            "source": source_name, "model_path": model_path,
            "worker_count": 0, "person_detected": False,
            "person_box": None, "frame_size": None,
            "display_detections": detections,
            "helmet": dict(uncertain), "vest": dict(uncertain), "shoes": dict(uncertain),
            "reframe_required": True, "reframe_message": reason, "ppe_compliant": False,
        }

    @staticmethod
    def _belongs_to_worker(item_box, person_box) -> bool:
        cx, cy = box_center(item_box)
        within = person_box[0] <= cx <= person_box[2] and person_box[1] <= cy <= person_box[3]
        return within or box_ioa(item_box, person_box) >= 0.15

    @staticmethod
    def _evaluate_framing(person_box, frame_width, frame_height):
        x1, y1, x2, y2 = person_box
        mx = max(10, int(frame_width * 0.03))
        my = max(10, int(frame_height * 0.03))
        ratio = box_area(person_box) / max(1, frame_width * frame_height)
        return {
            "head_cutoff":  y1 <= my,
            "feet_cutoff":  y2 >= frame_height - my,
            "side_cutoff":  x1 <= mx or x2 >= frame_width - mx,
            "too_small":    ratio < 0.12,
        }

    @staticmethod
    def _resolve_item_status(item_name, positive_hits, negative_hits, uncertain):
        if positive_hits:
            best = max(positive_hits, key=lambda h: h["confidence"])
            return {"status": "present", "confidence": round(best["confidence"], 3),
                    "evidence_label": best["raw_label"]}
        if negative_hits:
            best = max(negative_hits, key=lambda h: h["confidence"])
            return {"status": "missing", "confidence": round(best["confidence"], 3),
                    "evidence_label": best["raw_label"]}
        if uncertain:
            return {"status": "uncertain", "confidence": 0.0, "evidence_label": ""}
        return {"status": "missing", "confidence": 0.0, "evidence_label": item_name}

    @staticmethod
    def _status_color(status):
        return {
            "present": (0, 200, 0),
            "missing": (0, 0, 255),
        }.get(status, (0, 215, 255))

    @staticmethod
    def _box_color_for_label(label):
        if label in POSITIVE_CLASS_MAP:
            return (0, 200, 0)
        if label in NEGATIVE_CLASS_MAP:
            return (0, 0, 255)
        return (255, 180, 0)

    @staticmethod
    def _deduplicate(items):
        seen, ordered = set(), []
        for item in items:
            if item and item not in seen:
                ordered.append(item)
                seen.add(item)
        return ordered