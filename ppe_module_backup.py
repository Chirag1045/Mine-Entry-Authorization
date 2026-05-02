"""
Dataset-trained PPE detection utilities for live mine entry authorization.

This module expects a YOLOv8 model fine-tuned on the Ultralytics
Construction-PPE dataset. It does not use heuristic fallbacks or synthetic
logic for PPE inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - handled with a clear runtime error
    YOLO = None


PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_TRAINED_MODEL = MODELS_DIR / "best.pt"

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


def normalize_label(label: str) -> str:
    """Normalize raw class names from the YOLO model."""
    return label.strip().lower()


def box_area(box: List[int]) -> int:
    """Return the pixel area of a box."""
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def box_center(box: List[int]) -> Tuple[int, int]:
    """Return the center point of a box."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def box_ioa(inner_box: List[int], outer_box: List[int]) -> float:
    """Compute intersection over area for inner_box against outer_box."""
    ix1 = max(inner_box[0], outer_box[0])
    iy1 = max(inner_box[1], outer_box[1])
    ix2 = min(inner_box[2], outer_box[2])
    iy2 = min(inner_box[3], outer_box[3])

    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    inner_area = max(1, box_area(inner_box))
    return intersection / inner_area


class PPEDetector:
    """PPE detector backed by a YOLOv8 model fine-tuned on Construction-PPE."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
    ) -> None:
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.model_path = Path(model_path) if model_path else DEFAULT_TRAINED_MODEL
        self.model = self._load_model(self.model_path)
        self.model_names = self._normalized_model_names()
        self._validate_model_labels()

    def detect(self, image_path: str) -> Dict[str, object]:
        """Run PPE analysis on an image path."""
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        return self.detect_frame(image, source_name=image_path)

    def detect_frame(self, frame: np.ndarray, source_name: str = "camera_frame") -> Dict[str, object]:
        """Run PPE analysis on an in-memory frame."""
        if frame is None or frame.size == 0:
            raise ValueError("Input frame is empty.")

        result = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            verbose=False,
        )[0]
        detections = self._parse_detections(result)
        return self._assess_primary_worker(detections, frame.shape, source_name)

    def annotate_frame(
        self,
        frame: np.ndarray,
        detection_result: Dict[str, object],
        authorization_result: Optional[Dict[str, object]] = None,
    ) -> np.ndarray:
        """Draw detections, PPE status, and the final decision on a frame."""
        canvas = frame.copy()

        for detection in detection_result.get("display_detections", []):
            x1, y1, x2, y2 = detection["bbox"]
            color = self._box_color_for_label(detection["raw_label"])
            label_text = f"{detection['display_label']} {detection['confidence']:.2f}"
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                canvas,
                label_text,
                (x1, max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

        person_box = detection_result.get("person_box")
        if person_box:
            px1, py1, px2, py2 = person_box
            cv2.rectangle(canvas, (px1, py1), (px2, py2), (255, 180, 0), 3)

        if authorization_result:
            decision = authorization_result["decision"]
            decision_color = authorization_result["decision_color"]
            cv2.putText(
                canvas,
                f"FINAL DECISION: {decision}",
                (16, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                decision_color,
                3,
                cv2.LINE_AA,
            )

            status_lines = [
                ("Helmet", authorization_result["helmet_status"]),
                ("Vest", authorization_result["vest_status"]),
                ("Shoes", authorization_result["shoes_status"]),
            ]
            line_y = 76
            for label, status in status_lines:
                status_color = self._status_color(status)
                cv2.putText(
                    canvas,
                    f"{label}: {status.upper()}",
                    (16, line_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    status_color,
                    2,
                    cv2.LINE_AA,
                )
                line_y += 28

            if authorization_result.get("message"):
                cv2.putText(
                    canvas,
                    authorization_result["message"][:90],
                    (16, line_y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    decision_color,
                    2,
                    cv2.LINE_AA,
                )

        return canvas

    def _load_model(self, model_path: Path) -> "YOLO":
        """Load the trained YOLOv8 PPE model."""
        if YOLO is None:
            raise ImportError(
                "ultralytics is not installed. Install dependencies with "
                "`pip install -r requirements.txt`."
            )
        if not model_path.exists():
            raise FileNotFoundError(
                f"Trained PPE model not found at {model_path}. "
                "Run train.py first to create models/best.pt."
            )
        return YOLO(str(model_path))

    def _normalized_model_names(self) -> Dict[int, str]:
        """Return normalized class names for the loaded model."""
        names = getattr(self.model, "names", {})
        return {int(class_id): normalize_label(name) for class_id, name in names.items()}

    def _validate_model_labels(self) -> None:
        """Fail fast if the model is not a PPE-trained checkpoint."""
        known_labels = set(self.model_names.values())
        required_labels = {"helmet", "vest", "boots", "person"}
        if not required_labels.issubset(known_labels):
            raise ValueError(
                "The loaded model does not look like a Construction-PPE checkpoint. "
                "Use the trained model saved at models/best.pt."
            )

    def _parse_detections(self, result) -> List[Dict[str, object]]:
        """Convert Ultralytics output into normalized detection dictionaries."""
        detections: List[Dict[str, object]] = []
        if result.boxes is None:
            return detections

        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            raw_label = self.model_names.get(class_id, str(class_id))
            normalized_box = [int(value) for value in box.tolist()]
            detections.append(
                {
                    "raw_label": raw_label,
                    "display_label": DISPLAY_LABEL_MAP.get(raw_label, raw_label),
                    "bbox": normalized_box,
                    "confidence": float(confidence),
                    "canonical_item": POSITIVE_CLASS_MAP.get(raw_label) or NEGATIVE_CLASS_MAP.get(raw_label),
                    "is_negative": raw_label in NEGATIVE_CLASS_MAP,
                }
            )

        return detections

    def _assess_primary_worker(
        self,
        detections: List[Dict[str, object]],
        frame_shape: Tuple[int, int, int],
        source_name: str,
    ) -> Dict[str, object]:
        """Assess the worker nearest the camera and summarize PPE status."""
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

        primary_person = max(person_detections, key=lambda detection: box_area(detection["bbox"]))
        person_box = primary_person["bbox"]
        worker_related = [
            detection
            for detection in relevant_detections
            if detection["raw_label"] != "person" and self._belongs_to_worker(detection["bbox"], person_box)
        ]

        positive_hits = {item: [] for item in PPE_ITEMS}
        negative_hits = {item: [] for item in PPE_ITEMS}
        for detection in worker_related:
            item = detection["canonical_item"]
            if not item:
                continue
            if detection["is_negative"]:
                negative_hits[item].append(detection)
            else:
                positive_hits[item].append(detection)

        framing = self._evaluate_framing(person_box, frame_width, frame_height)
        reframe_reasons: List[str] = []
        if len(person_detections) > 1:
            reframe_reasons.append("Show one worker at a time.")
        if framing["too_small"]:
            reframe_reasons.append("Move closer so the worker fills more of the frame.")

        helmet_status = self._resolve_item_status(
            item_name="helmet",
            positive_hits=positive_hits["helmet"],
            negative_hits=negative_hits["helmet"],
            uncertain=framing["head_cutoff"] or framing["too_small"],
        )
        vest_status = self._resolve_item_status(
            item_name="vest",
            positive_hits=positive_hits["vest"],
            negative_hits=[],
            uncertain=framing["side_cutoff"] or framing["too_small"],
        )
        shoes_status = self._resolve_item_status(
            item_name="shoes",
            positive_hits=positive_hits["shoes"],
            negative_hits=negative_hits["shoes"],
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
                item_result["status"] == "present"
                for item_result in (helmet_status, vest_status, shoes_status)
            )
            and not bool(reframe_message),
        }

    def _empty_result(
        self,
        source_name: str,
        model_path: str,
        detections: List[Dict[str, object]],
        reason: str,
    ) -> Dict[str, object]:
        """Return a consistent response when no worker can be analyzed."""
        uncertain_item = {"status": "uncertain", "confidence": 0.0, "evidence_label": ""}
        return {
            "source": source_name,
            "model_path": model_path,
            "worker_count": 0,
            "person_detected": False,
            "person_box": None,
            "frame_size": None,
            "display_detections": detections,
            "helmet": dict(uncertain_item),
            "vest": dict(uncertain_item),
            "shoes": dict(uncertain_item),
            "reframe_required": True,
            "reframe_message": reason,
            "ppe_compliant": False,
        }

    @staticmethod
    def _belongs_to_worker(item_box: List[int], person_box: List[int]) -> bool:
        """Associate a PPE item to the primary worker."""
        center_x, center_y = box_center(item_box)
        within_center = (
            person_box[0] <= center_x <= person_box[2] and person_box[1] <= center_y <= person_box[3]
        )
        return within_center or box_ioa(item_box, person_box) >= 0.15

    @staticmethod
    def _evaluate_framing(person_box: List[int], frame_width: int, frame_height: int) -> Dict[str, bool]:
        """Check whether the worker is fully visible enough for PPE verification."""
        x1, y1, x2, y2 = person_box
        margin_x = max(10, int(frame_width * 0.03))
        margin_y = max(10, int(frame_height * 0.03))
        area_ratio = box_area(person_box) / max(1, frame_width * frame_height)
        return {
            "head_cutoff": y1 <= margin_y,
            "feet_cutoff": y2 >= frame_height - margin_y,
            "side_cutoff": x1 <= margin_x or x2 >= frame_width - margin_x,
            "too_small": area_ratio < 0.12,
        }

    @staticmethod
    def _resolve_item_status(
        item_name: str,
        positive_hits: List[Dict[str, object]],
        negative_hits: List[Dict[str, object]],
        uncertain: bool,
    ) -> Dict[str, object]:
        """Resolve an item's present/missing/uncertain state from model evidence."""
        if positive_hits:
            best = max(positive_hits, key=lambda hit: hit["confidence"])
            return {
                "status": "present",
                "confidence": round(best["confidence"], 3),
                "evidence_label": best["raw_label"],
            }

        if negative_hits:
            best = max(negative_hits, key=lambda hit: hit["confidence"])
            return {
                "status": "missing",
                "confidence": round(best["confidence"], 3),
                "evidence_label": best["raw_label"],
            }

        if uncertain:
            return {"status": "uncertain", "confidence": 0.0, "evidence_label": ""}

        return {"status": "missing", "confidence": 0.0, "evidence_label": item_name}

    @staticmethod
    def _status_color(status: str) -> Tuple[int, int, int]:
        """Return a display color for a PPE status."""
        if status == "present":
            return (0, 200, 0)
        if status == "missing":
            return (0, 0, 255)
        return (0, 215, 255)

    @staticmethod
    def _box_color_for_label(label: str) -> Tuple[int, int, int]:
        """Return a color for detection boxes."""
        if label in POSITIVE_CLASS_MAP:
            return (0, 200, 0)
        if label in NEGATIVE_CLASS_MAP:
            return (0, 0, 255)
        return (255, 180, 0)

    @staticmethod
    def _deduplicate(items: List[str]) -> List[str]:
        """Keep the first occurrence of each message."""
        seen = set()
        ordered: List[str] = []
        for item in items:
            if item and item not in seen:
                ordered.append(item)
                seen.add(item)
        return ordered
