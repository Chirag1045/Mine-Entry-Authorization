"""
Real-time webcam PPE detection for mine entry authorization.
"""

from __future__ import annotations

import argparse
import time

import cv2

from fusion_module import build_authorization_result
from ppe_module import PPEDetector


def build_parser() -> argparse.ArgumentParser:
    """Configure CLI arguments for live webcam detection."""
    parser = argparse.ArgumentParser(
        description="Run real-time PPE-based smart entry authorization from a webcam."
    )
    parser.add_argument("--model", type=str, default=None, help="Path to models/best.pt.")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--width", type=int, default=1280, help="Requested webcam width.")
    parser.add_argument("--height", type=int, default=720, help="Requested webcam height.")
    return parser


def main() -> int:
    """Start webcam capture and display live authorization decisions."""
    args = build_parser().parse_args()
    detector = PPEDetector(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        imgsz=args.imgsz,
    )

    camera = cv2.VideoCapture(args.camera)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not camera.isOpened():
        raise RuntimeError(f"Could not open webcam index {args.camera}")

    previous_time = time.time()

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break

            detection_result = detector.detect_frame(frame, source_name="webcam")
            authorization_result = build_authorization_result(detection_result)
            annotated = detector.annotate_frame(frame, detection_result, authorization_result)

            current_time = time.time()
            fps = 1.0 / max(current_time - previous_time, 1e-6)
            previous_time = current_time

            cv2.putText(
                annotated,
                f"FPS: {fps:.1f}",
                (16, annotated.shape[0] - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Real-Time PPE Smart Entry Authorization", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
