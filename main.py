"""
Static image inference helper for the trained PPE model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from fusion_module import build_authorization_result, format_terminal_report
from ppe_module import PPEDetector


def build_parser() -> argparse.ArgumentParser:
    """Configure CLI arguments for static image inference."""
    parser = argparse.ArgumentParser(description="Run PPE authorization on a single image.")
    parser.add_argument("--image", type=str, required=True, help="Path to an input image.")
    parser.add_argument("--model", type=str, default=None, help="Path to models/best.pt.")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional path for the annotated image output.",
    )
    return parser


def main() -> int:
    """Analyze a single image with the trained PPE detector."""
    args = build_parser().parse_args()
    detector = PPEDetector(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        imgsz=args.imgsz,
    )

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path))
    detection_result = detector.detect_frame(image, source_name=str(image_path))
    authorization_result = build_authorization_result(detection_result)
    print(format_terminal_report(authorization_result))

    if args.save:
        annotated = detector.annotate_frame(image, detection_result, authorization_result)
        output_path = Path(args.save)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), annotated)
        print(f"Annotated image saved to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
