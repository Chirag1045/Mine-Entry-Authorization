"""
Training script for the Construction-PPE based mine entry system.

This script fine-tunes YOLOv8 on the official Ultralytics Construction-PPE
dataset and copies the best weights to models/best.pt.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from zipfile import ZipFile

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - handled at runtime
    YOLO = None


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
DATASET_ARCHIVE = DATA_DIR / "construction-ppe.zip"
LOCAL_DATASET_YAML = DATA_DIR / "construction_ppe_local.yaml"
DEFAULT_BASE_MODEL = PROJECT_ROOT / "yolov8n.pt"

CLASS_NAMES = {
    0: "helmet",
    1: "gloves",
    2: "vest",
    3: "boots",
    4: "goggles",
    5: "none",
    6: "Person",
    7: "no_helmet",
    8: "no_goggle",
    9: "no_gloves",
    10: "no_boots",
}


def ensure_ultralytics_installed() -> None:
    """Raise a clear error when ultralytics is unavailable."""
    if YOLO is None:
        raise ImportError(
            "ultralytics is not installed. Install dependencies with "
            "`pip install -r requirements.txt`."
        )


def extract_dataset_if_needed() -> None:
    """Extract the dataset zip into data/ if the split folders are missing."""
    train_dir = DATA_DIR / "images" / "train"
    val_dir = DATA_DIR / "images" / "val"
    test_dir = DATA_DIR / "images" / "test"
    if train_dir.exists() and val_dir.exists() and test_dir.exists():
        return

    if not DATASET_ARCHIVE.exists():
        raise FileNotFoundError(
            f"Dataset archive not found at {DATASET_ARCHIVE}. "
            "Download the official Construction-PPE dataset first."
        )

    with ZipFile(DATASET_ARCHIVE, "r") as archive:
        archive.extractall(DATA_DIR)


def write_local_dataset_yaml() -> Path:
    """Write a workspace-local YOLO dataset yaml with the correct root path."""
    yaml_lines = [
        f"path: {DATA_DIR.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    for class_id, class_name in CLASS_NAMES.items():
        yaml_lines.append(f"  {class_id}: {class_name}")

    LOCAL_DATASET_YAML.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    return LOCAL_DATASET_YAML


def dataset_counts() -> dict:
    """Count images in each dataset split."""
    return {
        "train": len(list((DATA_DIR / "images" / "train").glob("*.*"))),
        "val": len(list((DATA_DIR / "images" / "val").glob("*.*"))),
        "test": len(list((DATA_DIR / "images" / "test").glob("*.*"))),
    }


def resolve_base_model(user_value: str | None) -> str:
    """Pick the starting YOLO checkpoint for fine-tuning."""
    if user_value:
        return user_value
    if DEFAULT_BASE_MODEL.exists():
        return str(DEFAULT_BASE_MODEL)
    return "yolov8n.pt"


def train_model(args: argparse.Namespace) -> Path:
    """Train the PPE detector and copy the best weights into models/best.pt."""
    ensure_ultralytics_installed()
    extract_dataset_if_needed()
    dataset_yaml = write_local_dataset_yaml()
    MODELS_DIR.mkdir(exist_ok=True)

    print("Dataset summary:")
    for split_name, count in dataset_counts().items():
        print(f"  {split_name}: {count} images")

    base_model = resolve_base_model(args.model)
    equivalent_command = (
        f"yolo task=detect mode=train model={base_model} "
        f"data={dataset_yaml.as_posix()} epochs={args.epochs} imgsz={args.imgsz} "
        f"batch={args.batch} fraction={args.fraction}"
    )
    print("\nEquivalent training command:")
    print(equivalent_command)

    model = YOLO(base_model)
    results = model.train(
        task="detect",
        data=str(dataset_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        project=str(PROJECT_ROOT / "runs"),
        name=args.run_name,
        exist_ok=True,
        fraction=args.fraction,
        device=args.device,
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"Expected best weights were not produced at {best_weights}")

    target_weights = MODELS_DIR / "best.pt"
    shutil.copy2(best_weights, target_weights)
    return target_weights


def build_parser() -> argparse.ArgumentParser:
    """Configure CLI arguments for model training."""
    parser = argparse.ArgumentParser(
        description="Train a YOLOv8 PPE detector for mine entry authorization."
    )
    parser.add_argument("--model", type=str, default=None, help="Base YOLOv8 checkpoint.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of fine-tuning epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=16, help="Training batch size.")
    parser.add_argument("--workers", type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.5,
        help="Fraction of the dataset to use for a quick demo fine-tune.",
    )
    parser.add_argument("--device", type=str, default=None, help="Device id, e.g. 0 or cpu.")
    parser.add_argument(
        "--run-name",
        type=str,
        default="construction_ppe_yolov8n",
        help="Run name inside the runs folder.",
    )
    return parser


def main() -> int:
    """CLI entry point."""
    args = build_parser().parse_args()
    best_model_path = train_model(args)
    print(f"\nTraining complete. Best model saved to: {best_model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
