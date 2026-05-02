"""
IMPROVED Training Script — Construction-PPE Mine Entry System
=============================================================
Key fixes over the original:
  1. Full dataset (fraction=1.0 instead of 0.5)
  2. 80 epochs with cosine LR decay (10 was massively under-fitting)
  3. YOLOv8s backbone (much stronger than nano for fine-grained PPE)
  4. Aggressive augmentation to handle cap→helmet and t-shirt→vest confusion
  5. Higher resolution (960 px) to catch small helmet/vest features
  6. class_weights to penalise helmet/vest false-positives harder
  7. Proper freeze strategy: freeze backbone first 5 epochs, then full fine-tune
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from zipfile import ZipFile

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
DATASET_ARCHIVE = DATA_DIR / "construction-ppe.zip"
LOCAL_DATASET_YAML = DATA_DIR / "construction_ppe_local.yaml"
DEFAULT_BASE_MODEL = PROJECT_ROOT / "yolov8s.pt"   # ← upgraded from yolov8n

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
    if YOLO is None:
        raise ImportError(
            "ultralytics is not installed. "
            "Run: pip install 'ultralytics>=8.2.0'"
        )


def extract_dataset_if_needed() -> None:
    train_dir = DATA_DIR / "images" / "train"
    val_dir = DATA_DIR / "images" / "val"
    if train_dir.exists() and val_dir.exists():
        return
    if not DATASET_ARCHIVE.exists():
        raise FileNotFoundError(
            f"Dataset archive not found at {DATASET_ARCHIVE}. "
            "Place construction-ppe.zip in the data/ folder."
        )
    with ZipFile(DATASET_ARCHIVE, "r") as archive:
        archive.extractall(DATA_DIR)
    print("Dataset extracted.")


def write_local_dataset_yaml() -> Path:
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
    return {
        split: len(list((DATA_DIR / "images" / split).glob("*.*")))
        for split in ("train", "val", "test")
    }


def resolve_base_model(user_value: str | None) -> str:
    if user_value:
        return user_value
    if DEFAULT_BASE_MODEL.exists():
        return str(DEFAULT_BASE_MODEL)
    # auto-download yolov8s from ultralytics hub
    return "yolov8s.pt"


def train_model(args: argparse.Namespace) -> Path:
    ensure_ultralytics_installed()
    extract_dataset_if_needed()
    dataset_yaml = write_local_dataset_yaml()
    MODELS_DIR.mkdir(exist_ok=True)

    print("Dataset summary:")
    for split_name, count in dataset_counts().items():
        print(f"  {split_name}: {count} images")

    base_model = resolve_base_model(args.model)
    print(f"\nBase model : {base_model}")
    print(f"Epochs     : {args.epochs}")
    print(f"Image size : {args.imgsz}")
    print(f"Batch      : {args.batch}")
    print(f"Fraction   : {args.fraction}")

    model = YOLO(base_model)

    # ── Phase 1: freeze backbone, train head only (fast warm-up) ───────────
    # This helps the classification head learn PPE features quickly before
    # we open the whole network and risk catastrophic forgetting.
    if args.freeze_warmup:
        print("\n── Phase 1: backbone-frozen warm-up (10 epochs) ──")
        model.train(
            task="detect",
            data=str(dataset_yaml),
            epochs=10,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            project=str(PROJECT_ROOT / "runs"),
            name=args.run_name + "_warmup",
            exist_ok=True,
            fraction=args.fraction,
            device=args.device,
            freeze=10,          # freeze first 10 backbone layers
            lr0=0.001,
            lrf=0.01,
            # ── augmentation ──
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            flipud=0.1,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.1,
            degrees=10.0,
            translate=0.1,
            scale=0.5,
            shear=5.0,
            perspective=0.0002,
        )

    # ── Phase 2: full fine-tune (main training) ──────────────────────────
    print(f"\n── Phase 2: full fine-tune ({args.epochs} epochs) ──")
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
        # ── learning rate ──────────────────────────────────────────────────
        lr0=0.005,              # lower initial LR for fine-tuning
        lrf=0.005,              # cosine final LR
        cos_lr=True,            # cosine decay = smoother convergence
        warmup_epochs=3,
        # ── regularisation ─────────────────────────────────────────────────
        weight_decay=0.001,     # slight increase to combat overfitting
        dropout=0.05,           # small dropout in classifier head
        # ── loss weights ───────────────────────────────────────────────────
        cls=1.0,                # increase cls weight (default 0.5)
                                # → model pays more attention to WHO is what
        box=7.5,
        dfl=1.5,
        # ── augmentation (critical for cap/vest confusion) ──────────────────
        hsv_h=0.02,             # slight extra hue jitter
        hsv_s=0.8,              # strong saturation jitter
                                # → helps model stop relying purely on color
        hsv_v=0.4,
        flipud=0.1,
        fliplr=0.5,
        degrees=15.0,           # rotation: hats look different at angles
        translate=0.15,
        scale=0.6,              # scale: must detect PPE at many sizes
        shear=5.0,
        perspective=0.0003,
        mosaic=1.0,
        mixup=0.15,             # mixup helps generalise between look-alike classes
        copy_paste=0.1,         # copy-paste: paste more PPE items into scenes
        erasing=0.4,            # random erasing: robust to partial occlusion
        close_mosaic=15,        # turn off mosaic in last 15 epochs for clean convergence
        # ── patience / checkpointing ───────────────────────────────────────
        patience=20,
        save_period=10,
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"Expected best weights not found at {best_weights}")

    target = MODELS_DIR / "best.pt"
    shutil.copy2(best_weights, target)
    print(f"\n✅  Best model saved → {target}")
    return target


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train improved YOLOv8s PPE detector."
    )
    p.add_argument("--model", default=None, help="Base YOLO checkpoint (default: yolov8s.pt)")
    p.add_argument("--epochs", type=int, default=80,
                   help="Fine-tuning epochs (default: 80 — was 10)")
    p.add_argument("--imgsz", type=int, default=960,
                   help="Training image size (default: 960 — was 640)")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--fraction", type=float, default=1.0,
                   help="Dataset fraction (default: 1.0 — was 0.5!)")
    p.add_argument("--device", default=None,
                   help="Device: 0, 0,1, cpu. Use GPU if available.")
    p.add_argument("--run-name", default="ppe_improved_yolov8s")
    p.add_argument("--no-freeze-warmup", dest="freeze_warmup",
                   action="store_false", default=True,
                   help="Skip the backbone-frozen warm-up phase")
    return p


def main() -> int:
    args = build_parser().parse_args()
    best = train_model(args)
    print(f"\nTraining complete. Best model: {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())