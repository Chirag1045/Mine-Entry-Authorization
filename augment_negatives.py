"""
Negative-Example Augmenter
===========================
The dataset has ZERO labelled examples of:
  • caps / baseball hats  (model mistakes them for helmets)
  • plain t-shirts        (model mistakes bright ones for vests)

This script auto-generates a small batch of negative-class training images
from the existing dataset by:
  1. Cropping confirmed "helmet" regions, converting them visually
     to look like caps (simple image transforms), and labelling them
     as class 7 (no_helmet).
  2. Cropping confirmed "vest" regions and stripping their high-vis
     color to simulate a t-shirt, labelling them as class 5 (none).

NOTE: This is a bootstrap augmentation to immediately improve the dataset.
For a production system, collect real cap/t-shirt photos and label them.

Usage:
  python augment_negatives.py --data-dir data/ --out-dir data/augmented/ --n 200
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


# ── Label constants ──────────────────────────────────────────────────────────
CLASS_HELMET     = 0
CLASS_VEST       = 2
CLASS_PERSON     = 6
CLASS_NO_HELMET  = 7   # "no hard hat" — we want caps labelled here
CLASS_NONE       = 5   # "no PPE" — plain shirt lives here


def read_yolo_labels(label_path: Path):
    """Return list of (class_id, cx, cy, w, h) from a YOLO txt file."""
    labels = []
    if not label_path.exists():
        return labels
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) == 5:
            labels.append((int(parts[0]), *[float(x) for x in parts[1:]]))
    return labels


def yolo_to_pixel(cx, cy, w, h, img_w, img_h):
    """Convert YOLO normalized coords to pixel bbox."""
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    return max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)


def desaturate_region(image: np.ndarray, box) -> np.ndarray:
    """Remove high-vis saturation from a region (vest → t-shirt simulation)."""
    x1, y1, x2, y2 = box
    roi = image[y1:y2, x1:x2].copy()
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= 0.25   # crush saturation
    hsv[:, :, 2] *= 0.85   # slightly darken
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    image[y1:y2, x1:x2] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return image


def add_brim_shadow(image: np.ndarray, box) -> np.ndarray:
    """
    Simulate a cap by darkening the lower half of the helmet bbox
    (hard hat brim area) and adding a slight shadow, making it look
    more like a peaked cap to a human reviewer.
    """
    x1, y1, x2, y2 = box
    mid_y = (y1 + y2) // 2
    image[mid_y:y2, x1:x2] = (image[mid_y:y2, x1:x2] * 0.55).astype(np.uint8)
    return image


def process_image(
    img_path: Path,
    label_path: Path,
    out_img_dir: Path,
    out_lbl_dir: Path,
    mode: str,   # "helmet" or "vest"
    idx: int,
):
    """Generate one augmented image with flipped negative labels."""
    image = cv2.imread(str(img_path))
    if image is None:
        return
    img_h, img_w = image.shape[:2]
    labels = read_yolo_labels(label_path)

    target_class = CLASS_HELMET if mode == "helmet" else CLASS_VEST
    negative_class = CLASS_NO_HELMET if mode == "helmet" else CLASS_NONE

    new_labels = []
    modified = False

    for lbl in labels:
        class_id, cx, cy, w, h = lbl
        if class_id == target_class:
            box = yolo_to_pixel(cx, cy, w, h, img_w, img_h)
            if mode == "helmet":
                image = add_brim_shadow(image, box)
            else:
                image = desaturate_region(image, box)
            # relabel: was positive PPE → now negative
            new_labels.append((negative_class, cx, cy, w, h))
            modified = True
        else:
            new_labels.append(lbl)

    if not modified:
        return

    # Random minor jitter to avoid exact duplicates
    image = cv2.flip(image, 1) if random.random() > 0.5 else image

    out_name = f"aug_{mode}_{idx:05d}.jpg"
    cv2.imwrite(str(out_img_dir / out_name), image)

    lbl_name = f"aug_{mode}_{idx:05d}.txt"
    lines = [f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, cx, cy, w, h in new_labels]
    (out_lbl_dir / lbl_name).write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description="Generate cap/t-shirt negative examples.")
    ap.add_argument("--data-dir", default="data/", help="Dataset root directory")
    ap.add_argument("--out-dir", default="data/augmented/", help="Output directory")
    ap.add_argument("--n", type=int, default=200, help="Number of augmented images to generate")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    out_img_dir = out_dir / "images" / "train"
    out_lbl_dir = out_dir / "labels" / "train"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    train_img_dir = data_dir / "images" / "train"
    train_lbl_dir = data_dir / "labels" / "train"

    all_imgs = sorted(train_img_dir.glob("*.jpg")) + sorted(train_img_dir.glob("*.jpeg"))
    random.shuffle(all_imgs)

    count = {"helmet": 0, "vest": 0}
    per_mode = args.n // 2

    for img_path in all_imgs:
        if count["helmet"] >= per_mode and count["vest"] >= per_mode:
            break
        lbl_path = train_lbl_dir / (img_path.stem + ".txt")
        labels = read_yolo_labels(lbl_path)
        classes_in_image = {l[0] for l in labels}

        for mode, target_cls in [("helmet", CLASS_HELMET), ("vest", CLASS_VEST)]:
            if count[mode] < per_mode and target_cls in classes_in_image:
                process_image(
                    img_path, lbl_path,
                    out_img_dir, out_lbl_dir,
                    mode, count[mode],
                )
                count[mode] += 1

    total = sum(count.values())
    print(f"Generated {total} augmented images → {out_dir}")
    print(f"  helmet negatives : {count['helmet']}")
    print(f"  vest   negatives : {count['vest']}")
    print()
    print("Next: merge the augmented/ folder into your training dataset YAML")
    print("and re-run train_improved.py")


if __name__ == "__main__":
    main()
