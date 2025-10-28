import os
import random
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageEnhance


ROOT = Path(__file__).parent
SRC_ICON = ROOT / "dataset" / "images" / "dl.png"
OUT_ROOT = ROOT / "dataset" / "ui_login"


def _rand_bg(size: Tuple[int, int]):
    """Create a random background image (RGB) with light texture."""
    w, h = size
    # Random smooth background
    base = np.random.randint(180, 240, (h, w, 3), dtype=np.uint8)
    noise = np.random.normal(0, 8, (h, w, 3))
    arr = np.clip(base + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _paste_icon(bg: Image.Image, icon: Image.Image):
    """
    Paste icon on background at a random position/scale/rotation.
    Returns bbox (x, y, w, h) in pixels.
    """
    W, H = bg.size

    # Random scaling: keep within a reasonable proportion of background
    min_w, max_w = int(W * 0.08), int(W * 0.25)
    target_w = random.randint(max(24, min_w), max_w)
    scale = target_w / icon.width
    target_h = int(icon.height * scale)
    icon_resized = icon.resize((target_w, target_h), Image.LANCZOS)

    # Small rotation for robustness
    angle = random.uniform(-6, 6)
    icon_rot = icon_resized.rotate(angle, resample=Image.BICUBIC, expand=True)

    # Random brightness/contrast tweaks
    icon_rot = ImageEnhance.Brightness(icon_rot).enhance(random.uniform(0.9, 1.1))
    icon_rot = ImageEnhance.Contrast(icon_rot).enhance(random.uniform(0.9, 1.1))

    iw, ih = icon_rot.size
    # Random position, ensure fully inside
    x = random.randint(0, max(0, W - iw))
    y = random.randint(int(H * 0.1), max(0, H - ih))

    # Composite (icon assumed RGBA)
    if icon_rot.mode != "RGBA":
        icon_rot = icon_rot.convert("RGBA")
    bg.paste(icon_rot, (x, y), mask=icon_rot)

    return x, y, iw, ih


def _bbox_to_yolo(cx, cy, w, h, W, H):
    return cx / W, cy / H, w / W, h / H


def generate_split(n_images: int, out_img_dir: Path, out_lbl_dir: Path, neg_ratio: float = 0.3):
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    icon = Image.open(SRC_ICON).convert("RGBA")

    for i in range(n_images):
        bg = _rand_bg((1280, 720))

        is_negative = random.random() < neg_ratio

        bboxes = []
        if not is_negative:
            x, y, w, h = _paste_icon(bg, icon)
            cx = x + w / 2
            cy = y + h / 2
            W, H = bg.size
            nx, ny, nw, nh = _bbox_to_yolo(cx, cy, w, h, W, H)
            bboxes.append((0, nx, ny, nw, nh))

        img_path = out_img_dir / f"ui_{i:05d}.jpg"
        lbl_path = out_lbl_dir / f"ui_{i:05d}.txt"
        bg.save(img_path, quality=92)

        # Write labels in YOLO format
        with open(lbl_path, "w", encoding="utf-8") as f:
            for cls, nx, ny, nw, nh in bboxes:
                f.write(f"{cls} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}\n")


def main():
    if not SRC_ICON.exists():
        raise FileNotFoundError(f"未找到按钮图: {SRC_ICON}")

    # Directory layout
    img_train = OUT_ROOT / "images" / "train"
    img_val = OUT_ROOT / "images" / "val"
    lbl_train = OUT_ROOT / "labels" / "train"
    lbl_val = OUT_ROOT / "labels" / "val"

    # Generate
    print("开始生成合成数据集 ...")
    generate_split(360, img_train, lbl_train, neg_ratio=0.3)
    generate_split(80, img_val, lbl_val, neg_ratio=0.3)
    print(f"完成！数据路径: {OUT_ROOT}")
    print("- 训练图像:", img_train)
    print("- 验证图像:", img_val)


if __name__ == "__main__":
    main()