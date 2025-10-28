import os
from pathlib import Path
import random
from typing import Tuple

import numpy as np
from PIL import Image, ImageEnhance


ROOT = Path(__file__).parent
BG_DIR = ROOT / 'screen'
ICON_LOGIN = ROOT / 'login.png'
ICON_FALLBACK = ROOT / 'dataset' / 'images' / 'dl.png'
OUT_ROOT = ROOT / 'dataset' / 'ui_login'


def list_backgrounds(bg_dir: Path):
    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    return [p for p in bg_dir.iterdir() if p.suffix.lower() in exts and not p.name.startswith('annot_')]


def load_icon() -> Image.Image:
    cand = ICON_LOGIN if ICON_LOGIN.exists() else ICON_FALLBACK
    icon = Image.open(cand).convert('RGBA')
    return icon


def paste_icon(bg: Image.Image, icon: Image.Image):
    W, H = bg.size
    min_w, max_w = int(W * 0.06), int(W * 0.22)
    target_w = random.randint(max(24, min_w), max(max_w, min_w + 1))
    scale = target_w / icon.width
    target_h = int(icon.height * scale)
    icon_resized = icon.resize((target_w, target_h), Image.LANCZOS)

    angle = random.uniform(-4, 4)
    icon_rot = icon_resized.rotate(angle, resample=Image.BICUBIC, expand=True)

    icon_rot = ImageEnhance.Brightness(icon_rot).enhance(random.uniform(0.92, 1.08))
    icon_rot = ImageEnhance.Contrast(icon_rot).enhance(random.uniform(0.92, 1.08))

    iw, ih = icon_rot.size
    x = random.randint(0, max(0, W - iw))
    y = random.randint(int(H * 0.1), max(0, H - ih))

    if icon_rot.mode != 'RGBA':
        icon_rot = icon_rot.convert('RGBA')
    bg.paste(icon_rot, (x, y), mask=icon_rot)

    return x, y, iw, ih


def bbox_to_yolo(cx, cy, w, h, W, H):
    return cx / W, cy / H, w / W, h / H


def ensure_dirs(root: Path):
    (root / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (root / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    (root / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    (root / 'labels' / 'val').mkdir(parents=True, exist_ok=True)


def generate_from_backgrounds(train_per_bg=3, val_per_bg=1):
    ensure_dirs(OUT_ROOT)
    icon = load_icon()
    bgs = list_backgrounds(BG_DIR)
    if not bgs:
        raise FileNotFoundError(f'背景截图为空：{BG_DIR}')

    tcount, vcount = 0, 0
    for bgp in bgs:
        base = Image.open(bgp).convert('RGB')
        for i in range(train_per_bg):
            img = base.copy()
            x, y, w, h = paste_icon(img, icon)
            cx = x + w / 2
            cy = y + h / 2
            W, H = img.size
            nx, ny, nw, nh = bbox_to_yolo(cx, cy, w, h, W, H)
            out_img = OUT_ROOT / 'images' / 'train' / f'{bgp.stem}_t{i}.jpg'
            out_lbl = OUT_ROOT / 'labels' / 'train' / f'{bgp.stem}_t{i}.txt'
            img.save(out_img, quality=92)
            with open(out_lbl, 'w', encoding='utf-8') as f:
                f.write(f'0 {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}\n')
            tcount += 1
        for j in range(val_per_bg):
            img = base.copy()
            x, y, w, h = paste_icon(img, icon)
            cx = x + w / 2
            cy = y + h / 2
            W, H = img.size
            nx, ny, nw, nh = bbox_to_yolo(cx, cy, w, h, W, H)
            out_img = OUT_ROOT / 'images' / 'val' / f'{bgp.stem}_v{j}.jpg'
            out_lbl = OUT_ROOT / 'labels' / 'val' / f'{bgp.stem}_v{j}.txt'
            img.save(out_img, quality=92)
            with open(out_lbl, 'w', encoding='utf-8') as f:
                f.write(f'0 {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}\n')
            vcount += 1
    return tcount, vcount


def main():
    print('开始基于真实截图背景合成数据集...')
    t, v = generate_from_backgrounds(train_per_bg=3, val_per_bg=1)
    print(f'完成！train={t}, val={v} 输出目录: {OUT_ROOT}')


if __name__ == '__main__':
    main()