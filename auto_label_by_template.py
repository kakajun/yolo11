import os
from pathlib import Path
import argparse
import random

import cv2
import numpy as np


ROOT = Path(__file__).parent


def yolo_line_from_bbox(x1, y1, x2, y2, W, H, cls=0):
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = (x2 - x1)
    h = (y2 - y1)
    return f"{cls} {cx/W:.6f} {cy/H:.6f} {w/W:.6f} {h/H:.6f}\n"


def multi_scale_match(image_gray, template_gray, scales, method=cv2.TM_CCOEFF_NORMED):
    best = {
        'score': -1.0,
        'loc': (0, 0),
        'size': (template_gray.shape[1], template_gray.shape[0])
    }
    for s in scales:
        th = int(template_gray.shape[0] * s)
        tw = int(template_gray.shape[1] * s)
        if th < 12 or tw < 12:
            continue
        tpl_resized = cv2.resize(template_gray, (tw, th), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(image_gray, tpl_resized, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        score = max_val if method in (cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED) else -min_val
        if score > best['score']:
            best['score'] = score
            best['loc'] = max_loc if method in (cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED) else min_loc
            best['size'] = (tw, th)
    return best


def build_dataset(src_dir: Path, template_path: Path, out_root: Path, val_ratio: float = 0.2, threshold: float = 0.75):
    out_images_train = out_root / 'images' / 'train'
    out_images_val = out_root / 'images' / 'val'
    out_labels_train = out_root / 'labels' / 'train'
    out_labels_val = out_root / 'labels' / 'val'
    for p in (out_images_train, out_images_val, out_labels_train, out_labels_val):
        p.mkdir(parents=True, exist_ok=True)

    tpl = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
    if tpl is None:
        raise FileNotFoundError(f"无法读取模板: {template_path}")

    # Collect source images
    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    imgs = [p for p in src_dir.iterdir() if p.suffix.lower() in exts and not p.name.startswith('annot_')]
    if not imgs:
        print(f"源目录无图片: {src_dir}")
        return 0, 0

    # Shuffle and split
    random.shuffle(imgs)
    n_val = max(1, int(len(imgs) * val_ratio))
    val_set = set(imgs[:n_val])

    scales = np.linspace(0.6, 1.4, 17)

    n_train, n_val_ok = 0, 0
    for img_path in imgs:
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        best = multi_scale_match(gray, tpl, scales)
        score = best['score']
        x, y = best['loc']
        tw, th = best['size']
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + tw), int(y + th)
        H, W = gray.shape

        if score < threshold:
            print(f"跳过 {img_path.name}: 匹配分数 {score:.3f} < 阈值 {threshold}")
            continue

        # Select split
        is_val = img_path in val_set
        img_out = (out_images_val if is_val else out_images_train) / img_path.name
        lbl_out = (out_labels_val if is_val else out_labels_train) / (img_path.stem + '.txt')

        # Save image
        cv2.imwrite(str(img_out), img)

        # Save label
        with open(lbl_out, 'w', encoding='utf-8') as f:
            f.write(yolo_line_from_bbox(x1, y1, x2, y2, W, H, cls=0))

        # Save visualization back to src_dir (optional)
        vis = img.copy()
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, f"score={score:.2f}", (x1, max(0, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(str(src_dir / f"annot_{img_path.stem}.jpg"), vis)

        if is_val:
            n_val_ok += 1
        else:
            n_train += 1

    return n_train, n_val_ok


def main():
    parser = argparse.ArgumentParser(description='使用模板匹配为真实截图自动生成YOLO标签并构建数据集')
    parser.add_argument('--src-dir', type=str, default=str(ROOT / 'screen'), help='真实截图目录')
    parser.add_argument('--template', type=str, default=str(ROOT / 'login.png'), help='登录按钮模板图片')
    parser.add_argument('--out', type=str, default=str(ROOT / 'dataset' / 'ui_login'), help='输出数据集根目录')
    parser.add_argument('--val-ratio', type=float, default=0.2, help='验证集比例')
    parser.add_argument('--threshold', type=float, default=0.75, help='匹配分数阈值')
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    template = Path(args.template)
    out_root = Path(args.out)

    print(f'源目录: {src_dir}')
    print(f'模板: {template}')
    print(f'输出: {out_root}')
    train_ok, val_ok = build_dataset(src_dir, template, out_root, args.val_ratio, args.threshold)
    print(f'完成。train={train_ok}, val={val_ok}')


if __name__ == '__main__':
    main()