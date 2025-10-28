import argparse
import os
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from PIL import ImageGrab


ROOT = Path(__file__).parent


def grab_screen(save_dir: Path | None):
    img = ImageGrab.grab()
    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    path = None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = save_dir / f"screen_{ts}.jpg"
        cv2.imwrite(str(path), frame)
    return frame, path


def multi_scale_match(image_gray, template_gray, scales, method=cv2.TM_CCOEFF_NORMED):
    best = {
        'score': -1.0,
        'loc': (0, 0),
        'size': (template_gray.shape[1], template_gray.shape[0])
    }
    for s in scales:
        th = int(template_gray.shape[0] * s)
        tw = int(template_gray.shape[1] * s)
        if th < 10 or tw < 10:
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


def annotate_and_save(image_bgr, box, score, out_path: Path):
    x1, y1, x2, y2 = box
    vis = image_bgr.copy()
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(vis, f"score={score:.2f}", (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)


def main():
    parser = argparse.ArgumentParser(description="使用模板匹配在屏幕或图像上检测登录按钮位置")
    parser.add_argument("--source", type=str, default="", help="图像路径；留空则截取全屏")
    parser.add_argument("--template", type=str, default=str(ROOT / "login.png"), help="模板图片路径")
    parser.add_argument("--save-dir", type=str, default=str(ROOT / "screen"), help="保存截屏与标注目录")
    parser.add_argument("--threshold", type=float, default=0.65, help="匹配分数阈值")
    parser.add_argument("--min-scale", type=float, default=0.6, help="最小缩放比")
    parser.add_argument("--max-scale", type=float, default=1.6, help="最大缩放比")
    parser.add_argument("--steps", type=int, default=21, help="缩放步数")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    template_path = Path(args.template)
    if not template_path.exists():
        raise FileNotFoundError(f"模板不存在: {template_path}")

    tpl = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
    if tpl is None:
        raise FileNotFoundError(f"无法读取模板图像: {template_path}")

    if args.source:
        img = cv2.imread(args.source, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"无法读取图像: {args.source}")
        screen_path = None
    else:
        img, screen_path = grab_screen(save_dir)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    scales = np.linspace(args.min_scale, args.max_scale, args.steps)
    best = multi_scale_match(gray, tpl, scales)
    score = best['score']
    x, y = best['loc']
    tw, th = best['size']
    x1, y1 = int(x), int(y)
    x2, y2 = int(x + tw), int(y + th)
    H, W = gray.shape

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    ann_path = save_dir / f"template_annot_{ts}.jpg"

    if score < args.threshold:
        annotate_and_save(img, (x1, y1, x2, y2), score, ann_path)
        print(f"未达到阈值: score={score:.3f} < {args.threshold:.2f}")
        if screen_path is not None:
            print(f"截屏: {screen_path}")
        print(f"标注图: {ann_path}")
        return

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    norm_x = cx / W
    norm_y = cy / H

    annotate_and_save(img, (x1, y1, x2, y2), score, ann_path)
    print("检测成功：")
    print(f"- 屏幕尺寸: {W}x{H}")
    if screen_path is not None:
        print(f"- 截屏: {screen_path}")
    print(f"- 标注图: {ann_path}")
    print(f"- 匹配分数: {score:.3f}")
    print(f"- 框(像素): x1={x1}, y1={y1}, x2={x2}, y2={y2}")
    print(f"- 中心(像素): x={int(cx)}, y={int(cy)}")
    print(f"- 中心(归一化): x={norm_x:.4f}, y={norm_y:.4f}")


if __name__ == "__main__":
    main()