import argparse
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


def build_button_mask_from_template(tpl_bgr, h_low=90, h_high=130, s_low=60, v_low=50):
    """
    Derive a binary mask of the blue login button from the template image.
    Returns (mask, bbox) where bbox is (x, y, w, h) in template coordinates of the largest masked region.
    """
    hsv = cv2.cvtColor(tpl_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([h_low, s_low, v_low], dtype=np.uint8)
    upper = np.array([h_high, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None
    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    # keep only region around button to avoid panel influence
    tight = np.zeros_like(mask)
    cv2.rectangle(tight, (x, y), (x + w, y + h), 255, -1)
    return tight, (x, y, w, h)


def multi_scale_masked_match(image_gray, template_gray, mask_gray, scales, method=cv2.TM_CCORR_NORMED):
    best = {
        'score': -1.0,
        'loc': (0, 0),
        'size': (template_gray.shape[1], template_gray.shape[0]),
        'scale': 1.0
    }
    for s in scales:
        th = max(10, int(template_gray.shape[0] * s))
        tw = max(10, int(template_gray.shape[1] * s))
        tpl_rs = cv2.resize(template_gray, (tw, th), interpolation=cv2.INTER_AREA)
        msk_rs = cv2.resize(mask_gray, (tw, th), interpolation=cv2.INTER_NEAREST)
        res = cv2.matchTemplate(image_gray, tpl_rs, method, mask=msk_rs)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        score = max_val
        if score > best['score']:
            best['score'] = score
            best['loc'] = max_loc
            best['size'] = (tw, th)
            best['scale'] = s
    return best


def annotate(image_bgr, box, score, out_path: Path):
    x1, y1, x2, y2 = box
    vis = image_bgr.copy()
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(vis, f"score={score:.2f}", (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)


def main():
    parser = argparse.ArgumentParser(description="使用带遮罩的模板匹配精确定位登录按钮（相对 panel 只取按钮区域）")
    parser.add_argument("--source", type=str, default="", help="要检测的图像；留空则截取全屏")
    parser.add_argument("--template", type=str, default=str(ROOT / "login.png"), help="模板（包含按钮）")
    parser.add_argument("--save-dir", type=str, default=str(ROOT / "screen"), help="保存目录")
    parser.add_argument("--threshold", type=float, default=0.55, help="匹配分数阈值")
    parser.add_argument("--min-scale", type=float, default=0.6, help="最小缩放比")
    parser.add_argument("--max-scale", type=float, default=1.6, help="最大缩放比")
    parser.add_argument("--steps", type=int, default=21, help="缩放步数")
    parser.add_argument("--h-low", type=int, default=90, help="HSV H 下限 (蓝) 0-179")
    parser.add_argument("--h-high", type=int, default=130, help="HSV H 上限 (蓝) 0-179")
    parser.add_argument("--s-low", type=int, default=60, help="HSV S 下限 0-255")
    parser.add_argument("--v-low", type=int, default=50, help="HSV V 下限 0-255")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    tpl_path = Path(args.template)
    if not tpl_path.exists():
        raise FileNotFoundError(f"模板不存在: {tpl_path}")

    tpl_bgr = cv2.imread(str(tpl_path), cv2.IMREAD_COLOR)
    if tpl_bgr is None:
        raise FileNotFoundError(f"无法读取模板图像: {tpl_path}")

    button_mask, mask_bbox = build_button_mask_from_template(
        tpl_bgr, h_low=args.h_low, h_high=args.h_high, s_low=args.s_low, v_low=args.v_low
    )
    if button_mask is None:
        raise RuntimeError("模板中未能提取到按钮掩码，请提供只包含按钮的模板或调整HSV范围。")

    tpl_gray = cv2.cvtColor(tpl_bgr, cv2.COLOR_BGR2GRAY)

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
    best = multi_scale_masked_match(gray, tpl_gray, button_mask, scales)
    score = best['score']
    x, y = best['loc']
    s = best['scale']
    # Convert template button bbox into target coordinates
    bx1 = x + int(mask_bbox[0] * s)
    by1 = y + int(mask_bbox[1] * s)
    bx2 = bx1 + int(mask_bbox[2] * s)
    by2 = by1 + int(mask_bbox[3] * s)
    H, W = gray.shape

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    ann_path = save_dir / f"button_annot_{ts}.jpg"

    if score < args.threshold:
        annotate(img, (bx1, by1, bx2, by2), score, ann_path)
        print(f"未达到阈值: score={score:.3f} < {args.threshold:.2f}")
        if screen_path is not None:
            print(f"截屏: {screen_path}")
        print(f"标注图: {ann_path}")
        return

    cx = (bx1 + bx2) / 2.0
    cy = (by1 + by2) / 2.0
    norm_x = cx / W
    norm_y = cy / H

    annotate(img, (bx1, by1, bx2, by2), score, ann_path)
    print("检测成功（按钮级）：")
    print(f"- 屏幕尺寸: {W}x{H}")
    if screen_path is not None:
        print(f"- 截屏: {screen_path}")
    print(f"- 标注图: {ann_path}")
    print(f"- 匹配分数: {score:.3f}")
    print(f"- 框(像素): x1={bx1}, y1={by1}, x2={bx2}, y2={by2}")
    print(f"- 中心(像素): x={int(cx)}, y={int(cy)}")
    print(f"- 中心(归一化): x={norm_x:.4f}, y={norm_y:.4f}")


if __name__ == "__main__":
    main()