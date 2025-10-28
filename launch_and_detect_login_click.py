import argparse
import os
import time
from datetime import datetime
from pathlib import Path
import subprocess
import ctypes

import numpy as np
from PIL import ImageGrab
import cv2


# 引入按钮级模板匹配的辅助函数
from template_detect_button import (
    build_button_mask_from_template,
    multi_scale_masked_match,
)

# 可选依赖（若不存在则退化为固定等待）
try:
    import win32gui
    import win32con
    import win32process
    import win32api
except Exception:
    win32gui = None
    win32con = None
    win32process = None
    win32api = None


ROOT = Path(__file__).parent


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def start_program(exe_path: str) -> subprocess.Popen:
    print(f"启动程序: {exe_path}")
    return subprocess.Popen([exe_path], shell=False)


def wait_for_window(pid: int, timeout: float = 30.0):
    """等待指定进程的可见窗口出现，返回 hwnd；若不可用或超时则返回 None。"""
    if win32gui is None or win32process is None:
        print("未安装 pywin32，无法确认窗口，退化为固定等待。")
        time.sleep(min(timeout, 8.0))
        return None

    start = time.time()
    target_hwnd = None

    def _enum_cb(hwnd, acc):
        try:
            if win32gui.IsWindowVisible(hwnd):
                _, wpid = win32process.GetWindowThreadProcessId(hwnd)
                if wpid == pid:
                    acc.append(hwnd)
        except Exception:
            pass

    while time.time() - start < timeout:
        found = []
        try:
            win32gui.EnumWindows(_enum_cb, found)
        except Exception:
            found = []
        if found:
            target_hwnd = found[0]
            break
        time.sleep(0.5)

    if target_hwnd:
        print(f"窗口已出现: hwnd={target_hwnd}")
    else:
        print("未能确认窗口（可能已出现但不可枚举），继续流程。")
    return target_hwnd


def activate_window(hwnd):
    if win32gui is None:
        return False
    try:
        if win32con is not None:
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        win32gui.SetForegroundWindow(hwnd)
        return True
    except Exception:
        return False


def grab_fullscreen(save_dir: Path):
    img = ImageGrab.grab()
    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = save_dir / f"screen_{ts}.jpg"
    cv2.imwrite(str(out_path), frame)
    print(f"已保存截屏: {out_path}")
    return frame, out_path


def detect_login(model_path: str, frame: np.ndarray, save_dir: Path):
    model = YOLO(model_path)
    results = model(frame, verbose=False)
    r0 = results[0]
    annotated = r0.plot()
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    ann_path = save_dir / f"annot_{ts}.jpg"
    cv2.imwrite(str(ann_path), annotated)
    print(f"已保存标注图: {ann_path}")

    boxes = getattr(r0, 'boxes', None)
    if boxes is None or len(boxes) == 0:
        return None, ann_path

    # 选择置信度最高的框
    conf = boxes.conf.detach().cpu().numpy()
    xyxy = boxes.xyxy.detach().cpu().numpy()
    idx = int(np.argmax(conf))

    x1, y1, x2, y2 = xyxy[idx].tolist()
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    h, w = frame.shape[:2]
    norm_x = cx / w
    norm_y = cy / h

    info = {
        'center_px': (int(cx), int(cy)),
        'center_norm': (float(norm_x), float(norm_y)),
        'bbox_px': (int(x1), int(y1), int(x2), int(y2)),
        'screen_size': (w, h),
        'conf': float(conf[idx]),
    }
    return info, ann_path


def move_mouse_and_click(x: int, y: int, delay: float = 0.08, double: bool = False):
    """将鼠标移动到 (x, y) 并点击；优先使用 pywin32，回退到 ctypes。"""
    x = int(x)
    y = int(y)
    # 常量：左键按下/抬起
    LEFTDOWN = getattr(win32con, 'MOUSEEVENTF_LEFTDOWN', 0x0002)
    LEFTUP = getattr(win32con, 'MOUSEEVENTF_LEFTUP', 0x0004)

    try:
        if win32api is not None:
            win32api.SetCursorPos((x, y))
            time.sleep(max(0.0, delay))
            win32api.mouse_event(LEFTDOWN, 0, 0, 0, 0)
            win32api.mouse_event(LEFTUP, 0, 0, 0, 0)
            if double:
                time.sleep(0.08)
                win32api.mouse_event(LEFTDOWN, 0, 0, 0, 0)
                win32api.mouse_event(LEFTUP, 0, 0, 0, 0)
            return True
    except Exception:
        pass

    # 回退：ctypes 调用 user32
    try:
        user32 = ctypes.windll.user32
        user32.SetCursorPos(x, y)
        time.sleep(max(0.0, delay))
        user32.mouse_event(LEFTDOWN, 0, 0, 0, 0)
        user32.mouse_event(LEFTUP, 0, 0, 0, 0)
        if double:
            time.sleep(0.08)
            user32.mouse_event(LEFTDOWN, 0, 0, 0, 0)
            user32.mouse_event(LEFTUP, 0, 0, 0, 0)
        return True
    except Exception:
        return False


def detect_login_by_button_template(template_path: Path, frame: np.ndarray, save_dir: Path,
                                    threshold: float = 0.6,
                                    min_scale: float = 0.6,
                                    max_scale: float = 1.6,
                                    steps: int = 21,
                                    h_low: int = 90,
                                    h_high: int = 130,
                                    s_low: int = 60,
                                    v_low: int = 50):
    """使用带掩码的模板匹配，仅输出登录按钮的精确框与中心。"""
    if not template_path.exists():
        raise FileNotFoundError(f"模板不存在: {template_path}")
    tpl_bgr = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
    if tpl_bgr is None:
        raise FileNotFoundError(f"无法读取模板图像: {template_path}")

    button_mask, mask_bbox = build_button_mask_from_template(
        tpl_bgr, h_low=h_low, h_high=h_high, s_low=s_low, v_low=v_low
    )
    if button_mask is None:
        return None, None

    tpl_gray = cv2.cvtColor(tpl_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    scales = np.linspace(min_scale, max_scale, steps)
    best = multi_scale_masked_match(gray, tpl_gray, button_mask, scales)
    score = best['score']
    x, y = best['loc']
    s = best['scale']

    bx1 = x + int(mask_bbox[0] * s)
    by1 = y + int(mask_bbox[1] * s)
    bx2 = bx1 + int(mask_bbox[2] * s)
    by2 = by1 + int(mask_bbox[3] * s)
    h, w = frame.shape[:2]

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    ann_path = save_dir / f"annot_button_{ts}.jpg"
    # 直接保存按钮级标注图
    vis = frame.copy()
    cv2.rectangle(vis, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
    cv2.putText(vis, f"score={score:.2f}", (bx1, max(0, by1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imwrite(str(ann_path), vis)

    if score < threshold:
        return None, ann_path

    cx = (bx1 + bx2) / 2.0
    cy = (by1 + by2) / 2.0
    norm_x = cx / w
    norm_y = cy / h

    info = {
        'center_px': (int(cx), int(cy)),
        'center_norm': (float(norm_x), float(norm_y)),
        'bbox_px': (int(bx1), int(by1), int(bx2), int(by2)),
        'screen_size': (w, h),
        'conf': float(score),  # 与 YOLO 接口保持一致，使用 score 作为置信度
    }
    return info, ann_path


def main():
    parser = argparse.ArgumentParser(description="启动/截屏后进行登录按钮定位并自动点击（支持 YOLO 或按钮级模板匹配）")
    parser.add_argument("--exe", type=str, default=r"D:\\Program Files (x86)\\5211game\\11Loader.exe", help="要启动的程序路径")
    parser.add_argument("--model", type=str, default=str(ROOT / "runs" / "detect" / "train" / "weights" / "best.pt"), help="YOLO 权重路径")
    parser.add_argument("--save-dir", type=str, default=str(ROOT / "screen"), help="截屏与标注保存目录")
    parser.add_argument("--timeout", type=float, default=30.0, help="等待窗口出现的超时时间（秒）")
    parser.add_argument("--wait", type=float, default=5.0, help="窗口出现后额外等待的时间（秒）")
    parser.add_argument("--no-launch", action="store_true", help="不启动程序，仅当前屏幕上检测")
    parser.add_argument("--source", type=str, default="", help="指定图像文件；留空则使用截屏")
    # 检测方式与模板参数
    parser.add_argument("--detect", type=str, choices=["yolo", "button"], default="button", help="检测方式：yolo 或 button(模板遮罩)")
    parser.add_argument("--template", type=str, default=str(ROOT / "login.png"), help="按钮模板图（包含按钮）")
    parser.add_argument("--threshold", type=float, default=0.6, help="模板匹配阈值")
    parser.add_argument("--min-scale", type=float, default=0.6, help="模板最小缩放比")
    parser.add_argument("--max-scale", type=float, default=1.6, help="模板最大缩放比")
    parser.add_argument("--steps", type=int, default=21, help="缩放搜索步数")
    parser.add_argument("--h-low", type=int, default=90, help="HSV H 下限 (蓝) 0-179")
    parser.add_argument("--h-high", type=int, default=130, help="HSV H 上限 (蓝) 0-179")
    parser.add_argument("--s-low", type=int, default=60, help="HSV S 下限 0-255")
    parser.add_argument("--v-low", type=int, default=50, help="HSV V 下限 0-255")
    # 点击控制
    parser.add_argument("--no-click", action="store_true", help="禁用自动点击（默认启用）")
    parser.add_argument("--double", action="store_true", help="使用双击")
    parser.add_argument("--click-delay", type=float, default=0.08, help="移动到中心后点击的等待秒数")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    ensure_dir(save_dir)

    proc = None
    hwnd = None

    if not args.no_launch:
        proc = start_program(args.exe)
        hwnd = wait_for_window(proc.pid, timeout=args.timeout)
        if hwnd:
            activate_window(hwnd)
        print(f"等待 {args.wait}s 以完成加载...")
        time.sleep(max(0.1, args.wait))
    else:
        print("跳过启动程序，直接截取当前屏幕。")

    # 获取检测帧：若提供 source 则直接读取，否则截屏
    if args.source:
        frame = cv2.imread(args.source, cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(f"无法读取图像: {args.source}")
        screen_path = args.source
        print(f"使用指定图像: {screen_path}")
    else:
        frame, screen_path = grab_fullscreen(save_dir)

    # 检测并输出位置
    if args.detect == "yolo":
        model_path = args.model
        if not os.path.exists(model_path):
            fallback = ROOT / "yolo11n.pt"
            print(f"未找到模型 {model_path}，回退到 {fallback}（可能无法识别登录按钮）")
            model_path = str(fallback)
        info, ann_path = detect_login(model_path, frame, save_dir)
    else:
        info, ann_path = detect_login_by_button_template(
            template_path=Path(args.template),
            frame=frame,
            save_dir=save_dir,
            threshold=args.threshold,
            min_scale=args.min_scale,
            max_scale=args.max_scale,
            steps=args.steps,
            h_low=args.h_low,
            h_high=args.h_high,
            s_low=args.s_low,
            v_low=args.v_low,
        )
    if info is None:
        print("未检测到登录按钮。")
        return

    center_px = info['center_px']
    center_norm = info['center_norm']
    bbox_px = info['bbox_px']
    screen_size = info['screen_size']
    conf = info['conf']

    print("检测结果：")
    print(f"- 屏幕尺寸: {screen_size[0]}x{screen_size[1]}")
    print(f"- 框(像素): x1={bbox_px[0]}, y1={bbox_px[1]}, x2={bbox_px[2]}, y2={bbox_px[3]}")
    print(f"- 中心(像素): x={center_px[0]}, y={center_px[1]}")
    print(f"- 中心(归一化): x={center_norm[0]:.4f}, y={center_norm[1]:.4f}")
    print(f"- 置信度/匹配分数: {conf:.3f}")
    print(f"- 截屏: {screen_path}")
    print(f"- 标注图: {ann_path}")

    # 自动点击：默认启用，除非 --no-click
    if not args.no_click:
        ok = move_mouse_and_click(center_px[0], center_px[1], delay=args.click_delay, double=args.double)
        if ok:
            print(f"已移动鼠标到中心点并点击: x={center_px[0]}, y={center_px[1]}{' (双击)' if args.double else ''}")
        else:
            print("尝试点击失败：你的系统可能缺少 pywin32，且 ctypes 调用未成功。")


if __name__ == "__main__":
    main()