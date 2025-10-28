import argparse
import os
import time
from datetime import datetime
from pathlib import Path
import subprocess

import numpy as np
from PIL import ImageGrab
import cv2
from ultralytics import YOLO

# 可选依赖（若不存在则退化为固定等待）
try:
    import win32gui
    import win32con
    import win32process
except Exception:
    win32gui = None
    win32con = None
    win32process = None


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


def main():
    parser = argparse.ArgumentParser(description="启动指定程序，等待加载后截屏并用YOLO识别登录按钮位置")
    parser.add_argument("--exe", type=str, default=r"D:\\Program Files (x86)\\5211game\\11Loader.exe", help="要启动的程序路径")
    parser.add_argument("--model", type=str, default=str(ROOT / "runs" / "detect" / "train" / "weights" / "best.pt"), help="YOLO 权重路径")
    parser.add_argument("--save-dir", type=str, default=str(ROOT / "screen"), help="截屏与标注保存目录")
    parser.add_argument("--timeout", type=float, default=30.0, help="等待窗口出现的超时时间（秒）")
    parser.add_argument("--wait", type=float, default=5.0, help="窗口出现后额外等待的时间（秒）")
    parser.add_argument("--no-launch", action="store_true", help="不启动程序，仅当前屏幕上检测")
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

    frame, screen_path = grab_fullscreen(save_dir)

    # 检测并输出位置
    model_path = args.model
    if not os.path.exists(model_path):
        fallback = ROOT / "yolo11n.pt"
        print(f"未找到模型 {model_path}，回退到 {fallback}（可能无法识别登录按钮）")
        model_path = str(fallback)

    info, ann_path = detect_login(model_path, frame, save_dir)
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
    print(f"- 置信度: {conf:.3f}")
    print(f"- 截屏: {screen_path}")
    print(f"- 标注图: {ann_path}")


if __name__ == "__main__":
    main()