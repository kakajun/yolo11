import os
import time
import csv
import sys
from datetime import datetime

import psutil
import numpy as np
import cv2
from PIL import ImageGrab
from ultralytics import YOLO

# 可选依赖：pywin32、uiautomation（未安装也能基本监控进程状态）
try:
    import win32gui  # 获取前台窗口标题
    import win32con
except Exception:
    win32gui = None

try:
    import uiautomation as auto  # 通过 UI 自动化尝试读取地址栏 URL
except Exception:
    auto = None


LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
CSV_PATH = os.path.join(LOG_DIR, 'edge_metrics.csv')
IMG_DIR_DEFAULT = os.path.join(os.path.dirname(__file__), 'img')


def ensure_log_setup():
    os.makedirs(LOG_DIR, exist_ok=True)
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'running', 'cpu_percent', 'memory_mb', 'active_title', 'active_url'
            ])


def find_edge_processes():
    procs = []
    for p in psutil.process_iter(['pid', 'name']):
        try:
            name = (p.info.get('name') or '').lower()
            if name == 'msedge.exe':
                procs.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return procs


def get_foreground_title():
    if win32gui is None:
        return None
    try:
        hwnd = win32gui.GetForegroundWindow()
        if not hwnd:
            return None
        title = win32gui.GetWindowText(hwnd)
        return title
    except Exception:
        return None


def get_foreground_edge_rect_and_title():
    """返回当前前台 Edge 窗口矩形 (left, top, right, bottom) 与标题；若非 Edge 或不可获取则返回 (None, title)。"""
    if win32gui is None:
        return None, None
    try:
        hwnd = win32gui.GetForegroundWindow()
        if not hwnd:
            return None, None
        title = win32gui.GetWindowText(hwnd)
        if 'Edge' not in title:
            return None, title
        try:
            rect = win32gui.GetWindowRect(hwnd)
            return rect, title
        except Exception:
            return None, title
    except Exception:
        return None, None


def grab_edge_frame():
    """抓取当前前台 Edge 窗口图像，若无法定位窗口则抓取整屏。返回 (frame_bgr, title, rect)。"""
    rect, title = get_foreground_edge_rect_and_title()
    try:
        if rect:
            left, top, right, bottom = rect
            img = ImageGrab.grab(bbox=(left, top, right, bottom))
        else:
            # 仅监控 Edge，不再回退到全屏
            return None, (title or ''), None
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return frame, (title or ''), rect
    except Exception:
        return None, (title or ''), rect


def get_edge_active_url_via_uia():
    """
    通过 UIAutomation 尝试获取当前活动 Edge 窗口的地址栏 URL。
    可能因系统语言/Edge 版本差异而失败，失败时返回 None。
    """
    if auto is None:
        return None
    try:
        # 寻找顶层 Edge 窗口（Chromium 基础窗口类名为 Chrome_WidgetWin_1）
        top_windows = auto.GetRootControl().GetChildren()
        edge_windows = [w for w in top_windows if getattr(w, 'ClassName', '') == 'Chrome_WidgetWin_1' and 'Edge' in getattr(w, 'Name', '')]
        target = edge_windows[0] if edge_windows else None

        if target is None:
            return None

        # 不同语言下地址栏控件名称可能不同，常见为中文/英文
        toolbar_names = ['地址和搜索栏', 'Address and search bar']
        for tb_name in toolbar_names:
            try:
                toolbar = target.ToolBarControl(Name=tb_name)
                if toolbar.Exists(0, 0):
                    edit = toolbar.EditControl()
                    if edit and edit.Exists(0, 0):
                        try:
                            return edit.GetValuePattern().Value
                        except Exception:
                            # 备用读取
                            return getattr(edit, 'Text', None)
            except Exception:
                continue

        # 兜底：尝试在窗口内搜索可能的 Edit 控件获取 URL（不保证稳定）
        try:
            edit = target.EditControl(searchDepth=20)
            if edit and edit.Exists(0, 0):
                try:
                    val = edit.GetValuePattern().Value
                except Exception:
                    val = getattr(edit, 'Text', None)
                if val and (val.startswith('http') or '.' in val):
                    return val
        except Exception:
            pass

        return None
    except Exception:
        return None


def sample_once():
    procs = find_edge_processes()
    running = len(procs) > 0

    cpu_percent = 0.0
    mem_mb = 0.0
    if running:
        # 轻量阻塞以更准确计算 CPU
        try:
            cpu_percent = sum(p.cpu_percent(interval=0.1) for p in procs)
        except Exception:
            cpu_percent = 0.0
        try:
            mem_mb = sum(p.memory_info().rss for p in procs) / (1024 * 1024)
        except Exception:
            mem_mb = 0.0

    # 仅记录 Edge 前台窗口标题
    rect, title = get_foreground_edge_rect_and_title()
    title = title or ''
    active_url = None
    if title:
        active_url = get_edge_active_url_via_uia()

    ts = datetime.now().isoformat(timespec='seconds')
    with open(CSV_PATH, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([ts, int(running), f'{cpu_percent:.2f}', f'{mem_mb:.2f}', title, active_url or ''])

    # 控制台打印简要信息
    print(f"[{ts}] Edge running={running} CPU={cpu_percent:.2f}% MEM={mem_mb:.2f}MB Title='{title}' URL='{active_url or ''}'")


def main():
    if sys.platform != 'win32':
        print('此脚本仅支持 Windows 环境（用于监控 Microsoft Edge）。')
        return

    import argparse
    parser = argparse.ArgumentParser(description='监控 Microsoft Edge：进程资源与窗口信息，并可对画面进行 YOLO 标注')
    parser.add_argument('--interval', type=float, default=2.0, help='采样间隔（秒），默认 2s')
    parser.add_argument('--mark', action='store_true', default=True, help='开启 YOLO 标注（默认开启）')
    parser.add_argument('--display', action='store_true', default=False, help='显示标注窗口（按 q 退出），默认关闭')
    parser.add_argument('--save-dir', type=str, default=IMG_DIR_DEFAULT, help='保存标注图像的目录，默认 yolo11_proj/img')
    parser.add_argument('--model', type=str, default='yolo11n.pt', help='YOLO 模型路径，默认使用 yolo11n.pt')
    parser.add_argument('--focus-edge', action='store_true', default=True, help='每次采样前激活 Edge 窗口（默认开启）')
    args = parser.parse_args()

    ensure_log_setup()
    os.makedirs(args.save_dir, exist_ok=True)

    print(f'开始监控 Edge，日志输出：{CSV_PATH}，采样间隔：{args.interval}s')
    print('提示：未安装 pywin32 或 uiautomation 时无法获取前台标题或 URL，但资源监控与截图仍可用。')

    # 初始化 YOLO 模型
    model = None
    if args.mark:
        model_path = args.model
        if not os.path.isabs(model_path):
            # 相对于脚本目录定位模型文件
            model_path = os.path.join(os.path.dirname(__file__), model_path)
        try:
            model = YOLO(model_path)
            print(f'YOLO 标注已开启，模型：{model_path}')
        except Exception as e:
            print(f'加载 YOLO 模型失败（已禁用标注）：{e}')
            model = None

    def _find_edge_hwnd_visible():
        if win32gui is None:
            return None, None
        try:
            hwnd_title_pairs = []
            def _enum_cb(hwnd, acc):
                try:
                    if win32gui.IsWindowVisible(hwnd):
                        title = win32gui.GetWindowText(hwnd)
                        cls = win32gui.GetClassName(hwnd)
                        if 'Edge' in (title or '') and cls == 'Chrome_WidgetWin_1':
                            acc.append((hwnd, title))
                except Exception:
                    pass
            win32gui.EnumWindows(_enum_cb, hwnd_title_pairs)
            return hwnd_title_pairs[0] if hwnd_title_pairs else (None, None)
        except Exception:
            return None, None

    def activate_edge_window():
        # 优先使用 UIAutomation 激活窗口
        if auto is not None:
            try:
                tops = auto.GetRootControl().GetChildren()
                for w in tops:
                    if getattr(w, 'ClassName', '') == 'Chrome_WidgetWin_1' and 'Edge' in getattr(w, 'Name', ''):
                        try:
                            w.SetActive()
                            return True
                        except Exception:
                            break
            except Exception:
                pass
        # 退化到 win32gui
        hwnd, _ = _find_edge_hwnd_visible()
        if hwnd and win32gui is not None:
            try:
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                win32gui.SetForegroundWindow(hwnd)
                return True
            except Exception:
                return False
        return False

    try:
        while True:
            if args.focus_edge:
                activate_edge_window()
            sample_once()
            # YOLO 标注：抓取当前前台 Edge 画面并绘制检测结果
            if args.mark and model is not None:
                frame, title, rect = grab_edge_frame()
                if frame is not None and rect is not None:
                    try:
                        results = model(frame, verbose=False)
                        annotated = results[0].plot()
                        # 保存到文件
                        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                        out_path = os.path.join(args.save_dir, f'edge_{ts}.jpg')
                        cv2.imwrite(out_path, annotated)
                        # 可选显示
                        if args.display:
                            cv2.imshow('Edge YOLO 标注', annotated)
                            # 按 q 退出显示循环（同时停止程序）
                            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                                raise KeyboardInterrupt
                    except Exception as e:
                        print(f'YOLO 标注失败：{e}')
            time.sleep(max(0.5, args.interval))
    except KeyboardInterrupt:
        print('\n已停止监控。')
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == '__main__':
    main()