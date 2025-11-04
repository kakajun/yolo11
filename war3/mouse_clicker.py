import time
import math
import random
import ctypes
from typing import Tuple

try:
    import win32api
    import win32con
    import win32gui
except Exception:
    win32api = None
    win32con = None
    win32gui = None


def _get_cursor_pos() -> Tuple[int, int]:
    """Return current cursor position (x, y)."""
    try:
        if win32api is not None:
            return win32api.GetCursorPos()
    except Exception:
        pass

    # ctypes fallback
    pt = ctypes.wintypes.POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y


def _set_cursor_pos(x: int, y: int):
    try:
        if win32api is not None:
            win32api.SetCursorPos((int(x), int(y)))
            return
    except Exception:
        pass
    # ctypes fallback
    ctypes.windll.user32.SetCursorPos(int(x), int(y))


def _mouse_event(down: bool = True):
    LEFTDOWN = getattr(win32con, 'MOUSEEVENTF_LEFTDOWN',
                       0x0002) if win32con is not None else 0x0002
    LEFTUP = getattr(win32con, 'MOUSEEVENTF_LEFTUP',
                     0x0004) if win32con is not None else 0x0004
    try:
        if win32api is not None:
            if down:
                win32api.mouse_event(LEFTDOWN, 0, 0, 0, 0)
            else:
                win32api.mouse_event(LEFTUP, 0, 0, 0, 0)
            return
    except Exception:
        pass

    user32 = ctypes.windll.user32
    if down:
        user32.mouse_event(LEFTDOWN, 0, 0, 0, 0)
    else:
        user32.mouse_event(LEFTUP, 0, 0, 0, 0)


def move_mouse_smooth_to(x: int, y: int, duration: float = 0.35, steps: int = 25,
                         jitter: float = 1.2, ease: str = 'ease_out_quad') -> None:
    """
    平滑移动鼠标到目标 (x, y)。

    - duration: 整体移动耗时（秒）
    - steps: 划分多少步
    - jitter: 每步的随机微抖幅度（像素），用于模拟人的不精确
    - ease: 缓动类型（支持 'linear', 'ease_in_quad', 'ease_out_quad', 'ease_in_out_cubic'）
    """
    x = int(x)
    y = int(y)
    try:
        cx, cy = _get_cursor_pos()
    except Exception:
        cx, cy = x, y

    def ease_fn(t: float) -> float:
        if ease == 'linear':
            return t
        if ease == 'ease_in_quad':
            return t * t
        if ease == 'ease_out_quad':
            return t * (2 - t)
        if ease == 'ease_in_out_cubic':
            if t < 0.5:
                return 4 * t * t * t
            return 1 - pow(-2 * t + 2, 3) / 2
        return t

    steps = max(1, int(steps))
    duration = max(0.0, float(duration))
    sleep_per = duration / steps if steps > 0 else 0.0

    for i in range(1, steps + 1):
        t = i / steps
        et = ease_fn(t)
        nx = cx + (x - cx) * et
        ny = cy + (y - cy) * et

        # 添加基于进度的抖动（靠近终点抖动减小）
        factor = (1 - abs(0.5 - t) * 2)  # 在中间抖动最大，接近两端较小
        jitter_amp = jitter * (1 - t) * 0.6 + jitter * 0.2
        rx = nx + random.uniform(-jitter_amp * factor, jitter_amp * factor)
        ry = ny + random.uniform(-jitter_amp * factor, jitter_amp * factor)

        _set_cursor_pos(int(round(rx)), int(round(ry)))
        # 增加少量随机化的等待，使轨迹更自然
        time.sleep(sleep_per * (0.9 + random.random() * 0.2))

    # 最终确保精确落在目标点
    _set_cursor_pos(x, y)


def click_at(double: bool = False, inter_delay: float = 0.08) -> bool:
    """对当前鼠标位置执行单击或双击。"""
    try:
        _mouse_event(down=True)
        _mouse_event(down=False)
        if double:
            time.sleep(inter_delay)
            _mouse_event(down=True)
            _mouse_event(down=False)
        return True
    except Exception:
        return False


def move_mouse_and_click(x: int, y: int, delay: float = 0.08, double: bool = False,
                         duration: float = 0.35, steps: int = 25) -> bool:
    """
    兼容原来接口的高层封装：先平滑移动，再等待 delay，之后点击（支持双击）。

    返回 True 表示点击动作已调起（不保证目标程序响应）。
    """
    try:
        move_mouse_smooth_to(x, y, duration=duration, steps=steps)
        time.sleep(max(0.0, delay))
        ok = click_at(double=double)
        return ok
    except Exception:
        # 退化为老的快速实现（尽量保证功能可用）
        try:
            _set_cursor_pos(x, y)
            time.sleep(max(0.0, delay))
            _mouse_event(down=True)
            _mouse_event(down=False)
            if double:
                time.sleep(0.08)
                _mouse_event(down=True)
                _mouse_event(down=False)
            return True
        except Exception:
            return False
