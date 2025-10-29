import os
import time
import cv2
import numpy as np
from ultralytics import YOLO

# 说明与假设：
# - 将 STREAM_URL 替换为你的实时摄像头地址（支持的协议取决于 OpenCV/FFmpeg 的构建：常见为 rtsp://, http://, rtmp:// 等）。
# - 你提供的原始地址包含 Web 界面片段和 WebSocket 编码（#/play/...），这类链接有时不能直接被 OpenCV 打开。
#   如果无法打开，请确认设备是否提供直接的 RTSP/HTTP 流地址（例如 rtsp://... 或 http://.../video）。

# 配置：把下面的 STREAM_URL 替换为你的可用流地址
# 使用你提供的 RTSP 地址（更适合直接用 OpenCV/FFmpeg 打开）
STREAM_URL = 'rtsp://192.168.0.102:554/rtp/34020000001110000001_34020000001320000002?originTypeStr=rtp_push'

# 只识别的目标类别名称（按模型提供的 names 字典匹配），例如 'person'
TARGET_CLASS = 'person'

# 输出目录
os.makedirs('img', exist_ok=True)

# 加载 YOLO11-nano 模型（自动下载，若未缓存）
model = YOLO('yolo11n.pt')


def run_stream(stream_url: str, infer_interval: float = 5.0, save_every_n: int = 150, save_on_infer: bool = True):
    """从 stream_url 打开摄像头流，实时推理并显示检测结果。

    输入:
      - stream_url: 视频流 URL
      - infer_interval: 每隔多少秒执行一次推理（浮点数，默认 5.0 秒）
      - save_every_n: 每多少帧保存一张带检测框的图片到 img/ 目录
    返回: 无
    """

    # 尝试使用 FFMPEG 后端打开（在许多 OpenCV 构建上，对 RTSP 更稳定）
    cap = None
    try:
        cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(stream_url)
    except Exception:
        # 后备：直接用默认方式打开
        cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print(f"无法打开视频流: {stream_url}\n请确认 URL 可访问，或尝试使用 RTSP/HTTP/RTMP 地址。")
        return

    # 方便地将类别 id 映射为名称（如果模型提供）
    names = getattr(model, 'names', None) or {}

    frame_idx = 0
    last_t = time.time()
    last_infer_time = 0.0
    fps = 0.0
    # 上一次推理检测到的目标数量（用于周期保存判定）
    last_person_count = 0

    # 创建一个可调整大小的窗口作为弹窗，并默认全屏显示
    cv2.namedWindow('YOLO11 Live', cv2.WINDOW_NORMAL)
    try:
        cv2.setWindowProperty(
            'YOLO11 Live', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    except Exception:
        # 某些 OpenCV 构建/平台可能不支持该属性，忽略异常
        pass

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("读取帧失败，流可能中断。正在退出...")
                break

            frame_idx += 1

            # 计算并显示简单 FPS
            if frame_idx % 10 == 0:
                now = time.time()
                fps = 10.0 / (now - last_t + 1e-6)
                last_t = now

            # 每隔 infer_interval 秒做一次推理以节约资源（时间驱动，减少卡顿）
            now = time.time()
            if now - last_infer_time >= infer_interval:
                last_infer_time = now
                # ultralytics 的 model 接受 RGB ndarray
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = model(rgb)  # 返回 Results 列表
                r = results[0]

                # 绘制检测框（仅对 TARGET_CLASS，例如 'person'）到原始 BGR 帧上
                xyxy = getattr(r.boxes, 'xyxy', None)
                person_count = 0
                if xyxy is not None and len(r.boxes) > 0:
                    xyxy = r.boxes.xyxy.cpu().numpy()  # (N,4)
                    conf = r.boxes.conf.cpu().numpy()  # (N,)
                    cls = r.boxes.cls.cpu().numpy()    # (N,)
                    for (x1, y1, x2, y2), c, cl in zip(xyxy, conf, cls):
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        cls_id = int(cl)
                        cls_name = names.get(cls_id, str(cls_id))
                        # 仅处理指定类别
                        if cls_name.lower() != TARGET_CLASS.lower():
                            continue
                        person_count += 1
                        label = f"{cls_name} {c:.2f}"
                        cv2.rectangle(frame, (x1, y1),
                                      (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, max(y1 - 6, 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # 记录上一次推理是否识别到目标（供周期性保存使用）
                last_person_count = person_count

                # 在每次推理后保存带标注的截图（仅当识别到目标且启用时）
                if save_on_infer and person_count > 0:
                    # 按 年-月-日-时-分-秒 格式保存，字段之间用中划线连接，例如: 2025-10-29-14-30-05
                    timestr = time.strftime(
                        '%Y-%m-%d_%H-%M-%S', time.localtime())
                    fname = os.path.join('img', f'result_infer_{timestr}.jpg')
                    try:
                        cv2.imwrite(fname, frame)
                        print(
                            f"已保存推理截图: {fname} (检测到 {person_count} 个 {TARGET_CLASS})")
                    except Exception as e:
                        print(f"保存截图失败: {e}")

            # 叠加信息
            cv2.putText(frame, f"FPS:{fps:.1f}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            # 在窗口中显示流地址（简短版）
            try:
                short_url = stream_url if len(
                    stream_url) < 60 else '...' + stream_url[-57:]
            except Exception:
                short_url = stream_url
            cv2.putText(
                frame, short_url, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.imshow('YOLO11 Live', frame)

            # 旧的按帧保存逻辑仍保留为备份：仅当开启周期保存且上次推理检测到目标时才保存
            if not save_on_infer and frame_idx % save_every_n == 0 and last_person_count > 0:
                timestr = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
                fname = os.path.join('img', f'result_frame_{timestr}.jpg')
                try:
                    cv2.imwrite(fname, frame)
                except Exception:
                    pass

            # 按 'q' 或 ESC 键退出
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 27 是 ESC
                if key == 27:
                    print('检测终止 (用户按 ESC)。')
                else:
                    print('检测终止 (用户请求)。')
                break

    except KeyboardInterrupt:
        print('检测终止 (KeyboardInterrupt)。')
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # 你可以在这里替换 STREAM_URL 为更简洁的 RTSP/HTTP 地址以便 OpenCV 能直接打开
    run_stream(STREAM_URL, infer_interval=2.0, save_every_n=300)
