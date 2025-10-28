import argparse
import os
from pathlib import Path

import numpy as np
from PIL import ImageGrab
import cv2
from ultralytics import YOLO


ROOT = Path(__file__).parent


def grab_screen(bbox=None):
    img = ImageGrab.grab(bbox=bbox)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def main():
    parser = argparse.ArgumentParser(description="使用训练好的模型进行UI登录按钮检测")
    parser.add_argument("--model", type=str, default=str(ROOT / "runs" / "detect" / "train" / "weights" / "best.pt"), help="模型权重路径")
    parser.add_argument("--source", type=str, default="", help="图像/视频路径（留空则抓取屏幕）")
    parser.add_argument("--save", type=str, default=str(ROOT / "img" / "result_ui.jpg"), help="结果保存路径")
    args = parser.parse_args()

    model = YOLO(args.model)

    if args.source:
        result = model(args.source)
        annotated = result[0].plot()
    else:
        frame = grab_screen()
        result = model(frame)
        annotated = result[0].plot()

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    cv2.imwrite(args.save, annotated)
    print("检测完成，结果保存到：", args.save)


if __name__ == "__main__":
    main()