import os
from pathlib import Path
import argparse

from ultralytics import YOLO


ROOT = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser(description="训练自定义UI登录按钮检测模型（YOLO11）")
    parser.add_argument("--data", type=str, default=str(ROOT / "dataset" / "ui_login.yaml"), help="数据配置YAML路径")
    parser.add_argument("--model", type=str, default=str(ROOT / "yolo11n.pt"), help="预训练模型权重（如 yolo11s.pt 更稳）")
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=640, help="训练图像尺寸")
    parser.add_argument("--batch", type=int, default=16, help="批大小（CPU上可减小）")
    parser.add_argument("--device", type=str, default="cpu", help="设备：cpu 或 cuda:0 等")
    parser.add_argument("--workers", type=int, default=2, help="DataLoader workers 数量")
    parser.add_argument("--mosaic", type=float, default=0.2, help="mosaic增强强度，单目标建议降低")

    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"未找到数据配置: {data_path}")

    model_path = Path(args.model)
    if not model_path.exists():
        # Ultralytics会自动下载官方权重；但这里提示一下
        print(f"提示：未找到 {model_path}，将尝试使用Ultralytics自动下载预训练权重。")

    print("开始训练 ...")
    print(f"- data: {data_path}")
    print(f"- model: {model_path}")
    print(f"- epochs: {args.epochs}, imgsz: {args.imgsz}, batch: {args.batch}, device: {args.device}")

    model = YOLO(str(model_path))
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        mosaic=args.mosaic,
        verbose=True,
    )

    # 结果权重路径
    try:
        weights = results.get('save_dir', None)
    except Exception:
        weights = None
    print("训练完成！")
    print("权重保存目录：", weights or "请查看 runs/detect/train/weights/")


if __name__ == "__main__":
    main()