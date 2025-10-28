from ultralytics import YOLO
import os

# 确保img目录存在
os.makedirs('img', exist_ok=True)

# 加载 YOLO11-nano 模型（自动下载，若未缓存）
model = YOLO('runs/detect/train2/weights/best.pt')  # 'yolo11n' 即 YOLO11-nano 的简写

# 测试推理（可选）
imgname='11.jpg'
# results = model('1.png')  # 替换为你的图像路径
results = model(imgname)  # 替换为你的图像路径
results[0].save(filename=f'img/result_{imgname}')  # 将检测结果保存到img目录下
