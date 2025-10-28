Edge 监控脚本
================

功能概述
- 监控 Microsoft Edge 是否运行、CPU 使用率、内存占用。
- 读取当前前台 Edge 窗口标题，并尽力读取地址栏 URL（依赖 UIAutomation）。
- 将采样结果以 CSV 形式写入 `yolo11_proj/logs/edge_metrics.csv`，同时在控制台输出简要信息。

环境依赖
- 必需：`psutil`（已在 requirements.txt）
- 可选：`pywin32`（用于获取前台窗口标题）、`uiautomation`（用于读取地址栏 URL）

安装依赖
```
pip install -r requirements.txt
```

运行示例（仅监控资源）
```
python yolo11_proj/edge_monitor.py --interval 2 --mark false
```

运行示例（启用 YOLO 标注）
```
python yolo11_proj/edge_monitor.py --interval 2 --mark --display --save-dir yolo11_proj/img --model yolo11n.pt
```
说明：
- `--mark` 开启标注（默认开启）。如需关闭请传 `--mark false`。
- `--display` 打开窗口显示标注画面，按 `q` 退出显示。
- `--save-dir` 指定标注图片保存目录，默认 `yolo11_proj/img`。
- `--model` 指定 YOLO 模型文件路径，默认脚本目录下的 `yolo11n.pt`。
- `--focus-edge` 每次采样前激活并聚焦 Edge 窗口（默认开启）。

说明与注意事项
- 如果未安装 `pywin32` 或 `uiautomation`，脚本仍可监控 Edge 进程与资源，但无法获取前台标题或 URL。
- 不同系统语言/Edge 版本下地址栏控件名称可能不同，当前脚本支持中文“地址和搜索栏”和英文“Address and search bar”，其他语言可能无法读取 URL。
- 日志文件路径：`yolo11_proj/logs/edge_metrics.csv`，字段依次为：`timestamp,running,cpu_percent,memory_mb,active_title,active_url`。

YOLO 标注说明
- 当前标注对象为前台 Edge 窗口（若无法定位则退化为全屏截图）。
- 标注图每次采样都会保存为 `edge_YYYYmmdd_HHMMSS.jpg` 到 `--save-dir`。
- 模型默认使用 `yolo11n.pt`（nano），可按需更换更精确的模型。