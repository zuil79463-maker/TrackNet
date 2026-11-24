# TrackNetV3 · RealSense D435i 实时羽毛球跟踪

基于 [官方 TrackNetV3](https://github.com/qaz812345/TrackNetV3) 的摄像头推理版本。默认通过 `pyrealsense2` SDK 选择已连接的 D435i，读取 1280×720、30 FPS 的 BGR 彩色流。仍支持普通 USB 摄像头和 RTSP/HTTP 视频流。保留 MIT 许可证和 Git 历史。

RealSense 采集参照 [官方 Python 彩色采集示例](https://github.com/realsenseai/librealsense/blob/master/wrappers/python/examples/opencv_viewer_example.py)。本工程仅开启彩色流，输出二维像素坐标；未开启深度和 IMU，不输出三维坐标。D435i 不在时会报错，不会自动切换到笔记本摄像头。

## 启动

在 `E:\ControlAlgorithm\Project\host` 目录运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\TrackNetV3\run_tracking.ps1

# 使用 D435i，并保存坐标
powershell -ExecutionPolicy Bypass -File .\TrackNetV3\run_tracking.ps1 -Csv .\output\d435i.csv

# 枚举 RealSense 设备（不打开流、不加载模型）
powershell -ExecutionPolicy Bypass -File .\TrackNetV3\run_tracking.ps1 -ListDevices

# 连接多台 D435i 时，按序列号选择其中一台
powershell -ExecutionPolicy Bypass -File .\TrackNetV3\run_tracking.ps1 -Serial '填写设备序列号'

# 可选：显式切换到普通摄像头 0（Windows 默认使用 DirectShow）
powershell -ExecutionPolicy Bypass -File .\TrackNetV3\run_tracking.ps1 -Camera 0

# 网络摄像头
powershell -ExecutionPolicy Bypass -File .\TrackNetV3\run_tracking.ps1 -Camera 'rtsp://192.168.1.10:554/stream'
```

启动后先采集 60 帧估计背景，请固定摄像头。随后显示羽毛球位置（红圈）和最近检测轨迹（黄点）。`B` 重新采集背景，`Q` / `Esc` 或关闭预览窗口退出；控制台 `Ctrl+C` 也可退出。无显示器运行可加 `-Headless`。

## 实时行为

官方模型使用连续 8 帧及背景图。采集线程只保留最新 8 帧，每次推理取最新窗口，仅输出该窗口最后一帧的坐标，不等待后续帧。推理速度慢于采集时会跳过旧窗口，避免无限排队；因此 CSV 帧号可能不连续，坐标对应的是该帧图像，不能当成每个采集帧都有结果。

当前为 TrackNet 在线定位流程，已移除离线 InpaintNet 修复及全视频融合。轨迹点表示实际检测结果，遮挡缺失不会补画。预览对应本次预测的帧；画面显示结果更新率、推理耗时和从主机收到该帧到输出结果的时间。后者不包含摄像头或网络内部缓冲延迟，也不是传感器曝光时间。

默认请求 1280×720、30 FPS，启动日志打印实际模式。RealSense 不支持所请求的模式时会报错；普通摄像头可能采用其他参数。D435i 可通过启动脚本修改模式：

```powershell
powershell -ExecutionPolicy Bypass -File .\TrackNetV3\run_tracking.ps1 -Width 640 -Height 480 -Fps 30

# 或使用 Python 入口
.\.venv\Scripts\python.exe .\TrackNetV3\predict.py --source realsense --width 1280 --height 720 --fps 30 --csv .\output\d435i.csv
```

性能取决于显卡、摄像头及曝光条件，当前实现不保证 30 次推理/秒。建议固定单球场机位，减少运动模糊。更换机位后按 `B` 重建背景。模型按比赛画面训练，实际摄像头识别精度需用现场画面验证。

## 坐标输出

控制台逐条输出检测结果。可选 CSV 包含 `Frame,TimestampUTC,Visibility,X,Y,PeakScore,InferenceMs,ResultAgeMs`。

- `Frame` 从 0 开始，包含背景采集阶段；跳号表示未对中间帧输出定位。
- `TimestampUTC` 为主机收到图像的 UTC 时间；`X,Y` 为原图像素坐标，左上角为原点。
- `Visibility=0` 时 `X,Y=0`，表示未检测到羽毛球。
- `PeakScore` 为热力图最大值，不是经过校准的概率。
- 轨迹长度按检测结果条数计，不代表固定时长。
- CSV 路径已存在时会覆盖；默认不保存 CSV 或视频。

## 环境和权重

本机 `.venv` 复用 `E:\Anaconda\install\envs\pythonProject` 中现有的 CUDA PyTorch。运行无需激活环境。重建环境：

```powershell
& 'E:\Anaconda\install\envs\pythonProject\python.exe' -m venv --system-site-packages .venv
.\.venv\Scripts\python.exe -m pip install -r .\TrackNetV3\requirements.txt
```

`ckpts/TrackNet_best.pt` 已安装。重新下载：

```powershell
.\.venv\Scripts\python.exe .\TrackNetV3\download_weights.py
# 或导入从官方 README 下载的模型压缩包
.\.venv\Scripts\python.exe .\TrackNetV3\download_weights.py --archive 'D:\Downloads\TrackNetV3_ckpts.zip'
```

保留的权重 SHA256：`df867641a02712b021f04548ff4b1208ddfdb47f629ab2094ceb978667e83b1a`。

## 验证

```powershell
.\.venv\Scripts\python.exe .\TrackNetV3\test_realtime.py
```

测试使用模拟摄像头检查连续帧缓存、坐标缩放、断流释放、CSV 输出和官方权重推理，不代表现场识别精度。真实硬件可用 `--headless --max-results 3` 做有限次数检查。

2026-11-24 已通过 RealSense SDK 识别本机 D435i（USB 3.2），以 1280×720/30 FPS 彩色模式完成背景采集、3 次 RTX 4060 官方模型推理并正常释放退出。SDK 设备选择、帧数据复制、超时及释放行为另有模拟测试。尚未以真实羽毛球场景验证精度。清理掉的旧文件保存在上一级 `.cleanup-backup/realtime-migration.zip`，便于恢复。

相机无法打开时，先关闭 RealSense Viewer 和其他占用相机的软件，确认 USB 连接正常，再用 `-ListDevices` 检查。SDK 已安装在项目 `.venv`，请使用上述启动入口。
