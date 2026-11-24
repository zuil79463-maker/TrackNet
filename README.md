# TrackNet · RealSense D435i 羽毛球实时识别

本仓库备份 `host` 工程的源码与配置，保持 `TrackNetV3/` 子目录结构。基于 [qaz812345/TrackNetV3](https://github.com/qaz812345/TrackNetV3)，保留原项目 MIT 许可证。

默认使用 RealSense D435i 的彩色流，通过 CUDA / CPU 执行羽毛球二维定位，支持轨迹预览和坐标 CSV 输出。当前仍为 FP32 推理版本；FP16、预处理缓存等性能优化尚未实施。

## 恢复环境

使用 Python 3.12。在仓库根目录运行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\TrackNetV3\requirements.txt
# 若未通过 Git LFS 下载模型，执行：
.\.venv\Scripts\python.exe .\TrackNetV3\download_weights.py
powershell -ExecutionPolicy Bypass -File .\TrackNetV3\run_tracking.ps1
```

需要 GPU 推理时，应在 `.venv` 安装适合本机显卡及驱动的 CUDA PyTorch。默认启用 D435i 彩色流，1280×720、30 FPS；按 `B` 校准背景，按 `Q` 退出。

## 备份范围

- `TrackNetV3/`：实时推理、RealSense 接入、模型结构、测试、依赖清单、启动脚本及许可证。
- `.vscode/settings.json`：当前编辑器配置。
- 不提交可重建的 `.venv`、Python 缓存、清理恢复包、输出文件和上游 Git 元数据。
- 官方模型权重 `TrackNet_best.pt`（约 136 MB）通过 Git LFS 备份。克隆前安装并启用 Git LFS，克隆后可运行 `git lfs pull` 恢复模型；也可使用上述官方下载脚本。模型来源与 SHA256 见子目录说明。

本机已验证 D435i 彩色采集及 RTX 4060 推理，实际羽毛球场景精度需现场验证。

完整说明见 [TrackNetV3/README.md](TrackNetV3/README.md)。
