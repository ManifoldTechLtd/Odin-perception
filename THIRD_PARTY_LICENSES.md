# 第三方组件许可

| 组件 | 用途 | 许可 | 说明 |
|---|---|---|---|
| Florence-2-base-ft (`microsoft/Florence-2-base-ft`) | 主检测后端 | MIT | 权重不随仓库分发，由 `scripts/download_models.sh` 获取 |
| transformers | 模型加载 | Apache-2.0 | — |
| torch | 推理运行时 | BSD-3-Clause | 自带 CUDA wheel 时的许可随发行版 |
| numpy | 数值计算 | BSD-3-Clause | — |
| Pillow | 图像解码 | HPND | — |
| PyYAML | 任务清单解析 | MIT | — |
| opencv-python-headless | 图像处理 | Apache-2.0 | 无 GUI 依赖版本 |
| huggingface_hub | 权重下载 | Apache-2.0 | 仅安装脚本使用 |

> 注意：本仓库**不包含**任何模型权重。若接入 AGPL-3.0 等强 copyleft 许可的第三方推理后端，
> 需单独评估许可义务，不应与 Apache-2.0 主体混放。
