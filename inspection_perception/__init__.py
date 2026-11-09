"""工业机器人巡检感知。

分层（每层只依赖下一层，可单独替换）::

    cli.py                命令行入口
    pipeline.py           取图 → 裁 ROI → 问模型 → 判定 → 存证
    ├─ capture/           取图层：ROS 1 / ROS 2 / HTTP / 本地文件
    ├─ vlm/               模型层：Moondream2 / Dummy
    ├─ checks/            判定层：prompt 拼装 + 答案解析（纯函数，可单测）
    ├─ tasks.py           任务清单加载（YAML/JSON）
    └─ schemas.py         数据契约：CheckItem / CheckResult / Evidence

顶层只导出数据契约。`capture` 和 `vlm` 各自有重量级可选依赖
（rclpy/rospy、torch/transformers），一律按需 import，
所以 `import inspection_perception` 在任何环境下都不会失败。
"""
from __future__ import annotations

from .schemas import CheckItem, CheckKind, CheckResult, CheckStatus, Evidence

__version__ = "0.2.0"

__all__ = [
    "CheckItem",
    "CheckKind",
    "CheckResult",
    "CheckStatus",
    "Evidence",
    "__version__",
]
