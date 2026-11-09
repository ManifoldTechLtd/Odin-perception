"""VLM 后端层。

    VLMBackend    后端协议（ask / caption / warmup / unload）
    make_backend  按名字构造后端
    VLMError      后端故障，pipeline 据此判成 ERROR 而非 FAIL

具体实现延迟导入：`moondream` 需要 torch/transformers，
只想用 `dummy` 跑测试的环境不应被迫安装它们。
"""
from __future__ import annotations

from .base import VLMBackend, VLMError, check_rgb
from .registry import AVAILABLE, make_backend

__all__ = ["VLMBackend", "VLMError", "check_rgb", "make_backend", "AVAILABLE"]
