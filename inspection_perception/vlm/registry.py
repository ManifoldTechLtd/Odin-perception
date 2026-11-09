"""后端注册表：按名字构造 VLM 后端。

保持 `pipeline` 与具体模型解耦——加一个模型只在这里加一行，
业务代码和配置文件里始终只出现字符串名字。
"""
from __future__ import annotations

from typing import Any

from .base import VLMError

AVAILABLE = {
    "moondream": "Moondream2 ~1.9B。profile A（平台侧）的模型，自由问答 + OCR，精度基准",
    "smolvlm": "SmolVLM-500M。profile B（端侧）推荐：体积 1/4 但保留自由问答，与 A 同构",
    "florence2": "Florence-2-base 0.23B。最小，OCR/计数更强，但**不能自由问答**，yes_no/state 会降级",
    "dummy": "桩后端，不加载模型。用于无 GPU/无权重环境下跑通链路与测试",
}

# 哪些后端是「只吃固定任务标签、不能自由问答」的。
# 这个集合不是给推理用的，是给启动时体检用：跑这类后端时如果清单里的
# yes_no/state/count 项没填 `target`，应当提前警告，而不是等到现场才发现全是 UNKNOWN。
TASK_TOKEN_BACKENDS = frozenset({"florence2"})


def make_backend(name: str = "moondream", **kwargs: Any):
    """构造后端。

    Args:
        name: AVAILABLE 中的键。
        **kwargs: 透传给具体后端的构造参数（如 model_dir、device、dtype）。
    """
    if name == "moondream":
        from .moondream import MoondreamBackend

        return MoondreamBackend(**kwargs)

    if name == "smolvlm":
        from .smolvlm import SmolVLMBackend

        return SmolVLMBackend(**kwargs)

    if name == "florence2":
        from .florence2 import Florence2Backend

        return Florence2Backend(**kwargs)

    if name == "dummy":
        from .dummy import DummyBackend

        return DummyBackend(**kwargs)

    options = "\n".join(f"  {k:12s} {v}" for k, v in AVAILABLE.items())
    raise VLMError(f"未知后端 {name!r}。可选：\n{options}")
