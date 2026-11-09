"""Moondream2 后端 —— 本方案的默认 VLM。

为什么选它
----------
巡检最基础的需求是「对着画面问一句话，拿到一个能判定的答案」，需要的是
自由问答 + OCR + 一点定位能力。在这个能力档位里 Moondream2 是最轻的：

    Moondream2        ~1.9B  自由问答 / OCR / point()   官方出 int8、int4 量化
    SmolVLM-500M      0.5B   也能自由问答，体积 1/4，端侧首选（见 smolvlm.py）
    Florence-2-base   0.23B  更小，但只吃固定任务标签，不能自由问答
    Qwen2.5-VL-3B     ~3.7B  更强，但显存和延迟都翻倍
    VLM-FO1           ~3B+   region 级最强，但要编 UPN 的自定义 CUDA op

Florence-2 看着更小，实际会逼着我们把每种检查都翻译成它认识的任务标签，
业务侧反而更重。Moondream2 是「能自由问答」这条线上的下限，所以先用它。
换模型只需换本文件的兄弟模块，业务层不动（见 `registry.py`）。

当前定位：**profile A（平台侧）的模型**。客户要求端侧部署时走 profile B，
见 `docs/profiles_a_b.md`。

依赖
----
    transformers>=4.44, torch, pillow, einops
    模型代码走 trust_remote_code（Moondream 的推理逻辑在仓库自带的 py 文件里）

模型目录默认 `models/moondream2`，由 `_bootstrap/setup_light.sh` 下载。
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from .base import AskHint, VLMError, check_rgb

DEFAULT_MODEL_DIR = "models/moondream2"

# Moondream 官方推荐用 revision 锁版本；本地目录加载时该参数不适用，
# 仅在从 HuggingFace Hub 直接拉取时生效。
DEFAULT_REVISION = "2025-06-21"


class MoondreamBackend:
    """Moondream2 的 `VLMBackend` 实现。

    模型在构造时就加载完（而不是首次调用时懒加载）——巡检是长驻服务，
    把 10~30 秒的加载耗时放在启动阶段，能避免第一个点位莫名超时。
    """

    def __init__(
        self,
        model_dir: str = DEFAULT_MODEL_DIR,
        device: Optional[str] = None,
        dtype: str = "auto",
        revision: str = DEFAULT_REVISION,
        do_warmup: bool = True,
    ) -> None:
        """
        Args:
            model_dir: 本地模型目录；不存在时回落到用 HuggingFace repo id 在线加载。
            device: "cuda" | "cpu" | None(自动)。
            dtype: "auto" | "float16" | "bfloat16" | "float32"。
                auto 在 GPU 上选 float16——Orin NX 的 fp16 吞吐是 fp32 的数倍，
                且 VQA 任务对精度不敏感。
            do_warmup: 构造后立刻跑一次预热推理。
        """
        try:
            import torch
            from transformers import AutoModelForCausalLM
        except ImportError as exc:
            raise VLMError(
                f"缺少依赖: {exc}。安装: uv pip install torch transformers pillow einops"
            ) from exc

        self._torch = torch
        self.model_dir = model_dir

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self._dtype = self._resolve_dtype(torch, dtype, device)

        source = model_dir if os.path.isdir(model_dir) else "vikhyatk/moondream2"
        from_hub = source != model_dir
        if from_hub:
            print(f"! 本地未找到 {model_dir}，改从 HuggingFace 加载 {source}（需要网络）")

        kwargs = {
            "trust_remote_code": True,
            "torch_dtype": self._dtype,
        }
        if from_hub:
            kwargs["revision"] = revision
        if device == "cuda":
            # device_map 让 accelerate 直接把权重搬到显卡，避免先在 CPU 上占一份内存。
            # NX 是统一内存，省掉这次拷贝对峰值内存很关键。
            kwargs["device_map"] = {"": "cuda"}

        try:
            self.model = AutoModelForCausalLM.from_pretrained(source, **kwargs)
        except Exception as exc:
            raise VLMError(f"加载 Moondream2 失败（source={source}）: {exc}") from exc

        if device != "cuda":
            self.model = self.model.to(device)
        self.model.eval()

        self.name = f"moondream2@{revision}" if from_hub else f"moondream2@{os.path.basename(model_dir)}"

        if do_warmup:
            self.warmup()

    @staticmethod
    def _resolve_dtype(torch, dtype: str, device: str):
        if dtype == "auto":
            return torch.float16 if device == "cuda" else torch.float32
        mapping = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if dtype not in mapping:
            raise VLMError(f"未知 dtype: {dtype!r}（可选 auto/float16/bfloat16/float32）")
        return mapping[dtype]

    # ------------------------------------------------------------------ 推理
    @staticmethod
    def _to_pil(rgb: np.ndarray):
        from PIL import Image

        return Image.fromarray(check_rgb(rgb))

    def ask(
        self,
        rgb: np.ndarray,
        question: str,
        max_new_tokens: int = 64,
        hint: Optional[AskHint] = None,
    ) -> str:
        """自由问答。`hint` 用不上——prompt 里的格式约束已经把类型信息表达完了。"""
        image = self._to_pil(rgb)
        try:
            with self._torch.inference_mode():
                out = self.model.query(
                    image, question, settings={"max_tokens": max_new_tokens}
                )
        except Exception as exc:
            raise VLMError(f"Moondream2 推理失败: {exc}") from exc
        # query() 返回 {"answer": "..."}；老版本直接返回字符串，两种都兼容
        answer = out.get("answer", "") if isinstance(out, dict) else str(out)
        return answer.strip()

    def caption(self, rgb: np.ndarray, length: str = "short") -> str:
        image = self._to_pil(rgb)
        try:
            with self._torch.inference_mode():
                out = self.model.caption(image, length=length)
        except Exception as exc:
            raise VLMError(f"Moondream2 caption 失败: {exc}") from exc
        text = out.get("caption", "") if isinstance(out, dict) else str(out)
        return text.strip()

    def warmup(self) -> None:
        dummy = np.zeros((224, 224, 3), dtype=np.uint8)
        try:
            self.ask(dummy, "What is in this image?", max_new_tokens=4)
        except VLMError as exc:
            # 预热失败要显式说出来，不能静默——否则真正的推理会在点位上才炸
            print(f"! 预热失败: {exc}")

    def unload(self) -> None:
        self.model = None
        if self.device == "cuda":
            self._torch.cuda.empty_cache()
