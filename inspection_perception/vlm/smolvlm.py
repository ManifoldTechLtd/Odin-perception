"""SmolVLM 后端 —— profile B 的推荐选项（端侧 + 保留自由问答）。

存在的理由
----------
Florence-2 虽然更小，但**不能自由问答**，yes_no 和 state 这两类检查在它上面
只能靠开词表检测硬凑，准确率明显下降（详见 `florence2.py` 顶部的能力对照表）。
而巡检清单里 yes_no + state 通常占一半以上。

SmolVLM 填的正是这个空档：它是真正的指令跟随 VQA 模型，`ask()` 完全等价于
Moondream2 的用法，业务侧和 prompt 一行都不用改，同时又小到能上 Orin NX：

    SmolVLM-256M-Instruct   0.26B   fp16 约 0.5GB   与 Florence-2-base 同档
    SmolVLM-500M-Instruct   0.5B    fp16 约 1.0GB   ← 默认，精度/体积平衡点
    SmolVLM-2.2B-Instruct   2.2B    fp16 约 4.4GB   与 Moondream2 同档
    Moondream2              1.86B   fp16 约 3.7GB

所以三条路的定位是：
    profile A   Moondream2      平台侧，精度基准
    profile B   SmolVLM-500M    端侧，能力与 A 同构，体积 1/4      ← 推荐
    profile B2  Florence-2-base 端侧，OCR/计数更强但不能自由问答

依赖
----
    torch, transformers>=4.46（SmolVLM 走 Idefics3 架构，4.46 才并入）, pillow
    py3.8 上 pip 最高装到 transformers 4.46.x，刚好够用，不能再低。

权重目录默认 `models/smolvlm-500m`，由 `_bootstrap/setup_edge_venv.sh` 下载。
"""
from __future__ import annotations

import os
import re
from typing import Optional

import numpy as np

from .base import AskHint, VLMError, check_rgb

DEFAULT_MODEL_DIR = "models/smolvlm-500m"
HUB_REPO = "HuggingFaceTB/SmolVLM-500M-Instruct"

# 生成结果里会带上完整对话，需要切掉 prompt 回显只留助手那段。
# 不同版本的 chat template 用的角色标记大小写不一致，所以用正则而不是固定字符串。
_ASSISTANT_RE = re.compile(r"assistant\s*:?\s*", re.IGNORECASE)


class SmolVLMBackend:
    """SmolVLM 的 `VLMBackend` 实现。

    Args:
        model_dir: 本地权重目录；不存在时回落到从 HuggingFace 在线加载。
        device: "cuda" | "cpu" | None(自动)。
        dtype: "auto" | "float16" | "bfloat16" | "float32"。
        do_warmup: 构造后立刻跑一次预热推理。
    """

    def __init__(
        self,
        model_dir: str = DEFAULT_MODEL_DIR,
        device: Optional[str] = None,
        dtype: str = "auto",
        do_warmup: bool = True,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except ImportError as exc:
            raise VLMError(
                f"缺少依赖: {exc}。安装见 _bootstrap/setup_edge_venv.sh"
            ) from exc

        self._torch = torch
        self.model_dir = model_dir

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._dtype = self._resolve_dtype(torch, dtype, device)

        source = model_dir if os.path.isdir(model_dir) else HUB_REPO
        if source != model_dir:
            print(f"! 本地未找到 {model_dir}，改从 HuggingFace 加载 {source}（需要网络）")

        try:
            self.processor = AutoProcessor.from_pretrained(source)
            self.model = AutoModelForVision2Seq.from_pretrained(
                source,
                torch_dtype=self._dtype,
                # 不用 flash_attention_2：Jetson 上装不了，eager 在这个体量下够快
                _attn_implementation="eager",
            )
        except Exception as exc:
            raise VLMError(f"加载 SmolVLM 失败（source={source}）: {exc}") from exc

        self.model = self.model.to(device)
        self.model.eval()
        self.name = f"smolvlm@{os.path.basename(source)}"

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
    def _generate(self, rgb: np.ndarray, text: str, max_new_tokens: int) -> str:
        from PIL import Image

        image = Image.fromarray(check_rgb(rgb))
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": text}],
            }
        ]
        try:
            prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True
            )
            inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with self._torch.inference_mode():
                ids = self.model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False
                )
            decoded = self.processor.batch_decode(ids, skip_special_tokens=True)[0]
        except Exception as exc:
            raise VLMError(f"SmolVLM 推理失败: {exc}") from exc

        # 只保留最后一段助手回复。切不出来时返回整串而不是空字符串——
        # 让 verdict 层去判 UNKNOWN，比这里静默吞掉答案好排查。
        parts = _ASSISTANT_RE.split(decoded)
        return (parts[-1] if len(parts) > 1 else decoded).strip()

    def ask(
        self,
        rgb: np.ndarray,
        question: str,
        max_new_tokens: int = 64,
        hint: Optional[AskHint] = None,
    ) -> str:
        """自由问答，`hint` 用不上（prompt 里的格式约束已经够了）。"""
        return self._generate(rgb, question, max_new_tokens)

    def caption(self, rgb: np.ndarray, length: str = "short") -> str:
        text = (
            "Describe this image in one sentence."
            if length == "short"
            else "Describe this image in detail."
        )
        return self._generate(rgb, text, 128 if length != "short" else 48)

    def warmup(self) -> None:
        dummy = np.zeros((224, 224, 3), dtype=np.uint8)
        try:
            self.ask(dummy, "What is in this image?", max_new_tokens=4)
        except VLMError as exc:
            print(f"! 预热失败: {exc}")

    def unload(self) -> None:
        self.model = None
        self.processor = None
        if self.device == "cuda":
            self._torch.cuda.empty_cache()
