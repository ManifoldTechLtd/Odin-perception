"""Florence-2 后端 —— profile B（端侧部署）的默认模型。

为什么它能放到 Orin NX 上
------------------------
    Florence-2-base      0.23B   fp16 约 0.5GB
    Florence-2-large     0.77B   fp16 约 1.6GB
    Moondream2           1.86B   fp16 约 3.7GB
    Qwen2.5-VL-3B        3.7B    fp16 约 7.5GB
base 比 Moondream2 小 8 倍，是目前能真正塞进 Orin NX 8GB 还留出余量给导航程序的档位。

必须先讲清楚的一个硬限制
------------------------
**Florence-2 不能自由问答，它没有 VQA 能力。**

它的输入不是问题，而是一组固定任务标签：`<CAPTION>`、`<OCR>`、`<OD>`、
`<OPEN_VOCABULARY_DETECTION>` 等等。喂它一句 "Is the valve closed?"
不会得到 yes/no，只会得到一段跟问题无关的描述——**这不是 prompt 写得不好，
是模型压根没这个能力**（官方模型卡的任务列表里就没有 VQA）。

HuggingFace 官方微调博客把这件事说得最直白：论文里说 Florence-2 能做 VQA，
但**放出来的权重不含 VQA 能力**；作者试过 `<VQA>` / `<vqa>` 等未公开标签，
结果不可用；DocVQA 上微调前的相似度是 0，微调 7 个 epoch 才到 57。

它真正强的地方，以及本方案怎么用
--------------------------------
它的能力全在**定位与读字**上：检测、开词表检测、指代分割、OCR、区域描述。
所以本方案不去跟它比问答，而是让它只做定位，把判断交给确定性算子：

    read_text  ->  <OCR>                              直接读，这是它的专长
    count      ->  <OPEN_VOCABULARY_DETECTION> + 去重  数框比让模型报数字可靠
    presence   ->  <OPEN_VOCABULARY_DETECTION> + 面积过滤
    姿态判断    ->  <REFERRING_EXPRESSION_SEGMENTATION> + 主轴角度（见 geometry.py）
    颜色判断    ->  定位一次 + 色相统计（见 checks/color.py）
    describe   ->  <MORE_DETAILED_CAPTION>

"阀门手柄是否与管道垂直" 在这条路上不再是语言问题，而是「拿到手柄轮廓、算主轴
角度、看落在哪个区间」——确定性、可单测、判错了能指出是框错还是阈值不对。
这也是为什么本方案在没有现场图的情况下就能验证大部分逻辑：算子可以用合成图证明。

本文件只提供**原料**（框、轮廓、文本），怎么判在 `checks/ops.py`。
`ask()` 保留下来是为了兼容不带 `rule` 的老清单，走的是「有框=yes」的简化路径，
精度不如算子路径，清单里该写 `rule` 就别省。

能力对照，选 profile 时看这张表
------------------------------
    检查类型      Moondream2(A)   Florence-2(B) + 算子
    read_text     可用            **更好**，OCR 是它的专长
    count         一般            **更好**，数框比让模型报数字稳
    describe      好              相当
    yes_no(有无)  好              相当，靠面积过滤压幻觉
    yes_no(姿态)  一般            **更好**，算角度比问模型"是否垂直"可靠
    state(颜色)   一般            **更好**，色相统计确定且只需 1 次推理
    state(其他)   好              取决于能不能翻译成定位问题

仍然存在的短板，不要在现场才发现：
  1. Florence-2 的检测是**生成式**的，倾向于「你让它找什么它就给个框」。
     找不到目标时最典型的行为是返回一个几乎覆盖整图的框，所以 expect=no 的
     检查项**必须**在 rule 里写 `max_area_frac`（0.9 左右），否则稳定误报 FAIL。
  2. 无法表达跟具体数值挂钩的语义判断（"压力是否偏高"），这类只能靠 OCR 读数 + 阈值。
  3. 没有置信度输出，过滤只能靠几何约束（面积、位置、细长度）。

结论：清单能翻译成「定位 + 几何/颜色」的部分 → B 比 A 更可靠；
真需要开放式语义理解的部分 → 回传平台走 A（VLM-FO1）。

依赖
----
    torch, transformers<4.50, pillow, einops, timm
    权重目录默认 `models/florence2-base-ft`，由 `_bootstrap/setup_edge_venv.sh` 下载。

两个已知的环境坑，本文件都已处理：
  - Florence-2 的 remote code 无条件 `import flash_attn`，而 Jetson 上装不了
    flash-attn。这里 patch 掉 transformers 的依赖检查（见 `_patch_flash_attn`）。
  - transformers>=4.50 改了 remote code 依赖的内部 API，Florence-2 会加载失败，
    必须钉版本。py3.8 环境下 pip 最高只能装到 4.46.x，正好符合要求。
"""
from __future__ import annotations

import contextlib
import os
import re
from typing import List, Optional, Tuple

import numpy as np

from ..geometry import Box, dedupe
from .base import AskHint, VLMError, check_rgb

DEFAULT_MODEL_DIR = "models/florence2-base-ft"
HUB_REPO = "microsoft/Florence-2-base-ft"

# 任务标签。带 <> 的字面量必须原样传给 processor，不能改大小写或加空格。
TASK_OCR = "<OCR>"
TASK_CAPTION = "<CAPTION>"
TASK_DETAILED_CAPTION = "<MORE_DETAILED_CAPTION>"
TASK_OVD = "<OPEN_VOCABULARY_DETECTION>"
TASK_GROUNDING = "<CAPTION_TO_PHRASE_GROUNDING>"
TASK_SEG = "<REFERRING_EXPRESSION_SEGMENTATION>"

# 两种「按短语找目标」的任务标签，语义不同，不能混用：
#   OVD      : 开词表检测。目标不存在时**倾向于**不给框（但也可能给整图框）。
#   GROUNDING: 把一句 caption 里的短语对到图上。它假定短语描述的东西**就在图里**，
#              所以几乎总会给框——用它做「有没有渗漏」等于问「渗漏在哪」，必然假阳性。
# 因此 presence 类检查只准用 OVD；GROUNDING 只适合「已知目标存在，要定位它」的场景，
# 比如先确认了有手柄再去框手柄算角度。
DETECT_MODES = ("ovd", "grounding")

# 从 `question` 里回落抽取检测短语时，要剥掉的疑问句式和格式约束尾巴。
# `build_prompt()` 会在问题后面接一长串 "Answer with only one word: ..."，
# 整段喂给 OVD 会让它去找 "word" 这种东西，必须切掉。
_INSTRUCTION_TAIL_RE = re.compile(
    r"\b(?:answer|respond|reply)\b.*$", re.IGNORECASE | re.DOTALL
)
_QUESTION_HEAD_RE = re.compile(
    r"^(?:is|are|was|were|does|do|did|has|have|can|could|will|would|should"
    r"|what|which|where|how many|how much|how)\b\s*",
    re.IGNORECASE,
)
# "Is there a box" 剥掉 is 后还剩 "there a box"，把 there 结构也剥掉：
# there is a / there are / there's a -> 剩下真正的目标词
_THERE_RE = re.compile(r"^there(?:'s|\s+(?:is|are))?\s+(?:a|an|the)\s+", re.IGNORECASE)
# 命令式开头（"count the robots" -> "the robots"）
_IMPERATIVE_HEAD_RE = re.compile(
    r"^(?:count|find|list|show|detect|locate|search|look for)\b\s*", re.IGNORECASE
)


@contextlib.contextmanager
def _patch_flash_attn():
    """让 Florence-2 的 remote code 在没有 flash-attn 的机器上也能加载。

    Florence-2 仓库里的 modeling 文件顶部无条件 `import flash_attn`，
    transformers 加载 remote code 前会扫描 import 列表并要求这些包都存在。
    Jetson/aarch64 上 flash-attn 没有可用 wheel，编译又要几十分钟且依赖 CUDA
    开发环境——而这个 import 在 eager attention 路径下根本用不到。
    所以这里在加载期间把它从依赖清单里摘掉，加载完就还原。
    """
    try:
        from transformers.dynamic_module_utils import get_imports
        import transformers.dynamic_module_utils as dmu
    except ImportError:  # transformers 太老/太新，没这个模块就不 patch
        yield
        return

    def filtered(filename):
        imports = get_imports(filename)
        return [imp for imp in imports if imp != "flash_attn"]

    original = dmu.get_imports
    dmu.get_imports = filtered
    try:
        yield
    finally:
        dmu.get_imports = original


def _patch_fsdp_on_jetson():
    """Jetson 上 torch.distributed 不可用，补丁 transformers 推理链。

    NVIDIA Jetson 的 PyTorch wheel 不包含 torch._C._distributed_c10d 扩展，
    而 transformers 的 generate() 方法会无条件调用 is_fsdp_managed_module()
    并尝试 import torch.distributed.fsdp，导致 ModuleNotFoundError。

    这里直接替换 generation/utils.py 中的局部引用（`from ..integrations.fsdp import ...`
    会在模块命名空间创建局部绑定，改模块属性不影响它）。
    """
    try:
        import transformers.generation.utils as gen_utils

        gen_utils.is_fsdp_managed_module = lambda *a, **kw: False
        gen_utils.is_deepspeed_zero3_enabled = lambda: False
    except (ImportError, AttributeError):
        pass  # transformers 版本太老/太新，没有这个模块，不需要 patch


class Florence2Backend:
    """Florence-2 的 `VLMBackend` 实现（任务标签映射式）。

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
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError as exc:
            raise VLMError(
                f"缺少依赖: {exc}。安装见 _bootstrap/setup_edge_venv.sh"
            ) from exc

        self._torch = torch
        self.model_dir = model_dir
        self.last_prompt = ""
        """上一次实际发给模型的任务标签串。

        流水线会优先把它写进 Evidence.prompt。因为对本后端来说，
        `build_prompt()` 拼出来的那句自然语言**并没有真的发出去**，
        存证里记它等于记了假信息，出争议时会把人带到错误方向。
        """

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._dtype = self._resolve_dtype(torch, dtype, device)

        source = model_dir if os.path.isdir(model_dir) else HUB_REPO
        if source != model_dir:
            print(f"! 本地未找到 {model_dir}，改从 HuggingFace 加载 {source}（需要网络）")

        try:
            with _patch_flash_attn():
                self.model = AutoModelForCausalLM.from_pretrained(
                    source,
                    trust_remote_code=True,
                    torch_dtype=self._dtype,
                    # eager 是唯一在无 flash-attn 环境下确定可用的实现
                    attn_implementation="eager",
                )
                self.processor = AutoProcessor.from_pretrained(
                    source, trust_remote_code=True
                )
        except Exception as exc:
            raise VLMError(f"加载 Florence-2 失败（source={source}）: {exc}") from exc

        self.model = self.model.to(device=device, dtype=self._dtype)
        self.model.eval()
        _patch_fsdp_on_jetson()
        # 不暴露具体模型名（对内标识见 registry；对外展示统一用中性名）
        self.name = "edge-model"

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

    # ------------------------------------------------------------ 底层单次推理
    def _run_task(
        self,
        rgb: np.ndarray,
        task: str,
        text_input: str = "",
        max_new_tokens: int = 128,
    ) -> dict:
        """跑一个任务标签，返回 post_process 之后的字典。

        Florence-2 的输入是 `任务标签 + 可选的文本参数`，输出是带 location token
        的原始串，必须经 `post_process_generation` 才能变成结构化结果。
        """
        from PIL import Image

        image = Image.fromarray(check_rgb(rgb))
        prompt = task + text_input
        self.last_prompt = prompt

        try:
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")
            inputs = {
                k: (v.to(self.device, self._dtype) if v.dtype.is_floating_point
                    else v.to(self.device))
                for k, v in inputs.items()
            }
            with self._torch.inference_mode():
                ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=max_new_tokens,
                    num_beams=3,
                    do_sample=False,
                )
            raw = self.processor.batch_decode(ids, skip_special_tokens=False)[0]
            return self.processor.post_process_generation(
                raw, task=task, image_size=(image.width, image.height)
            )
        except Exception as exc:
            raise VLMError(f"Florence-2 推理失败（task={task}）: {exc}") from exc

    def detect(
        self, rgb: np.ndarray, phrase: str, mode: str = "ovd"
    ) -> List[Box]:
        """按短语找目标，返回像素坐标的框。找不到返回空列表。

        Args:
            mode: "ovd" 开词表检测（默认，presence 类只准用这个）；
                "grounding" 短语定位，几乎总会给框，仅适合已知目标存在时用。
                两者区别见文件头 `DETECT_MODES` 处的说明。
        """
        if mode not in DETECT_MODES:
            raise VLMError(f"未知 detect mode: {mode!r}（可选 {DETECT_MODES}）")
        phrase = (phrase or "").strip()
        if not phrase:
            return []

        task = TASK_OVD if mode == "ovd" else TASK_GROUNDING
        out = self._run_task(rgb, task, text_input=phrase, max_new_tokens=128)
        result = out.get(task, {}) or {}
        if not isinstance(result, dict):
            return []

        bboxes = result.get("bboxes") or []
        # OVD 用 bboxes_labels，grounding 用 labels。不能只取一个，
        # 否则换 mode 之后标签全变空，而标签过滤会静默失效。
        labels = result.get("bboxes_labels") or result.get("labels") or []
        boxes: List[Box] = []
        for i, raw in enumerate(bboxes):
            if raw is None or len(raw) < 4:
                continue
            label = str(labels[i]) if i < len(labels) else phrase
            boxes.append(
                Box(float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]), label)
            )
        return boxes

    def segment(self, rgb: np.ndarray, phrase: str) -> List[List[Tuple[float, float]]]:
        """指代分割：返回多边形轮廓列表，每个是 [(x, y), ...]。

        Florence-2 把轮廓给成拉平的 [x1, y1, x2, y2, ...]，而且多嵌一层
        （每个实例可能有多个环）。这里拉平成一组环，不保留实例归属——
        巡检里看的是单个部位的姿态，上层取面积最大的环就够了。
        """
        phrase = (phrase or "").strip()
        if not phrase:
            return []
        out = self._run_task(rgb, TASK_SEG, text_input=phrase, max_new_tokens=512)
        result = out.get(TASK_SEG, {}) or {}
        if not isinstance(result, dict):
            return []

        rings: List[List[Tuple[float, float]]] = []
        for instance in result.get("polygons") or []:
            for ring in instance or []:
                flat = [float(v) for v in ring]
                if len(flat) < 6:  # 不足 3 个点，算不出面积也算不出主轴
                    continue
                pts = list(zip(flat[0::2], flat[1::2]))
                rings.append([(x, y) for x, y in pts])
        return rings

    def ocr(self, rgb: np.ndarray) -> str:
        """读出画面里的文字。读不到返回空字符串。"""
        out = self._run_task(rgb, TASK_OCR, max_new_tokens=256)
        return str(out.get(TASK_OCR, "") or "").strip()

    # ------------------------------------------------------------------ 对外
    def ask(
        self,
        rgb: np.ndarray,
        question: str,
        max_new_tokens: int = 64,
        hint: Optional[AskHint] = None,
    ) -> str:
        """按 hint 里的检查类型选任务标签，把结果合成成可解析的文本。

        返回值刻意做成和自由问答模型同构（"yes"/"no"/"unclear"/数字/文本），
        这样 `checks/verdict.py` 一行都不用改。
        """
        kind = (hint.kind if hint else "") or ""
        target = (hint.target if hint else "") or ""
        options = list(hint.options) if hint and hint.options else []

        if kind == "read_text":
            return self.ocr(rgb) or "unclear"

        if kind == "describe":
            return self.caption(rgb, length="normal")

        phrase = target or self._phrase_from_question(question)

        if kind == "count":
            if not phrase:
                return "unclear"
            # 去重是必选项：生成式检测对同一目标常吐出多个重叠框，
            # 不去重的计数会稳定偏大，而这种错在现场极难发现。
            return str(len(dedupe(self.detect(rgb, phrase))))

        if kind == "yes_no":
            if not phrase:
                return "unclear"
            return "yes" if self.detect(rgb, phrase) else "no"

        if kind == "state":
            return self._ask_state(rgb, options, phrase)

        # 没给 hint（比如直接调 ask 做探测）——只能降级成描述，并说清楚。
        # 不静默返回描述文本，否则 yes_no 的解析会在描述里瞎命中关键词。
        return (
            f"unclear (Florence-2 不能自由问答，本次没收到 hint.kind，"
            f"无法选择任务标签；原问题: {question[:60]})"
        )

    def _ask_state(self, rgb: np.ndarray, options: List[str], phrase: str) -> str:
        """多选一的简化路径：逐个 option 检测，恰好命中一个才作数。

        代价是 N 次推理，而且检测器对 "green" 这种形容词本来就不敏感。
        **颜色类状态应当用 `rule: {op: region_color}`**：只需 1 次推理，
        而且色相统计是确定性的。本方法只为不带 rule 的老清单兜底。
        """
        if not options:
            return "unclear (state 类型缺 options)"
        hits = []
        for opt in options:
            # 把状态词拼到目标短语前面，例如 "green indicator light"。
            probe = f"{opt} {phrase}".strip() if phrase else opt
            if self.detect(rgb, probe):
                hits.append(opt)
        # 命中零个或多个都不猜——verdict 层对 state 也是这个原则，这里保持一致
        return hits[0] if len(hits) == 1 else f"unclear (命中 {hits or '零个'} 个选项)"

    @staticmethod
    def _phrase_from_question(question: str) -> str:
        """`target` 没填时，从问句里勉强抽一个检测短语出来。

        这是兜底，不是正常路径：抽出来的短语质量取决于问句怎么写，
        所以 profile B 的清单应当显式写 `target`。
        """
        text = _INSTRUCTION_TAIL_RE.sub("", question or "").strip()
        text = text.rstrip("?？.。").strip()
        # 疑问词/命令词可能叠着剥（"where is the sign" -> 剥 where 后剩 "is the sign"）
        for _ in range(3):
            new = _QUESTION_HEAD_RE.sub("", text).strip()
            new = _THERE_RE.sub("", new).strip()
            new = _IMPERATIVE_HEAD_RE.sub("", new).strip()
            if new == text:
                break
            text = new
        return text[:80]

    def caption(self, rgb: np.ndarray, length: str = "short") -> str:
        task = TASK_CAPTION if length == "short" else TASK_DETAILED_CAPTION
        out = self._run_task(rgb, task, max_new_tokens=128)
        return str(out.get(task, "") or "").strip()

    def warmup(self) -> None:
        dummy = np.zeros((224, 224, 3), dtype=np.uint8)
        try:
            self.caption(dummy, length="short")
        except VLMError as exc:
            print(f"! 预热失败: {exc}")

    def unload(self) -> None:
        self.model = None
        self.processor = None
        if self.device == "cuda":
            self._torch.cuda.empty_cache()
