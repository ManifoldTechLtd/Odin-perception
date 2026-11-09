"""profile A/B/B2 相关的测试：hint 透传、Florence-2 任务分派、清单 target 字段。

跟 `test_pipeline.py` 一样不需要 GPU、不需要权重、不需要 ROS。
Florence-2 的测试用 `object.__new__` 绕过 `__init__`（那里面要 import torch），
只测 `ask()` 的分派逻辑——这是新增代码里最容易写错、又最难在现场发现的部分：
分派错了不会报异常，只会安静地返回一个语义不对的答案。

    python3 -m unittest discover -s tests -v
"""
from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout

import numpy as np

from inspection_perception.capture.decode import encode_jpeg
from inspection_perception.capture.file_source import FileFrameSource
from inspection_perception.geometry import Box
from inspection_perception.cli import build_argparser, warn_missing_target
from inspection_perception.pipeline import InspectionPipeline
from inspection_perception.schemas import CheckItem, CheckKind
from inspection_perception.tasks import load_check_items
from inspection_perception.vlm.base import AskHint
from inspection_perception.vlm.dummy import DummyBackend
from inspection_perception.vlm.florence2 import (
    TASK_OCR,
    TASK_OVD,
    Florence2Backend,
)
from inspection_perception.vlm.registry import AVAILABLE, TASK_TOKEN_BACKENDS, make_backend


def _encodable() -> bool:
    try:
        encode_jpeg(np.zeros((8, 8, 3), dtype=np.uint8))
        return True
    except Exception:
        return False


ENCODABLE = _encodable()
SKIP_REASON = "需要 opencv 或 pillow 才能编解码图像"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# =============================================================================
class TestRegistry(unittest.TestCase):
    def test_three_real_backends_registered(self):
        for name in ("moondream", "smolvlm", "florence2", "dummy"):
            self.assertIn(name, AVAILABLE)

    def test_task_token_backends_only_florence2(self):
        """这个集合驱动启动时的 target 体检。往里加后端时必须同步 cli 的提示文案，
        所以钉住当前内容，加成员时会失败提醒。"""
        self.assertEqual(set(TASK_TOKEN_BACKENDS), {"florence2"})

    def test_unknown_backend_lists_options(self):
        with self.assertRaises(Exception) as ctx:
            make_backend("gpt5")
        self.assertIn("florence2", str(ctx.exception))

    def test_cli_backend_choices_follow_registry(self):
        """启动脚本会传 --backend florence2，argparse 的 choices 必须跟得上注册表。"""
        action = next(
            a for a in build_argparser()._actions if a.dest == "backend"
        )
        self.assertEqual(set(action.choices), set(AVAILABLE))

    def test_model_dir_has_no_default(self):
        """默认值必须留空，否则 florence2 会去 models/moondream2 找权重。"""
        args = build_argparser().parse_args(["--tasks", "x.yaml"])
        self.assertIsNone(args.model_dir)


# =============================================================================
class TestTargetField(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ip_target_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_target_loaded_from_manifest(self):
        path = os.path.join(self.tmp, "t.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "items": [
                        {
                            "id": "leak",
                            "question": "Is there any oil leak",
                            "kind": "yes_no",
                            "target": "oil leak",
                        }
                    ]
                },
                fh,
            )
        items = load_check_items(path)
        self.assertEqual(items[0].target, "oil leak")

    def test_target_defaults_empty(self):
        """老清单没有这个字段，不能因此加载失败——A/B 都不需要它。"""
        path = os.path.join(self.tmp, "t.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"items": [{"id": "a", "question": "Is it closed"}]}, fh)
        self.assertEqual(load_check_items(path)[0].target, "")

    def test_demo_manifest_fills_target_for_detectable_kinds(self):
        """demo.yaml 是三条 profile 共用的清单，B2 依赖 target。
        没有 PyYAML 的环境跳过。"""
        try:
            import yaml  # noqa: F401
        except ImportError:
            self.skipTest("需要 PyYAML")
        items = load_check_items(os.path.join(REPO_ROOT, "configs/tasks/demo.yaml"))
        need = [
            it
            for it in items
            if it.kind in (CheckKind.YES_NO, CheckKind.STATE, CheckKind.COUNT)
        ]
        self.assertTrue(need, "demo 清单里应当有检测类的项")
        for it in need:
            self.assertTrue(it.target, f"{it.id} 缺 target，profile B2 会判成 UNKNOWN")


# =============================================================================
class TestWarnMissingTarget(unittest.TestCase):
    def _run(self, items, backend: str) -> str:
        buf = io.StringIO()
        with redirect_stdout(buf):
            warn_missing_target(items, backend)
        return buf.getvalue()

    def test_silent_for_free_form_backends(self):
        item = CheckItem(id="a", question="Is it closed", kind=CheckKind.YES_NO)
        for backend in ("moondream", "smolvlm", "dummy"):
            self.assertEqual(self._run([item], backend), "")

    def test_warns_and_names_the_items(self):
        items = [
            CheckItem(id="no_target", question="Is it closed", kind=CheckKind.YES_NO),
            CheckItem(id="ok", question="Any leak", kind=CheckKind.YES_NO, target="oil leak"),
        ]
        out = self._run(items, "florence2")
        self.assertIn("no_target", out)
        self.assertNotIn("ok,", out)  # 填了的不该被点名

    def test_read_text_and_describe_need_no_target(self):
        """这两类走 <OCR> / caption，不需要检测词条，不该被误报。"""
        items = [
            CheckItem(id="plate", question="Read the text", kind=CheckKind.READ_TEXT),
            CheckItem(id="scene", question="Describe it", kind=CheckKind.DESCRIBE),
        ]
        self.assertEqual(self._run(items, "florence2"), "")


# =============================================================================
@unittest.skipUnless(ENCODABLE, SKIP_REASON)
class TestHintPlumbing(unittest.TestCase):
    """流水线必须把检查类型传到后端。

    这条链路断了不会有任何异常：Florence-2 收不到 kind 就走「降级成描述」分支，
    整批检查静默变成 UNKNOWN。所以在这里用 DummyBackend 把它钉住。
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ip_hint_")
        img_dir = os.path.join(self.tmp, "imgs")
        os.makedirs(img_dir)
        with open(os.path.join(img_dir, "a.jpg"), "wb") as fh:
            fh.write(encode_jpeg(np.full((240, 320, 3), 128, dtype=np.uint8)))
        self.source = FileFrameSource(img_dir, loop=True)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _pipeline(self, backend):
        return InspectionPipeline(
            source=self.source, backend=backend, evidence_dir=None, max_age_ms=0
        )

    def test_hint_carries_kind_target_options(self):
        backend = DummyBackend(default="green")
        item = CheckItem(
            id="lamp",
            question="What colour is the indicator",
            kind=CheckKind.STATE,
            options=["red", "green"],
            target="indicator light",
            expect="green",
        )
        self._pipeline(backend).run_item(item)

        hint = backend.last_hint
        self.assertIsNotNone(hint, "pipeline 没把 hint 传下去")
        self.assertEqual(hint.kind, "state")
        self.assertEqual(hint.target, "indicator light")
        self.assertEqual(hint.options, ["red", "green"])

    def test_hint_kind_is_plain_string(self):
        """hint.kind 必须是字符串而不是 CheckKind——`vlm/` 层不 import schemas，
        传枚举进去会让后端里的 `kind == "yes_no"` 比较恒为 False。"""
        backend = DummyBackend(answers={"closed": "yes"})
        item = CheckItem(id="v", question="Is it closed", kind=CheckKind.YES_NO, expect="yes")
        self._pipeline(backend).run_item(item)
        self.assertIsInstance(backend.last_hint.kind, str)
        self.assertEqual(backend.last_hint.kind, "yes_no")

    def test_evidence_records_backend_reported_prompt(self):
        """后端暴露 last_prompt 时，存证要记它发出去的那句，而不是业务层拼的自然语言。"""

        class TaskTokenBackend(DummyBackend):
            last_prompt = "<OPEN_VOCABULARY_DETECTION>oil leak"

        backend = TaskTokenBackend(default="no")
        item = CheckItem(
            id="leak", question="Is there any oil leak", kind=CheckKind.YES_NO,
            expect="no", target="oil leak",
        )
        pipeline = InspectionPipeline(
            source=self.source, backend=backend,
            evidence_dir=os.path.join(self.tmp, "ev"), max_age_ms=0,
        )
        result = pipeline.run_item(item)
        self.assertEqual(result.evidence.prompt, "<OPEN_VOCABULARY_DETECTION>oil leak")

    def test_evidence_falls_back_to_built_prompt(self):
        """没有 last_prompt 的后端仍然记 build_prompt 的结果，不能变成空串。"""
        backend = DummyBackend(answers={"closed": "yes"})
        item = CheckItem(id="v", question="Is it closed", kind=CheckKind.YES_NO, expect="yes")
        pipeline = InspectionPipeline(
            source=self.source, backend=backend,
            evidence_dir=os.path.join(self.tmp, "ev2"), max_age_ms=0,
        )
        result = pipeline.run_item(item)
        self.assertIn("Is it closed", result.evidence.prompt)


# =============================================================================
class _FakeFlorence(Florence2Backend):
    """只替掉两个推理入口，其余分派逻辑用真的。

    不 mock `ask()` 本身——要测的正是它怎么选任务标签。
    """

    def __init__(self, ocr_text: str = "", detections=None) -> None:  # noqa: D107
        self.name = "florence2@fake"
        self.last_prompt = ""
        self._ocr_text = ocr_text
        self._detections = detections or {}
        self.detect_calls = []
        self.task_calls = []

    def _run_task(self, rgb, task, text_input="", max_new_tokens=128):
        self.task_calls.append((task, text_input))
        self.last_prompt = task + text_input
        if task == TASK_OCR:
            return {TASK_OCR: self._ocr_text}
        return {task: f"caption of {rgb.shape[1]}x{rgb.shape[0]}"}

    def detect(self, rgb, phrase, mode="ovd"):
        # 真实后端的 detect 是公开的 GroundingBackend 协议方法——桩必须覆写它，
        # 覆写一个没人调的 _detect 会让 ask() 静默走回真实现（测试曾因此全灭）
        self.detect_calls.append(phrase)
        self.last_prompt = TASK_OVD + phrase
        return list(self._detections.get(phrase, []))


def _box(x0: float = 0.0, label: str = "x") -> Box:
    """10x10 假检测框。要多个框时错开 x0：dedupe 合并 IoU>=0.6 的框，
    挪在一起的框会被数成一个。"""
    return Box(x0, 0.0, x0 + 10.0, 10.0, label)


class TestFlorence2Dispatch(unittest.TestCase):
    def setUp(self):
        self.rgb = np.zeros((64, 64, 3), dtype=np.uint8)

    def test_read_text_uses_ocr(self):
        be = _FakeFlorence(ocr_text="SN-88123")
        out = be.ask(self.rgb, "Read the nameplate", hint=AskHint(kind="read_text"))
        self.assertEqual(out, "SN-88123")
        self.assertEqual(be.task_calls, [(TASK_OCR, "")])
        self.assertEqual(be.detect_calls, [], "read_text 不该触发检测")

    def test_empty_ocr_is_unclear_not_empty_string(self):
        """空串会让 verdict 层拿不到任何信息又不报错，必须显式表达「看不清」。"""
        be = _FakeFlorence(ocr_text="   ")
        self.assertEqual(be.ask(self.rgb, "Read it", hint=AskHint(kind="read_text")), "unclear")

    def test_yes_no_uses_target_as_detection_phrase(self):
        be = _FakeFlorence(detections={"oil leak": [_box()]})
        out = be.ask(
            self.rgb, "Is there any oil leak",
            hint=AskHint(kind="yes_no", target="oil leak"),
        )
        self.assertEqual(out, "yes")
        self.assertEqual(be.detect_calls, ["oil leak"])

    def test_yes_no_no_detection_means_no(self):
        be = _FakeFlorence(detections={})
        out = be.ask(
            self.rgb, "Is there any oil leak",
            hint=AskHint(kind="yes_no", target="oil leak"),
        )
        self.assertEqual(out, "no")

    def test_count_returns_box_count(self):
        be = _FakeFlorence(
            detections={"fire extinguisher": [_box(0.0), _box(20.0), _box(40.0)]}
        )
        out = be.ask(
            self.rgb, "How many extinguishers",
            hint=AskHint(kind="count", target="fire extinguisher"),
        )
        self.assertEqual(out, "3")

    def test_state_probes_each_option(self):
        be = _FakeFlorence(detections={"green indicator light": [_box()]})
        out = be.ask(
            self.rgb, "What colour",
            hint=AskHint(kind="state", options=["red", "green"], target="indicator light"),
        )
        self.assertEqual(out, "green")
        # 逐个 option 各一次推理——这就是 B2 下 state 延迟翻 N 倍的来源
        self.assertEqual(be.detect_calls, ["red indicator light", "green indicator light"])

    def test_state_multiple_hits_is_unclear(self):
        be = _FakeFlorence(detections={"red light": [_box()], "green light": [_box()]})
        out = be.ask(
            self.rgb, "What colour",
            hint=AskHint(kind="state", options=["red", "green"], target="light"),
        )
        self.assertIn("unclear", out)

    def test_state_without_options_is_unclear(self):
        be = _FakeFlorence()
        out = be.ask(self.rgb, "What colour", hint=AskHint(kind="state", target="light"))
        self.assertIn("unclear", out)

    def test_missing_hint_returns_unclear_not_a_caption(self):
        """降级必须显式说「不知道」。若返回整图描述，yes_no 的解析会在描述文本里
        瞎命中 yes/no 关键词，把无效结果伪装成有效结论——那比 UNKNOWN 危险得多。"""
        be = _FakeFlorence()
        out = be.ask(self.rgb, "Is the valve handle vertical")
        self.assertTrue(out.startswith("unclear"))
        self.assertEqual(be.detect_calls, [])

    def test_describe_uses_caption(self):
        be = _FakeFlorence()
        out = be.ask(self.rgb, "Describe the scene", hint=AskHint(kind="describe"))
        self.assertIn("caption of", out)

    def test_phrase_fallback_strips_question_head_and_instruction_tail(self):
        """target 没填时的兜底：必须把 build_prompt 追加的格式约束切掉，
        否则 OVD 会去画面里找 "one word" 这种不存在的东西。"""
        phrase = Florence2Backend._phrase_from_question(
            "Is there any oil leak on the pump? Answer with only one word: yes, no, or unclear."
        )
        self.assertNotIn("Answer", phrase)
        self.assertNotIn("?", phrase)
        self.assertTrue(phrase.lower().startswith("there any oil leak")
                        or phrase.lower().startswith("any oil leak")
                        or "oil leak" in phrase.lower())

    def test_target_beats_question_fallback(self):
        be = _FakeFlorence(detections={"oil leak": [_box()]})
        be.ask(
            self.rgb, "Is the pump leaking badly",
            hint=AskHint(kind="yes_no", target="oil leak"),
        )
        self.assertEqual(be.detect_calls, ["oil leak"], "填了 target 就不该再去猜问句")


if __name__ == "__main__":
    unittest.main()
