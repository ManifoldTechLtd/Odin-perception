"""算子路径的单元测试 —— 全部用**合成图**，不需要 GPU / 权重 / ROS / 现场图。

为什么这套测试重要
------------------
用户没有现场图，但这不阻塞核心验证：几何和颜色算子是**确定性**的，可以用合成图
完全证明正确性——画一根已知角度的杆去验主轴角度，画一块纯色去验色相投票。
只有「模型框得准不准」需要真实图像，那部分要等 Go2 在线实测（文档 §8）。

运行::

    python3 -m unittest discover -s tests -v
"""
from __future__ import annotations

import unittest

import numpy as np

from inspection_perception.checks.color import classify_region, resolve_palette, rgb_to_hsv
from inspection_perception.checks.ops import RuleError, answer_check, validate_rule
from inspection_perception.checks.verdict import judge
from inspection_perception.geometry import (
    AMBIGUOUS,
    Box,
    HORIZONTAL,
    VERTICAL,
    angle_to_orientation,
    dedupe,
    filter_boxes,
    polygon_orientation,
)
from inspection_perception.schemas import CheckItem, CheckKind, CheckStatus


# =============================================================================
# 合成后端：只按构造时给的固定结果作答，让算子逻辑与真实模型完全解耦。
# =============================================================================
class FakeGrounding:
    """定位型后端替身：detect / segment / ocr 返回预置结果。"""

    name = "fake-grounding"

    def __init__(self, boxes=None, rings=None, text=""):
        self._boxes = list(boxes or [])
        self._rings = list(rings or [])
        self._text = text
        self.last_prompt = ""

    def detect(self, rgb, phrase, mode="ovd"):
        self.last_prompt = f"[detect:{mode}]{phrase}"
        return list(self._boxes)

    def segment(self, rgb, phrase):
        self.last_prompt = f"[seg]{phrase}"
        return list(self._rings)

    def ocr(self, rgb):
        return self._text


class FakeVQA:
    """自由问答后端替身：只有 ask()，没有任何定位能力。"""

    name = "fake-vqa"

    def __init__(self, answer="yes"):
        self._answer = answer
        self.last_prompt = "vqa-prompt"

    def ask(self, rgb, prompt, max_new_tokens=64, hint=None):
        return self._answer


def _item(**kw) -> CheckItem:
    kw.setdefault("id", "t")
    kw.setdefault("question", "q")
    return CheckItem(**kw)


def _solid(rgb, size=100):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, :] = rgb
    return img


# =============================================================================
# 几何：主轴角度（合成杆）
# =============================================================================
class TestGeometryOrientation(unittest.TestCase):
    VERTICAL_RECT = [(45, 10), (55, 10), (55, 90), (45, 90)]
    HORIZONTAL_RECT = [(10, 45), (90, 45), (90, 55), (10, 55)]
    SQUARE = [(30, 30), (70, 30), (70, 70), (30, 70)]
    DIAGONAL = [(10, 20), (20, 10), (90, 80), (80, 90)]

    def test_vertical_bar(self):
        angle, elong = polygon_orientation(self.VERTICAL_RECT)
        self.assertGreater(elong, 1.25)
        self.assertEqual(angle_to_orientation(angle, elong), VERTICAL)

    def test_horizontal_bar(self):
        angle, elong = polygon_orientation(self.HORIZONTAL_RECT)
        self.assertGreater(elong, 1.25)
        self.assertEqual(angle_to_orientation(angle, elong), HORIZONTAL)

    def test_square_is_ambiguous(self):
        # 近正方形：elongation≈1，主轴方向是噪声，必须判 ambiguous 而不是硬选
        angle, elong = polygon_orientation(self.SQUARE)
        self.assertLess(elong, 1.25)
        self.assertEqual(angle_to_orientation(angle, elong), AMBIGUOUS)

    def test_diagonal_is_ambiguous(self):
        # 斜 45°：落在死区带，不该被判成竖直或水平
        angle, elong = polygon_orientation(self.DIAGONAL)
        self.assertEqual(angle_to_orientation(angle, elong, tol_deg=30.0), AMBIGUOUS)

    def test_deadzone_prevents_flip(self):
        # 44° 与 46° 不能给出相反结论——tol 30° 时两者都落在 ambiguous
        self.assertEqual(angle_to_orientation(44.0, 5.0, tol_deg=30.0), AMBIGUOUS)
        self.assertEqual(angle_to_orientation(46.0, 5.0, tol_deg=30.0), AMBIGUOUS)


# =============================================================================
# 几何：框过滤与去重
# =============================================================================
class TestBoxOps(unittest.TestCase):
    def test_max_area_frac_drops_fullframe_box(self):
        # Florence-2 找不到目标时返回整图框，必须被 max_area_frac 滤掉
        full = Box(0, 0, 100, 100)
        small = Box(10, 10, 30, 30)
        kept = filter_boxes([full, small], 100, 100, max_area_frac=0.9)
        self.assertEqual(kept, [small])

    def test_dedupe_merges_overlaps(self):
        a = Box(10, 10, 50, 50)
        b = Box(12, 12, 52, 52)   # 与 a 高度重叠
        c = Box(60, 60, 80, 80)   # 独立
        self.assertEqual(len(dedupe([a, b, c], iou_thr=0.6)), 2)


# =============================================================================
# 颜色：色相投票（合成纯色块）
# =============================================================================
class TestColor(unittest.TestCase):
    def test_green(self):
        self.assertEqual(classify_region(_solid((0, 200, 0))).name, "green")

    def test_red(self):
        self.assertEqual(classify_region(_solid((220, 0, 0))).name, "red")

    def test_red_wraparound(self):
        # 色相约 349°，落在红色的第二段 [348,360)，单区间实现会漏掉这一半
        self.assertEqual(classify_region(_solid((220, 0, 40))).name, "red")

    def test_off_when_dark(self):
        # 灯灭：亮度中位数低于 off_v，色相是噪声不能投票
        self.assertEqual(classify_region(_solid((10, 10, 12))).name, "off")

    def test_gray_and_white(self):
        self.assertEqual(classify_region(_solid((128, 128, 128))).name, "gray")
        self.assertEqual(classify_region(_solid((240, 240, 240))).name, "white")

    def test_restricted_palette(self):
        # 只投红/绿两色，绿块仍判绿
        self.assertEqual(classify_region(_solid((0, 200, 0)), colors=["red", "green"]).name, "green")

    def test_rgb_to_hsv_ranges(self):
        h, s, v = rgb_to_hsv(_solid((0, 200, 0)))
        self.assertTrue((h >= 0).all() and (h < 360).all())
        self.assertAlmostEqual(float(h.mean()), 120.0, delta=2.0)

    def test_unknown_color_name_raises(self):
        with self.assertRaises(ValueError):
            resolve_palette(["greeen"])


# =============================================================================
# 算子分派：answer_check 的完整路径
# =============================================================================
class TestOpsDispatch(unittest.TestCase):
    IMG = np.zeros((100, 100, 3), dtype=np.uint8)

    def test_presence_yes(self):
        be = FakeGrounding(boxes=[Box(10, 10, 30, 30, "leak")])
        item = _item(kind=CheckKind.YES_NO, rule={"op": "presence", "target": "leak"})
        out = answer_check(self.IMG, item, be)
        self.assertEqual(out.answer, "yes")

    def test_presence_no_when_only_fullframe_box(self):
        # 只有整图框：max_area_frac 默认 0.9 会滤掉它 -> no（关键的假阳性防线）
        be = FakeGrounding(boxes=[Box(0, 0, 100, 100, "leak")])
        item = _item(kind=CheckKind.YES_NO, rule={"op": "presence", "target": "leak"})
        self.assertEqual(answer_check(self.IMG, item, be).answer, "no")

    def test_count_dedupes(self):
        boxes = [Box(10, 10, 50, 50), Box(12, 12, 52, 52), Box(60, 60, 80, 80)]
        be = FakeGrounding(boxes=boxes)
        item = _item(kind=CheckKind.COUNT, rule={"op": "count", "target": "extinguisher"})
        self.assertEqual(answer_check(self.IMG, item, be).answer, "2")

    def test_segment_orientation_to_business_answer(self):
        rings = [[(45, 10), (55, 10), (55, 90), (45, 90)]]  # 竖直杆
        be = FakeGrounding(rings=rings)
        item = _item(
            kind=CheckKind.YES_NO,
            expect="yes",
            rule={
                "op": "segment_orientation",
                "target": "valve handle",
                "map": {"vertical": "yes", "horizontal": "no"},
            },
        )
        out = answer_check(self.IMG, item, be)
        self.assertEqual(out.answer, "yes")
        # 端到端：judge 也应判 PASS，且不用为算子路径改一行
        status, answer, _ = judge(item, out.answer)
        self.assertEqual(status, CheckStatus.PASS)

    def test_segment_orientation_angle_map_wraparound(self):
        rings = [[(10, 45), (90, 45), (90, 55), (10, 55)]]  # 水平杆 -> angle≈0 -> off 区间
        be = FakeGrounding(rings=rings)
        item = _item(
            kind=CheckKind.STATE,
            options=["on", "off"],
            rule={
                "op": "segment_orientation",
                "target": "breaker",
                "angle_map": [
                    {"from": 60, "to": 120, "answer": "on"},
                    {"from": 120, "to": 60, "answer": "off"},  # 环绕区间覆盖近水平
                ],
            },
        )
        self.assertEqual(answer_check(self.IMG, item, be).answer, "off")

    def test_region_color(self):
        be = FakeGrounding()  # locate=false，不需要 detect
        img = _solid((0, 200, 0))
        item = _item(
            kind=CheckKind.STATE,
            options=["green", "red", "off"],
            rule={"op": "region_color", "colors": ["green", "red"], "locate": False},
        )
        self.assertEqual(answer_check(img, item, be).answer, "green")

    def test_ocr_text_uses_backend_ocr(self):
        be = FakeGrounding(text="AB-1234")
        item = _item(kind=CheckKind.READ_TEXT, rule={"op": "ocr_text"})
        self.assertEqual(answer_check(self.IMG, item, be).answer, "AB-1234")

    def test_ocr_text_falls_back_to_vqa(self):
        # 后端没有 ocr()，ocr_text 应回落自由问答，而不是报错
        be = FakeVQA(answer="XY-99")
        item = _item(kind=CheckKind.READ_TEXT, rule={"op": "ocr_text"})
        out = answer_check(self.IMG, item, be)
        self.assertEqual(out.answer, "XY-99")
        self.assertIn("fallback", out.detail)

    def test_detail_written_for_presence(self):
        be = FakeGrounding(boxes=[Box(10, 10, 30, 30, "x")])
        item = _item(kind=CheckKind.YES_NO, rule={"op": "presence", "target": "x"})
        out = answer_check(self.IMG, item, be)
        self.assertIn("boxes", out.detail)  # 存证里能看到框，判错了可回溯

    def test_missing_detect_capability_raises_ruleerror(self):
        # 用不能定位的后端跑 presence：明确报错，不能悄悄给答案
        be = FakeVQA()
        item = _item(kind=CheckKind.YES_NO, rule={"op": "presence", "target": "leak"})
        with self.assertRaises(RuleError):
            answer_check(self.IMG, item, be)

    def test_no_rule_goes_vqa(self):
        be = FakeVQA(answer="no")
        item = _item(kind=CheckKind.YES_NO)  # 没有 rule
        self.assertEqual(answer_check(self.IMG, item, be).answer, "no")


# =============================================================================
# 加载期校验：validate_rule
# =============================================================================
class TestValidateRule(unittest.TestCase):
    def test_none_ok(self):
        validate_rule(None)  # 不抛异常

    def test_valid_ok(self):
        validate_rule({"op": "segment_orientation", "map": {"vertical": "yes"}})

    def test_unknown_op(self):
        with self.assertRaises(ValueError):
            validate_rule({"op": "teleport"})

    def test_unknown_param_key(self):
        # max_area_fraction 手滑拼错，静默忽略会导致现场偶发误报，必须当场报错
        with self.assertRaises(ValueError):
            validate_rule({"op": "presence", "max_area_fraction": 0.9})

    def test_orientation_needs_map(self):
        with self.assertRaises(ValueError):
            validate_rule({"op": "segment_orientation", "target": "x"})

    def test_bad_map_key(self):
        with self.assertRaises(ValueError):
            validate_rule({"op": "segment_orientation", "map": {"verticle": "yes"}})

    def test_bool_map_value_rejected(self):
        """map: {vertical: yes} 裸写 yes 会被 YAML 1.1 解析成 True——必须加载期拦住，
        否则算子把 True 当答案吐给判定层，炸成看不懂的 TypeError。"""
        with self.assertRaises(ValueError):
            validate_rule({"op": "segment_orientation", "map": {"vertical": True}})

    def test_bool_angle_map_answer_rejected(self):
        """angle_map 的 answer 裸写 on/off 同理。"""
        with self.assertRaises(ValueError):
            validate_rule(
                {"op": "segment_orientation",
                 "angle_map": [{"from": 60, "to": 120, "answer": True}]}
            )

    def test_bad_color_name(self):
        with self.assertRaises(ValueError):
            validate_rule({"op": "region_color", "colors": ["greeen"]})


if __name__ == "__main__":
    unittest.main()
