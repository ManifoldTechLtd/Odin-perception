"""判定逻辑的单元测试。

这层最需要测试，因为它处理的是模型输出的不确定性——而模型输出无法在代码里穷举，
只能靠这些真实见过的变体样本来守住解析的正确性。

用 stdlib unittest，不引入 pytest：这些测试要能在 NX 上直接跑。
    python3 -m unittest discover -s tests -v
"""
from __future__ import annotations

import unittest

from inspection_perception.checks import build_prompt, judge, parse_state, parse_text, parse_yes_no
from inspection_perception.schemas import CheckItem, CheckKind, CheckStatus


class TestParseYesNo(unittest.TestCase):
    def test_bare_words(self):
        self.assertEqual(parse_yes_no("yes"), "yes")
        self.assertEqual(parse_yes_no("No"), "no")
        self.assertEqual(parse_yes_no("Yes."), "yes")

    def test_full_sentences(self):
        self.assertEqual(parse_yes_no("Yes, the valve is closed."), "yes")
        self.assertEqual(parse_yes_no("No, there are no visible leaks."), "no")

    def test_substring_traps(self):
        """子串匹配会在这些用例上翻车，词边界匹配不会。"""
        # "not"、"nothing"、"unknown" 里都含 "no"
        self.assertIsNone(parse_yes_no("The handle is not visible in this image."))
        self.assertIsNone(parse_yes_no("Nothing unusual here."))

    def test_hedge_wins_over_answer(self):
        """既有犹豫又有答案时，必须判不确定。"""
        self.assertIsNone(parse_yes_no("I cannot tell if it is closed."))
        self.assertIsNone(parse_yes_no("unclear"))
        self.assertIsNone(parse_yes_no("The image is too blurry, but yes probably."))

    def test_contradiction(self):
        """yes 和 no 同时出现说明解析不可靠。"""
        self.assertIsNone(parse_yes_no("Yes, but no leak is visible."))

    def test_chinese(self):
        self.assertEqual(parse_yes_no("是的"), "yes")
        self.assertIsNone(parse_yes_no("看不清"))


class TestParseState(unittest.TestCase):
    OPTIONS = ["green", "red", "yellow", "off"]

    def test_single_hit(self):
        self.assertEqual(parse_state("green", self.OPTIONS), "green")
        self.assertEqual(parse_state("The light is red.", self.OPTIONS), "red")

    def test_multiple_hits_rejected(self):
        """提到多个选项时不能随便挑一个。"""
        self.assertIsNone(parse_state("It is either red or green.", self.OPTIONS))

    def test_no_hit(self):
        self.assertIsNone(parse_state("blue", self.OPTIONS))

    def test_hedge(self):
        self.assertIsNone(parse_state("unclear", self.OPTIONS))

    def test_word_boundary(self):
        """"off" 不应被 "offset"/"officer" 命中。"""
        self.assertIsNone(parse_state("There is an officer nearby.", ["off"]))


class TestParseText(unittest.TestCase):
    def test_strips_prefix_and_quotes(self):
        self.assertEqual(parse_text('The text reads: "PUMP-01"'), "PUMP-01")
        self.assertEqual(parse_text("It says 0.45"), "0.45")
        self.assertEqual(parse_text("Answer: A-102."), "A-102")

    def test_plain(self):
        self.assertEqual(parse_text("0.45"), "0.45")

    def test_hedge(self):
        self.assertIsNone(parse_text("I cannot read the label."))

    def test_unanswerable_is_hedge_not_a_reading(self):
        """Florence-2 对无文字 ROI 的 OCR 固定回 "unanswerable"——这是「没读到」，
        不能当成一个读数去匹配 expect 正则，否则「没读到」会被误判成「不合格」。"""
        self.assertIsNone(parse_text("unanswerable"))


class TestJudge(unittest.TestCase):
    def test_yes_no_pass_and_fail(self):
        item = CheckItem(id="v", question="Is it closed", kind=CheckKind.YES_NO, expect="yes")
        self.assertEqual(judge(item, "yes")[0], CheckStatus.PASS)
        self.assertEqual(judge(item, "no")[0], CheckStatus.FAIL)

    def test_expect_no_inverts_verdict(self):
        """渗漏检查这类「期望没有」的项，expect=no 时 no 才是合格。"""
        item = CheckItem(id="leak", question="Any leak", kind=CheckKind.YES_NO, expect="no")
        self.assertEqual(judge(item, "no")[0], CheckStatus.PASS)
        self.assertEqual(judge(item, "yes")[0], CheckStatus.FAIL)

    def test_yes_no_default_expect_is_yes(self):
        item = CheckItem(id="v", question="Is it closed", kind=CheckKind.YES_NO)
        self.assertEqual(judge(item, "yes")[0], CheckStatus.PASS)

    def test_unknown_when_unparseable(self):
        item = CheckItem(id="v", question="Is it closed", kind=CheckKind.YES_NO, expect="yes")
        status, answer, reason = judge(item, "The image is too dark to tell.")
        self.assertEqual(status, CheckStatus.UNKNOWN)
        self.assertEqual(answer, "")
        self.assertTrue(reason)

    def test_empty_answer_is_unknown(self):
        item = CheckItem(id="v", question="Is it closed", kind=CheckKind.YES_NO)
        self.assertEqual(judge(item, "")[0], CheckStatus.UNKNOWN)

    def test_state_without_expect_only_records(self):
        item = CheckItem(
            id="s", question="What color", kind=CheckKind.STATE, options=["green", "red"]
        )
        status, answer, _ = judge(item, "red")
        self.assertEqual(status, CheckStatus.PASS)
        self.assertEqual(answer, "red")

    def test_read_text_expect_as_regex(self):
        """expect 当正则用，可以表达合格区间。"""
        item = CheckItem(
            id="g",
            question="What number",
            kind=CheckKind.READ_TEXT,
            expect=r"^0?\.[3-6]\d*$",
        )
        self.assertEqual(judge(item, "0.45")[0], CheckStatus.PASS)
        self.assertEqual(judge(item, "0.85")[0], CheckStatus.FAIL)

    def test_bad_regex_is_error_not_fail(self):
        """配置写错是系统问题，要判 ERROR，不能算设备不合格。"""
        item = CheckItem(id="g", question="What", kind=CheckKind.READ_TEXT, expect="[invalid")
        self.assertEqual(judge(item, "123")[0], CheckStatus.ERROR)

    def test_bad_expect_for_yes_no_is_error(self):
        item = CheckItem(id="v", question="Is it", kind=CheckKind.YES_NO, expect="maybe")
        self.assertEqual(judge(item, "yes")[0], CheckStatus.ERROR)

    def test_count(self):
        item = CheckItem(id="c", question="How many", kind=CheckKind.COUNT, expect="2")
        self.assertEqual(judge(item, "There are 2 extinguishers.")[0], CheckStatus.PASS)
        self.assertEqual(judge(item, "3")[0], CheckStatus.FAIL)

    def test_describe_never_fails(self):
        item = CheckItem(id="d", question="Describe", kind=CheckKind.DESCRIBE)
        status, answer, _ = judge(item, "A pump with some rust on the flange.")
        self.assertEqual(status, CheckStatus.PASS)
        self.assertIn("rust", answer)


class TestBuildPrompt(unittest.TestCase):
    def test_yes_no_adds_format_constraint(self):
        item = CheckItem(id="v", question="Is the valve closed", kind=CheckKind.YES_NO)
        prompt = build_prompt(item)
        self.assertIn("Is the valve closed?", prompt)
        self.assertIn("yes", prompt)
        self.assertIn("unclear", prompt)

    def test_state_lists_options(self):
        item = CheckItem(
            id="s", question="What color", kind=CheckKind.STATE, options=["green", "red"]
        )
        prompt = build_prompt(item)
        self.assertIn("green, red", prompt)

    def test_state_without_options_rejected(self):
        item = CheckItem(id="s", question="What color", kind=CheckKind.STATE)
        with self.assertRaises(ValueError):
            build_prompt(item)

    def test_empty_question_rejected(self):
        item = CheckItem(id="x", question="   ", kind=CheckKind.YES_NO)
        with self.assertRaises(ValueError):
            build_prompt(item)


class TestCheckResultStatus(unittest.TestCase):
    def test_unknown_needs_attention(self):
        """看不清必须进复核清单——漏判比多看一眼危险得多。"""
        from inspection_perception.schemas import CheckResult

        for status in (CheckStatus.FAIL, CheckStatus.UNKNOWN, CheckStatus.ERROR):
            self.assertTrue(CheckResult(item_id="x", status=status).needs_attention)
        self.assertFalse(CheckResult(item_id="x", status=CheckStatus.PASS).needs_attention)


if __name__ == "__main__":
    unittest.main()
