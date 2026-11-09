"""端到端链路测试：取图 → 裁 ROI → 问模型 → 判定 → 存证。

用 `FileFrameSource` + `DummyBackend`，因此**不需要 ROS、不需要 GPU、不需要模型权重**。
它守的是「各层能否正确拼装」，而不是模型准确率——准确率要靠现场样本回归。

    python3 -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest

import numpy as np

from inspection_perception.capture.decode import crop_roi, encode_jpeg
from inspection_perception.capture.file_source import FileFrameSource
from inspection_perception.pipeline import InspectionPipeline, dump_results, summarize
from inspection_perception.schemas import CheckItem, CheckKind, CheckStatus
from inspection_perception.tasks import load_check_items
from inspection_perception.vlm.dummy import DummyBackend


def _encodable() -> bool:
    """当前环境是否有 cv2 或 PIL。都没有的话整个链路无法跑图。"""
    try:
        encode_jpeg(np.zeros((8, 8, 3), dtype=np.uint8))
        return True
    except Exception:
        return False


ENCODABLE = _encodable()
SKIP_REASON = "需要 opencv 或 pillow 才能编解码图像"

# 从本文件位置推仓库根，而不依赖当前工作目录——测试不应因为换个目录跑就失败
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestCropRoi(unittest.TestCase):
    def test_no_roi_returns_original(self):
        rgb = np.zeros((100, 200, 3), dtype=np.uint8)
        self.assertIs(crop_roi(rgb, None), rgb)

    def test_crop_geometry(self):
        rgb = np.zeros((100, 200, 3), dtype=np.uint8)
        out = crop_roi(rgb, (0.25, 0.25, 0.75, 0.75), pad=0.0)
        self.assertEqual(out.shape, (50, 100, 3))  # h=100*0.5, w=200*0.5

    def test_pad_expands(self):
        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        tight = crop_roi(rgb, (0.4, 0.4, 0.6, 0.6), pad=0.0)
        padded = crop_roi(rgb, (0.4, 0.4, 0.6, 0.6), pad=0.5)
        self.assertGreater(padded.shape[0], tight.shape[0])

    def test_pad_clamped_at_border(self):
        """贴边的 ROI 外扩后不能越界。"""
        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        out = crop_roi(rgb, (0.0, 0.0, 0.2, 0.2), pad=1.0)
        self.assertLessEqual(out.shape[0], 100)
        self.assertLessEqual(out.shape[1], 100)

    def test_tiny_roi_not_empty(self):
        """极窄 ROI 四舍五入后也要至少 1 像素。"""
        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        out = crop_roi(rgb, (0.5, 0.5, 0.5001, 0.5001), pad=0.0)
        self.assertGreater(out.size, 0)

    def test_invalid_roi_rejected(self):
        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        for bad in [(0.5, 0.0, 0.2, 1.0), (-0.1, 0.0, 0.5, 0.5), (0.0, 0.0, 1.5, 0.5)]:
            with self.assertRaises(ValueError):
                crop_roi(rgb, bad)


@unittest.skipUnless(ENCODABLE, SKIP_REASON)
class TestFileFrameSource(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ip_test_")
        for index in range(3):
            rgb = np.full((60, 80, 3), index * 40, dtype=np.uint8)
            with open(os.path.join(self.tmp, f"img_{index}.jpg"), "wb") as fh:
                fh.write(encode_jpeg(rgb))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_iterates_and_decodes(self):
        src = FileFrameSource(self.tmp)
        for _ in range(3):
            frame = src.get_latest()
            self.assertIsNotNone(frame)
            self.assertEqual(frame.as_rgb().shape, (60, 80, 3))
        self.assertEqual(src.stats()["files"], 3)

    def test_loop_wraps(self):
        src = FileFrameSource(self.tmp, loop=True)
        # 3 张图连取 7 次，期望 0,1,2,0,1,2,0。
        # 断言完整序列而不是「取 N 次后应回到第一张」——后者很容易把
        # off-by-one 写进期望里（本用例第一版就是这么错的）。
        names = [os.path.basename(src.get_latest().topic) for _ in range(7)]
        self.assertEqual(
            names,
            [
                "img_0.jpg",
                "img_1.jpg",
                "img_2.jpg",
                "img_0.jpg",
                "img_1.jpg",
                "img_2.jpg",
                "img_0.jpg",
            ],
        )

    def test_no_loop_terminates(self):
        src = FileFrameSource(self.tmp, loop=False)
        count = 0
        while src.get_latest() is not None:
            count += 1
            self.assertLess(count, 10, "no-loop 模式应当能自然结束")
        self.assertEqual(count, 3)

    def test_missing_path(self):
        with self.assertRaises(FileNotFoundError):
            FileFrameSource(os.path.join(self.tmp, "nope"))


@unittest.skipUnless(ENCODABLE, SKIP_REASON)
class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ip_pipe_")
        self.img_dir = os.path.join(self.tmp, "imgs")
        os.makedirs(self.img_dir)
        rgb = np.full((240, 320, 3), 128, dtype=np.uint8)
        with open(os.path.join(self.img_dir, "a.jpg"), "wb") as fh:
            fh.write(encode_jpeg(rgb))
        self.source = FileFrameSource(self.img_dir)
        self.evidence_dir = os.path.join(self.tmp, "evidence")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _pipeline(self, backend):
        return InspectionPipeline(
            source=self.source,
            backend=backend,
            evidence_dir=self.evidence_dir,
            max_age_ms=0,  # 文件源没有新鲜度概念
        )

    def test_pass_path_writes_evidence(self):
        backend = DummyBackend(answers={"closed": "yes"})
        item = CheckItem(id="valve", question="Is it closed", kind=CheckKind.YES_NO, expect="yes")
        result = self._pipeline(backend).run_item(item)

        self.assertEqual(result.status, CheckStatus.PASS)
        self.assertEqual(result.answer, "yes")
        self.assertTrue(os.path.isfile(result.evidence.image_path))
        self.assertEqual(result.evidence.raw_answer, "yes")
        self.assertEqual(result.evidence.model, "dummy@v1")
        self.assertIn("unclear", result.evidence.prompt)  # 格式约束确实拼上了
        self.assertGreater(result.latency_ms, 0)

    def test_fail_path(self):
        backend = DummyBackend(answers={"closed": "no"})
        item = CheckItem(id="valve", question="Is it closed", kind=CheckKind.YES_NO, expect="yes")
        result = self._pipeline(backend).run_item(item)
        self.assertEqual(result.status, CheckStatus.FAIL)
        self.assertTrue(result.needs_attention)

    def test_unknown_path(self):
        backend = DummyBackend(default="I cannot tell.")
        item = CheckItem(id="valve", question="Is it closed", kind=CheckKind.YES_NO, expect="yes")
        result = self._pipeline(backend).run_item(item)
        self.assertEqual(result.status, CheckStatus.UNKNOWN)

    def test_backend_error_becomes_error_not_fail(self):
        """模型炸了要判 ERROR——不能让系统故障伪装成设备异常。"""

        class BrokenBackend(DummyBackend):
            def ask(self, rgb, question, max_new_tokens=64, hint=None):
                from inspection_perception.vlm.base import VLMError

                raise VLMError("CUDA out of memory")

        item = CheckItem(id="valve", question="Is it closed", kind=CheckKind.YES_NO)
        result = self._pipeline(BrokenBackend()).run_item(item)
        self.assertEqual(result.status, CheckStatus.ERROR)
        self.assertIn("CUDA out of memory", result.reason)

    def test_roi_is_applied_before_inference(self):
        """存证图应当是裁剪后的尺寸，证明 ROI 真的生效了。"""
        backend = DummyBackend(answers={"uniform": "yes"})
        item = CheckItem(
            id="roi",
            question="Is the region uniform",
            kind=CheckKind.YES_NO,
            roi=(0.25, 0.25, 0.75, 0.75),
        )
        result = self._pipeline(backend).run_item(item)
        from inspection_perception.capture.decode import decode_jpeg

        with open(result.evidence.image_path, "rb") as fh:
            saved = decode_jpeg(fh.read())
        self.assertLess(saved.shape[1], 320)  # 原图宽 320，裁过就该更小

    def test_batch_shares_one_frame(self):
        backend = DummyBackend(answers={"first": "yes", "second": "yes"})
        items = [
            CheckItem(id="a", question="Is the first thing there", kind=CheckKind.YES_NO),
            CheckItem(id="b", question="Is the second thing there", kind=CheckKind.YES_NO),
        ]
        results = self._pipeline(backend).run_items(items, share_frame=True)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r.status is CheckStatus.PASS for r in results))
        # 同一帧问两次，取图源只被读了一次
        self.assertEqual(self.source.stats()["frames_total"], 1)

    def test_one_bad_item_does_not_break_batch(self):
        backend = DummyBackend(answers={"good": "yes"})
        items = [
            CheckItem(id="good", question="Is the good thing there", kind=CheckKind.YES_NO),
            # question 为空会在 build_prompt 抛错，应被吞成 ERROR 而不中断整批
            CheckItem(id="bad", question="  ", kind=CheckKind.YES_NO),
            CheckItem(id="good2", question="Is the good thing still there", kind=CheckKind.YES_NO),
        ]
        results = self._pipeline(backend).run_items(items)
        self.assertEqual([r.status for r in results][1], CheckStatus.ERROR)
        self.assertEqual(results[0].status, CheckStatus.PASS)
        self.assertEqual(results[2].status, CheckStatus.PASS)

    def test_summarize_and_dump(self):
        backend = DummyBackend(answers={"closed": "yes", "leak": "yes"})
        items = [
            CheckItem(id="ok", question="Is it closed", kind=CheckKind.YES_NO, expect="yes"),
            CheckItem(id="bad", question="Any leak", kind=CheckKind.YES_NO, expect="no"),
        ]
        results = self._pipeline(backend).run_items(items)
        stats = summarize(results)
        self.assertEqual(stats["total"], 2)
        self.assertEqual(stats["counts"]["pass"], 1)
        self.assertEqual(stats["counts"]["fail"], 1)
        self.assertEqual(stats["needs_attention"], ["bad"])

        out = dump_results(results, os.path.join(self.tmp, "runs", "r.json"))
        with open(out, encoding="utf-8") as fh:
            payload = json.load(fh)
        self.assertEqual(payload["summary"]["total"], 2)
        self.assertEqual(payload["results"][0]["status"], "pass")


class TestTaskLoading(unittest.TestCase):
    def test_load_smoke_json(self):
        items = load_check_items(os.path.join(REPO_ROOT, "configs/tasks/smoke.json"))
        self.assertGreater(len(items), 0)
        self.assertTrue(all(item.id for item in items))

    def test_load_demo_yaml_if_pyyaml_available(self):
        try:
            import yaml  # noqa: F401
        except ImportError:
            self.skipTest("未安装 PyYAML")
        items = load_check_items(os.path.join(REPO_ROOT, "configs/tasks/demo.yaml"))
        self.assertGreater(len(items), 0)
        by_id = {item.id: item for item in items}
        self.assertIsNotNone(by_id["panel_01_indicator"].roi)
        self.assertEqual(by_id["panel_01_indicator"].kind, CheckKind.STATE)
        # defaults 应当被合并进去
        self.assertEqual(by_id["breaker_01_position"].severity, "normal")

    def test_duplicate_id_rejected(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as fh:
            json.dump(
                {"items": [{"id": "x", "question": "q"}, {"id": "x", "question": "q2"}]}, fh
            )
            path = fh.name
        try:
            with self.assertRaises(ValueError):
                load_check_items(path)
        finally:
            os.unlink(path)

    def test_unknown_field_rejected(self):
        """配置里写错字段名必须报错，不能静默忽略。"""
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as fh:
            json.dump({"items": [{"id": "x", "question": "q", "quesion": "typo"}]}, fh)
            path = fh.name
        try:
            with self.assertRaises(ValueError):
                load_check_items(path)
        finally:
            os.unlink(path)

    def test_unquoted_yaml_bool_in_options_rejected(self):
        """options: [on, off] 不加引号会被 YAML 1.1 解析成 [True, False]，溜到判定层
        才炸成「decoding to str: need a bytes-like object, bool found」这种看不懂的
        TypeError——加载期必须拦住并提示加引号。（JSON 的 true 与 YAML 的裸 on 等价）"""
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as fh:
            json.dump({"items": [{"id": "x", "question": "q", "kind": "state",
                                  "options": ["on", True]}]}, fh)
            path = fh.name
        try:
            with self.assertRaises(ValueError) as ctx:
                load_check_items(path)
            self.assertIn("引号", str(ctx.exception))
        finally:
            os.unlink(path)

    def test_unquoted_yaml_bool_in_expect_rejected(self):
        """expect: yes 不加引号同理（→ True）。配置错误必须报在加载期。"""
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as fh:
            json.dump({"items": [{"id": "x", "question": "q", "kind": "yes_no",
                                  "expect": True}]}, fh)
            path = fh.name
        try:
            with self.assertRaises(ValueError):
                load_check_items(path)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
