"""巡检检查流水线：把取图 → 裁 ROI → 问模型 → 判定 → 存证串起来。

这是唯一同时知道「取图源」和「VLM 后端」的模块，两侧都通过协议接入，
所以 ROS1/ROS2/HTTP 任意取图源 × Moondream/Dummy 任意后端都能自由组合。

设计取舍
--------
**一次取图、多问几句。** 同一个点位往往要查好几件事（阀门关没关 + 有没有渗漏 +
压力表读数）。这里对同一帧连续提问，而不是每个问题重新取图——既省时间，
也保证同一批结论描述的是同一时刻的现场，报表才自洽。

**单条检查失败不中断整批。** 每条 item 独立 try/except，一个点位的模型抽风
不该让整轮巡检白跑。异常统一落成 `CheckStatus.ERROR`。
"""
from __future__ import annotations

import json
import os
import time
from typing import Iterable, List, Optional

from .capture.decode import crop_roi, encode_jpeg, resize_long_side
from .checks.ops import RuleError, answer_check
from .checks.verdict import judge
from .schemas import CheckItem, CheckResult, CheckStatus, Evidence
from .vlm.base import VLMError

# VLM 的视觉塔通常在 384~768 之间，喂 1080p 只是白烧算力。
# 裁完 ROI 再缩到这个长边，是延迟与细节的平衡点。
DEFAULT_INFER_LONG_SIDE = 768


class InspectionPipeline:
    """执行检查任务。

    Args:
        source: 实现 `FrameSource` 的取图源。
        backend: 实现 `VLMBackend` 的模型后端。
        evidence_dir: 存证目录。None 表示不存图（只跑不留痕，用于压测）。
        infer_long_side: 送进模型前的长边上限。
        max_age_ms: 取图新鲜度上限。巡检是停下来看，1 秒足够宽松。
        roi_pad: ROI 外扩比例，见 `crop_roi`。
        annotate_url: 取图服务的地址（如 http://127.0.0.1:8730）。非空时把
            每条检查的框与文字结论画到图上并 POST /annotate，最终发布到
            `/inspection/annotated/compressed`。空串表示不发布。
        max_text_lines: 底部文字面板最多显示几行（默认 3）。
        text_scale: 字号系数。现场嫌字小就调大（如 1.4），不用改代码。
    """

    def __init__(
        self,
        source,
        backend,
        evidence_dir: Optional[str] = "runs/evidence",
        infer_long_side: int = DEFAULT_INFER_LONG_SIDE,
        max_age_ms: float = 1000.0,
        roi_pad: float = 0.05,
        annotate_url: str = "",
        max_text_lines: int = 3,
        text_scale: float = 1.0,
    ) -> None:
        self.source = source
        self.backend = backend
        self.evidence_dir = evidence_dir
        self.infer_long_side = infer_long_side
        self.max_age_ms = max_age_ms
        self.roi_pad = roi_pad
        self.annotate_url = (annotate_url or "").rstrip("/")
        self.max_text_lines = max_text_lines
        self.text_scale = text_scale

        if evidence_dir:
            os.makedirs(evidence_dir, exist_ok=True)

    # ------------------------------------------------------------------ 单条
    def run_item(self, item: CheckItem, frame=None) -> CheckResult:
        """执行一条检查。

        Args:
            frame: 复用已取到的帧。None 时自己取一帧。
        """
        started = time.perf_counter()

        def elapsed_ms() -> float:
            return (time.perf_counter() - started) * 1000.0

        try:
            if frame is None:
                frame = self.source.get_latest(timeout_s=3.0, max_age_ms=self.max_age_ms)
            if frame is None:
                return CheckResult(
                    item_id=item.id,
                    status=CheckStatus.ERROR,
                    reason=f"取不到 {self.max_age_ms:.0f}ms 内的新鲜图像",
                    latency_ms=elapsed_ms(),
                )

            rgb = crop_roi(frame.as_rgb(), item.roi, pad=self.roi_pad)
            rgb = resize_long_side(rgb, self.infer_long_side)

            # 一个入口处理两条路径：写了 rule 走确定性算子，否则走自由问答。
            # judge() 对两者同一套解析——算子返回值刻意做成与模型回答同构。
            outcome = answer_check(rgb, item, self.backend)
            status, answer, reason = judge(item, outcome.answer)

            evidence = Evidence(
                image_path=self._save_evidence(item, rgb),
                raw_answer=outcome.raw or outcome.answer,
                prompt=outcome.prompt,
                detail=outcome.detail,
                model=getattr(self.backend, "name", "unknown"),
                frame_stamp_ns=frame.stamp_ns,
            )
            # 把框 + 文字结论画到模型实际看到的那张图上，回送取图节点发布
            self._publish_annotation(rgb, item, status, answer, outcome.detail)
            return CheckResult(
                item_id=item.id,
                status=status,
                answer=answer,
                reason=reason,
                latency_ms=elapsed_ms(),
                evidence=evidence,
            )

        except RuleError as exc:
            # 算子负不了（后端缺定位能力、rule 与 item 对不上）是配置/系统问题，不是设备不合格
            return CheckResult(
                item_id=item.id,
                status=CheckStatus.ERROR,
                reason=f"算子无法执行: {exc}",
                latency_ms=elapsed_ms(),
            )
        except VLMError as exc:
            # 模型故障是系统问题，不能算设备不合格
            return CheckResult(
                item_id=item.id,
                status=CheckStatus.ERROR,
                reason=f"模型故障: {exc}",
                latency_ms=elapsed_ms(),
            )
        except Exception as exc:
            return CheckResult(
                item_id=item.id,
                status=CheckStatus.ERROR,
                reason=f"{type(exc).__name__}: {exc}",
                latency_ms=elapsed_ms(),
            )

    def _publish_annotation(self, rgb, item, status, answer, detail) -> None:
        """把框 + 文字结论画到图上，POST 回取图节点的 /annotate。

        标注是**旁路能力**：网络不通、字体缺失、编码失败都只默默跳过，
        绝不能让一条检查因为"画不出图"而变成 ERROR。
        """
        if not self.annotate_url:
            return
        try:
            import urllib.request

            from .annotation import encode_jpeg as _encode_annotated
            from .annotation import iter_boxes, render_result

            lines = ["%s: %s" % (item.name or item.id, answer)]
            if item.location:
                lines.append("位置: %s" % item.location)
            status_text = getattr(status, "value", status)
            lines.append("状态: %s" % status_text)

            img = render_result(
                rgb,
                lines,
                boxes=iter_boxes(detail),
                status=status,
                max_lines=self.max_text_lines,
                scale=self.text_scale,
            )
            body = _encode_annotated(img)
            req = urllib.request.Request(
                self.annotate_url + "/annotate",
                data=body,
                headers={"Content-Type": "image/jpeg"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                resp.read(0)
        except Exception:
            pass

    def _save_evidence(self, item: CheckItem, rgb) -> str:
        """保存模型实际看到的那张图（已裁 ROI、已缩放）。

        存裁剪后的图而不是原图，是为了让人复核时看到的画面和模型看到的一致——
        存原图的话，「模型为什么判错」永远查不清。
        """
        if not self.evidence_dir:
            return ""
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in item.id)
        path = os.path.join(self.evidence_dir, f"{int(time.time())}_{safe_id}.jpg")
        try:
            with open(path, "wb") as fh:
                fh.write(encode_jpeg(rgb, quality=88))
        except Exception as exc:
            print(f"! 存证写入失败 {path}: {exc}")
            return ""
        return path

    # ------------------------------------------------------------------ 批量
    def run_items(self, items: Iterable[CheckItem], share_frame: bool = True) -> List[CheckResult]:
        """执行一批检查。

        Args:
            share_frame: True 时整批共用同一帧（同一点位的多个检查项应当如此）；
                False 时每条各取新帧（跨点位或需要重新对焦时用）。
        """
        items = list(items)
        frame = None
        if share_frame and items:
            frame = self.source.get_latest(timeout_s=3.0, max_age_ms=self.max_age_ms)
        return [self.run_item(item, frame=frame) for item in items]


def summarize(results: List[CheckResult]) -> dict:
    """汇总一轮结果。给上层调度和报表用。"""
    counts = {status.value: 0 for status in CheckStatus}
    for result in results:
        counts[result.status.value] += 1
    latencies = [r.latency_ms for r in results]
    return {
        "total": len(results),
        "counts": counts,
        "needs_attention": [r.item_id for r in results if r.needs_attention],
        "latency_ms": {
            "mean": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
            "max": round(max(latencies), 1) if latencies else 0.0,
        },
    }


def dump_results(results: List[CheckResult], path: str) -> str:
    """把结果写成 JSON。目录不存在会自动建。"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "summary": summarize(results),
        "results": [r.to_dict() for r in results],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return path
