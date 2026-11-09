"""命令行入口：跑一轮巡检检查。

三种典型用法
------------
**离线回归**（无 ROS、无 GPU，只验链路）::

    python3 -m inspection_perception.cli \\
        --tasks configs/tasks/demo.yaml \\
        --source file --file-path _bootstrap/ros_probe \\
        --backend dummy

**开发机上用真模型验准确率**（有 GPU，图从本地目录来）::

    python3 -m inspection_perception.cli \\
        --tasks configs/tasks/demo.yaml \\
        --source file --file-path samples/ \\
        --backend moondream

**NX 上端到端**（取图进程另起，见 frame_server 的说明）::

    # 进程 1（ROS 环境）
    python3 -m inspection_perception.capture.ros1_odin_node --serve --port 8730
    # 进程 2（模型环境，不需要 ROS）
    python3 -m inspection_perception.cli --tasks configs/tasks/demo.yaml \\
        --source http --backend moondream

日常不必手敲这些参数——`scripts/start_inspection.sh --profile A|B|B2` 会按
profile 把上面两个进程连起来，参数都在 `configs/profiles/*.env` 里。
本文件是那层脚本的底座，也保留直接调用，方便临时试参数。
"""
from __future__ import annotations

import argparse
import sys
import time

from .capture.factory import make_frame_source
from .pipeline import InspectionPipeline, dump_results, summarize
from .tasks import load_check_items
from .vlm.registry import AVAILABLE, TASK_TOKEN_BACKENDS, make_backend


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="巡检感知：按任务清单跑一轮检查",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--tasks", required=True, help="任务清单 YAML/JSON")

    grp = ap.add_argument_group("取图")
    grp.add_argument(
        "--source",
        default="auto",
        choices=["auto", "ros1", "ros2", "http", "file"],
        help="auto 会按环境探测 ROS 版本",
    )
    grp.add_argument("--file-path", help="--source file 时的图片文件或目录")
    grp.add_argument("--http-url", default="http://127.0.0.1:8730")
    grp.add_argument("--prefer", default="compressed", choices=["compressed", "raw", "both"])
    grp.add_argument("--max-age-ms", type=float, default=1000.0, help="取图新鲜度上限")

    grp = ap.add_argument_group("模型")
    grp.add_argument(
        "--backend",
        default="moondream",
        choices=sorted(AVAILABLE),
        help="; ".join(f"{k}: {v}" for k, v in AVAILABLE.items()),
    )
    # 不给默认值：每个后端有自己的 DEFAULT_MODEL_DIR，写死 moondream 的路径会让
    # `--backend florence2` 在用户没显式传 --model-dir 时去 models/moondream2 找权重。
    grp.add_argument("--model-dir", help="默认用所选后端自己的权重目录")
    grp.add_argument("--device", choices=["cuda", "cpu"], help="默认自动选择")
    grp.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="float32",
        help="推理精度，默认 float32。⚠️ Florence-2 的 remote code 里有 CPU 算子"
        "（`torch.exp`），用 float16 会在加载阶段直接报 "
        "`\"exp_vml_cpu\" not implemented for 'Half'`。除非你确认后端支持半精度，"
        "否则保持 float32。",
    )
    grp.add_argument("--infer-long-side", type=int, default=768)

    grp = ap.add_argument_group("输出")
    grp.add_argument("--evidence-dir", default="runs/evidence", help="空字符串表示不存图")
    grp.add_argument("--out", help="结果 JSON 路径，默认 runs/<时间戳>.json")

    grp = ap.add_argument_group("标注发布（可选）")
    grp.add_argument(
        "--annotate-url",
        default="",
        help="取图服务地址（如 http://127.0.0.1:8730）；非空时把检测框与文字结论画到图上，"
        "POST /annotate 发布到 /inspection/annotated/compressed。",
    )
    grp.add_argument(
        "--max-text-lines",
        type=int,
        default=3,
        help="底部文字面板最多显示几行（默认 3）",
    )
    grp.add_argument(
        "--text-scale",
        type=float,
        default=1.0,
        help="字号系数；现场嫌字小就调大（如 1.4）",
    )
    grp.add_argument(
        "--no-share-frame",
        action="store_true",
        help="每条检查各取一帧新图（默认整批共用一帧）",
    )
    return ap


# 需要一个明确检测词条才能工作的检查类型。read_text 走 <OCR>、describe 走
# caption，都不需要 target，所以不在这个集合里。
_TARGET_REQUIRED_KINDS = frozenset({"yes_no", "state", "count"})


def warn_missing_target(items, backend: str) -> None:
    """跑「只吃任务标签」的后端前先体检清单。

    Florence-2 这类后端没有自由问答能力，yes_no/state/count 全靠把 `target`
    当检测词条去做开放词汇检测。清单里漏填 `target` 时它只能猜——从自然语言问句里
    抠名词短语，抠不准就返回 unclear。等到现场才发现一整批 UNKNOWN，排查成本很高，
    所以在加载模型之前（这一步要几十秒）就把话说清楚。

    只警告不报错：清单可能是给 profile A 写的，临时换 B2 跑一下看看效果是合理需求。
    """
    if backend not in TASK_TOKEN_BACKENDS:
        return
    missing = [
        it.id
        for it in items
        if it.kind.value in _TARGET_REQUIRED_KINDS
        and not (it.target or "").strip()
        # 有 rule 的项走「定位 + 算子」路径，target 在 rule 里，不看顶层 target
        and not it.rule
    ]
    if not missing:
        return
    print(
        f"!! 后端 {backend} 不能自由问答，yes_no/state/count 依赖清单里的 target 字段，"
        f"以下 {len(missing)} 条没填，结果大概率是 UNKNOWN："
    )
    print(f"   {', '.join(missing)}")
    print("   给这些项补上 target（要检测的英文词条，如 oil leak / red indicator light）")


def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)

    items = load_check_items(args.tasks)
    print(f"清单 {args.tasks}: {len(items)} 条检查")

    source = make_frame_source(
        backend=args.source,
        prefer=args.prefer,
        http_url=args.http_url,
        file_path=args.file_path,
    )

    warn_missing_target(items, args.backend)

    # dummy 不吃这些参数，其余真模型后端签名一致。
    # 只传用户真给了的，没给的交给后端自己的默认值。
    backend_kwargs = {}
    if args.backend != "dummy":
        if args.model_dir:
            backend_kwargs["model_dir"] = args.model_dir
        if args.dtype:
            backend_kwargs["dtype"] = args.dtype
        if args.device:
            backend_kwargs["device"] = args.device
    print(f"加载后端 {args.backend} ...")
    load_started = time.perf_counter()
    backend = make_backend(args.backend, **backend_kwargs)
    print(f"后端就绪: {backend.name}  ({(time.perf_counter() - load_started):.1f}s)")

    pipeline = InspectionPipeline(
        source=source,
        backend=backend,
        evidence_dir=args.evidence_dir or None,
        infer_long_side=args.infer_long_side,
        max_age_ms=args.max_age_ms,
        annotate_url=getattr(args, "annotate_url", "") or "",
        max_text_lines=getattr(args, "max_text_lines", 3),
        text_scale=getattr(args, "text_scale", 1.0),
    )

    try:
        results = pipeline.run_items(items, share_frame=not args.no_share_frame)
    finally:
        source.close()
        backend.unload()

    print("")
    for result in results:
        print(result.summary_line())

    stats = summarize(results)
    print("")
    print(f"合计 {stats['total']} 条: {stats['counts']}")
    print(f"延迟 均值 {stats['latency_ms']['mean']}ms  最大 {stats['latency_ms']['max']}ms")
    if stats["needs_attention"]:
        print(f"需人工复核: {', '.join(stats['needs_attention'])}")

    out = args.out or f"runs/check_{time.strftime('%Y%m%d_%H%M%S')}.json"
    print(f"结果已写入 {dump_results(results, out)}")

    # 有需要复核的项就返回非 0，方便上层脚本据此决定是否告警
    return 1 if stats["needs_attention"] else 0


if __name__ == "__main__":
    sys.exit(main())
