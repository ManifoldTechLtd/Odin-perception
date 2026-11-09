"""ROS 2 取图 + 标注发布（薄适配层），供推理侧通过 HTTP 使用。

一条命令起两个能力
------------------
    python -m inspection_perception.adapters.ros2_capture --serve \
        --raw-topic /odin1/image/undistorted --prefer raw --long-side 768 --port 8730

  GET  /latest.jpg        推理侧取最新帧（可带 max_age_ms / long_side / quality）
  GET  /meta.json         帧元信息（话题、时间戳、尺寸、新鲜度）
  GET  /healthz           就绪探针
  POST /annotate          推理侧把画好标注的 JPEG 送回 → 本节点发布到
                          /inspection/annotated/compressed（RViz / rosbridge 可看）

为什么默认是 undistorted
------------------------
Odin 是鱼眼镜头，去畸变后的针孔图对"读表盘/铭牌/直线裂纹"这类检查更可靠。
注意：`/odin1/image/undistorted` 需要在 odin 驱动的 config 里开启（`sendrgbundistort: 1`），
否则该话题只在 advertise、没有数据；未开启时请改用 `--raw-topic /odin1/image`
或 `--prefer compressed`。

推理侧怎么接
------------
    perception-run --source http --http-url http://127.0.0.1:8730 \
        --backend florence2 --tasks <清单>

本模块不加载任何模型，也不做判据——那是推理侧的事。ROS 只在这里出现，
换 ROS 版本时只需另写一个同接口的取图源（`get_latest/wait_for_first/stats/close/annotate`）。
"""
from __future__ import annotations

import argparse
import time
from typing import Any, Dict, Optional

from ..capture.base import Frame
from ..capture.ros_odin_node import (
    DEFAULT_COMPRESSED_TOPIC,
    UNDISTORTED_TOPIC,
    OdinFrameSource,
    probe_and_save,
)
from .annotation import DEFAULT_ANNOTATE_TOPIC, NullAnnotationSink, Ros2AnnotationPublisher


class Ros2CaptureSource:
    """`OdinFrameSource` + 标注发布：满足 frame_server 需要的鸭子协议。

    对外只暴露四个数据方法（`get_latest / wait_for_first_frame / stats / close`）
    加一个回送方法 `annotate`——和 HTTP 服务的契约一一对应。
    """

    def __init__(
        self,
        raw_topic: str = UNDISTORTED_TOPIC,
        compressed_topic: str = DEFAULT_COMPRESSED_TOPIC,
        prefer: str = "raw",
        long_side: Optional[int] = 768,
        annotate_topic: Optional[str] = DEFAULT_ANNOTATE_TOPIC,
    ) -> None:
        # 订阅侧沿用 capture/ros_odin_node.py：自带 executor 线程，构造后即收帧
        self._src = OdinFrameSource(
            raw_topic=raw_topic,
            compressed_topic=compressed_topic,
            prefer=prefer,
            long_side=long_side,
        )
        # 复用取图源已创建的节点（同包内约定：OdinFrameSource 持有 self._node）
        node = getattr(self._src, "_node", None)
        if annotate_topic and node is not None:
            self._sink: Any = Ros2AnnotationPublisher(node, annotate_topic)
        else:
            self._sink = NullAnnotationSink()
            if annotate_topic:
                print("[adapters] ! 取图源未暴露节点，标注发布已禁用")

    # ---- frame_server 需要的四个方法 ----
    def get_latest(self, timeout_s: float = 2.0, max_age_ms: float = 1000.0) -> Optional[Frame]:
        return self._src.get_latest(timeout_s=timeout_s, max_age_ms=max_age_ms)

    def wait_for_first_frame(self, timeout_s: float = 10.0) -> bool:
        return self._src.wait_for_first_frame(timeout_s=timeout_s)

    def stats(self) -> Dict[str, Any]:
        merged = dict(self._src.stats())
        merged["annotate"] = self._sink.stats()
        return merged

    def close(self) -> None:
        self._src.close()

    # ---- 标注回送（frame_server 的 POST /annotate 会调这里）----
    def annotate(self, jpeg_bytes: bytes) -> None:
        self._sink.annotate(jpeg_bytes)


def build_argparser(desc: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=desc)
    g = ap.add_argument_group("取图")
    g.add_argument("--raw-topic", default=UNDISTORTED_TOPIC,
                   help="原始/去畸变 Image 话题（默认 Odin 去畸变话题；需驱动侧开启 sendrgbundistort）")
    g.add_argument("--compressed-topic", default=DEFAULT_COMPRESSED_TOPIC)
    g.add_argument("--prefer", default="raw", choices=["compressed", "raw", "both"],
                   help="订阅哪路；端侧默认 raw（=去畸变图），带宽紧张时用 compressed")
    g.add_argument("--long-side", type=int, default=768,
                   help="raw 帧长边缩放，0 表示不缩放；默认 768 与推理侧 --infer-long-side 对齐")
    g.add_argument("--probe", type=int, metavar="N", help="只保存 N 帧后退出（用于验证图源）")
    g.add_argument("--out-dir", default="runs/probe", help="--probe 的图片输出目录")

    g = ap.add_argument_group("HTTP 服务")
    g.add_argument("--host", default="0.0.0.0")
    g.add_argument("--port", type=int, default=8730)
    g.add_argument("--wait-first", type=float, default=15.0,
                   help="起服务前等待首帧的秒数（0 表示不等待）")

    g = ap.add_argument_group("标注发布")
    g.add_argument("--annotate-topic", default=DEFAULT_ANNOTATE_TOPIC,
                   help="标注图发布话题；传空字符串则不发布")
    return ap


def main(argv=None) -> int:
    args = build_argparser("ROS 2 取图 + 标注发布（供推理侧 HTTP 调用）").parse_args(argv)

    src = Ros2CaptureSource(
        raw_topic=args.raw_topic,
        compressed_topic=args.compressed_topic,
        prefer=args.prefer,
        long_side=args.long_side or None,
        annotate_topic=args.annotate_topic or None,
    )
    try:
        if args.probe:
            return probe_and_save(src._src, args.probe, args.out_dir)

        from ..capture.frame_server import serve

        if args.wait_first > 0:
            print(f"[adapters] 等待首帧（最多 {args.wait_first:.0f}s）...")
            if not src.wait_for_first_frame(timeout_s=args.wait_first):
                print("[adapters] ! 未收到首帧，仍启动 HTTP 服务；请检查话题与驱动开关（sendrgbundistort）")
        print(f"[adapters] 取图: {src.stats()}")
        print(f"[adapters] 推理侧用法: perception-run --source http "
              f"--http-url http://127.0.0.1:{args.port} --backend florence2 --tasks <清单>")
        serve(src, host=args.host, port=args.port)
        return 0
    except KeyboardInterrupt:
        print(f"\n[adapters] 停止（统计 {src.stats()}）")
        return 0
    finally:
        src.close()


if __name__ == "__main__":
    raise SystemExit(main())
