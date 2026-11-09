"""Odin 图像话题取图 —— ROS 2（rclpy）实现。

适配话题（话题名 2026-08-01 在 Go2 NX 上实测确认）：
    /odin1/image             sensor_msgs/msg/Image
    /odin1/image/compressed  sensor_msgs/msg/CompressedImage

注意 compressed 的全名是 `/odin1/image/compressed`，不是 `/odin1/compressed_image`——
ODIN 驱动用的是 image_transport 的标准命名（在 raw 话题名后追加 `/compressed`）。

ROS 1 版本见 `ros1_odin_node.py`；两者对外接口一致，
用 `make_frame_source()`（`factory.py`）可自动选择。

关键决策
--------
**默认优先 compressed。** raw Image 在 1080p@10fps 下约 500 MB/s，会吃满
NX 网口并拖垮 DDS；compressed 约 5 MB/s，且能零解码透传给远端推理服务。
巡检检查是「到点位取一帧」，JPEG 有损对指示灯/铭牌/表盘的影响可忽略。

**订阅侧 QoS 固定 BEST_EFFORT + depth=1。** 相机驱动通常用 SensorDataQoS
发布；订阅端 BEST_EFFORT 能收 RELIABLE 发布者的数据，反之不成立，所以
BEST_EFFORT 是唯一「两种发布者都兼容」的选择。depth=1 保证拿到的是最新帧。

用法
----
    source /opt/ros/<distro>/setup.bash
    python3 -m inspection_perception.capture.ros_odin_node --probe 5
    python3 -m inspection_perception.capture.ros_odin_node --serve --port 8730
"""
from __future__ import annotations

import argparse
import os
import threading
import time
from typing import Optional

from .base import Frame, LatestFrameStore
from .decode import raw_to_rgb, resize_long_side

DEFAULT_RAW_TOPIC = "/odin1/image"
# image_transport 约定：compressed 话题挂在 raw 话题名下面，而不是平级另起一个名字。
DEFAULT_COMPRESSED_TOPIC = "/odin1/image/compressed"
# 鱼眼去畸变后的图。Odin 是鱼眼镜头（工作区里有 fish2pinhole 包），
# 读表盘/铭牌这类需要看清细节和直线的检查项，用去畸变图比原始鱼眼图更靠谱。
UNDISTORTED_TOPIC = "/odin1/image/undistorted"


class OdinFrameSource:
    """把 Odin 图像话题包装成 `FrameSource`（ROS 2 实现）。

    本类自带 executor 线程，构造后即开始收帧，调用方只需要 `get_latest()`——
    pipeline 侧因此完全不需要写 ROS 代码。
    """

    def __init__(
        self,
        raw_topic: str = DEFAULT_RAW_TOPIC,
        compressed_topic: str = DEFAULT_COMPRESSED_TOPIC,
        prefer: str = "compressed",
        long_side: Optional[int] = None,
        node_name: str = "inspection_capture_odin",
        spin_in_thread: bool = True,
    ) -> None:
        """
        Args:
            prefer: "compressed" | "raw" | "both"。both 会都订阅，取更新的那个。
            long_side: raw 帧收下时即按长边缩放，省内存与后续算力。
                compressed 帧不在此处解码，缩放留到 `as_rgb()` 之后由上层做。
            spin_in_thread: False 时由调用方自己 spin（例如挂到外部 executor）。
        """
        import rclpy
        from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
        from sensor_msgs.msg import CompressedImage, Image

        if prefer not in ("compressed", "raw", "both"):
            raise ValueError(f"prefer 只能是 compressed/raw/both，收到 {prefer!r}")

        self._long_side = long_side
        self._store = LatestFrameStore()

        self._owns_context = not rclpy.ok()
        if self._owns_context:
            rclpy.init()

        self._node = rclpy.create_node(node_name)
        self._rclpy = rclpy

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        self._subs = []
        if prefer in ("compressed", "both"):
            self._subs.append(
                self._node.create_subscription(
                    CompressedImage, compressed_topic, self._on_compressed, qos
                )
            )
            self._node.get_logger().info(f"订阅 CompressedImage: {compressed_topic}")
        if prefer in ("raw", "both"):
            self._subs.append(self._node.create_subscription(Image, raw_topic, self._on_raw, qos))
            self._node.get_logger().info(f"订阅 Image: {raw_topic}")

        self._spin_thread: Optional[threading.Thread] = None
        if spin_in_thread:
            self._spin_thread = threading.Thread(target=self._spin, daemon=True)
            self._spin_thread.start()

    # ---------------------------------------------------------------- 内部
    def _spin(self) -> None:
        try:
            self._rclpy.spin(self._node)
        except Exception:  # 关闭时 context 失效会抛，属正常退出路径
            pass

    @staticmethod
    def _stamp_ns(header) -> int:
        return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)

    def _on_compressed(self, msg) -> None:
        # 不解码，只搬字节。format 形如 "jpeg" / "rgb8; jpeg compressed bgr8"
        self._store.put(
            Frame(
                seq=0,
                stamp_ns=self._stamp_ns(msg.header),
                frame_id=msg.header.frame_id,
                topic="compressed",
                encoding=str(msg.format),
                jpeg=bytes(msg.data),
            ),
            "compressed",
        )

    def _on_raw(self, msg) -> None:
        try:
            rgb = raw_to_rgb(bytes(msg.data), msg.height, msg.width, msg.step, msg.encoding)
        except Exception as exc:
            self._node.get_logger().error(f"raw Image 转换失败: {exc}")
            return
        rgb = resize_long_side(rgb, self._long_side)
        self._store.put(
            Frame(
                seq=0,
                stamp_ns=self._stamp_ns(msg.header),
                frame_id=msg.header.frame_id,
                topic="raw",
                width=rgb.shape[1],
                height=rgb.shape[0],
                encoding=msg.encoding,
                rgb=rgb,
            ),
            "raw",
        )

    # ---------------------------------------------------------------- 对外
    def get_latest(self, timeout_s: float = 2.0, max_age_ms: float = 1000.0) -> Optional[Frame]:
        return self._store.get_latest(timeout_s=timeout_s, max_age_ms=max_age_ms)

    def wait_for_first_frame(self, timeout_s: float = 10.0) -> bool:
        return self._store.wait_for_first(timeout_s=timeout_s)

    def stats(self) -> dict:
        return self._store.stats()

    def close(self) -> None:
        try:
            self._node.destroy_node()
        finally:
            if self._owns_context and self._rclpy.ok():
                self._rclpy.shutdown()


# -------------------------------------------------------------------- CLI
NO_FRAME_HINT = (
    "！超时没收到任何帧。依次检查：\n"
    "  1. ros2 topic list | grep odin              话题是否存在\n"
    "  2. ros2 topic hz /odin1/image/compressed    是否真的在发\n"
    "  3. echo $ROS_DOMAIN_ID                      两端是否一致\n"
    "  4. 跨机时确认 RMW 实现与组播是否连通"
)


def probe_and_save(src, count: int, out_dir: str, hint: str = NO_FRAME_HINT) -> int:
    """保存若干帧并打印诊断信息。ROS 1/ROS 2 两个实现共用。"""
    os.makedirs(out_dir, exist_ok=True)
    if not src.wait_for_first_frame(timeout_s=15.0):
        print(hint)
        return 1

    saved, last_seq = 0, -1
    while saved < count:
        frame = src.get_latest(timeout_s=5.0, max_age_ms=0)
        if frame is None:
            break
        if frame.seq == last_seq:
            time.sleep(0.02)
            continue
        last_seq = frame.seq
        rgb = frame.as_rgb()  # 顺带验证 compressed 能否正确解码
        path = os.path.join(out_dir, f"odin_{frame.seq:04d}.jpg")
        with open(path, "wb") as fh:
            fh.write(frame.as_jpeg())
        print(f"{frame.describe()} -> {path}  (解码后 {rgb.shape})")
        saved += 1

    print("统计:", src.stats())
    return 0


def build_argparser(desc: str) -> argparse.ArgumentParser:
    """ROS 1/ROS 2 两个 CLI 共用的参数定义。"""
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--raw-topic", default=DEFAULT_RAW_TOPIC)
    ap.add_argument("--compressed-topic", default=DEFAULT_COMPRESSED_TOPIC)
    ap.add_argument(
        "--prefer",
        default="both",
        choices=["compressed", "raw", "both"],
        help="订阅哪个话题；探测阶段建议 both，上线建议 compressed",
    )
    ap.add_argument("--long-side", type=int, default=0, help="raw 帧长边缩放，0 表示不缩放")
    ap.add_argument("--probe", type=int, metavar="N", help="保存 N 帧后退出")
    ap.add_argument("--out-dir", default="_bootstrap/ros_probe", help="--probe 的输出目录")
    ap.add_argument("--serve", action="store_true", help="常驻并起 HTTP 取图服务")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8730)
    return ap


def main(argv=None) -> int:
    args = build_argparser("Odin 图像话题取图节点（ROS 2）").parse_args(argv)

    src = OdinFrameSource(
        raw_topic=args.raw_topic,
        compressed_topic=args.compressed_topic,
        prefer=args.prefer,
        long_side=args.long_side or None,
    )
    try:
        if args.probe:
            return probe_and_save(src, args.probe, args.out_dir)
        if args.serve:
            from .frame_server import serve

            src.wait_for_first_frame(timeout_s=15.0)
            serve(src, host=args.host, port=args.port)
            return 0
        print("需要指定 --probe N 或 --serve")
        return 2
    except KeyboardInterrupt:
        print("\n已中断")
    finally:
        src.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
