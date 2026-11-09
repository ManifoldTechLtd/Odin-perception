"""Odin 图像话题取图 —— ROS 1（rospy）实现。

Go2 上的 `go2_start_roscore.sh` 指向 `roscore`，说明机载端很可能是 ROS 1。
本模块与 `ros_odin_node.py`（ROS 2）对外接口完全一致，
由 `factory.make_frame_source()` 按实际环境自动选择，业务层不感知差异。

ROS 1 特有的两个坑，这里都处理了
--------------------------------
**① `buff_size` 必须显式设大。** rospy 的 Subscriber 默认 `buff_size=65536`，
远小于一帧图像。当消息大于缓冲区时，数据会在 TCP 接收缓冲里排队，
表现为「话题帧率正常，但拿到的图延迟几秒且越来越旧」——这是 ROS 1 图像订阅
最经典的陷阱。设成 2**24（16MB）可容纳一帧 4K raw。

**② `tcp_nodelay=True`。** 关掉 Nagle 算法，避免小包被攒着发导致额外延迟。

用法
----
    source /opt/ros/noetic/setup.bash
    source $HOME/k02_nav/ros_ws_base_inspection/ros_ws/devel/setup.bash
    python3 -m inspection_perception.capture.ros1_odin_node --probe 5
    python3 -m inspection_perception.capture.ros1_odin_node --serve --port 8730
"""
from __future__ import annotations

from typing import Optional

from .base import Frame, LatestFrameStore
from .decode import raw_to_rgb, resize_long_side
from .ros_odin_node import (
    DEFAULT_COMPRESSED_TOPIC,
    DEFAULT_RAW_TOPIC,
    build_argparser,
    probe_and_save,
)

NO_FRAME_HINT_ROS1 = (
    "！超时没收到任何帧。依次检查：\n"
    "  1. rostopic list | grep odin            话题是否存在\n"
    "  2. rostopic hz /odin1/image/compressed  是否真的在发\n"
    "  3. echo $ROS_MASTER_URI                 是否指向正确的 master\n"
    "  4. 确认 go2_start_roscore.sh 与 go2_start_odin.sh 都已启动"
)

# rospy 默认 buff_size=65536，远小于一帧图像，会导致帧越拿越旧。16MB 可容纳一帧 4K raw。
IMAGE_BUFF_SIZE = 2 ** 24


class OdinFrameSourceRos1:
    """把 Odin 图像话题包装成 `FrameSource`（ROS 1 实现）。"""

    def __init__(
        self,
        raw_topic: str = DEFAULT_RAW_TOPIC,
        compressed_topic: str = DEFAULT_COMPRESSED_TOPIC,
        prefer: str = "compressed",
        long_side: Optional[int] = None,
        node_name: str = "inspection_capture_odin",
        init_node: bool = True,
    ) -> None:
        """
        Args:
            prefer: "compressed" | "raw" | "both"。
            long_side: raw 帧收下时即按长边缩放。
            init_node: False 时假定调用方已 `rospy.init_node()`。
                ROS 1 一个进程只能有一个节点，所以嵌入到别的节点里时要传 False。
        """
        import rospy
        from sensor_msgs.msg import CompressedImage, Image

        if prefer not in ("compressed", "raw", "both"):
            raise ValueError(f"prefer 只能是 compressed/raw/both，收到 {prefer!r}")

        self._long_side = long_side
        self._store = LatestFrameStore()
        self._rospy = rospy

        if init_node:
            # disable_signals: 让 KeyboardInterrupt 由我们自己的 CLI 处理
            rospy.init_node(node_name, anonymous=True, disable_signals=True)

        self._subs = []
        if prefer in ("compressed", "both"):
            self._subs.append(
                rospy.Subscriber(
                    compressed_topic,
                    CompressedImage,
                    self._on_compressed,
                    queue_size=1,
                    buff_size=IMAGE_BUFF_SIZE,
                    tcp_nodelay=True,
                )
            )
            rospy.loginfo(f"订阅 CompressedImage: {compressed_topic}")
        if prefer in ("raw", "both"):
            self._subs.append(
                rospy.Subscriber(
                    raw_topic,
                    Image,
                    self._on_raw,
                    queue_size=1,
                    buff_size=IMAGE_BUFF_SIZE,
                    tcp_nodelay=True,
                )
            )
            rospy.loginfo(f"订阅 Image: {raw_topic}")

        # 标注图发布器：推理进程通过 POST /annotate 把标注图送回来，
        # 本节点负责发布到 /inspection/annotated/compressed（RViz 可看）
        self._CompressedImage = CompressedImage
        self._annotate_pub = rospy.Publisher(
            "/inspection/annotated/compressed",
            CompressedImage,
            queue_size=1,
        )
        rospy.loginfo("标注发布器就绪: /inspection/annotated/compressed")

    # ---------------------------------------------------------------- 内部
    @staticmethod
    def _stamp_ns(header) -> int:
        return int(header.stamp.secs) * 1_000_000_000 + int(header.stamp.nsecs)

    def _on_compressed(self, msg) -> None:
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
            self._rospy.logerr(f"raw Image 转换失败: {exc}")
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
        for sub in self._subs:
            try:
                sub.unregister()
            except Exception:
                pass
        if self._annotate_pub is not None:
            self._annotate_pub.unregister()

    def annotate(self, jpeg_bytes: bytes) -> None:
        """把标注图发布到 /inspection/annotated/compressed（由推理进程通过 HTTP POST 送达）。"""
        msg = self._CompressedImage()
        msg.header.stamp = self._rospy.Time.now()
        msg.format = "jpeg"
        msg.data = jpeg_bytes
        self._annotate_pub.publish(msg)


def main(argv=None) -> int:
    args = build_argparser("Odin 图像话题取图节点（ROS 1）").parse_args(argv)

    src = OdinFrameSourceRos1(
        raw_topic=args.raw_topic,
        compressed_topic=args.compressed_topic,
        prefer=args.prefer,
        long_side=args.long_side or None,
    )
    try:
        if args.probe:
            return probe_and_save(src, args.probe, args.out_dir, hint=NO_FRAME_HINT_ROS1)
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
