"""标注回送出口：把推理侧画好的图交回 ROS 侧发布。

统一契约是**一个方法** `annotate(jpeg_bytes)`：`frame_server` 的 `POST /annotate`
就是按这个鸭子类型调用取图源的。推理侧不需要知道 ROS 存在，ROS 侧也不需要知道
模型是什么——两边只有一包 JPEG 字节的往来。

为什么不做成"推理侧直接发 ROS 话题"
------------------------------------
推理跑在独立 venv（torch），继承 rospy/rclpy 会带来 numpy 版本冲突；
端侧 NX 上系统 python 还有导航程序在用。让 ROS 只出现在取图进程里，
是这套结构能同时兼容 ROS 1 / ROS 2 的前提。
"""
from __future__ import annotations

import time
from typing import Any, Dict, Protocol

DEFAULT_ANNOTATE_TOPIC = "/inspection/annotated/compressed"


class AnnotationSink(Protocol):
    """标注出口协议：实现这一个方法即可被 frame_server 使用。"""

    def annotate(self, jpeg_bytes: bytes) -> None:
        ...


class NullAnnotationSink:
    """不发布（无 ROS 环境、或只跑离线检查时用）。"""

    def annotate(self, jpeg_bytes: bytes) -> None:
        return None

    def stats(self) -> Dict[str, Any]:
        return {"published": 0, "topic": None}


class Ros2AnnotationPublisher:
    """把 JPEG 字节发布成 `sensor_msgs/CompressedImage`（ROS 2）。

    Args:
        node: 已创建的 rclpy 节点（由取图源提供）。
        topic: 发布话题，默认 `/inspection/annotated/compressed`（RViz / rosbridge 可看）。
        frame_id: 可选，留空则由下游按需处理（Odin 图像话题本身也不发 CameraInfo）。

    QoS 用 depth=1：标注图是"最新一张"的语义，积压旧帧没有意义。
    """

    def __init__(self, node: Any, topic: str = DEFAULT_ANNOTATE_TOPIC, frame_id: str = "") -> None:
        # 延迟 import：本模块在无 ROS 的机器上也要能 import
        from sensor_msgs.msg import CompressedImage  # noqa: PLC0415

        self._CompressedImage = CompressedImage
        self._node = node
        self._topic = topic
        self._frame_id = frame_id
        self._pub = node.create_publisher(CompressedImage, topic, 1)
        self._published = 0
        self._last_publish_ns = 0
        print(f"[adapters] 标注发布器就绪: {topic}")

    def annotate(self, jpeg_bytes: bytes) -> None:
        if not jpeg_bytes:
            return
        msg = self._CompressedImage()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id
        msg.format = "jpeg"
        msg.data = bytes(jpeg_bytes)
        self._pub.publish(msg)
        self._published += 1
        self._last_publish_ns = time.time_ns()

    def stats(self) -> Dict[str, Any]:
        return {
            "published": self._published,
            "topic": self._topic,
            "last_publish_ms_ago": (
                round((time.time_ns() - self._last_publish_ns) / 1e6, 1)
                if self._last_publish_ns else None
            ),
        }
