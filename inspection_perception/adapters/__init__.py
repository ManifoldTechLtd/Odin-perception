"""ROS 适配层（薄）：把 ROS 话题接成「取帧服务 + 标注回送」，业务零 ROS。

为什么单独一层
--------------
核心包（`capture/` 的 file/http 源、`vlm/`、`checks/`、`pipeline`）**不 import 任何 ROS**，
所以它能在没有 ROS 的机器、容器、CI 里跑，也能被纯 Python 调用。

ROS 相关的东西只在两处出现：
  1. 取图侧：订阅图像话题 → 写进 `LatestFrameStore` → 由 `frame_server` 以 HTTP 暴露；
  2. 回送侧：推理进程把画好的标注图 `POST /annotate` 回来 → 由本层发布成 ROS 话题。

这样 ROS 1 / ROS 2（将来 ROS 3）的差异只落在取图侧的几个文件里，业务与判据完全不动。
这也是它不写成一个大而全的 "bridge 节点" 的原因——节点越厚，换 ROS 版本时要改的地方越多。

组成
----
    annotation.py   AnnotationSink 协议 + ROS2 发布实现 + 空实现
    ros2_capture.py Ros2CaptureSource（订阅 + 标注发布，供 frame_server 使用）+ CLI

约定
----
`frame_server` 收到 `POST /annotate` 时会调用 `source.annotate(jpeg_bytes)`（鸭子类型），
所以取图源只要实现这一个方法，标注就能发布出去；不实现也不报错，只是不会发布。

依赖
----
本层需要 ROS 环境（`rclpy` / `sensor_msgs`），均为**函数内延迟 import**，
因此 `import inspection_perception.adapters` 在无 ROS 的机器上也不会失败。
"""
