#!/usr/bin/env python3
"""对一张图片自由提问（端侧定位型后端）。

当前模型没有 VQA 能力，不能直接问「这图里是什么颜色」这种开放问题——
它只认固定任务标签。本脚本把自然语言问题路由到它能做的三类动作：

    描述类   "图里有什么" / "describe" / "what is in"  -> <CAPTION> 整图描述
    读字类   "有什么文字" / "read" / "ocr"             -> <OCR> 读全部文字
    找目标   "有没有自行车" / "where is the sign" / "count"
                                                      -> <OPEN_VOCABULARY_DETECTION>
             返回坐标框 + 在图上画框存证（runs/ask/），肉眼核对模型框得准不准

用法：
    # 静态提问：对一张本地图片提问
    .venv-edge/bin/python3 scripts/ask_image.py --image runs/evidence/1785808510_pump_01_leak.jpg
    # 实时提问：每次提问都从取图服务拉最新一帧（odin 画面）
    .venv-edge/bin/python3 scripts/ask_image.py --http-url http://127.0.0.1:8730
        # 本机提问连 NX 上的取图服务，把地址换成 NX 的 IP 即可
    # 单次提问（--question），适合脚本调用
    .venv-edge/bin/python3 scripts/ask_image.py --image xxx.jpg --question "Is there a bicycle"
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from typing import Optional

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from inspection_perception.vlm.florence2 import Florence2Backend

# 三类意图的匹配规则。顺序即优先级：count > text > describe，
# 都没命中则按「找目标」处理（最接近普通提问的语义）。
_COUNT_RE = re.compile(
    r"\b(how many|count|number of)\b|几个|多少|数量", re.IGNORECASE
)
_TEXT_RE = re.compile(
    r"\b(text|ocr|read|words|printed|writing|label|number(s)?|reading)\b"
    r"|文字|字符|写的|铭牌|读数", re.IGNORECASE
)
_DESCRIBE_RE = re.compile(
    r"\b(describe|what is in|what's in|what can you see|see in|contains)\b"
    r"|里面有什么|有什么|图里|画面里|图片里|是什么|描述了|看到什么|内容", re.IGNORECASE
)


def route(question: str) -> str:
    if _COUNT_RE.search(question):
        return "count"
    if _TEXT_RE.search(question):
        return "text"
    if _DESCRIBE_RE.search(question):
        return "describe"
    return "find"


def load_image(path: str) -> np.ndarray:
    from PIL import Image

    img = Image.open(path).convert("RGB")
    rgb = np.asarray(img)
    print(f"图片: {path}  ({rgb.shape[1]}x{rgb.shape[0]})")
    return rgb


def draw_boxes(rgb: np.ndarray, boxes, phrase: str):
    """把检测框画到图上，返回 PIL Image（**不存盘**，由调用方决定落盘或发布）。"""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except OSError:
        font = None
    for box in boxes:
        draw.rectangle(
            [box.x0, box.y0, box.x1, box.y1], outline=(255, 0, 0), width=4
        )
        label = f"{box.label} ({box.x0:.0f},{box.y0:.0f})"
        draw.text((box.x0, max(0, box.y0 - 30)), label, fill=(255, 0, 0), font=font)
    return img


def _save_evidence(img, phrase: str) -> str:
    """存证图到 runs/ask/，返回路径。"""
    out_dir = os.path.join(REPO, "runs", "ask")
    os.makedirs(out_dir, exist_ok=True)
    safe = re.sub(r"[^0-9A-Za-z_-]+", "_", phrase)[:40] or "find"
    out = os.path.join(out_dir, f"{int(time.time())}_{safe}.jpg")
    img.save(out, quality=92)
    return out


def _post_annotated(img, http_url: str) -> None:
    """把标注图 POST 到取图服务，由它发布到 ROS 话题 /inspection/annotated/compressed。"""
    import io
    import urllib.request

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    jpeg_bytes = buf.getvalue()

    url = http_url.rstrip("/") + "/annotate"
    try:
        req = urllib.request.Request(url, data=jpeg_bytes, method="POST")
        req.add_header("Content-Type", "application/octet-stream")
        with urllib.request.urlopen(req, timeout=5):
            pass
    except Exception as exc:
        print(f"  ! 发布标注图失败: {exc}")


def dedupe_boxes(boxes) -> list:
    """合并重复框 + 嵌套框（Florence-2 生成式检测的典型输出问题）。

    纯 IoU 阈值挡不住嵌套框：小框被大框完全包含时 IoU 可能只有 0.5 左右
    （交集/小框面积很高但 /并集不高）。这里按面积从大到小保留，
    交集占**自身**面积过半即视为重复，专治「同一目标吐两三个框」。
    """
    kept: list = []
    for box in sorted(boxes, key=lambda b: b.area, reverse=True):
        dup = False
        for other in kept:
            ix0 = max(box.x0, other.x0)
            iy0 = max(box.y0, other.y0)
            ix1 = min(box.x1, other.x1)
            iy1 = min(box.y1, other.y1)
            inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
            if box.area > 0 and inter / box.area >= 0.5:
                dup = True
                break
        if not dup:
            kept.append(box)
    return kept


def answer(backend: Florence2Backend, rgb: np.ndarray, question: str,
           http_url: Optional[str] = None, save_evidence: bool = False) -> None:
    intent = route(question)
    t0 = time.perf_counter()
    print(f"\n问: {question}    [路由 -> {intent}]")

    if intent == "text":
        text = backend.ocr(rgb)
        print(f"读到的文字: {text!r}" if text else "没读到文字")

    elif intent == "describe":
        print("描述:", backend.caption(rgb, length="normal"))

    elif intent == "count":
        phrase = backend._phrase_from_question(question)
        boxes = dedupe_boxes(backend.detect(rgb, phrase))
        print(f"检测短语: {phrase!r}  命中 {len(boxes)} 个")
        for b in boxes:
            print(f"  - {b.label}: ({b.x0:.0f},{b.y0:.0f})-({b.x1:.0f},{b.y1:.0f})")
        if boxes:
            annotated = draw_boxes(rgb, boxes, phrase)
            if http_url:
                _post_annotated(annotated, http_url)
            if save_evidence:
                print("存证图:", _save_evidence(annotated, phrase))

    else:  # find —— 默认最像「提问」的行为
        phrase = backend._phrase_from_question(question)
        boxes = dedupe_boxes(backend.detect(rgb, phrase))
        if boxes:
            print(f"找到 {len(boxes)} 个「{phrase}」:")
            for b in boxes:
                print(f"  - {b.label}: ({b.x0:.0f},{b.y0:.0f})-({b.x1:.0f},{b.y1:.0f})")
            annotated = draw_boxes(rgb, boxes, phrase)
            if http_url:
                _post_annotated(annotated, http_url)
            if save_evidence:
                print("存证图:", _save_evidence(annotated, phrase))
        else:
            print(f"没找到「{phrase}」。")
            print("注: 当前模型暂不支持开放问答，已自动降级为整图描述 ->")
            print("   ", backend.caption(rgb, length="normal"))

    print(f"耗时 {(time.perf_counter() - t0) * 1000:.0f}ms")


def fetch_latest(http_src) -> Optional[np.ndarray]:
    """从取图服务拉最新一帧，失败返回 None。"""
    try:
        frame = http_src.get_latest(timeout_s=5.0)
    except RuntimeError as exc:
        print(f"!! {exc}")
        return None
    if frame is None:
        print("!! 取图服务在线但暂无新鲜帧（odin 在发图吗？）")
        return None
    rgb = frame.as_rgb()
    print(f"  实时帧: seq={frame.seq} age={frame.age_ms:.0f}ms 编码={frame.encoding} "
          f"{rgb.shape[1]}x{rgb.shape[0]}")
    return rgb


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="本地图片路径（静态提问）")
    src.add_argument("--http-url", help="实时取图服务地址，如 http://127.0.0.1:8730（每次提问拉最新帧）")
    ap.add_argument("--question", help="单次提问（不传则进入交互循环）")
    ap.add_argument("--model-dir", default="models/florence2-base-ft")
    ap.add_argument("--device", choices=["cuda", "cpu"])
    ap.add_argument("--dtype", default="float32", choices=["auto", "float16", "bfloat16", "float32"],
                    help="模型精度（NX 上默认 float32 最稳，本机 GPU 可改 auto 或 float16 提速）")
    args = ap.parse_args(argv)

    http_src = None
    if args.http_url:
        from inspection_perception.capture.http_source import HttpFrameSource

        http_src = HttpFrameSource(args.http_url)
        rgb = fetch_latest(http_src)
        if rgb is None:
            return 1
    else:
        rgb = load_image(args.image)

    t0 = time.perf_counter()
    print("加载模型 ...")
    backend = Florence2Backend(
        model_dir=args.model_dir, device=args.device, dtype=args.dtype, do_warmup=True
    )
    print(f"模型就绪  ({(time.perf_counter() - t0):.1f}s)")

    # 实时模式（有取图服务）→ 标注图自动 POST 发布，不存盘；
    # 静态模式（--image）→ 保持存盘兼容
    save_evidence = args.http_url is None

    try:
        if args.question:
            answer(backend, rgb, args.question, http_url=args.http_url, save_evidence=save_evidence)
            return 0

        print("\n" + "=" * 60)
        print("可问的问题类型（共 4 类，中英文均可，按关键词路由）")
        print("=" * 60)
        print("")
        print("【1】描述类 — 整图描述")
        print("    关键词: describe / what is in / see in / contains / 里面有什么 / 是什么 / 看到什么 / 内容")
        print("    示例:   \"图里有什么\" / \"describe the scene\" / \"what can you see\"")
        print("    返回:   一段自然语言场景描述（如: a clean room with a circular banner...）")
        print("")
        print("【2】读字类 — 读出图中全部文字")
        print("    关键词: text / ocr / read / words / 文字 / 字符 / 铭牌 / 读数")
        print("    示例:   \"读出所有文字\" / \"read the text\" / \"what is printed on the sign\"")
        print("    返回:   图中所有可见文字（如: MANIFOLD TECH 留形科技）")
        print("")
        print("【3】计数类 — 数几个目标")
        print("    关键词: how many / count / number of / 几个 / 多少 / 数量")
        print("    示例:   \"有几个纸箱\" / \"how many people\" / \"count the bicycles\"")
        print("    返回:   检测到 N 个目标 + 坐标框；标注图已发布到 /inspection/annotated/compressed")
        print("")
        print("【4】找目标类 — 有没有/在哪里（默认，不匹配以上三类时自动走此路）")
        print("    关键词: 不限，从问句里自动提取检测短语")
        print("    示例:   \"Is there a bicycle\" / \"where is the sign\" / \"找到 logo\" / \"banner\"")
        print("    返回:   找到则显示坐标框 + 发布标注图；找不到则降级为整图描述")
        print("")
        if http_src is not None:
            print("实时模式: 每次提问前自动拉取最新帧，标注图发布到 ROS 话题 /inspection/annotated/compressed")
        print("提示: 检测词条用英文效果最好；q/quit/exit 退出")
        print("-" * 60)
        print("")
        while True:
            try:
                q = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                continue
            if q.lower() in ("q", "quit", "exit"):
                break
            if http_src is not None:
                rgb = fetch_latest(http_src)
                if rgb is None:
                    continue
            answer(backend, rgb, q, http_url=args.http_url, save_evidence=save_evidence)
    finally:
        backend.unload()
    return 0


if __name__ == "__main__":
    sys.exit(main())
