#!/usr/bin/env bash
# =============================================================================
# 一键问答：对 odin 实时画面提问（Odin-perception + Orin NX GPU 加速）
#
# 前置（仅需一次）：odin ros driver 已在跑（roscore + 相机驱动）
#   本脚本自动完成：
#     1. 检查取图服务 :8730，没跑就启动（订阅去畸变话题 /odin1/image/undistorted）
#     2. 确认有实时帧
#     3. 进入交互提问（GPU 推理，约 1.2s/条）
#
# 用法（在 NX 上）：bash scripts/ask_live.sh
#   单次提问：      bash scripts/ask_live.sh "What text is printed in the image"
# =============================================================================
set -uo pipefail

PORT="${ASK_PORT:-8730}"
BASE="http://127.0.0.1:${PORT}"
DEVICE="${ASK_DEVICE:-cuda}"
DTYPE="${ASK_DTYPE:-float32}"

# 定位仓库根（脚本自推导优先，兼容两种部署位置）
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ ! -d "$REPO/inspection_perception" ]; then
  REPO=""
  for cand in "$HOME/Odin-perception" "$HOME/base_inspection/perception"; do
    [ -d "$cand/inspection_perception" ] && { REPO="$cand"; break; }
  done
fi
[ -z "$REPO" ] && { echo "!! 找不到 Odin-perception 仓库" >&2; exit 1; }

VENV_PY="$REPO/.venv-edge/bin/python3"
[ -x "$VENV_PY" ] || { echo "!! 缺少 venv: $VENV_PY（先按部署文档建环境）" >&2; exit 1; }

# 取图侧必须用【系统 python3.8】（带 rospy）；本机 conda 的 python3 没有 rospy
SYS_PY="/usr/bin/python3"

start_capture() {
  local tag="$1"
  echo "[1/3] ${tag}，启动取图服务（去畸变话题）..."
  pkill -f "[r]os1_odin_node" 2>/dev/null
  sleep 1
  (
    set +u
    source /opt/ros/noetic/setup.bash
    set -u
    cd "$REPO" || exit 1
    export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}"
    exec "$SYS_PY" -m inspection_perception.capture.ros1_odin_node \
      --serve --host 0.0.0.0 --port "$PORT" \
      --prefer raw --raw-topic /odin1/image/undistorted
  ) > /tmp/odin_capture.log 2>&1 &
  for _ in $(seq 1 15); do
    sleep 2
    curl -sf -m 2 "$BASE/meta.json" 2>/dev/null | grep -q '"has_frame": true' && return 0
  done
  return 1
}

# --- [1/3] 取图服务 -----------------------------------------------------------
if curl -sf -m 3 "$BASE/meta.json" >/dev/null 2>&1; then
  TOPIC=$(curl -sf -m 3 "$BASE/meta.json" 2>/dev/null | grep -o '"topic": "[^"]*"' | head -1)
  if echo "$TOPIC" | grep -qE '"topic": "(raw|undistorted)"'; then
    echo "[1/3] 取图服务已在运行 (:$PORT，$TOPIC)"
  else
    start_capture "取图服务话题不匹配（$TOPIC）"
  fi
else
  start_capture "取图服务未运行"
fi

# --- [2/3] 确认有实时帧 -------------------------------------------------------
META=$(curl -sf -m 3 "$BASE/meta.json" 2>/dev/null) || {
  echo "!! 取图服务起不来，查看: cat /tmp/odin_capture.log" >&2; exit 1; }
if ! echo "$META" | grep -q '"has_frame": true'; then
  echo "!! 取图服务在线但暂无帧（odin 驱动在发图吗？）" >&2
  echo "   检查: rostopic hz /odin1/image/undistorted" >&2
  echo "   若 odin 掉了（xHCI 崩溃），执行: echo 3610000.xhci | sudo tee /sys/bus/platform/drivers/tegra-xusb/unbind && sleep 2 && echo 3610000.xhci | sudo tee /sys/bus/platform/drivers/tegra-xusb/bind" >&2
  exit 1
fi
echo "[2/3] 取图正常: $(echo "$META" | grep -o '"fps": [0-9.]*')  $(echo "$META" | grep -o '"topic": "[^"]*"')"

# --- [3/3] 提问（默认 GPU）----------------------------------------------------
GPU_NOTE=""
"$VENV_PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null \
  || { GPU_NOTE=" (CUDA 不可用，回落 CPU，会很慢)"; DEVICE="cpu"; }

if [ "$#" -gt 0 ]; then
  echo "[3/3] 单次提问（device=$DEVICE dtype=$DTYPE）${GPU_NOTE}"
  cd "$REPO" || exit 1
  exec "$VENV_PY" scripts/ask_image.py --http-url "$BASE" \
    --device "$DEVICE" --dtype "$DTYPE" --question "$*"
else
  echo "[3/3] 进入交互提问（device=$DEVICE dtype=$DTYPE）${GPU_NOTE}  输入 q 退出"
  cd "$REPO" || exit 1
  exec "$VENV_PY" scripts/ask_image.py --http-url "$BASE" \
    --device "$DEVICE" --dtype "$DTYPE"
fi
