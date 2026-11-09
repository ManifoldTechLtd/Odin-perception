<p align="center">
  <img src="logo.png" alt="Manifold Tech" width="200">
</p>

# Odin-perception

On-device inspection perception for **[Odin-Nav-Stack2](https://github.com/ManifoldTechLtd/Odin-Nav-Stack2)**.
Feed one image to a vision model, get back **decidable conclusions** against a structured
checklist
(`yes/no`, state, count, read-text, describe) — plus evidence image and latency.

The current release ships the **Florence-2 (grounding)** pipeline.

> 中文文档见 [`README.zh-CN.md`](README.zh-CN.md)。

---

## What's New

| Date | New feature |
|---|---|
| **2026-11-09** | Annotation overlay on the published image (detection boxes + text verdict, configurable line count and font size) |
| **2026-11-09** | GPU inference support on Jetson (NVIDIA platform wheel, `--dtype` flag) |
| **2026-11-09** | One-command live Q&A script (`scripts/ask_live.sh`) |

---

## Capabilities and boundaries

| Backend | Capability | Status | Notes |
|---|---|---|---|
| `florence2` | Boxes / segmentation / OCR (**no free-form VQA**) | ✅ recommended | 0.23B, runs on CPU and GPU; `yes_no / state / count` require a `target` phrase |
| `dummy` | No model loaded | ✅ | Verify the plumbing before downloading weights |
| `moondream` / `smolvlm` | Free-form Q&A + OCR | ⚠️ code present | Bring your own weights; not part of the verified matrix |
| ROS 2 capture / annotation publishing | `adapters/ros2_capture` | ✅ | Subscribe to camera topics → HTTP frame server; inference posts annotations back |

**Not in this repository**: model weights (downloaded separately), third-party VLM toolkits,
and ROS nodes / task orchestration (those belong to the navigation stack).

---

## Installation

### Standard (machine with internet)

```bash
# 1) Create an isolated environment — do NOT install into system python
bash scripts/setup.sh
source .venv/bin/activate

# 2) Self-check: confirm the plumbing works before downloading weights
python scripts/doctor.py

# 3) Run the pipeline without a model
perception-run --backend dummy --source file --file-path ./demo.jpg \
  --tasks configs/tasks/smoke.json --out /tmp/smoke.json

# 4) Download Florence-2 weights (~888 MB)
bash scripts/download_models.sh florence2-base-ft
python scripts/doctor.py            # florence2 should now report "yes"

# 5) Real run
perception-run --backend florence2 --source file --file-path ./demo.jpg \
  --tasks configs/tasks/demo_florence2.yaml --out /tmp/result.json
```

### Offline (edge device without internet)

Resolve and download wheels on a machine that has network access, then copy them over:

```bash
# On the online machine — match the target platform exactly
pip download -r requirements.txt "numpy<1.25" \
  --platform manylinux2014_aarch64 --python-version 38 \
  --only-binary=:all: -d ./whl
# Build helpers are required or 'pip install .' fails with bdist_wheel missing
pip download wheel setuptools --platform any --python-version 38 \
  --only-binary=:all: -d ./whl

# On the device
python3 -m venv .venv-edge && source .venv-edge/bin/activate
pip install --no-index --find-links=./whl wheel setuptools
pip install --no-index --find-links=./whl -r requirements.txt
pip install --no-index --no-build-isolation .   # NOT -e: pip on py3.8 lacks PEP 660
```

Copy the `models/florence2-base-ft` directory separately (~888 MB) — weights are
platform independent, binaries are not.

---

## Dependency constraints (important)

Florence-2 is loaded through transformers **remote code** and is version sensitive:

| Package | Constraint | Why |
|---|---|---|
| `transformers` | **==4.46.3** | 5.x lacks `forced_bos_token_id`; 4.47+ changes the processor output and inference fails with `'NoneType' object has no attribute 'shape'` |
| `timm` | **==1.0.9** | Matches the Florence-2 modeling file |
| `einops` | >=0.7 | Required by remote code |
| `torch` | **CPU: ==2.4.1**<br>**Jetson: NVIDIA wheel** | ⚠️ Do **not** write `torch>=1.13`: pip resolves to 2.1.0, which **segfaults (exit 139) inside the Florence-2 DaViT convolution with no error output at all**. 2.4.1 is the newest CPU wheel for py3.8 |
| `torchvision` | **must match torch** | `timm 1.0.9` needs it (`from torchvision.ops.misc import FrozenBatchNorm2d`); without it Florence-2 fails to load. torch 2.0.0 ↔ tv 0.15.1, torch 2.4.1 ↔ tv 0.19.1 |

`scripts/setup.sh` installs exactly what `requirements.txt` pins. If you changed versions
and inference broke, run `pip install -r requirements.txt --force-reinstall` to get back
inside the constraints; `scripts/doctor.py` warns when they drift.

---

## Enabling GPU on Jetson / Orin NX (≈38× faster)

A plain `pip install torch` gives you the **PyPI CPU wheel** (`torch.cuda.is_available() == False`).
On Jetson you must install NVIDIA's platform-specific wheel, chosen by JetPack version:

```bash
# JetPack 5.1.1 (L4T 35.3.1) ships CUDA 11.4
curl -LO "https://developer.download.nvidia.com/compute/redist/jp/v511/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl"
pip install --no-deps --force-reinstall "torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl"
pip install --no-deps "torchvision==0.15.1"     # required by timm
```

Measured on Orin NX 8 GB, Florence-2 base, 1600×1296 input:

| | CPU | GPU |
|---|---|---|
| Per-check inference | **46 800 ms** | **1 200 ms** (≈38×) |
| Model load | 27–44 s | 15–17 s |

Two GPU-specific gotchas:

1. **`--dtype float16` does not work with Florence-2** — `"exp_vml_cpu" not implemented for 'Half'`
   (the remote code contains CPU operators). Use `--device cuda --dtype float32`.
2. Check `python -c "import torch;print(torch.cuda.is_available())"` is `True` before
   going further, otherwise you are just running CPU slower.

---

## Integrating with [Odin-Nav-Stack2](https://github.com/ManifoldTechLtd/Odin-Nav-Stack2)

The perception component is **decoupled from navigation**. Integration is two processes
talking over HTTP, which means you never have to merge the model environment into the
ROS workspace.

```
Odin-Nav-Stack (ROS)                                   Odin-perception
  odin driver ──► /odin1/image/undistorted
                        │
                        ▼
              ros1_capture / ros2_capture  ──HTTP──►  perception-run
              (frame server :8730)                       (Florence-2, venv)
                        ▲                                    │
                        └────── POST /annotate ◄─────────────┘
                        │
                        ▼
              /inspection/annotated/compressed  ──►  Foxglove / RViz / rosbridge
```

### Step 1 — Make sure the navigation stack publishes images

The perception side can only see what the driver publishes. On the Odin driver,
check the stream switches first:

```bash
grep -E "^  (sendrgb|sendrgbundistort):" <odin_ros_ws>/src/odin_ros_driver/config/control_command.yaml
```

`sendrgb` must be `1`; `sendrgbundistort: 1` **depends on `sendrgb`** and does nothing on
its own. Then confirm data actually flows:

```bash
rostopic hz /odin1/image/undistorted      # expect ~10 Hz
```

### Step 2 — Install the perception side next to the stack

Follow [Installation](#installation). Keep it in its own venv — the navigation stack
already uses the system python and torch, and mixing the two is the single most common
cause of hard-to-diagnose failures on edge devices.

### Step 3 — Start the frame server (ROS side)

ROS 1 (Noetic):

```bash
source /opt/ros/noetic/setup.bash
cd <path>/Odin-perception

# Use the SYSTEM interpreter here: it carries rospy. A conda python3 will not have it.
PYTHONPATH="/usr/lib/python3/dist-packages:.${PYTHONPATH:+:$PYTHONPATH}" \
/usr/bin/python3 -m inspection_perception.capture.ros1_odin_node --serve \
  --prefer raw --raw-topic /odin1/image/undistorted --port 8730
```

ROS 2 (Jazzy / Humble):

```bash
python -m inspection_perception.adapters.ros2_capture --serve \
  --raw-topic /odin1/image/undistorted --prefer raw --long-side 768 --port 8730
```

Verify before moving on:

```bash
curl -s http://127.0.0.1:8730/meta.json     # "has_frame": true
curl -s -o /tmp/f.jpg -w "%{http_code}\n" http://127.0.0.1:8730/latest.jpg
```

### Step 4 — Run inference against the live feed

```bash
source <venv>/bin/activate
cd <path>/Odin-perception

perception-run \
  --source http --http-url http://127.0.0.1:8730 \
  --backend florence2 --model-dir models/florence2-base-ft \
  --device cuda --dtype float32 \
  --tasks configs/tasks/demo_florence2.yaml \
  --annotate-url http://127.0.0.1:8730 \
  --out /tmp/result.json
```

`--annotate-url` turns on overlay publishing (boxes + verdict text). Omit it if you only
want JSON output.

### Step 5 — Verify the annotation topic

```bash
# Subscribe FIRST: this topic is not latched, so echo-after-publish shows nothing
rostopic echo -n1 /inspection/annotated/compressed --noarr
```

A `sensor_msgs/CompressedImage` with `format: "jpeg"` should arrive for each check.

### One-command variant

`scripts/ask_live.sh` wires steps 3–4 together and defaults to GPU:

```bash
bash scripts/ask_live.sh                                          # interactive
bash scripts/ask_live.sh "What text is printed on the nameplate"   # single question
```

### Integration checklist

| Item | Value |
|---|---|
| Topic consumed | `/odin1/image/undistorted` (raw preferred); `--prefer compressed` falls back to `/odin1/image/compressed` (~5 MB/s instead of ~500 MB/s) |
| Topic produced | `/inspection/annotated/compressed` (`sensor_msgs/CompressedImage`) |
| HTTP endpoints | `GET /latest.jpg`, `GET /meta.json`, `GET /healthz`, `POST /annotate` |
| Default port | 8730 — check `ss -lntp \| grep 8730` for stale instances |
| Output | One `CheckResult` per item (`item_id / status / answer / reason / latency_ms / evidence / checked_at`) plus a `summarize` block |

---

## Annotation overlay

With `--annotate-url` set, every check renders its **boxes** and **verdict text** onto the
image the model actually saw, then posts it back for publishing.

```bash
--annotate-url http://127.0.0.1:8730   # enable publishing
--max-text-lines 3                     # bottom panel line cap (default 3, overflow → …)
--text-scale 1.4                       # font-size multiplier (default 1.0)
```

Layout rules:

- The text panel is pinned to the **bottom** (dark background, white text) so it never
  covers the subject being inspected.
- At most `--max-text-lines` lines are drawn — the image height does not grow unbounded.
- Font size scales with the image long side, then multiplies by `--text-scale`.
- Box color follows the verdict: green = pass, red = fail, yellow = unknown, grey = error.

Implementation: `inspection_perception/annotation.py` (`render_result`, `draw_text_panel`,
`iter_boxes`). Pillow only — no new dependency. Fonts are auto-detected (Noto CJK / WenQuanYi),
falling back to the Pillow default (CJK renders as boxes, Latin and digits stay readable).

> Overlay is a **side channel**: network errors, missing fonts or encode failures are
> swallowed and never turn a check into `ERROR`.

---

## CLI reference

```text
perception-run --tasks <yaml|json> \
  --source auto|ros1|ros2|http|file [--file-path ...] [--http-url ...] \
  [--prefer compressed|raw|both] [--max-age-ms 1000] \
  --backend florence2|moondream|smolvlm|dummy [--model-dir ...] \
  [--device cuda|cpu] [--dtype auto|float32|float16|bfloat16] \
  [--infer-long-side 768] [--evidence-dir runs/evidence] \
  [--annotate-url URL] [--max-text-lines 3] [--text-scale 1.0] \
  --out result.json
```

| Flag | Meaning |
|---|---|
| `--tasks` | Checklist (YAML/JSON); each item has `id/question/kind`, detection kinds also need `target` |
| `--source` | `file` offline image, `http` frame server, `ros1/ros2` subscribe directly, `auto` autodetect |
| `--backend` | **Defaults to `moondream` — always pass `--backend florence2` explicitly** |
| `--dtype` | Default `float32`. Keep it that way for Florence-2 |
| `--evidence-dir` | Evidence image directory; empty string disables |

## Task lists

See `configs/tasks/`:

- `smoke.json` — minimal smoke test (works with `dummy`)
- `demo_florence2.yaml` — Florence-2 detection examples (`kind` and `target` in pairs)

`kind` values: `yes_no` / `state` / `count` / `read_text` / `describe`.
Florence-2 only understands fixed task tokens, so the first three **require** an English
`target` phrase, otherwise the result degrades to `UNKNOWN`.

## Repository layout

```text
inspection_perception/   # core package: vlm/ capture/ checks/ pipeline cli schemas annotation
configs/                 # task / profile / capture examples
scripts/                 # setup, download_models, doctor, ask_image, ask_live
tests/                   # zero-dependency unit tests (no weights, no ROS)
adapters/                # ROS thin adapters (optional): capture + annotation publishing
```

---

## ROS 1 pitfalls (encountered during on-device validation)

Validated on Jetson NX + ROS 1 Noetic + python 3.8. Do these up front and you skip all of them:

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | Inference produces **no output at all**, exit code 139 | `torch>=1.13` resolved to 2.1.0; convolution segfaults | Pin `torch==2.4.1` (CPU) or the NVIDIA wheel |
| 2 | `pip install -e .` complains about setup.py | pip on py3.8 lacks PEP 660 | `pip install --no-build-isolation .` |
| 3 | `invalid command 'bdist_wheel'` | `wheel` missing in the venv | Install `wheel` + `setuptools` first |
| 4 | Frame node: `No module named 'rospy'` | miniconda shadows `python3` | Use **`/usr/bin/python3`** for the capture side |
| 5 | Still no rospy after switching interpreter | `export PYTHONPATH=.` **overwrites** the ROS path | Append instead: `export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}"` |
| 6 | `import rospy` fails on `rospkg` | `rospkg` lives in `/usr/lib/python3/dist-packages` | Add that directory to `PYTHONPATH` too |
| 7 | Image topic exists but `rostopic hz` shows nothing | driver has `sendrgb: 0`, and `sendrgbundistort` depends on it | Set `sendrgb: 1` and `sendrgbundistort: 1`, restart the driver |
| 8 | Frame server returns a stale frame (`age_ms` in the tens of thousands) | the capture node's subscription died when the driver restarted | Restart the capture node to re-subscribe |

---

## Deployment notes (edge NX / Jetson)

| # | Item | Note |
|---|---|---|
| 1 | **torch must be a platform wheel** | aarch64 Jetson wheel (validated `2.0.0+nv23.05`, CUDA 11.4); **x86 binaries cannot be copied over** |
| 2 | **transformers pinned to 4.46.3** | plus `timm==1.0.9`, `einops` |
| 3 | **python 3.8** | Keep `from __future__ import annotations`; avoid 3.9+ syntax |
| 4 | **weights are portable** | The 888 MB directory can be copied; verify `config.json`, `preprocessor_config.json` and the remote-code `.py` files are present |
| 5 | **undistorted topic needs the driver switch** | `sendrgbundistort: 1`, or the topic is advertised with no data |
| 6 | **port conflicts** | Frame server defaults to 8730; check `ss -lntp \| grep 8730` |
| 7 | **disk** | venv (~1.5 GB with torch) + 888 MB weights — run `df -h` first |
| 8 | **never install into system python** | The navigation stack uses the system interpreter; always isolate in a venv |

## FAQ

- **Weights are not in the repo**: `models/` is gitignored; use `scripts/download_models.sh`.
  On an offline device, download them on another machine and copy the directory.
- **No CUDA**: `--device cpu` works; expect seconds rather than ~1 s per check.
- **No ROS**: `--source file/http` does not touch ROS at all; ROS modules are lazily imported.
- **All `yes_no` come back `UNKNOWN`**: confirm `--backend florence2` was passed and each item has a `target`.
- **Slow or failing downloads**: `export HF_ENDPOINT=https://hf-mirror.com`, then re-run; the script verifies file completeness.

## License

Apache-2.0 — see `LICENSE`. Third-party licenses in `THIRD_PARTY_LICENSES.md`.
Model weights are not distributed here; their licenses come from upstream
(Florence-2 is MIT).
