<p align="center">
  <img src="logo.png" alt="Manifold Tech" width="200">
</p>

# Odin-perception

**[Odin-Nav-Stack2](https://github.com/ManifoldTechLtd/Odin-Nav-Stack2) 的端侧巡检感知组件**：把一帧图像交给视觉模型，
按**结构化检查清单**输出可判定的结论（`yes/no`、状态、计数、读字、描述），并保留证据图与耗时。

当前版本提供 **Florence-2（grounding）** 识别链路。

> English documentation: [`README.md`](README.md)

---

## 更新日志（What's New）

| 日期 | 新增功能 |
|---|---|
| **2026-11-09** | 标注图上屏（检测框 + 文字结论，行数与字号可配） |
| **2026-11-09** | Jetson GPU 推理支持（NVIDIA 平台 wheel、`--dtype` 参数） |
| **2026-11-09** | 一键实时问答脚本（`scripts/ask_live.sh`） |

---

## 能力与边界

| 后端 | 能力 | 状态 | 说明 |
|---|---|---|---|
| `florence2` | 框选 / 分割 / OCR（**不能自由问答**） | ✅ 推荐 | 0.23B，CPU/GPU 均可；`yes_no / state / count` 必须给 `target` 英文词条 |
| `dummy` | 不加载模型 | ✅ | 装完先用它验证链路是否通 |
| `moondream` / `smolvlm` | 自由问答 + OCR | ⚠️ 代码保留 | 需自备权重，不在已验证矩阵内 |
| ROS 2 取图 / 标注发布 | `adapters/ros2_capture` | ✅ | 订阅相机话题 → HTTP 取帧；推理侧回送标注 |

**不在本仓库**：模型权重（另行下载）、第三方 VLM 工具包、ROS 节点与任务编排（属导航栈）。

---

## 安装

### 常规安装（机器可联网）

```bash
# 1) 建隔离环境 —— 不要装进系统 python
bash scripts/setup.sh
source .venv/bin/activate

# 2) 自检：先确认链路通，再下权重
python scripts/doctor.py

# 3) 不加载模型跑一遍
perception-run --backend dummy --source file --file-path ./demo.jpg \
  --tasks configs/tasks/smoke.json --out /tmp/smoke.json

# 4) 下载 Florence-2 权重（约 888 MB；国内可先 export HF_ENDPOINT=https://hf-mirror.com）
bash scripts/download_models.sh florence2-base-ft
python scripts/doctor.py            # 此时 florence2 应显示「是」

# 5) 实跑
perception-run --backend florence2 --source file --file-path ./demo.jpg \
  --tasks configs/tasks/demo_florence2.yaml --out /tmp/result.json
```

### 离线安装（端侧设备无外网）

在有网的机器上把 wheel 下齐（平台与 python 版本必须与目标机一致），再拷过去：

```bash
# 联网机器上执行
pip download -r requirements.txt "numpy<1.25" \
  --platform manylinux2014_aarch64 --python-version 38 \
  --only-binary=:all: -d ./whl
# 构建工具也必须带上，否则 pip install . 会报 bdist_wheel 缺失
pip download wheel setuptools --platform any --python-version 38 \
  --only-binary=:all: -d ./whl

# 在设备上执行
python3 -m venv .venv-edge && source .venv-edge/bin/activate
pip install --no-index --find-links=./whl wheel setuptools
pip install --no-index --find-links=./whl -r requirements.txt
pip install --no-index --no-build-isolation .   # 不要用 -e：py3.8 的 pip 不支持 PEP 660
```

权重目录 `models/florence2-base-ft`（约 888 MB）单独拷贝即可——**权重跨平台通用，二进制不行**。

---

## 依赖版本约束（重要）

Florence-2 通过 transformers 的 **remote code** 加载，对版本非常敏感：

| 包 | 约束 | 违反后的现象 |
|---|---|---|
| `transformers` | **==4.46.3** | 5.x 缺 `forced_bos_token_id`；4.47+ processor 返回结构变化，报 `'NoneType' object has no attribute 'shape'` |
| `timm` | **==1.0.9** | 与 Florence-2 建模文件配套 |
| `einops` | >=0.7 | remote code 必需 |
| `torch` | **CPU：==2.4.1**<br>**Jetson：NVIDIA 官方 wheel** | ⚠️ **不要写 `torch>=1.13`**：pip 会挑到 2.1.0，在 Florence-2 的卷积层直接**段错误（exit 139）且没有任何报错输出** |
| `torchvision` | **必须与 torch 配套** | `timm 1.0.9` 强依赖；torch 2.0.0 ↔ tv 0.15.1，torch 2.4.1 ↔ tv 0.19.1 |

`scripts/setup.sh` 会按 `requirements.txt` 精确安装。若手工改过版本导致推理失败，
执行 `pip install -r requirements.txt --force-reinstall` 回到约束内；`scripts/doctor.py` 会告警。

---

## Jetson / Orin NX 启用 GPU（实测快约 38 倍）

直接 `pip install torch` 装到的是 **PyPI 的 CPU wheel**（`torch.cuda.is_available() == False`）。
Jetson 上必须换 NVIDIA 官方特供 wheel，按 JetPack 版本选路径：

```bash
# JetPack 5.1.1（L4T 35.3.1）自带 CUDA 11.4
curl -LO "https://developer.download.nvidia.com/compute/redist/jp/v511/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl"
pip install --no-deps --force-reinstall "torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl"
pip install --no-deps "torchvision==0.15.1"     # timm 需要
```

实测（Orin NX 8GB，Florence-2 base，1600×1296 输入）：

| | CPU | GPU |
|---|---|---|
| 单条推理 | **46 800 ms** | **1 200 ms**（约快 38 倍） |
| 模型加载 | 27~44 s | 15~17 s |

两个 GPU 相关的坑：

1. **`--dtype float16` 在 Florence-2 上不可用**，报 `"exp_vml_cpu" not implemented for 'Half'`
   （remote code 里有 CPU 算子）。**用 `--device cuda --dtype float32`**。
2. 先确认 `python -c "import torch;print(torch.cuda.is_available())"` 为 `True` 再往下走。

---

## 接入导航栈（[Odin-Nav-Stack2](https://github.com/ManifoldTechLtd/Odin-Nav-Stack2)）

感知组件与导航**解耦**：接入方式是两个进程通过 HTTP 通信，因此**不需要把模型环境合并进 ROS 工作区**。

```
Odin-Nav-Stack（ROS）                                  Odin-perception
  odin 驱动 ──► /odin1/image/undistorted
                     │
                     ▼
           ros1_capture / ros2_capture  ──HTTP──►  perception-run
           （取图服务 :8730）                          （Florence-2，独立 venv）
                     ▲                                  │
                     └────── POST /annotate ◄───────────┘
                     │
                     ▼
           /inspection/annotated/compressed  ──►  Foxglove / RViz / rosbridge
```

### 第 1 步：确认导航栈的图像话题有数据

感知只能看到驱动推出来的画面。先查 Odin 驱动的图像流开关：

```bash
grep -E "^  (sendrgb|sendrgbundistort):" <odin_ros_ws>/src/odin_ros_driver/config/control_command.yaml
```

`sendrgb` 必须是 `1`；`sendrgbundistort: 1` **依赖 `sendrgb`**，单独开没有意义。然后确认有数据：

```bash
rostopic hz /odin1/image/undistorted      # 期望约 10 Hz
```

### 第 2 步：在导航栈同一台设备上装感知侧

按 [安装](#安装) 执行。**务必独立 venv**——导航栈在用系统 python 与系统 torch，
混用是端侧最难排查的故障来源。

### 第 3 步：起取图服务（ROS 侧）

ROS 1（Noetic）：

```bash
source /opt/ros/noetic/setup.bash
cd <path>/Odin-perception

# 这里必须用【系统解释器】，它才带 rospy；conda 的 python3 没有
PYTHONPATH="/usr/lib/python3/dist-packages:.${PYTHONPATH:+:$PYTHONPATH}" \
/usr/bin/python3 -m inspection_perception.capture.ros1_odin_node --serve \
  --prefer raw --raw-topic /odin1/image/undistorted --port 8730
```

ROS 2（Jazzy / Humble）：

```bash
python -m inspection_perception.adapters.ros2_capture --serve \
  --raw-topic /odin1/image/undistorted --prefer raw --long-side 768 --port 8730
```

**先验证再往下**：

```bash
curl -s http://127.0.0.1:8730/meta.json     # "has_frame": true
curl -s -o /tmp/f.jpg -w "%{http_code}\n" http://127.0.0.1:8730/latest.jpg
```

### 第 4 步：对实时画面推理

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

加 `--annotate-url` 才会把框与文字画到图上并回送发布；只要 JSON 结果就不加。

### 第 5 步：验证标注话题

```bash
# 必须先订阅：该话题不是 latched，先发后看永远看不到
rostopic echo -n1 /inspection/annotated/compressed --noarr
```

每条检查应收到一个 `sensor_msgs/CompressedImage`，`format` 为 `"jpeg"`。

### 一条命令的替代方案

`scripts/ask_live.sh` 把第 3、4 步串起来，默认走 GPU：

```bash
bash scripts/ask_live.sh                                          # 交互提问
bash scripts/ask_live.sh "What text is printed on the nameplate"   # 单次提问
```

### 接入清单

| 项 | 值 |
|---|---|
| 订阅话题 | `/odin1/image/undistorted`（优先原图）；`--prefer compressed` 回落到 `/odin1/image/compressed`（约 5 MB/s，原图约 500 MB/s） |
| 发布话题 | `/inspection/annotated/compressed`（`sensor_msgs/CompressedImage`） |
| HTTP 接口 | `GET /latest.jpg`、`GET /meta.json`、`GET /healthz`、`POST /annotate` |
| 默认端口 | 8730 —— 起服务前 `ss -lntp \| grep 8730` 查残留 |
| 输出 | 每条一个 `CheckResult`（`item_id / status / answer / reason / latency_ms / evidence / checked_at`）+ `summarize` 汇总 |

---

## 标注图上屏

加 `--annotate-url` 后，每条检查会把**检测框**和**文字结论**画到模型实际看到的那张图上，
再 POST 回来发布。

```bash
--annotate-url http://127.0.0.1:8730   # 开启发布
--max-text-lines 3                     # 底部面板最多几行（默认 3，超出用 … 收尾）
--text-scale 1.4                       # 字号系数（默认 1.0）
```

排版规则：

- 文字面板固定在最**底部**（深底白字），不会遮挡被检目标；
- 最多画 `--max-text-lines` 行，图片高度不会无限增长；
- 字号按图片长边自适应，再乘 `--text-scale`；
- 框颜色随结论：绿=通过、红=不通过、黄=未知、灰=错误。

实现：`inspection_perception/annotation.py`（`render_result` / `draw_text_panel` / `iter_boxes`）。
只用 Pillow，不引入新依赖；字体自动探测 Noto CJK / 文泉驿，都没有时回落 Pillow 默认字体
（中文显方块，英文与数字仍可读）。

> 标注是**旁路能力**：网络不通、字体缺失、编码失败都只默默跳过，不会让一条检查变成 `ERROR`。

---

## 命令行接口

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

| 参数 | 说明 |
|---|---|
| `--tasks` | 检查清单（YAML/JSON），每条含 `id/question/kind`，检测类还需 `target` |
| `--source` | `file` 离线图片、`http` 取图服务、`ros1/ros2` 直接订阅、`auto` 自动探测 |
| `--backend` | **默认是 `moondream`，用 Florence-2 时务必显式传 `--backend florence2`** |
| `--dtype` | 默认 `float32`。Florence-2 请保持这个值 |
| `--evidence-dir` | 证据图目录，传空字符串不存图 |

## 任务清单

见 `configs/tasks/`：

- `smoke.json`：最小冒烟（`dummy` 也能跑）
- `demo_florence2.yaml`：Florence-2 检测示例（`kind` 与 `target` 成对）

`kind` 取值：`yes_no` / `state` / `count` / `read_text` / `describe`。
Florence-2 只认固定任务标签，因此前三种**必须**给英文 `target`，否则结果退化为 `UNKNOWN`。

## 目录结构

```text
inspection_perception/   # 核心包：vlm/ capture/ checks/ pipeline cli schemas annotation
configs/                 # 任务 / profile / capture 示例
scripts/                 # setup、download_models、doctor、ask_image、ask_live
tests/                   # 零依赖单测（无需权重与 ROS）
adapters/                # ROS 薄适配层（可选）：取图 + 标注发布
```

---

## ROS 1 实测坑（端侧验证时按撞到顺序）

在 Jetson NX + ROS 1 Noetic + python 3.8 上逐项验证遇到的问题，照做可避开：

| # | 现象 | 根因 | 解决 |
|---|---|---|---|
| 1 | 推理**无任何输出**、退出码 139 | `torch>=1.13` 解析到 2.1.0，卷积段错误 | 钉 `torch==2.4.1`（CPU）或 NVIDIA wheel |
| 2 | `pip install -e .` 报需要 setup.py | py3.8 的 pip 不支持 PEP 660 | 改用 `pip install --no-build-isolation .` |
| 3 | `invalid command 'bdist_wheel'` | venv 里没有 `wheel` | 先装 `wheel` + `setuptools` |
| 4 | 取图节点报 `No module named 'rospy'` | 装了 miniconda，`python3` 解析到 conda | 取图侧必须用 **`/usr/bin/python3`** |
| 5 | 换了解释器仍缺 rospy | `export PYTHONPATH=.` **覆盖**了 ROS 路径 | 改成追加：`export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}"` |
| 6 | `import rospy` 缺 `rospkg` | `rospkg` 在 `/usr/lib/python3/dist-packages` | 把该目录也加进 `PYTHONPATH` |
| 7 | 图像话题存在但 `rostopic hz` 无数据 | 驱动 `sendrgb: 0`，而 `sendrgbundistort` 依赖它 | 同时置 `sendrgb: 1` 与 `sendrgbundistort: 1` 并重启驱动 |
| 8 | 取图返回旧帧（`age_ms` 上万） | 驱动重启后取图节点的订阅已断 | 重启取图节点重新订阅 |

---

## 端侧部署注意事项（NX / Jetson）

| # | 事项 | 说明 |
|---|---|---|
| 1 | **torch 必须装平台 wheel** | aarch64 Jetson wheel（实测 `2.0.0+nv23.05`，CUDA 11.4）；**x86 二进制不能拷过来** |
| 2 | **transformers 钉死 4.46.3** | 另需 `timm==1.0.9`、`einops` |
| 3 | **python 3.8** | 保持 `from __future__ import annotations`，不用 3.9+ 语法 |
| 4 | **权重可跨平台** | 888 MB 目录可直接拷；注意 `config.json`、`preprocessor_config.json` 与 remote code 的 `.py` 要齐 |
| 5 | **去畸变话题需要驱动开开关** | `sendrgbundistort: 1`，否则只 advertise 无数据 |
| 6 | **端口冲突** | 取图服务默认 8730，起前 `ss -lntp \| grep 8730` |
| 7 | **磁盘** | venv（含 torch 约 1.5G）+ 权重 888M，先 `df -h` |
| 8 | **不要装进系统 python** | 导航栈在用系统解释器，务必独立 venv |

## 常见问题

- **仓库里没有权重**：`models/` 已被 `.gitignore` 忽略；用 `scripts/download_models.sh` 获取。
  离线设备请先在别处下好再拷目录。
- **没有 CUDA**：`--device cpu` 可用，单条从约 1 秒变成数十秒。
- **没有 ROS**：`--source file/http` 完全不碰 ROS；ROS 模块是懒加载的，未装 ROS 不影响安装与单测。
- **`yes_no` 全是 `UNKNOWN`**：确认传了 `--backend florence2`，且每条任务里有 `target`。
- **下载慢或失败**：`export HF_ENDPOINT=https://hf-mirror.com` 后重跑，脚本会校验文件完整性。

## 许可

Apache-2.0，见 `LICENSE`；第三方组件许可见 `THIRD_PARTY_LICENSES.md`。
模型权重不随本仓库分发，许可由上游模型决定（Florence-2 为 MIT）。
