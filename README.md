# memory_cam：物体级空间记忆与场景理解原型系统

面向日常动态场景的物体级空间记忆与场景理解——课程设计配套代码。
从第一视角视频（或普通摄像头画面）中持续提取物体与空间信息，构建持久的
物体级记忆，并支持自然语言查询、变化回顾与人机协同修正。

系统为「感知层 → 记忆层 → 交互层」三层架构：从第一视角视频流中持续提取物体与
空间信息，构建持久的物体级记忆，并支持自然语言查询、变化回顾与人机协同修正。

> **实现说明**：仿真核心仅依赖 Python 标准库（无 GPU），摄像头与视频演示需要 OpenCV 和 NumPy。
> 感知模块采用**可插拔后端**设计——默认使用仿真后端（SimBackend）以保证实验可复现，
> 真实模型（YOLO / DepthAnything / ORB-SLAM3 / CLIP / VLM-LLM）只需替换对应类的
> `predict()` 实现，接口签名不变，上层管线无需改动。

## 1. 目录结构

```
memory_cam/
├── osm/                      # 核心包
│   ├── config.py             # 全局配置与超参数（消融实验按名覆盖）
│   ├── types.py              # 数据结构：Observation / ObjectNode / Event
│   ├── utils.py              # 向量运算、匈牙利匹配、统计量
│   ├── perception.py         # 感知层：检测、深度反投影、位姿、外观编码、重识别
│   ├── memory.py             # 记忆层：物体级记忆构建、更新、人机协同修正
│   ├── interaction.py        # 交互层：查询解析、混合检索、答案生成
│   ├── metrics.py            # 评价指标：IDF1、重识别、定位误差、F1、NASA-TLX
│   ├── runtime.py            # 日志配置、逐帧控制与限速
│   └── pipeline.py           # 端到端管线：三层串联 + 人机协同模拟
├── demo/
│   └── webcam_demo.py        # 实时演示：电脑/外接摄像头 → 空间记忆与场景理解
├── experiments/
│   ├── sim_dataset.py        # Sim-Scene-12 合成动态场景数据集生成
│   └── run_all.py            # 一键复现：系统性能 + 消融 + 用户实验
├── run_demo.py               # 单场景演示（打印记忆库与问答示例）
├── tests/                    # 运行控制、视频输入与时间戳回归测试
├── docs/ROADMAP.md            # 开发路线图
├── results/                  # 实验结果（自动生成）
│   ├── metrics.json          # 全部原始指标
│   ├── tables.md             # 论文用表格
│   └── ablation.md           # 消融实验表
├── requirements.txt       # 仅摄像头 Demo 需要（核心包零依赖）
└── README.md
```

## 2. 运行

```bash
git clone https://github.com/tomacx/memory_cam.git
cd memory_cam

python run_demo.py --fps -1      # 单场景演示：不限速运行并打印结果
python run_demo.py --step        # 仿真逐帧：第一帧后回车推进，q 退出
python -m experiments.run_all   # 复现全部实验，约 15 s，结果写入 results/
```

环境要求：Python ≥ 3.9，核心功能**零第三方依赖**；摄像头 Demo 需要
`pip install -r requirements.txt`（opencv-python 与 numpy）。
随机种子固定为 20260924，结果完全可复现。

## 2.1 摄像头与视频演示

`demo/webcam_demo.py` 把**摄像头或本地视频文件**接入同一条
ObjectMemory 记忆管线，可直接演示日常动态场景的物体级空间记忆与场景理解：

```bash
pip install -r requirements.txt           # 摄像头模式需要 opencv-python

python demo/webcam_demo.py                # 自动检测摄像头，失败回退仿真模式
python demo/webcam_demo.py --camera 1     # 使用外接 USB 摄像头
python demo/webcam_demo.py --mode sim     # 仿真模式（无摄像头/无 OpenCV 也能演示）
python demo/webcam_demo.py --headless     # 不弹窗，仅终端输出（服务器/录屏）
python demo/webcam_demo.py --selftest     # 离线自检 demo 依赖的 osm API
python demo/webcam_demo.py --video input.mp4 --step --log-level DEBUG
python demo/webcam_demo.py --video input.mp4 --fps 30
python demo/webcam_demo.py --video input.mp4 --headless --fps -1
python demo/webcam_demo.py --video input.mp4 --headless --step
```

摄像头与视频输入互斥；`--video` 不能与 `--mode sim` 组合。视频文件缺失、无法打开或
缺少图像处理依赖时明确报错，不会回退成仿真。视频保留原始尺寸和顺序。

感知后端：运行期指数滑动平均背景建模 + 帧差提取前景 blob，HSV 直方图
（12×4×2=96 维）作外观特征，像素坐标按可调「像素—米比例」映射为平面位置；
静止超过 45 帧的前景自动并入背景（对应「日常物体大多数时候静止」的先验）。

| 按键 | 功能 | 对应系统机制 |
| --- | --- | --- |
| . | 推进一帧并暂停 | 单步调试 |
| 空格 | 播放 / 暂停 | 运行控制 |
| n | 给选中/最新物体命名 | 一次性命名（user_label） |
| c | 裁决当前待确认事件（y/n） | 被动裁决（resolve_pending） |
| m | 主动纠正物体位置 | 主动纠正（user_correct_position） |
| f | 自然语言查询 | 混合检索 + 答案生成 |
| s | 打印记忆库快照 | 场景回顾 |
| + / - | 调整像素—米比例 | 尺度标定 |
| 鼠标左键 | 选中画面中的物体 | 交互对象选择 |
| q / Esc | 退出 | — |

演示建议流程：空画面 1–2 秒学习背景 → 把水杯放到桌上（added 入库）→ 按 n 命名
「我的水杯」→ 移动水杯（触发 moved / pending_confirm，按 c 裁决）→ 按 f 查询
「我的水杯在哪里？」→ 按 s 查看记忆库快照。

## 2.2 日志、逐帧与帧率控制

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--log-level INFO\|DEBUG` | INFO | 三个入口均支持；INFO 为运行状态与关键事件，DEBUG 增加观测、关联判定和逐帧耗时 |
| `--step` | 关闭 | 两个演示入口支持；首帧自动处理后等待手动推进 |
| `--fps FLOAT` | 30 | 两个演示入口支持；正数为处理帧率上限，`-1` 不主动限速 |

窗口模式使用句点键单步、空格播放/暂停；暂停时仍可命名、确认、纠正和查询。
仿真与无窗口单步模式使用终端回车推进，`q`、输入结束或 Ctrl-C 退出。
等待期间不读取下一帧、不更新背景和记忆。单步不受帧率限制，窗口恢复播放后按配置限速。
限速包括处理与显示耗时，处理较慢时按实际速度运行，不跳帧或累计追赶。
`-1` 模式仍保留窗口必要的事件处理，实际速度受解码、计算和摄像头采集能力限制。

日志输出到 stderr，问答、快照和交互提示输出到终端。现有 `--verbose` 兼容为 DEBUG，
显式 `--log-level` 优先。批量实验保持不限速：

```bash
python -m experiments.run_all --log-level INFO
python demo/webcam_demo.py --video input.mp4 --log-level DEBUG --fps -1 2>debug.log
```

播放速度与观测时间分离：视频使用源时间戳，不可用时按帧索引和源帧率换算；两者均不可用则报错。
摄像头使用采集时间，仿真使用场景时间。调速与暂停不改写已处理帧的时间，“上次看到”按输入时间计算。
记忆状态的超时阈值目前仍按帧计算。

测试无需打开摄像头；视频测试使用临时生成的视频，缺少可导入的 OpenCV/NumPy 时跳过：

```bash
python -m unittest discover -s tests -v
```

## 3. 数据集

`Sim-Scene-12` 为合成动态室内场景数据集，用于在没有真实第一视角数据时装定
量评测环境：

| 属性 | 取值 |
| --- | --- |
| 场景序列数 | 12 |
| 每序列帧数 | 200（帧间隔 3 s，即 10 分钟） |
| 每场景物体数 | 18（12 类高频物品，含同类多实例） |
| 动态事件 | 每序列 3 次物体移动（位移 0.8–2.0 m） |
| 干扰因素 | 临时遮挡、出画、检测漏检、类别混淆、外观观测噪声、单目深度与位姿误差 |

替换为真实数据：保持 `frames` 字段结构（`frame_id / timestamp / pose / visible`），
把 `Detector.predict()` 换成真实检测器即可，其余代码无需改动。

## 4. 关键机制

1. **分级重识别判定**（`perception.Reidentifier.identify`）
   外观证据强 → 直接认定为同一物体（允许位置变化）；外观中等 + 位置突变 → 转人工确认；
   两项证据均不足 → 新建记忆节点。该机制解决了"物体被移动后几何证据失效"的核心难点。
2. **记忆更新与变化确认**（`memory.ObjectMemory`）
   位置用滑动平均融合；位移需连续 2 帧确认后才写入变化事件，抑制观测噪声引起的误报；
   低置信的推断不自动改写记忆，而是挂起为待确认事件。
3. **人机协同闭环**（`memory.resolve_pending` / `user_correct_position`）
   一次性命名标注、被动确认/拒绝、主动位置纠正三种交互，均写入记忆并影响后续匹配。
4. **类别投票修正**（`memory._apply_update`）
   以多数票修正单帧分类抖动，用户纠正后锁定类别不再自动改写。

## 5. 实验结果摘要（Sim-Scene-12，种子 20260924）

| 指标 | 中期基线 | 最终系统 | 目标 |
| --- | --- | --- | --- |
| 检测 mAP@0.5 | 93.9% | 93.9% | ≥ 75% |
| 跟踪 IDF1 | 82.4% | 98.4% | ≥ 80% |
| 跨时段重识别准确率 | 74.8% | 97.5% | ≥ 85% |
| 定位中位误差 | 0.101 m | 0.056 m | ≤ 0.20 m |
| 记忆更新一致率 | 75.7% | 90.2% | ≥ 88% |
| 变化检测 F1 | 0.0% | 83.3% | ≥ 80% |
| 待确认/裁决 | 0.00 次/10 min | 0.50 次/10 min | ≤ 2 次 |
| 查询成功率 Top-3 | 79.2% | 84.5% | ≥ 85% |
| NASA-TLX | 24.2 | 26.9 | ≤ 40 |

完整表格见 `results/tables.md`，原始数值见 `results/metrics.json`。

> **重要提示**：上述结果来自**仿真数据集**，用于验证方法有效性与系统可运行性，
> 不等同于真实场景性能；真实数据集（如自采第一视角视频）上的结果仍需实测补充。

## 6. 接入真实模型的改造点

| 模块 | 当前（仿真后端） | 真实实现建议 |
| --- | --- | --- |
| `Detector` | 按可见性采样 + 噪声 | YOLOv8 / Grounding-DINO |
| `DepthEstimator` | 真值深度 + 6% 噪声 | DepthAnything / Metric3D + 相机内参 |
| `PoseEstimator` | 真值位姿 + 噪声/漂移 | ORB-SLAM3 / VINS-Mono（开启回环） |
| `AppearanceEncoder` | 原型 + 噪声 | CLIP / DINOv2 图像编码器 |
| `AnswerGenerator` | 模板生成 | VLM 视觉校验 + LLM 组织语言 |
