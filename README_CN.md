# SimpleSort

[![English](https://img.shields.io/badge/Language-English-blue)](README.md)

在线多目标追踪算法，基于 YOLO 检测 + Kalman 滤波 + ReID 特征。核心创新：**KEMA** — 自适应在线聚类，用于外观特征管理。

## 概述

SimpleSort 是在线多目标追踪器，整合了 YOLO 行人检测器、8 自由度 Kalman 滤波器和 FastReID 外观特征提取模型。其核心创新是 **KEMA** — 一种自适应在线聚类算法，为每条轨迹维护紧凑的外观特征中心，并自动完成分裂决策，取代了 DeepSORT 和 StrongSORT 中依赖人工阈值的特征图库。

**开发场景**：面向固定监控下的建筑工人追踪，人物重叠较少、统一着装外观相近、目标因遮挡或离开画面而短暂丢失（可达数十秒）。无相机运动补偿，不适合移动相机场景。其他场景未经充分验证。

```
YOLO → 检测 → 去重 → ReID 特征 → 三级级联匹配 → Kalman 更新
         ↑                                              ↓
         └────────── KEMA 在线外观模型 ─────────────────┘
```

## 快速开始

**Python API** — 供代码集成：

```python
from SimpleSort import Detector, ReID, Matcher, Tracker, VideoFrameReader

tracker = Tracker(
    Detector("yolo26x.pt"),
    ReID("msmt_sbs_R101-ibn.pth"),
    Matcher(appearance_thresh=0.4, strict_iou_thresh=0.5, loose_iou_thresh=0.3),
)

for frame_idx, frame in VideoFrameReader("video.mp4"):
    tracker.update(frame)
    annotated = tracker.visual(frame)
```

**命令行** — 一行搞定：

```bash
cd SimpleSort
python run.py video.mp4
```

完整参数：

```bash
python run.py video.mp4 \
    --yolo yolo26x.pt \
    --reid msmt_sbs_R101-ibn.pth \
    --output output/run \
    --no-video           # 跳过输出视频，仅生成 FPS 图和统计数据
```

模型权重首次使用时会自动下载。YOLO 权重由 Ultralytics 自动获取；ReID 权重从 [FastReID 模型库](https://github.com/JDAI-CV/fast-reid/releases) 下载。

## 三者对比：DeepSORT vs StrongSORT vs SimpleSort

| | DeepSORT | StrongSORT | SimpleSort |
|---|---|---|---|
| **状态机** | Tentative → Confirmed → Deleted | Tentative → Confirmed → Deleted | Tentative → Confirmed → **Lost → Recovering** → Confirmed |
| **匹配阶段 1** | 级联：外观 + Mahalanobis（confirmed 轨迹，按 `time_since_update` 分层） | 级联：外观 + Mahalanobis 门控（confirmed 轨迹，按 `time_since_update` 分层） | 严格 IoU（`tsu ≤ 1`，排除 Recovering）+ **外观门控** |
| **匹配阶段 2** | IoU（unconfirmed + `tsu = 1` 的未匹配 confirmed） | IoU（unconfirmed + `tsu = 1` 的未匹配 confirmed） | 宽松 IoU（剩余，排除 Lost & Recovering）+ **外观门控** |
| **匹配阶段 3** | — | — | **外观 NN**（全部剩余，含 Lost） |
| **外观模型** | `NearestNeighborDistanceMetric` — 存储全部特征（预算限制，最旧淘汰），最小 NN 距离 | `NearestNeighborDistanceMetric` — 同上，可选 **EMA**（单平滑特征） | **KEMA** — 自适应在线聚类，K 个 EMA 活跃中心，自动分裂，无需人工阈值 |
| **运动门控** | Mahalanobis 距离（χ² 95%）作用于联合代价矩阵 | Mahalanobis 距离（χ² 95%）作用于外观代价矩阵 | **外观门控 IoU** — KEMA 中心与检测的余弦距离门控代价矩阵（无 Mahalanobis） |
| **Kalman 滤波器** | 单一共享 KF 实例 | 每轨迹独立 KF 实例，**NSA**（置信度噪声缩放）、**ECC** 相机补偿、**MC**（运动代价混合） | 无状态类方法，**NSA**（置信度噪声缩放） |
| **丢失轨迹恢复** | 无 Lost 状态。未匹配轨迹递增 age，超过 `max_age` 删除 | 同 DeepSORT | **Recovering 考察期** — 重找回轨迹需 `n_confirm` 次命中；一次未匹配 → 回到 Lost；排除在 IoU 匹配之外 |
| **后处理** | — | AFLink（全局连接）+ GSI（高斯平滑插值） | — |
| **ReID 骨干** | 自训 CNN（Market-1501） | — | FastReID SBS（ResNet + GeM + BNNeck） |
| **检测器** | 外部（任意） | 外部（任意） | 内置 YOLO（Ultralytics） |

## 核心创新

### 1. KEMA — 自适应在线聚类

不保存所有检测特征（DeepSORT），也不使用固定阈值 EMA（StrongSORT）。KEMA 为每条轨迹学习一组紧凑的特征中心（受 K-means 思想启发，但采用 EMA 增量更新、无迭代优化）：

- **分裂判定** — 新特征离最近中心足够远（> mean + `n_std * pooled_std`）→ 创建新聚类
- **无需人工阈值** — 分裂标准相对于每个聚类自身的余弦距离分布，使用跨轨迹的 pooled 标准差
- **匹配中心** — 只有足够活跃的聚类（≥ `min_hits` 次命中）才对匹配器暴露，防止新轨迹的噪声干扰

### 2. 外观门控 IoU 匹配

DeepSORT 和 StrongSORT 用 Mahalanobis 距离基于运动过滤误匹配。SimpleSort 反过来：**外观过滤位置**。在第 1、2 阶段（严格/宽松 IoU），若轨迹的 KEMA 图库到检测的余弦距离超过阈值，直接拒绝匹配。计算更轻量，在运动不确定时更可靠。

### 3. Recovering 考察期状态

失踪轨迹重新找回后，不会立即恢复为 Confirmed，而是进入 **Recovering** 考察期：

- 一次未匹配 → 回到 Lost（防止闪烁）
- 连续 `n_confirm` 次命中 → 恢复 Confirmed

Recovering 轨迹被排除在 IoU 匹配（阶段 1-2）之外，防止其抢夺稳定轨迹的检测。

### 4. 检测去重

在特征提取之前，检测器进行按面积排序的 NMS：重叠检测框合并，保留最大者。解决 YOLO 对同一目标偶尔产生双重检测的问题。

## 匹配流程

| 阶段 | 候选轨迹 | 匹配方式 | 说明 |
|------|---------|---------|------|
| 1. 严格 IoU | `time_since_update ≤ 1`，非 Recovering | IoU ≥ `strict_iou_thresh` + 外观门控 | 近期更新的轨迹优先 |
| 2. 宽松 IoU | 剩余（排除 Lost、Recovering） | IoU ≥ `loose_iou_thresh` + 外观门控 | 未匹配轨迹的第二次机会 |
| 3. 外观 NN | 全部剩余（含 Lost） | 余弦距离 ≤ `appearance_thresh` | 纯外观匹配，找回丢失轨迹 |
| 4. 初始化 | 未匹配的检测 | — | 创建新的 Tentative 轨迹 |
| 5. 清理 | — | — | 从活跃集合移除 |

## 文件结构

```
SimpleSort/
├── __init__.py        包入口，导出所有公共类
├── detector.py        YOLO 行人检测器 + Detection 数据类
├── kalmanfilter.py    8 维 Kalman 滤波器（匀速模型）
├── kema.py            在线聚类，自适应余弦距离门控
├── matcher.py         IoU 代价、余弦代价、Hungarian 匹配
├── reid.py            FastReID SBS 特征提取器
├── track.py           轨迹类（状态机 + KEMA）
├── tracker.py         主跟踪器，三级级联匹配
├── video_reader.py    视频 I/O，支持上下文管理器
├── run.py             CLI 入口，FPS 图与 ID 统计
├── requirements.txt   Python 依赖
├── README.md          英文文档
├── README_CN.md       中文文档
├── LICENSE            GNU AGPL-3.0
├── ultralytics/       Ultralytics YOLO（AGPL-3.0）
└── fast-reid/         FastReID（Apache 2.0）
```

## 许可证

SimpleSort 基于 **GNU AGPL-3.0** 协议开源。详见 [LICENSE](LICENSE)。

本项目引用了以下第三方代码：
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)（AGPL-3.0）
- [FastReID](https://github.com/JDAI-CV/fast-reid)（Apache 2.0）

## 引用

如果你在研究中使用了 SimpleSort，请引用：

```bibtex
@software{nivendel2026simplesort,
  author       = {Nivendel},
  title        = {SimpleSort},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/nivendel/SimpleSort},
}
```

## 致谢

本项目在 Claude Code（Anthropic）和 DeepSeek-V4 的协助下开发完成。

## 参考文献

```bibtex
@inproceedings{wojke2017simple,
  author    = {Nicolai Wojke and Alex Bewley and Dietrich Paulus},
  title     = {Simple Online and Realtime Tracking with a Deep Association Metric},
  booktitle = {2017 IEEE International Conference on Image Processing (ICIP)},
  year      = {2017},
  publisher = {IEEE},
}

@article{du2023strongsort,
  title     = {Strongsort: Make DeepSort Great Again},
  author    = {Du, Yunhao and Zhao, Zhicheng and Song, Yang and Zhao, Yanyun and Su, Fei and Gong, Tao and Meng, Hongying},
  journal   = {IEEE Transactions on Multimedia},
  year      = {2023},
  publisher = {IEEE},
}

@article{he2020fastreid,
  title     = {FastReID: A Pytorch Toolbox for General Instance Re-identification},
  author    = {He, Lingxiao and Liao, Xingyu and Liu, Wu and Liu, Xinchen and Cheng, Peng and Mei, Tao},
  journal   = {arXiv preprint arXiv:2006.02631},
  year      = {2020},
}

@software{ultralytics2023yolo,
  author    = {Glenn Jocher and Jing Qiu and Ayush Chaurasia},
  title     = {Ultralytics YOLO},
  url       = {https://github.com/ultralytics/ultralytics},
  version   = {8.0.0},
  year      = {2023},
}
```
