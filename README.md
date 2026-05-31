# SimpleSort

[![中文](https://img.shields.io/badge/语言-中文-red)](README_CN.md)

Online multi-object tracking with YOLO detection, Kalman filtering, and ReID appearance features. Core innovation: **KEMA** — adaptive online clustering for appearance feature management.

## Overview

SimpleSort is an online multi-object tracker that combines a YOLO person detector, an 8-DOF Kalman filter, and a FastReID appearance extractor. Its core innovation is **KEMA** — an adaptive online clustering algorithm that maintains a compact set of appearance feature centres per track with automatic split decisions, replacing the hand-tuned feature galleries of DeepSORT and StrongSORT.

**Developed for** construction worker tracking with fixed surveillance cameras, where person overlap is infrequent, individuals wear similar uniforms, and targets may temporarily disappear due to occlusion or leaving the frame for up to tens of seconds. No camera motion compensation — not designed for moving cameras. Performance in other scenarios has not been validated.

```
YOLO → Detection → Dedup → ReID features → 3-stage matching → Kalman update
         ↑                                                    ↓
         └─────────── KEMA online appearance model ───────────┘
```

## Quick Start

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

Model weights are downloaded automatically on first use. YOLO weights are fetched by Ultralytics; ReID weights are fetched from the [FastReID model zoo](https://github.com/JDAI-CV/fast-reid/releases).

## Comparison: DeepSORT vs StrongSORT vs SimpleSort

| | DeepSORT | StrongSORT | SimpleSort |
|---|---|---|---|
| **State machine** | Tentative → Confirmed → Deleted | Tentative → Confirmed → Deleted | Tentative → Confirmed → **Lost → Recovering** → Confirmed |
| **Matching stage 1** | Cascade: appearance + Mahalanobis (confirmed tracks, by `time_since_update` level) | Cascade: appearance + Mahalanobis gating (confirmed tracks, by `time_since_update` level) | Strict IoU (`tsu ≤ 1`, excl. Recovering) + **appearance gate** |
| **Matching stage 2** | IoU (unconfirmed + `tsu = 1` unmatched confirmed) | IoU (unconfirmed + `tsu = 1` unmatched confirmed) | Loose IoU (remaining, excl. Lost & Recovering) + **appearance gate** |
| **Matching stage 3** | — | — | **Appearance NN** (all remaining, incl. Lost) |
| **Appearance model** | `NearestNeighborDistanceMetric` — stores all features per track (budget-limited, oldest evicted), min NN distance | `NearestNeighborDistanceMetric` — same, with optional **EMA** (single smoothed feature per track) | **KEMA** — adaptive online clustering, K EMA-active centres, auto split, no hand-tuned threshold |
| **Motion gating** | Mahalanobis distance (χ² 95%) on combined cost matrix | Mahalanobis distance (χ² 95%) on appearance cost matrix | **Appearance gates IoU** — cosine distance from KEMA centres to detection gates the cost matrix (no Mahalanobis) |
| **Kalman filter** | Single shared KF instance | Per-track KF instance, **NSA** (confidence-based noise scaling), **ECC** camera compensation, **MC** (motion cost mixing) | Stateless classmethods, **NSA** (confidence-based noise scaling) |
| **Lost track recovery** | No Lost state. Unmatched tracks increment age, deleted at `max_age` | Same as DeepSORT | **Recovering probation** — re-acquired tracks need `n_confirm` hits; one miss → back to Lost; excluded from IoU stages |
| **Post-processing** | — | AFLink (global link) + GSI (Gaussian interpolation) | — |
| **ReID backbone** | Custom CNN (Market-1501) | — | FastReID SBS (ResNet + GeM + BNNeck) |
| **Detector** | External (any) | External (any) | Built-in YOLO (Ultralytics) |

## Key Innovations

### 1. KEMA — Adaptive online clustering

Instead of storing every detection feature (DeepSORT) or applying a fixed-threshold EMA (StrongSORT), KEMA learns a compact set of feature centres per track. It is conceptually inspired by K-means but uses EMA-based incremental updates (no iterative Lloyd's algorithm):

- **Split** — a new feature is far enough from the nearest centre (> mean + `n_std * pooled_std`) → it seeds a new cluster
- **No hand-tuned absolute threshold** — the split criterion is relative to each cluster's own similarity distribution, using the pooled within-cluster standard deviation across all tracks
- **Match centres** — only active clusters (≥ `min_hits` hits) are exposed to the matcher, preventing noise from young tracks

### 2. Appearance-gated IoU matching

DeepSORT and StrongSORT use Mahalanobis gating to discard unlikely matches based on motion. SimpleSort reverses this: **appearance gates position**. In stages 1–2 (strict and loose IoU), a candidate match is rejected if the cosine distance from the track's KEMA gallery to the detection exceeds `appearance_thresh`. This is cheaper to compute and more reliable when motion is unpredictable.

### 3. Recovering state

When a Lost track is re-acquired, it does not immediately return to Confirmed. Instead it enters a **Recovering** probation:

- One missed frame → back to Lost (prevents flickering)
- `n_confirm` consecutive hits → Confirmed again

Recovering tracks are excluded from IoU matching (stages 1–2) to prevent them from stealing detections from well-tracked targets.

### 4. Detection deduplication

Before feature extraction, the detector runs an area-prioritised NMS: overlapping detections are merged, keeping the largest box. This removes double detections that YOLO occasionally produces for the same person.

## Pipeline Detail

| Stage | Candidates | Matcher | Description |
|-------|-----------|---------|-------------|
| 1. Strict IoU | `time_since_update ≤ 1`, not Recovering | IoU ≥ `strict_iou_thresh` + appearance gate | Recently updated tracks get first pick |
| 2. Loose IoU | Remaining (excl. Lost & Recovering) | IoU ≥ `loose_iou_thresh` + appearance gate | Give unmatched tracks a second chance |
| 3. Appearance NN | All remaining (incl. Lost) | Cosine distance ≤ `appearance_thresh` | Pure appearance for recovering lost tracks |
| 4. Initiation | Unmatched detections | — | Seed new Tentative tracks |
| 5. Cleanup | Deleted tracks | — | Remove from active set |

## File Structure

```
SimpleSort/
├── __init__.py        Package entry, exports all public classes
├── detector.py        YOLO person detector + Detection data class
├── kalmanfilter.py    8-dim Kalman filter (constant velocity)
├── kema.py            Online clustering with adaptive gating
├── matcher.py         IoU cost, cosine cost, Hungarian assignment
├── reid.py            FastReID SBS feature extractor
├── track.py           Track class (state machine + KEMA)
├── tracker.py         Main Tracker with 3-stage pipeline
├── video_reader.py    Video I/O with context manager
├── requirements.txt   Python dependencies
├── README.md          English documentation
├── README_CN.md       Chinese documentation
├── LICENSE            GNU AGPL-3.0
├── ultralytics/       Ultralytics YOLO (AGPL-3.0)
└── fast-reid/         FastReID (Apache 2.0)
```

## License

SimpleSort is licensed under **GNU AGPL-3.0**. See [LICENSE](LICENSE).

This project incorporates:
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (AGPL-3.0)
- [FastReID](https://github.com/JDAI-CV/fast-reid) (Apache 2.0)

## Citation

If you use SimpleSort in your research, please cite:

```bibtex
@software{SimpleSort,
  author       = {Nivendel},
  title        = {SimpleSort},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/nivendel/SimpleSort},
  note         = {With assistance from Claude Code and deepseek-v4-pro[1m]},
}
```

## References

```bibtex
@inproceedings{wojke2017simple,
  author    = {Nicolai Wojke and Alex Bewley and Dietrich Paulus},
  title     = {Simple Online and Realtime Tracking with a Deep Association Metric},
  booktitle = {2017 IEEE International Conference on Image Processing (ICIP)},
  year      = {2017},
  publisher = {IEEE},
}

@article{du2023strongsort,
  author    = {Yunhao Du and Yang Song and Bo Yang and Yanyun Zhao},
  title     = {StrongSORT: Make DeepSORT Great Again},
  journal   = {IEEE Transactions on Multimedia},
  volume    = {25},
  pages     = {8725--8737},
  year      = {2023},
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
