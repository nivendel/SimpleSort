"""YOLO person detector.

Input:  frame (H, W, 3) BGR  →  np.ndarray
Output: list[Detection]

Detection fields:
    tlwh : np.ndarray (4,)  — [x, y, w, h]
    xyah : np.ndarray (4,)  — [x, y, aspect_ratio, h]
    conf : float            — confidence score
    feat : np.ndarray (D,) | None  — set later by ReID

Copyright (C) 2026 Nivendel, College of Civil Engineering, Tongji University
With assistance from Claude Code and deepseek-v4-pro[1m]
SPDX-License-Identifier: AGPL-3.0-or-later
"""

import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "ultralytics"))
from ultralytics import YOLO


class Detection:
    """Detection box with confidence and optional feature."""

    __slots__ = ("tlwh", "conf", "feat", "xyah")

    def __init__(
        self,
        tlwh: np.ndarray | list,
        conf: float,
        feat: np.ndarray | list | None = None,
    ):
        self.tlwh = np.asarray(tlwh, dtype=np.float64)
        x, y, w, h = self.tlwh
        self.xyah = np.asarray([x, y, w / max(h, 1e-6), h], dtype=np.float64)
        self.conf = conf
        self.feat = None if feat is None else np.asarray(feat, dtype=np.float32)

    def set_feat(self, feat: np.ndarray | list) -> None:
        self.feat = np.asarray(feat, dtype=np.float32)


class Detector:
    """YOLO person detector with optional deduplication.

    Parameters
    ----------
    model_path : str = "yolo26x.pt"
    conf : float = 0.328
        YOLO confidence threshold.
    iou : float = 0.552
        YOLO NMS IoU threshold.
    dedup_iou_thresh : float = 0.6
        Dedup IoU threshold (≥ 1.0 disables).
    """

    def __init__(
        self,
        model_path: str = "yolo26x.pt",
        conf: float = 0.328,
        iou: float = 0.552,
        dedup_iou_thresh: float = 0.6,
    ):
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.dedup_iou_thresh = dedup_iou_thresh

    @staticmethod
    def _dedup_by_area(dets: list[Detection], iou_thresh: float) -> list[Detection]:
        """Remove overlapping detections, keeping the largest per group."""
        if len(dets) <= 1:
            return dets

        idx_sorted = sorted(
            range(len(dets)),
            key=lambda i: dets[i].tlwh[2] * dets[i].tlwh[3],
            reverse=True,
        )
        keep = []
        for i in idx_sorted:
            x1, y1, w1, h1 = dets[i].tlwh
            area1 = w1 * h1
            dup = False
            for j in keep:
                x2, y2, w2, h2 = dets[j].tlwh
                xi1, yi1 = max(x1, x2), max(y1, y2)
                xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
                inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                union = area1 + w2 * h2 - inter
                if inter / max(union, 1e-6) > iou_thresh:
                    dup = True
                    break
            if not dup:
                keep.append(i)

        return [dets[i] for i in keep]

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run YOLO on *frame*, return person detections."""
        results = self.model(
            frame, conf=self.conf, iou=self.iou, classes=0, verbose=False
        )
        dets = []
        if results[0].boxes is not None:
            xyxy = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            for box, conf in zip(xyxy, confs):
                tlwh = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
                dets.append(Detection(tlwh, float(conf)))

        if self.dedup_iou_thresh < 1.0:
            dets = self._dedup_by_area(dets, self.dedup_iou_thresh)
        return dets

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        return self.detect(frame)

    def visual(self, frame: np.ndarray, dets: list[Detection]) -> np.ndarray:
        """Draw detection boxes with confidence scores (RGB)."""
        annotated = frame.copy()
        for det in dets:
            x, y, w, h = map(int, det.tlwh)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                annotated, f"{det.conf:.2f}", (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
            )
        return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
