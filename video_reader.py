"""Video frame reader with context-manager support.

Input:  path (str | Path)
Yield:  (frame_idx: int, frame: np.ndarray (H, W, 3) BGR)

Copyright (C) 2026 Nivendel
With assistance from Claude Code and deepseek-v4-pro[1m]
SPDX-License-Identifier: AGPL-3.0-or-later
"""

import cv2
import numpy as np
from pathlib import Path


class VideoFrameReader:
    """Iterable video reader with random-access ``read_frame``.

    Parameters
    ----------
    path : str | Path
    """

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Video not found: {path}")
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")

        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration_sec = self.total_frames / self.fps if self.fps > 0 else 0

    def __len__(self) -> int:
        return self.total_frames

    def __iter__(self):
        """Yield (frame_idx, frame) from the beginning."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame_idx, frame
            frame_idx += 1

    def read_frame(self, frame_idx: int | None = None) -> tuple[bool, np.ndarray | None]:
        """Read a single frame, optionally seeking first.

        Returns (ret, frame) — ret=False if past end.
        """
        if frame_idx is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        return ret, frame if ret else None

    def get_info(self) -> dict:
        """Return video metadata dict (fps, width, height, …)."""
        return {
            "path": str(self.path),
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "total_frames": self.total_frames,
            "duration_sec": self.duration_sec,
        }

    def __str__(self) -> str:
        lines = [
            f"── Video Info ",
            f"  Path:      {self.path}",
            f"  FPS:       {self.fps}",
            f"  Size:      {self.width} x {self.height}",
            f"  Frames:    {self.total_frames:,}",
            f"  Duration:  {self.duration_sec:.1f}s",
        ]
        max_len = max(len(l) for l in lines)
        lines[0] = f"── Video Info " + "─" * (max_len - len(lines[0]))
        lines.append("─" * max_len)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"VideoFrameReader({self.path!r})"

    def release(self) -> None:
        if self.cap:
            self.cap.release()

    def __enter__(self) -> "VideoFrameReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.release()
        return False
