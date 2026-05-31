"""SimpleSort — multi-object tracking with YOLO + Kalman + ReID.

>>> from SimpleSort import Detector, ReID, Matcher, Tracker, VideoFrameReader
>>> tracker = Tracker(Detector("yolo26x.pt"), ReID("msmt_sbs_R101-ibn.pth"), Matcher())
>>> for frame_idx, frame in VideoFrameReader("video.mp4"):
...     tracker.update(frame)
...     annotated = tracker.visual(frame)

Copyright (C) 2026 Nivendel, College of Civil Engineering, Tongji University
With assistance from Claude Code and deepseek-v4-pro[1m]
SPDX-License-Identifier: AGPL-3.0-or-later
"""

from .detector import Detection, Detector
from .reid import ReID
from .matcher import Matcher
from .tracker import Tracker
from .track import Track, CONFIRMED, TENTATIVE, LOST, DELETED, RECOVERING
from .video_reader import VideoFrameReader
from .kalmanfilter import KalmanFilter
from .kema import KEMA
