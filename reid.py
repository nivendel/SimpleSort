"""ReID feature extractor — FastReID SBS (IBN-ResNet + GeM + BNNeck).

Input:  frame (H, W, 3) BGR,  tlwhs (N, 4) or list[np.ndarray]
Output: feats (N, 2048)  L2-normalised

Copyright (C) 2026 Nivendel
With assistance from Claude Code and deepseek-v4-pro[1m]
SPDX-License-Identifier: AGPL-3.0-or-later
"""

import sys
from pathlib import Path
from tqdm import tqdm
import urllib.request
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent / "fast-reid"))
from fastreid.config import get_cfg
from fastreid.modeling.backbones import build_resnet_backbone
from fastreid.layers.pooling import GeneralizedMeanPoolingP

MODEL_ZOO = [
    "duke_agw_R101-ibn.pth",
    "duke_agw_R50-ibn.pth",
    "duke_agw_R50.pth",
    "duke_agw_S50.pth",
    "duke_bot_R101-ibn.pth",
    "duke_bot_R50-ibn.pth",
    "duke_bot_R50.pth",
    "duke_bot_S50.pth",
    "duke_mgn_R50-ibn.pth",
    "duke_sbs_R101-ibn.pth",
    "duke_sbs_R50-ibn.pth",
    "duke_sbs_R50.pth",
    "duke_sbs_S50.pth",
    "market_agw_R101-ibn.pth",
    "market_agw_R50-ibn.pth",
    "market_agw_R50.pth",
    "market_agw_S50.pth",
    "market_bot_R101-ibn.pth",
    "market_bot_R50-ibn.pth",
    "market_bot_R50.pth",
    "market_bot_S50.pth",
    "market_mgn_R50-ibn.pth",
    "market_sbs_R101-ibn.pth",
    "market_sbs_R50-ibn.pth",
    "market_sbs_R50.pth",
    "market_sbs_S50.pth",
    "msmt_agw_R101-ibn.pth",
    "msmt_agw_R50-ibn.pth",
    "msmt_agw_R50.pth",
    "msmt_agw_S50.pth",
    "msmt_bot_R101-ibn.pth",
    "msmt_bot_R50-ibn.pth",
    "msmt_bot_R50.pth",
    "msmt_bot_S50.pth",
    "msmt_sbs_R101-ibn.pth",
    "msmt_sbs_R50-ibn.pth",
    "msmt_sbs_R50.pth",
    "msmt_sbs_S50.pth",
    "vehicleid_bot_R50-ibn.pth",
    "veri_sbs_R50-ibn.pth",
    "veriwild_bot_R50-ibn.pth",
]


class _TqdmHook:
    """Progress bar hook for urllib.request.urlretrieve."""

    def __init__(self, total: int | None = None) -> None:
        self.pbar = None
        self.total = total

    def __call__(self, block_num: int, block_size: int, total_size: int) -> None:
        if self.pbar is None:
            if total_size > 0:
                self.total = total_size
            self.pbar = tqdm(total=self.total, unit="B", unit_scale=True)
        self.pbar.update(block_size)
        if block_num * block_size >= self.total:
            self.pbar.close()


class ReID(nn.Module):
    """FastReID SBS feature extractor.  ``self.dim`` = 2048.

    Parameters
    ----------
    weights : str = "msmt_sbs_R50-ibn.pth"
        Model name (from MODEL_ZOO) or path to a ``.pth`` file.
    """

    def __init__(self, weights: str = "msmt_sbs_R50-ibn.pth"):
        super().__init__()

        self.dim = 2048

        path = self._resolve(weights)
        name = Path(path).stem
        cfg = get_cfg()
        cfg.MODEL.BACKBONE.DEPTH = "101x" if "R101" in name else "50x"
        cfg.MODEL.BACKBONE.WITH_IBN = "ibn" in name
        cfg.MODEL.BACKBONE.LAST_STRIDE = 1
        cfg.MODEL.BACKBONE.NORM = "BN"
        cfg.MODEL.BACKBONE.PRETRAIN = True

        self.backbone = build_resnet_backbone(cfg)
        self.pool = GeneralizedMeanPoolingP()
        self.bottleneck = nn.BatchNorm2d(self.dim)
        nn.init.constant_(self.bottleneck.weight, 1)
        nn.init.constant_(self.bottleneck.bias, 0)
        self.bottleneck.bias.requires_grad_(False)

        state = torch.load(path, map_location="cpu", weights_only=True)
        self.load_state_dict(state["model"], strict=False)

        self.eval()
        self.cuda() if torch.cuda.is_available() else self.cpu()
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((256, 128)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    @staticmethod
    def _resolve(weights: str) -> str:
        """Resolve *weights* to a local path, downloading if needed."""
        p = Path(weights)
        if p.exists():
            return str(p)
        if p.name not in MODEL_ZOO:
            raise KeyError(f"Unknown model '{p.name}'. Choose from: {MODEL_ZOO}")
        url = f"https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/{p.name}"
        local = Path(__file__).parent / Path(url).name
        if not local.exists():
            local.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading ...\r\nFrom: {url}\r\nTo: {local}")
            urllib.request.urlretrieve(url, local, reporthook=_TqdmHook())
        return str(local)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Backbone → GeM pool → BNNeck → flatten to (B, dim)."""
        x = self.backbone(x)
        x = self.pool(x)
        x = self.bottleneck(x)
        return x[..., 0, 0]

    def __call__(self, frame: np.ndarray, tlwhs: np.ndarray | list) -> np.ndarray:
        """Extract L2-normalised features for each detection box.

        frame : (H, W, 3) BGR
        tlwhs : (N, 4) or list of [x, y, w, h]
        →  (N, 2048)  L2-normalised
        """
        crops = []
        for tlwh in tlwhs:
            x, y, w, h = map(int, tlwh)
            crop = frame[y : y + h, x : x + w]
            if crop.size == 0:
                crop = np.zeros((256, 128, 3), dtype=np.uint8)
            crops.append(self.transform(crop))

        if not crops:
            return np.empty((0, self.dim), dtype=np.float32)

        tensor = torch.stack(crops)
        if torch.cuda.is_available():
            tensor = tensor.cuda()
        with torch.no_grad():
            feats = self.forward(tensor).cpu().numpy()

        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        feats /= norms
        return feats

    def visual(
        self, frame: np.ndarray, tlwhs: np.ndarray, feats: np.ndarray
    ) -> np.ndarray:
        """Tiled crops fed to the ReID model (RGB).  *feats* is ignored."""
        tiles = []
        for tlwh in tlwhs:
            x, y, w, h = map(int, tlwh)
            crop = frame[y : y + h, x : x + w]
            if crop.size == 0:
                crop = np.zeros((256, 128, 3), dtype=np.uint8)
            crop = cv2.resize(crop, (128, 256))
            tiles.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

        if not tiles:
            return np.zeros((256, 128, 3), dtype=np.uint8)

        return np.hstack(tiles)
