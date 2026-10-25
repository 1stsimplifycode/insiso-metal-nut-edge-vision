"""Minimal PatchCore (Roth et al., CVPR 2022): normal-only anomaly detection from pretrained features."""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class PatchCore:
    def __init__(self, device: str = "cuda", size: int = 256, coreset_ratio: float = 0.1, seed: int = 0):
        import torchvision
        self.device = device
        self.size = size
        self.ratio = coreset_ratio
        self.seed = seed
        net = torchvision.models.wide_resnet50_2(weights="IMAGENET1K_V1").eval().to(device)
        self.net = net
        self.bank: torch.Tensor | None = None

    def _tensor(self, imgs_bgr: list[np.ndarray]) -> torch.Tensor:
        xs = []
        for im in imgs_bgr:
            im = cv2.resize(im, (self.size, self.size), interpolation=cv2.INTER_AREA)[:, :, ::-1].astype(np.float32) / 255
            xs.append(((im - MEAN) / STD).transpose(2, 0, 1))
        return torch.from_numpy(np.stack(xs)).to(self.device)

    @torch.no_grad()
    def features(self, imgs_bgr: list[np.ndarray]) -> torch.Tensor:
        """(B, H*W, C) locally-aware patch features from layer2 + upsampled layer3."""
        x = self._tensor(imgs_bgr)
        n = self.net
        x = n.maxpool(n.relu(n.bn1(n.conv1(x))))
        l2 = n.layer2(n.layer1(x))
        l3 = n.layer3(l2)
        l2 = F.avg_pool2d(l2, 3, 1, 1)
        l3 = F.interpolate(F.avg_pool2d(l3, 3, 1, 1), size=l2.shape[-2:], mode="bilinear", align_corners=False)
        f = torch.cat([l2, l3], 1)  # (B, 1536, h, w)
        self.fmap_hw = f.shape[-2:]
        return f.flatten(2).transpose(1, 2)

    @torch.no_grad()
    def fit(self, imgs_bgr: list[np.ndarray], batch: int = 8) -> None:
        feats = torch.cat([self.features(imgs_bgr[i:i + batch]).reshape(-1, 1536)
                           for i in range(0, len(imgs_bgr), batch)])
        self.bank = self._coreset(feats)

    @torch.no_grad()
    def _coreset(self, feats: torch.Tensor) -> torch.Tensor:
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        n_keep = max(1, int(len(feats) * self.ratio))
        proj = torch.randn(feats.shape[1], 128, generator=g).to(feats.device)
        z = feats @ proj
        idx = [int(torch.randint(len(z), (1,), generator=g))]
        mind = torch.cdist(z, z[idx[-1]][None]).squeeze(1)
        for _ in range(n_keep - 1):
            i = int(mind.argmax())
            idx.append(i)
            mind = torch.minimum(mind, torch.cdist(z, z[i][None]).squeeze(1))
        return feats[idx]

    @torch.no_grad()
    def predict(self, img_bgr: np.ndarray) -> tuple[float, np.ndarray]:
        """Image score (max patch NN distance) and anomaly map at source resolution."""
        f = self.features([img_bgr])[0]
        d = torch.cdist(f, self.bank).min(1).values
        h, w = self.fmap_hw
        amap = d.reshape(1, 1, h, w)
        amap = F.interpolate(amap, size=img_bgr.shape[:2], mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
        amap = cv2.GaussianBlur(amap, (0, 0), 4)
        return float(d.max()), amap
