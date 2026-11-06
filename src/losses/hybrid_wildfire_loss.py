"""FP32-Safe Hybrid Loss for Severe Class Imbalance & Boundary Refinement in Wildfire Segmentation."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Compute gradient of the Lovasz extension w.r.t sorted errors (in FP32)."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1.0 - gt_sorted.float()).cumsum(0)
    jaccard = 1.0 - intersection / union.clamp_min(1e-12)
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


class WildfireHybridLoss(nn.Module):
    """Hybrid Wildfire Segmentation Loss operating strictly in FP32 for numerical stability under FP16 AMP.

    Combines three complementary objectives:
      1. Multi-Class Focal Loss (`gamma = 2.0`, `alpha_class = [0.10, 0.35, 0.55]`):
         Penalizes hard-to-classify smoke-obscured and sub-canopy fire pixels.
      2. Multi-Class Focal Tversky Loss (`alpha_FP = 0.30`, `beta_FN = 0.70`, `gamma_Tversky = 1.3333`):
         Heavily penalizes false negatives on thin active-fire flame fronts.
      3. Multi-Class Lovasz-Softmax Loss (`L_Lovasz`):
         Directly optimizes the discrete Jaccard (mIoU) submodular surrogate.

    Total loss:
      `L_total = w_0 * L_Focal + w_1 * L_FocalTversky + w_2 * L_Lovasz`
      with default weights `[0.40, 0.40, 0.20]`.
    """

    def __init__(
        self,
        num_classes: int = 3,
        loss_weights: Sequence[float] = (0.40, 0.40, 0.20),
        alpha_class: Sequence[float] = (0.10, 0.35, 0.55),
        focal_gamma: float = 2.0,
        tversky_alpha_fp: float = 0.30,
        tversky_beta_fn: float = 0.70,
        tversky_gamma: float = 1.3333,
        smooth: float = 1e-6,
        ignore_index: Optional[int] = None,
    ) -> None:
        super().__init__()
        if len(loss_weights) != 3:
            raise ValueError(f"Expected 3 loss weights [Focal, FocalTversky, Lovasz], got {loss_weights}")
        if len(alpha_class) != num_classes:
            raise ValueError(f"Expected {num_classes} class weights in `alpha_class`, got {alpha_class}")

        self.num_classes = int(num_classes)
        self.loss_weights: Tuple[float, float, float] = (
            float(loss_weights[0]),
            float(loss_weights[1]),
            float(loss_weights[2]),
        )
        self.focal_gamma = float(focal_gamma)
        self.tversky_alpha_fp = float(tversky_alpha_fp)
        self.tversky_beta_fn = float(tversky_beta_fn)
        self.tversky_gamma = float(tversky_gamma)
        self.smooth = float(smooth)
        self.ignore_index = ignore_index

        alpha_tensor = torch.tensor(alpha_class, dtype=torch.float32)
        self.register_buffer("alpha_class", alpha_tensor)
        self.last_components: Dict[str, float] = {}

    def _compute_multiclass_focal_loss(
        self,
        logits_fp32: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute class-balanced multi-class Focal Loss in FP32."""
        log_probs = F.log_softmax(logits_fp32, dim=1)  # (B, C, H, W)
        probs = torch.exp(log_probs)

        targets_clamped = targets.clamp(0, self.num_classes - 1).long()
        target_unsqueezed = targets_clamped.unsqueeze(1)  # (B, 1, H, W)

        log_pt = log_probs.gather(1, target_unsqueezed).squeeze(1)  # (B, H, W)
        pt = probs.gather(1, target_unsqueezed).squeeze(1).clamp(1e-7, 1.0 - 1e-7)

        alpha_t = self.alpha_class.to(logits_fp32.device)[targets_clamped]  # (B, H, W)
        focal_term = torch.pow(1.0 - pt, self.focal_gamma)
        loss_map = -alpha_t * focal_term * log_pt

        if valid_mask is not None:
            loss_map = loss_map * valid_mask.float()
            return loss_map.sum() / valid_mask.float().sum().clamp_min(1.0)
        return loss_map.mean()

    def _compute_focal_tversky_loss(
        self,
        probs_fp32: torch.Tensor,
        targets_one_hot: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute multi-class Focal Tversky Loss in FP32 (`alpha_FP=0.30, beta_FN=0.70, gamma=1.3333`)."""
        if valid_mask is not None:
            vm = valid_mask.unsqueeze(1).float()
            probs_fp32 = probs_fp32 * vm
            targets_one_hot = targets_one_hot * vm

        dims = (0, 2, 3)
        tp = torch.sum(probs_fp32 * targets_one_hot, dim=dims)
        fp = torch.sum(probs_fp32 * (1.0 - targets_one_hot), dim=dims)
        fn = torch.sum((1.0 - probs_fp32) * targets_one_hot, dim=dims)

        tversky_index = (tp + self.smooth) / (
            tp + self.tversky_alpha_fp * fp + self.tversky_beta_fn * fn + self.smooth
        )
        focal_tversky_per_class = torch.pow(
            (1.0 - tversky_index).clamp_min(1e-7),
            self.tversky_gamma,
        )

        # Weight class contributions by normalized alpha_class
        alpha_norm = self.alpha_class.to(probs_fp32.device) / self.alpha_class.sum().clamp_min(1e-7)
        return torch.sum(alpha_norm * focal_tversky_per_class)

    def _compute_lovasz_softmax_loss(
        self,
        probs_fp32: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute multi-class Lovasz-Softmax loss in FP32."""
        b, c, h, w = probs_fp32.shape
        probs_flat = probs_fp32.permute(0, 2, 3, 1).reshape(-1, c)  # (N, C)
        targets_flat = targets.reshape(-1)  # (N,)

        if valid_mask is not None:
            vm_flat = valid_mask.reshape(-1)
            probs_flat = probs_flat[vm_flat]
            targets_flat = targets_flat[vm_flat]

        if probs_flat.numel() == 0:
            return probs_fp32.sum() * 0.0

        losses = []
        for class_idx in range(c):
            fg = (targets_flat == class_idx).float()
            if fg.sum() == 0 and c > 1:
                # Still include smooth surrogate if no class is present in mini-batch
                pass
            class_pred = probs_flat[:, class_idx]
            errors = (fg - class_pred).abs()
            errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
            fg_sorted = fg[perm]
            grad = _lovasz_grad(fg_sorted)
            losses.append(torch.dot(errors_sorted, grad))

        return torch.stack(losses).mean()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        return_components: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Compute FP32 hybrid segmentation loss from raw logits `(B, C, H, W)` and targets `(B, H, W)`."""
        device_type = logits.device.type if logits.device.type in ("cuda", "cpu") else "cpu"
        with torch.amp.autocast(device_type=device_type, enabled=False):
            logits_fp32 = logits.float()
            targets_long = targets.long()

            valid_mask: Optional[torch.Tensor] = None
            if self.ignore_index is not None:
                valid_mask = targets_long != self.ignore_index

            probs_fp32 = F.softmax(logits_fp32, dim=1)
            targets_clamped = targets_long.clamp(0, self.num_classes - 1)
            targets_one_hot = (
                F.one_hot(targets_clamped, num_classes=self.num_classes)
                .permute(0, 3, 1, 2)
                .float()
            )

            l_focal = self._compute_multiclass_focal_loss(logits_fp32, targets_clamped, valid_mask)
            l_tversky = self._compute_focal_tversky_loss(probs_fp32, targets_one_hot, valid_mask)
            l_lovasz = self._compute_lovasz_softmax_loss(probs_fp32, targets_clamped, valid_mask)

            w0, w1, w2 = self.loss_weights
            total_loss = w0 * l_focal + w1 * l_tversky + w2 * l_lovasz

            self.last_components = {
                "loss_total": float(total_loss.detach().cpu().item()),
                "loss_focal": float(l_focal.detach().cpu().item()),
                "loss_focal_tversky": float(l_tversky.detach().cpu().item()),
                "loss_lovasz": float(l_lovasz.detach().cpu().item()),
            }

            if return_components:
                return total_loss, {
                    "loss_total": total_loss,
                    "loss_focal": l_focal,
                    "loss_focal_tversky": l_tversky,
                    "loss_lovasz": l_lovasz,
                }
            return total_loss

