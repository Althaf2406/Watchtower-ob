"""Smoke-Conditioned Segmentation & Calibration Evaluator (mIoU, F1, Area Ratio, NLL, Brier, ECE, AUSE)."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F


class SmokeObscuredSegmentationAndCalibrationEvaluator:
    """Evaluator for 3-class wildfire segmentation and probabilistic calibration under smoke obscuration.

    Computes:
      - Per-class `IoU`, `F1`, `Precision`, `Recall`, `Area_Ratio`
      - Macro metrics: `mIoU`, `macro_F1`
      - Probabilistic calibration metrics (strictly in FP32) conditioned on the smoke-FCER mask
        (`raw_aerosol > 1.0` unioned with a `31x31` max-pool fire boundary buffer):
          * `NLL` (Negative Log-Likelihood)
          * `Brier_Score` (Multi-class Brier Score)
          * `ECE` (Expected Calibration Error across `num_bins` confidence bins)
          * `AUSE` (Area Under the Sparsification Error Curve relative to the Oracle curve)
    """

    def __init__(
        self,
        num_classes: int = 3,
        class_names: Sequence[str] = ("Background_Smoke", "Burn_Scar", "Active_Flame_Front"),
        ece_bins: int = 15,
        smoke_aerosol_threshold: float = 1.0,
        boundary_kernel_size: int = 31,
        sparsification_steps: int = 100,
        eps: float = 1e-7,
    ) -> None:
        self.num_classes = int(num_classes)
        self.class_names = list(class_names)
        self.ece_bins = int(ece_bins)
        self.smoke_aerosol_threshold = float(smoke_aerosol_threshold)
        self.boundary_kernel_size = int(boundary_kernel_size)
        self.sparsification_steps = int(sparsification_steps)
        self.eps = float(eps)
        self.reset()

    def reset(self) -> None:
        """Reset accumulated confusion matrix and smoke-FCER calibration buffers."""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self._probs_list: List[np.ndarray] = []
        self._targets_list: List[np.ndarray] = []
        self._uncertainties_list: List[np.ndarray] = []

    def build_smoke_fcer_mask(
        self,
        targets: torch.Tensor,
        raw_aerosol: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Construct the smoke-FCER mask: `(raw_aerosol > 1.0)` unioned with a `31x31` max-pool fire boundary buffer.

        Args:
            targets: Ground-truth class tensor `(B, H, W)`.
            raw_aerosol: Optional unstandardized S5P UV Aerosol Index tensor `(B, H, W)` or `(B, 1, H, W)`.

        Returns:
            Boolean mask tensor `(B, H, W)`.
        """
        fire_binary = (targets > 0).float().unsqueeze(1)  # (B, 1, H, W)
        pad = self.boundary_kernel_size // 2
        dilated_fire = F.max_pool2d(
            fire_binary,
            kernel_size=self.boundary_kernel_size,
            stride=1,
            padding=pad,
        )
        # Ensure exact spatial shape match if kernel_size is even
        dilated_fire = dilated_fire[:, :, : targets.shape[-2], : targets.shape[-1]]
        fire_buffer_mask = dilated_fire.squeeze(1) > 0.5  # (B, H, W)

        if raw_aerosol is not None:
            aero = raw_aerosol.to(device=targets.device, dtype=torch.float32)
            if aero.ndim == 4:
                aero = aero.squeeze(1)
            smoke_mask = aero > self.smoke_aerosol_threshold
            fcer_mask = smoke_mask | fire_buffer_mask
        else:
            fcer_mask = fire_buffer_mask

        # If neither smoke nor fire pixels exist in the sample, evaluate over the full spatial grid
        if not bool(fcer_mask.any().item()):
            fcer_mask = torch.ones_like(targets, dtype=torch.bool)

        return fcer_mask

    @torch.no_grad()
    def update(
        self,
        probs: torch.Tensor,
        targets: torch.Tensor,
        raw_aerosol: Optional[torch.Tensor] = None,
        uncertainty: Optional[torch.Tensor] = None,
    ) -> None:
        """Accumulate a batch of predictions `(B, C, H, W)` and ground-truth labels `(B, H, W)`."""
        probs_fp32 = probs.detach().float()
        # Normalize if logits or unnormalized probabilities were passed
        if (probs_fp32.min() < 0.0) or not torch.allclose(
            probs_fp32.sum(dim=1), torch.ones_like(probs_fp32[:, 0]), atol=1e-2
        ):
            probs_fp32 = F.softmax(probs_fp32, dim=1)

        targets_long = targets.detach().long().clamp(0, self.num_classes - 1)
        preds_long = torch.argmax(probs_fp32, dim=1)

        # Update confusion matrix over all pixels
        y_true_flat = targets_long.cpu().numpy().reshape(-1)
        y_pred_flat = preds_long.cpu().numpy().reshape(-1)
        idx = y_true_flat * self.num_classes + y_pred_flat
        bincount = np.bincount(idx, minlength=self.num_classes * self.num_classes)
        self.confusion_matrix += bincount.reshape(self.num_classes, self.num_classes)

        # Build smoke-FCER mask for calibration evaluation
        fcer_mask = self.build_smoke_fcer_mask(targets_long, raw_aerosol=raw_aerosol)
        fcer_flat = fcer_mask.cpu().numpy().reshape(-1)

        probs_np = probs_fp32.permute(0, 2, 3, 1).cpu().numpy().reshape(-1, self.num_classes)
        if uncertainty is not None:
            unc_np = uncertainty.detach().float().cpu().numpy().reshape(-1)
        else:
            probs_safe = np.clip(probs_np, self.eps, 1.0)
            unc_np = -np.sum(probs_np * np.log(probs_safe), axis=1)

        probs_sel = probs_np[fcer_flat]
        targets_sel = y_true_flat[fcer_flat]
        unc_sel = unc_np[fcer_flat]

        # Subsample if very large to keep memory bounded during 60-epoch evaluation
        max_pixels_per_batch = 65536
        if probs_sel.shape[0] > max_pixels_per_batch:
            rng = np.random.RandomState(42)
            sel_idx = rng.choice(probs_sel.shape[0], size=max_pixels_per_batch, replace=False)
            probs_sel = probs_sel[sel_idx]
            targets_sel = targets_sel[sel_idx]
            unc_sel = unc_sel[sel_idx]

        self._probs_list.append(probs_sel.astype(np.float32))
        self._targets_list.append(targets_sel.astype(np.int64))
        self._uncertainties_list.append(unc_sel.astype(np.float32))

    def _compute_calibration_metrics(
        self,
        probs: np.ndarray,
        targets: np.ndarray,
        uncertainties: np.ndarray,
    ) -> Dict[str, float]:
        """Compute NLL, Brier_Score, ECE, and AUSE (Oracle Sparsification Error) in float32."""
        n = probs.shape[0]
        if n == 0:
            return {"NLL": 0.0, "Brier_Score": 0.0, "ECE": 0.0, "AUSE": 0.0}

        probs_f32 = probs.astype(np.float32)
        probs_safe = np.clip(probs_f32, self.eps, 1.0)

        # 1. Negative Log-Likelihood (NLL)
        true_class_probs = probs_safe[np.arange(n), targets]
        nll = float(-np.mean(np.log(true_class_probs)))

        # 2. Multi-class Brier Score
        one_hot = np.eye(self.num_classes, dtype=np.float32)[targets]
        brier = float(np.mean(np.sum(np.square(probs_f32 - one_hot), axis=1)))

        # 3. Expected Calibration Error (ECE)
        confidences = np.max(probs_f32, axis=1)
        predictions = np.argmax(probs_f32, axis=1)
        accuracies = (predictions == targets).astype(np.float32)

        bin_boundaries = np.linspace(0.0, 1.0, self.ece_bins + 1, dtype=np.float32)
        ece = 0.0
        for b_idx in range(self.ece_bins):
            low, high = bin_boundaries[b_idx], bin_boundaries[b_idx + 1]
            if b_idx == 0:
                in_bin = (confidences >= low) & (confidences <= high)
            else:
                in_bin = (confidences > low) & (confidences <= high)
            prop_in_bin = float(np.mean(in_bin))
            if prop_in_bin > 0.0:
                acc_in_bin = float(np.mean(accuracies[in_bin]))
                conf_in_bin = float(np.mean(confidences[in_bin]))
                ece += abs(acc_in_bin - conf_in_bin) * prop_in_bin

        # 4. Area Under the Sparsification Error Curve (AUSE) via Oracle Sparsification
        errors = 1.0 - accuracies  # Per-pixel 0/1 classification error (or Brier residual)
        # Sort descending by uncertainty (most uncertain removed first)
        model_order = np.argsort(-uncertainties, kind="mergesort")
        # Oracle sorts descending by true error (highest error removed first)
        oracle_order = np.argsort(-errors, kind="mergesort")

        errors_model_sorted = errors[model_order]
        errors_oracle_sorted = errors[oracle_order]

        fractions = np.linspace(0.0, 0.99, self.sparsification_steps, dtype=np.float32)
        model_curve = np.zeros_like(fractions)
        oracle_curve = np.zeros_like(fractions)

        for idx, frac in enumerate(fractions):
            start_idx = min(int(round(frac * n)), n - 1)
            model_curve[idx] = float(np.mean(errors_model_sorted[start_idx:]))
            oracle_curve[idx] = float(np.mean(errors_oracle_sorted[start_idx:]))

        base_err = max(float(model_curve[0]), self.eps)
        model_curve_norm = model_curve / base_err
        oracle_curve_norm = oracle_curve / base_err

        ause = float(np.trapezoid(np.abs(model_curve_norm - oracle_curve_norm), fractions))

        return {
            "NLL": nll,
            "Brier_Score": brier,
            "ECE": float(ece),
            "AUSE": ause,
        }

    def compute(self) -> Dict[str, object]:
        """Compute all segmentation and smoke-conditioned calibration metrics."""
        cm = self.confusion_matrix.astype(np.float64)
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        gt_area = cm.sum(axis=1)
        pred_area = cm.sum(axis=0)

        per_class_metrics: Dict[str, Dict[str, float]] = {}
        ious: List[float] = []
        f1s: List[float] = []

        for c_idx in range(self.num_classes):
            c_name = self.class_names[c_idx] if c_idx < len(self.class_names) else f"Class_{c_idx}"
            iou = float(tp[c_idx] / max(tp[c_idx] + fp[c_idx] + fn[c_idx], self.eps))
            precision = float(tp[c_idx] / max(tp[c_idx] + fp[c_idx], self.eps))
            recall = float(tp[c_idx] / max(tp[c_idx] + fn[c_idx], self.eps))
            f1 = float((2.0 * tp[c_idx]) / max(2.0 * tp[c_idx] + fp[c_idx] + fn[c_idx], self.eps))
            area_ratio = float(pred_area[c_idx] / max(gt_area[c_idx], 1.0))

            ious.append(iou)
            f1s.append(f1)
            per_class_metrics[c_name] = {
                "IoU": iou,
                "F1": f1,
                "Precision": precision,
                "Recall": recall,
                "Area_Ratio": area_ratio,
            }

        miou = float(np.mean(ious))
        macro_f1 = float(np.mean(f1s))

        if len(self._probs_list) > 0:
            probs_all = np.concatenate(self._probs_list, axis=0)
            targets_all = np.concatenate(self._targets_list, axis=0)
            unc_all = np.concatenate(self._uncertainties_list, axis=0)
            calib = self._compute_calibration_metrics(probs_all, targets_all, unc_all)
        else:
            calib = {"NLL": 0.0, "Brier_Score": 0.0, "ECE": 0.0, "AUSE": 0.0}

        return {
            "mIoU": miou,
            "macro_F1": macro_f1,
            "NLL": calib["NLL"],
            "Brier_Score": calib["Brier_Score"],
            "ECE": calib["ECE"],
            "AUSE": calib["AUSE"],
            "per_class": per_class_metrics,
        }

