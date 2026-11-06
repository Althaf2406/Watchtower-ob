"""Isolated MC Dropout & Deep Ensemble Epistemic/Aleatoric Uncertainty Decomposition (BALD)."""

from __future__ import annotations

from typing import Dict, Literal, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

UQMode = Literal["mc_dropout", "deep_ensemble", "hybrid_ensemble_mc"]


def enable_mc_dropout_only(module: nn.Module) -> nn.Module:
    """Put `module` in `.eval()` mode and selectively enable ONLY Dropout layers (`Dropout`, `Dropout2d`, `Dropout3d`).

    CRITICAL HARDWARE & STATISTICAL INVARIANT:
    Never call `model.train()` globally during stochastic MC inference, as doing so corrupts
    `BatchNorm2d` running mean/variance statistics and degrades calibration.
    """
    module.eval()
    for m in module.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train(True)
    return module


@torch.no_grad()
def predict_with_uncertainty(
    models: Union[nn.Module, Sequence[nn.Module]],
    x: torch.Tensor,
    mode: UQMode = "hybrid_ensemble_mc",
    num_mc_passes: int = 20,
    use_amp: bool = True,
    eps: float = 1e-7,
) -> Dict[str, torch.Tensor]:
    """Perform Bayesian / Ensemble predictive inference with FP32 entropy & BALD mutual information decomposition.

    Modes:
      - `"mc_dropout"`: Single model (or first model in `models`) with `enable_mc_dropout_only` over `num_mc_passes`.
      - `"deep_ensemble"`: Deterministic `.eval()` forward pass across `M = len(models)` ensemble members.
      - `"hybrid_ensemble_mc"`: `M` ensemble members each executing `num_mc_passes` stochastic passes (`S = M * T`).

    Returns a dictionary of FP32/Int64 tensors:
      - `mean_probs`:         `(B, C, H, W)` float32 mean posterior probability
      - `pred_mask`:          `(B, H, W)` int64 MAP class prediction
      - `pred_variance`:      `(B, C, H, W)` float32 sample variance across stochastic passes
      - `predictive_entropy`: `(B, H, W)` float32 total Shannon entropy `H[E[p]]`
      - `aleatoric_entropy`:  `(B, H, W)` float32 expected Shannon entropy `E[H[p]]`
      - `epistemic_mi`:       `(B, H, W)` float32 BALD Mutual Information `H[E[p]] - E[H[p]] >= 0`
    """
    if isinstance(models, nn.Module):
        model_list = [models]
    else:
        model_list = list(models)
        if len(model_list) == 0:
            raise ValueError("`models` sequence cannot be empty.")

    if mode not in ("mc_dropout", "deep_ensemble", "hybrid_ensemble_mc"):
        raise ValueError(
            f"Unsupported uncertainty mode {mode!r}. Expected 'mc_dropout', 'deep_ensemble', or 'hybrid_ensemble_mc'."
        )

    device = x.device
    use_cuda_amp = bool(use_amp and device.type == "cuda" and torch.cuda.is_available())

    prob_samples = []

    if mode == "mc_dropout":
        active_model = model_list[0].to(device)
        enable_mc_dropout_only(active_model)
        passes = max(1, int(num_mc_passes))
        for _ in range(passes):
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_cuda_amp):
                logits = active_model(x)
            probs_fp32 = F.softmax(logits.float(), dim=1)
            prob_samples.append(probs_fp32)
        active_model.eval()

    elif mode == "deep_ensemble":
        for m in model_list:
            m = m.to(device)
            m.eval()
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_cuda_amp):
                logits = m(x)
            probs_fp32 = F.softmax(logits.float(), dim=1)
            prob_samples.append(probs_fp32)

    elif mode == "hybrid_ensemble_mc":
        passes = max(1, int(num_mc_passes))
        for m in model_list:
            m = m.to(device)
            enable_mc_dropout_only(m)
            for _ in range(passes):
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_cuda_amp):
                    logits = m(x)
                probs_fp32 = F.softmax(logits.float(), dim=1)
                prob_samples.append(probs_fp32)
            m.eval()

    # Stack samples: (S, B, C, H, W) strictly in FP32
    stacked_probs = torch.stack(prob_samples, dim=0).float()
    num_samples = stacked_probs.shape[0]

    mean_probs = torch.mean(stacked_probs, dim=0)  # (B, C, H, W)
    if num_samples > 1:
        pred_variance = torch.var(stacked_probs, dim=0, unbiased=False)
    else:
        pred_variance = torch.zeros_like(mean_probs)

    pred_mask = torch.argmax(mean_probs, dim=1).to(dtype=torch.int64)  # (B, H, W)

    # All entropy calculations explicitly in float32 to prevent FP16 log underflow
    mean_probs_safe = mean_probs.clamp(min=eps, max=1.0)
    predictive_entropy = -torch.sum(mean_probs * torch.log(mean_probs_safe), dim=1)  # (B, H, W)

    stacked_probs_safe = stacked_probs.clamp(min=eps, max=1.0)
    per_pass_entropy = -torch.sum(stacked_probs * torch.log(stacked_probs_safe), dim=2)  # (S, B, H, W)
    aleatoric_entropy = torch.mean(per_pass_entropy, dim=0)  # (B, H, W)

    # BALD Mutual Information I[y; theta | x] = H[E[p]] - E[H[p]] (clamped >= 0 for numerical precision)
    epistemic_mi = torch.clamp(predictive_entropy - aleatoric_entropy, min=0.0)

    return {
        "mean_probs": mean_probs,
        "pred_mask": pred_mask,
        "pred_variance": pred_variance,
        "predictive_entropy": predictive_entropy,
        "aleatoric_entropy": aleatoric_entropy,
        "epistemic_mi": epistemic_mi,
    }

