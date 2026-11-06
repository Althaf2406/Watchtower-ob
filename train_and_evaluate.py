"""End-to-End Watchtower Training, Uncertainty Quantification, Attribution & Carbon Accounting CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from src.attribution.captum_explainer import compute_spectral_feature_attributions
from src.carbon.gfw_carbon_accounting import (
    estimate_burn_scar_carbon_loss,
    track_pipeline_carbon_footprint,
)
from src.data.sen2fire_dataset import Sen2FireMultispectralDataset
from src.evaluation.calibration_metrics import (
    SmokeObscuredSegmentationAndCalibrationEvaluator,
)
from src.losses.hybrid_wildfire_loss import WildfireHybridLoss
from src.models.attention_unet_r50 import SSL4EOAttentionUNetResNet50
from src.models.compact_vit_s16 import SSL4EOCompactViTS16Seg
from src.uncertainty.mc_ensemble_uq import predict_with_uncertainty


def build_model(
    architecture: str,
    ssl_source: str,
    in_channels: int = 17,
    num_classes: int = 3,
    dropout_p: float = 0.15,
    patch_size: int = 256,
) -> nn.Module:
    """Instantiate either `SSL4EOAttentionUNetResNet50` or `SSL4EOCompactViTS16Seg`."""
    arch = architecture.lower().strip()
    if arch == "attention_unet_r50":
        return SSL4EOAttentionUNetResNet50(
            in_channels=in_channels,
            num_classes=num_classes,
            ssl_source=ssl_source,  # type: ignore[arg-type]
            dropout_p=dropout_p,
        )
    if arch == "compact_vit_s16":
        return SSL4EOCompactViTS16Seg(
            in_channels=in_channels,
            num_classes=num_classes,
            img_size=patch_size,
            ssl_source=ssl_source,  # type: ignore[arg-type]
            dropout_p=dropout_p,
        )
    raise ValueError(
        f"Unsupported architecture {architecture!r}. Expected 'attention_unet_r50' or 'compact_vit_s16'."
    )


def run_pipeline(
    config: Dict[str, Any],
    architecture: str,
    ssl_source: str,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Execute training, Bayesian UQ evaluation, Captum attribution, and GFW carbon accounting."""
    train_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    loss_cfg = config.get("loss", {})
    uq_cfg = config.get("uncertainty", {})
    eval_cfg = config.get("evaluation", {})

    seed = int(config.get("project", {}).get("seed", 42))
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = bool(train_cfg.get("fp16", True) and device.type == "cuda")

    batch_size = 2 if dry_run else int(train_cfg.get("batch_size", 16))
    patch_size = int(train_cfg.get("patch_size", 256))
    epochs = 1 if dry_run else int(train_cfg.get("epochs", 60))
    lr = float(train_cfg.get("lr", 3e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    dropout_p = float(uq_cfg.get("dropout_p", 0.15))
    mc_passes = 4 if dry_run else int(uq_cfg.get("mc_passes", 20))

    # 1. Datasets & DataLoaders (Zero Data Leakage: stats strictly from scene1 + scene2)
    train_dataset = Sen2FireMultispectralDataset(
        root_dir=data_cfg.get("sen2fire_root", "data/sen2fire"),
        scenes=data_cfg.get("train_scenes", ["scene1", "scene2"]),
        patch_size=patch_size,
        is_train=True,
        positive_oversample_prob=float(train_cfg.get("positive_oversample_prob", 0.70)),
        stats_json_path=data_cfg.get("stats_path", "configs/sen2fire_train_stats.json"),
        synthetic_fallback_samples=4 if dry_run else 16,
    )
    val_dataset = Sen2FireMultispectralDataset(
        root_dir=data_cfg.get("sen2fire_root", "data/sen2fire"),
        scenes=data_cfg.get("val_scenes", ["scene3"]),
        patch_size=patch_size,
        is_train=False,
        stats_json_path=data_cfg.get("stats_path", "configs/sen2fire_train_stats.json"),
        synthetic_fallback_samples=2 if dry_run else 8,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 2. Model, FP32 Hybrid Loss, Optimizer, AMP GradScaler
    model = build_model(
        architecture=architecture,
        ssl_source=ssl_source,
        in_channels=int(data_cfg.get("in_channels", 17)),
        num_classes=int(data_cfg.get("num_classes", 3)),
        dropout_p=dropout_p,
        patch_size=patch_size,
    ).to(device)

    criterion = WildfireHybridLoss(
        num_classes=int(data_cfg.get("num_classes", 3)),
        loss_weights=loss_cfg.get("weights", [0.40, 0.40, 0.20]),
        alpha_class=loss_cfg.get("alpha_class", [0.10, 0.35, 0.55]),
        focal_gamma=float(loss_cfg.get("focal_gamma", 2.0)),
        tversky_alpha_fp=float(loss_cfg.get("tversky_alpha_fp", 0.30)),
        tversky_beta_fn=float(loss_cfg.get("tversky_beta_fn", 0.70)),
        tversky_gamma=float(loss_cfg.get("tversky_gamma", 1.3333)),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    evaluator = SmokeObscuredSegmentationAndCalibrationEvaluator(
        num_classes=int(data_cfg.get("num_classes", 3)),
        ece_bins=int(eval_cfg.get("ece_bins", 15)),
        smoke_aerosol_threshold=float(eval_cfg.get("smoke_aerosol_threshold", 1.0)),
        boundary_kernel_size=int(eval_cfg.get("fire_boundary_kernel_size", 31)),
    )

    last_train_loss = 0.0
    with track_pipeline_carbon_footprint(
        project_suffix=f"{architecture}_{ssl_source}",
        enabled=not dry_run,
    ) as compute_emissions:
        for _epoch in range(epochs):
            model.train()
            running_loss = 0.0
            num_batches = 0

            for batch in train_loader:
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_fp16):
                    logits = model(images)

                # Loss strictly in FP32 outside FP16 autocast
                loss = criterion(logits.float(), masks)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=float(train_cfg.get("grad_clip_norm", 1.0)),
                )
                scaler.step(optimizer)
                scaler.update()

                running_loss += float(loss.detach().cpu().item())
                num_batches += 1
                if dry_run:
                    break

            last_train_loss = running_loss / max(1, num_batches)

        # 3. Validation with MC Dropout Isolation & Calibration Evaluation
        evaluator.reset()
        sample_uq_outputs: Dict[str, torch.Tensor] = {}
        sample_images: torch.Tensor = torch.empty(0)

        for val_batch in val_loader:
            val_images = val_batch["image"].to(device)
            val_masks = val_batch["mask"].to(device)
            val_aerosol = val_batch["raw_aerosol"].to(device)

            uq_out = predict_with_uncertainty(
                models=model,
                x=val_images,
                mode="mc_dropout",
                num_mc_passes=mc_passes,
                use_amp=use_fp16,
            )
            evaluator.update(
                probs=uq_out["mean_probs"],
                targets=val_masks,
                raw_aerosol=val_aerosol,
                uncertainty=uq_out["predictive_entropy"],
            )
            sample_uq_outputs = uq_out
            sample_images = val_images
            if dry_run:
                break

        eval_metrics = evaluator.compute()

        # 4. Spectral Attribution via Captum
        attr_out = compute_spectral_feature_attributions(
            model=model,
            x=sample_images[:1],
            target_class=2,
            mode="class_logit",
            ig_steps=4 if dry_run else 16,
        )

        # 5. Uncertainty-Propagated Carbon Accounting
        carbon_report = estimate_burn_scar_carbon_loss(
            mean_probs=sample_uq_outputs["mean_probs"][0].cpu().numpy(),
            epistemic_mi=sample_uq_outputs["epistemic_mi"][0].cpu().numpy(),
            pred_variance=sample_uq_outputs["pred_variance"][0].cpu().numpy(),
        )

    summary = {
        "architecture": architecture,
        "ssl_source": ssl_source,
        "device": str(device),
        "fp16_enabled": use_fp16,
        "final_train_loss": round(last_train_loss, 6),
        "evaluation": {
            "mIoU": round(float(eval_metrics["mIoU"]), 4),
            "macro_F1": round(float(eval_metrics["macro_F1"]), 4),
            "NLL": round(float(eval_metrics["NLL"]), 4),
            "Brier_Score": round(float(eval_metrics["Brier_Score"]), 4),
            "ECE": round(float(eval_metrics["ECE"]), 4),
            "AUSE": round(float(eval_metrics["AUSE"]), 4),
        },
        "attribution_shares_sum": round(float(attr_out["channel_importance_shares"].sum().item()), 6),
        "carbon_accounting": carbon_report,
        "pipeline_emissions_kg_co2eq": compute_emissions["emissions_kg_co2eq"],
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Watchtower: Uncertainty-Aware Wildfire Segmentation & Carbon Accounting Pipeline"
    )
    parser.add_argument(
        "--architecture",
        type=str,
        choices=["attention_unet_r50", "compact_vit_s16"],
        default="attention_unet_r50",
        help="Segmentation backbone architecture.",
    )
    parser.add_argument(
        "--ssl_source",
        type=str,
        choices=["ssl4eo_l", "ssl4eo_s12"],
        default="ssl4eo_s12",
        help="SSL4EO self-supervised pretraining sensor source.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_16gb_fp16.yaml",
        help="Path to YAML hyperparameter configuration file.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run a single-batch synthetic verification pass.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if config_path.is_file():
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    summary = run_pipeline(
        config=config,
        architecture=args.architecture,
        ssl_source=args.ssl_source,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

