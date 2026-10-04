#!/usr/bin/env python3
"""Frozen, quantitative explanation-fidelity evaluation for FedEvi-DG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from fedevi_dg.dataset import ManifestImageDataset
from fedevi_dg.preprocessing import SourceNormalization
from train_fedevi import build_pretrained_model


def _rebuild(run_dir: Path, device: torch.device):
    resolved = json.loads((run_dir / "resolved_config.json").read_text())
    if not resolved.get("lesion_evidence_head", False):
        raise ValueError("run is not a FedEvi-DG evidence model")
    stats = json.loads((run_dir / "normalization.json").read_text())
    model = build_pretrained_model(
        SourceNormalization(mean=stats["mean"], std=stats["std"]),
        with_evidence_heads=True,
        use_evidence_bottleneck=bool(
            resolved.get("diagnostic_evidence_bottleneck", True)
        ),
        monotonic_ordinal=bool(resolved.get("monotonic_ordinal_evidence", False)),
    )
    state = torch.load(
        run_dir / "best_global.pt", map_location="cpu", weights_only=False
    )
    model.load_state_dict(state)
    return model.to(device).eval()


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if len(np.unique(labels)) != 2:
        return None
    return float(roc_auc_score(labels, scores))


def _difference(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    # Avoid severe OpenMP oversubscription when multiple seed evaluations run
    # concurrently on high-core hosts. This does not change model arithmetic.
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)

    frame = pd.read_csv(args.manifest)
    dataset = ManifestImageDataset(
        frame, args.data_root, input_size=256, train=False, seed=0
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    device = torch.device(args.device)
    model = _rebuild(args.run_dir, device)

    rows: list[dict[str, object]] = []
    offset = 0
    dice_values: list[float] = []
    iou_values: list[float] = []
    with torch.no_grad():
        for batch in loader:
            raw = torch.as_tensor(batch["raw_image"]).to(device)
            content = torch.as_tensor(batch["content_mask"]).to(device).bool()
            evidence = model.forward_with_evidence(raw, content_mask=content)
            predicted_mask = (torch.sigmoid(evidence.lesion_logits) >= 0.5) & content
            lesion_only = raw.masked_fill(~predicted_mask, 0.0)
            lesion_erased = raw.masked_fill(predicted_mask, 0.0)
            only_evidence = model.forward_with_evidence(
                lesion_only, content_mask=content
            )
            erased_evidence = model.forward_with_evidence(
                lesion_erased, content_mask=content
            )
            original_score = torch.softmax(evidence.binary_logits, dim=1)[:, 1]
            lesion_only_score = torch.softmax(only_evidence.binary_logits, dim=1)[:, 1]
            lesion_erased_score = torch.softmax(erased_evidence.binary_logits, dim=1)[
                :, 1
            ]
            low_risk_logits = model.classify_from_evidence(
                evidence.lesion_features,
                torch.zeros(len(raw), device=device),
            )
            high_risk_logits = model.classify_from_evidence(
                evidence.lesion_features,
                torch.ones(len(raw), device=device),
            )
            low_risk_score = torch.softmax(low_risk_logits, dim=1)[:, 1]
            high_risk_score = torch.softmax(high_risk_logits, dim=1)[:, 1]
            low_risk_log_odds = low_risk_logits[:, 1] - low_risk_logits[:, 0]
            high_risk_log_odds = high_risk_logits[:, 1] - high_risk_logits[:, 0]
            ordinal_risk = torch.sigmoid(evidence.ordinal_logits)

            available = torch.as_tensor(batch["lesion_mask_available"]).bool()
            truth = torch.as_tensor(batch["lesion_mask"]).bool()
            predicted_cpu = predicted_mask.cpu()
            for prediction, target, valid in zip(
                predicted_cpu, truth, available, strict=True
            ):
                if not valid:
                    continue
                intersection = int((prediction & target).sum())
                union = int((prediction | target).sum())
                total = int(prediction.sum() + target.sum())
                dice_values.append((2 * intersection + 1) / (total + 1))
                iou_values.append((intersection + 1) / (union + 1))

            batch_count = len(raw)
            for index in range(batch_count):
                source = frame.iloc[offset + index]
                rows.append(
                    {
                        "manifest_row": offset + index,
                        "dataset": source["dataset"],
                        "patient_id": source["patient_id"],
                        "image_path": source["image_path"],
                        "label": int(source["label"]),
                        "birads_ordinal": source.get("birads_ordinal", np.nan),
                        "original_score": float(original_score[index]),
                        "lesion_only_score": float(lesion_only_score[index]),
                        "lesion_erased_score": float(lesion_erased_score[index]),
                        "ordinal_risk": float(ordinal_risk[index]),
                        "ordinal_clamp_low_score": float(low_risk_score[index]),
                        "ordinal_clamp_high_score": float(high_risk_score[index]),
                        "ordinal_clamp_low_log_odds": float(low_risk_log_odds[index]),
                        "ordinal_clamp_high_log_odds": float(high_risk_log_odds[index]),
                        "predicted_lesion_fraction": float(
                            predicted_mask[index].float().mean()
                        ),
                        "lesion_mask_available": bool(available[index]),
                    }
                )
            offset += batch_count

    predictions = pd.DataFrame(rows)
    labels = predictions["label"].to_numpy(dtype=int)
    original = predictions["original_score"].to_numpy(dtype=float)
    lesion_only = predictions["lesion_only_score"].to_numpy(dtype=float)
    lesion_erased = predictions["lesion_erased_score"].to_numpy(dtype=float)
    high = predictions["ordinal_clamp_high_score"].to_numpy(dtype=float)
    low = predictions["ordinal_clamp_low_score"].to_numpy(dtype=float)
    high_log_odds = predictions["ordinal_clamp_high_log_odds"].to_numpy(dtype=float)
    low_log_odds = predictions["ordinal_clamp_low_log_odds"].to_numpy(dtype=float)
    malignant = labels == 1
    original_auc = _safe_auc(labels, original)
    lesion_only_auc = _safe_auc(labels, lesion_only)
    lesion_erased_auc = _safe_auc(labels, lesion_erased)
    finite_birads = pd.to_numeric(
        predictions["birads_ordinal"], errors="coerce"
    ).notna()
    spearman = None
    if finite_birads.sum() >= 2:
        spearman = float(
            predictions.loc[finite_birads, "ordinal_risk"].corr(
                pd.to_numeric(predictions.loc[finite_birads, "birads_ordinal"]),
                method="spearman",
            )
        )
    report = {
        "n": len(predictions),
        "original_auc": original_auc,
        "lesion_only_auc": lesion_only_auc,
        "lesion_erased_auc": lesion_erased_auc,
        "lesion_sufficiency_auc_gap": _difference(original_auc, lesion_only_auc),
        "lesion_necessity_auc_drop": _difference(original_auc, lesion_erased_auc),
        "malignant_score_drop_after_lesion_erasure": (
            float(np.mean(original[malignant] - lesion_erased[malignant]))
            if malignant.any()
            else None
        ),
        "ordinal_high_minus_low_mean": float(np.mean(high - low)),
        "ordinal_positive_intervention_fraction": float(np.mean(high > low)),
        "ordinal_non_decreasing_intervention_fraction": float(np.mean(high >= low)),
        "ordinal_negative_intervention_count": int(np.sum(high < low)),
        "ordinal_log_odds_high_minus_low_mean": float(
            np.mean(high_log_odds - low_log_odds)
        ),
        "ordinal_log_odds_positive_intervention_fraction": float(
            np.mean(high_log_odds > low_log_odds)
        ),
        "ordinal_risk_birads_spearman": spearman,
        "ordinal_birads_samples": int(finite_birads.sum()),
        "lesion_dice": float(np.mean(dice_values)) if dice_values else None,
        "lesion_iou": float(np.mean(iou_values)) if iou_values else None,
        "lesion_evaluation_samples": len(dice_values),
        "interpretation": {
            "lesion_sufficiency_auc_gap": "closer to zero is better",
            "lesion_necessity_auc_drop": "positive and larger supports reliance on lesion evidence",
            "ordinal_high_minus_low_mean": "positive supports monotone diagnostic use of ordinal evidence",
            "ordinal_negative_intervention_count": "must be zero for the monotonic candidate",
        },
    }
    args.output_dir.mkdir(parents=True)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
