# Experiment reproduction

Execute commands from the repository root with Python 3.12 and the pinned environment. Obtain datasets and verify split fingerprints first. Examples are instructions, **not evidence that a new full training campaign has already been run**. Outputs remain local. ImageNet initialization is downloaded by torchvision for real training; the synthetic demo does not download weights.

## Methods and controls

| CLI identifier | Role |
|---|---|
| `fedevi_monotonic` | Final FedEvi-DG: lesion pooling, ordinal branch, nonnegative ordinal coefficient, valid-label-aware aggregation |
| `fedavg` | Binary classification baseline |
| `fedevi_mtl` | Auxiliary branches without the final evidence fusion |
| `fedevi_seg_only` | Segmentation auxiliary task only |
| `fedavg_ord` | Ordinal auxiliary task only |
| `fedevi_shuffle` | Shuffled BI-RADS supervision control |
| `fedevi_mask_shift` | Spatially shifted lesion-mask control |
| `fedevi_dg` | R0: unconstrained feature/ordinal concatenation classifier |
| `fedevi_no_taskagg` | Ordinary aggregation instead of valid-label-aware auxiliary aggregation |

Exact behavior is preserved in the model, losses and trainer. Historical physics/frequency code is retained for import compatibility, not presented as the final method.

## One smoke run

```bash
python scripts/run_fedevi.py --manifest-root splits/prepared/lodo \
  --data-root /path/to/provider_data --target UDIAT \
  --method fedevi_monotonic --seed 42 --device cuda:0 \
  --smoke --output-root outputs/smoke
```

Smoke results are not paper results. Use a separate output folder. Successful runs contain `COMPLETE`, `best_global.pt`, source/target metrics, local target predictions, resolved configuration, normalization, source manifests' digests and a captured implementation snapshot. Failed or partial runs must not be included in aggregation. Do not publish the local predictions, source metadata or checkpoint by default.

## Five held-out domains, three seeds

The source-only defaults in `configs/paper_protocol.json` are enforced by the preserved runner/config. Each held-out domain is untouched during training and checkpoint selection. Each campaign uses one clean code revision. Five domain rotations are **not random five-fold CV**.

```bash
# Prints the 135 planned runs; does NOT execute training.
python scripts/run_campaign.py --manifest-root splits/prepared/lodo \
  --data-root /path/to/provider_data \
  --methods fedevi_monotonic fedavg fedevi_mtl fedevi_no_taskagg \
  fedevi_shuffle fedevi_mask_shift fedevi_dg fedevi_seg_only fedavg_ord \
  --calibrate

# Add --execute only when ready for the full GPU campaign.
```

The script runs sequentially, skips completed runs, refuses to overwrite incomplete run artifacts through the underlying runner, and requires a clean Git checkout. Run folders follow the layout in `scripts/analyze_fedevi_final.py::METHOD_ROOTS`; legacy directory names do not imply additional protocol stages. Do not mix previous and fresh campaigns in the same folders.

```bash
python scripts/analyze_lodo.py --output-root outputs/new_lodo_analysis --calibration
```

This requires all nine methods × five domains × three seeds. It produces seed/ensemble AUROC, paired bootstrap intervals, sensitivity at 90% specificity and patient-maximum-probability sensitivity analyses; `--calibration` additionally produces calibration and decision curves. It reuses the original analysis functions while disabling historical seed-42 replacement paths for fresh campaigns. The original `analyze_fedevi_final.py` remains available for the archived output layout and its complete ordinal/spatial audit.

Ensemble metrics average aligned seed predictions before scoring. Reliable patient IDs are aggregated before the primary metrics. A directional gate is not statistical significance. Sensitivity at 90% specificity is a retrospective ROC endpoint, not a target-selected deployment threshold. Calibrators use only source validation.

## Lesion and ordinal explanation evaluation

```bash
python scripts/evaluate_fedevi_explanations.py \
  --run-dir outputs/fedevi_monotonic_gate/target_BUSBRA/fedevi_monotonic/seed_42 \
  --manifest splits/prepared/lodo/target_BUSBRA/target_test.csv \
  --data-root /path/to/provider_data \
  --output-dir outputs/fedevi_monotonic_gate_explanations_v2/target_BUSBRA/seed_42
```

Repeat for seeds 43/44 and other domains as needed. Outputs include local intervention predictions and aggregate metrics for lesion-only/erased images, mask agreement and conditional ordinal clamping. They are probes of the fitted model, not causal clinical explanations. A ground-truth mask is not required for inference. The final ordinal/spatial aggregation additionally needs BUS-CoT explanation predictions and mask-shift control metrics; these are not fabricated if unavailable.

## External validation

### QAMEBI

1. Obtain QAMEBI originals and both original class archives. Run `audit_qamebi_external.py --help`. Its original overlap screen also requires locally acquired BrEaST, BUS-CoT, BCMID and BUS-UCLM inputs. These dependencies are intentionally documented rather than silently omitted. Individual manual near-duplicate decisions must be recorded locally; without those frozen decisions, the reconstructed screen is not claimed identical to the paper.
2. Freeze the audit manifest and summary before scoring. Build the five-source bundle with `build_qamebi_final_bundle.py`:

```bash
python scripts/build_qamebi_final_bundle.py --lodo-root splits/prepared/lodo \
  --qamebi-manifest /path/to/audit/frozen_manifest.csv \
  --qamebi-audit-summary /path/to/audit/audit_summary.json \
  --protocol configs/qamebi_external_protocol.json \
  --output-root splits/qamebi
```

3. Train main, FedAvg and MTL on all five sources, each seed 42/43/44:

```bash
python scripts/run_fedevi.py --manifest-root splits/qamebi \
  --data-root /path/to/common_data_parent --target QAMEBI \
  --method fedevi_monotonic --seed 42 --rounds 30 \
  --output-root outputs/fedevi_qamebi_final_v1/runs --require-clean-git
```

The data root must contain all absolute image paths in the locally generated source and target manifests. Inspect path containment before running; this is read-only image access. QAMEBI is not a sixth client. Fit source-only calibration separately if requested, and run explanation evaluation on its `target_test.csv` into `outputs/fedevi_qamebi_final_v1/explanations/seed_SEED`.

```bash
python scripts/aggregate_qamebi_external.py \
  --result-root outputs/fedevi_qamebi_final_v1/runs/target_QAMEBI \
  --output-dir outputs/fedevi_qamebi_final_v1/summary
python scripts/aggregate_qamebi_explanations.py \
  --explanation-root outputs/fedevi_qamebi_final_v1/explanations \
  --output outputs/fedevi_qamebi_final_v1/summary/explanation_summary.json
python scripts/analyze_qamebi_clinical_utility.py \
  --predictions outputs/fedevi_qamebi_final_v1/summary/ensemble_predictions.csv \
  --output-dir outputs/fedevi_qamebi_final_v1/summary/clinical_utility
```

QAMEBI does not supply BI-RADS grades; ordinal accuracy cannot be evaluated there. The reference AUROC gap versus FedAvg is small and its interval crosses zero.

### BUS-CoT supplementary evaluation

Use `audit_buscot_external.py` on the provider archive/metadata and development bundle, then `extract_frozen_buscot.py` on the frozen, screened manifest. This selection is distinct from QAMEBI. The paper's source combination is the **GDPH-held-out** model trained on BUSBRA, BUSI_WHU, SYSUCC and UDIAT, not a five-source retraining. Evaluate seeds 42/43/44 with `evaluate_fedevi_explanations.py`, using the extracted local manifest/data root, and write to `outputs/fedevi_monotonic_gate_buscot_reasoning_v3/seed_SEED`. Masks/BI-RADS/patient IDs are used only for post-training evaluation. Do not characterize this supplementary annotation benchmark as prospective clinical confirmation.

## Complete archived analysis and plots

When all classification, source calibration, BUSBRA and BUS-CoT explanation outputs are present in the documented layout:

```bash
python scripts/analyze_fedevi_final.py --output-root outputs/fedevi_final_analysis_v3_audit
python scripts/generate_paper_figures.py
```

The full plot script additionally reads QAMEBI aggregate outputs, local ensemble predictions, reliability and decision-curve tables; see its file reads. It writes quantitative panels to `docs/paper/figures/`. It also retains a programmatic framework illustration, which is **not** the subsequently selected editorial framework image. Medical image panels/AI-drawn manuscript artwork are not distributed. Aggregate-only reference CSVs let users inspect reported metrics without datasets, but do not replace independently regenerating predictions.

For exact historical model-source inspection, see [provenance](PROVENANCE.md). New hardware/library versions may change numerical results even when split digests match.
