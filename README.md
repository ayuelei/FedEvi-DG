# FedEvi-DG

Code-release candidate for **Federated Domain Generalization for Breast Ultrasound Classification with Lesion-Guided Feature Pooling and an Ordinal-Score Monotonicity Constraint under Heterogeneous Clinical Annotations**.

This repository contains the model, federated training, matched controls, data-audit and split builders, external-evaluation tools, statistical analysis, and quantitative plotting code. **Datasets, masks, clinical metadata, per-record predictions and trained weights are not redistributed.** Obtain the data from the original providers and follow their terms.

The code identifier for the paper's main method is **`fedevi_monotonic`**. `fedevi_dg` is the unconstrained feature/ordinal concatenation comparator, not the final proposed model.

## Installation

The collected development environment used Python 3.12.6, PyTorch 2.4.1 and torchvision 0.19.1. GPU installation example:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m pytest -q
```

CPU-only PyTorch can be used for unit tests and the synthetic demo. CUDA and hardware differences may prevent bitwise-identical numerical reproduction.

## Start without medical data

```bash
python scripts/demo_model.py
```

This uses random tensors and untrained weights to check the forward path and fixed-feature ordinal monotonicity. It downloads nothing and is **not** a diagnostic example or a performance experiment.

## Reproduce with locally obtained data

1. Read [data preparation](docs/DATA_PREPARATION.md); recreate the frozen source selection and verify the split fingerprints.
2. Read [experiment commands](docs/REPRODUCTION.md); run one smoke experiment, then the five held-out domains and seeds 42/43/44.
3. Fit calibration using source validation only and run explanation evaluation.
4. Reproduce the independent QAMEBI evaluation using five source clients. BUS-CoT is a separately screened supplementary evaluation with a different model source combination.
5. Aggregate predictions and regenerate quantitative plots locally. Check the [reference aggregate results](reference_results/README.md).

No target images, labels or statistics are used to train, normalize, choose checkpoints, fit temperature or choose deployment thresholds. Five-domain LODO means **five held-out dataset domains**, not random five-fold cross-validation. Reliable patient IDs are used for BUSBRA and BUSI_WHU; other domains use record-level evaluation.

## Layout

```text
fedusdg_mvp/                # Preserved implementation; historical package name
scripts/                   # Preparation, training, calibration, evaluation, analysis
configs/                   # Frozen protocol and data-free filtering rules
tests/fedusdg_mvp/          # Synthetic/unit regression tests
reference_results/         # Aggregate metrics only, not predictions or dataset rows
reproducibility/            # Source hashes, split digests, historical candidate source
docs/                      # Data instructions, commands and release boundaries
```

The shared training engine imports spectral/physics support modules from earlier development. They are retained to keep imports and source provenance intact; those modules are **not** the FedEvi-DG paper's proposed mechanism. The public experiment entry point restricts the available methods to the paper's methods and controls.

## Scientific scope

The five-domain results support the overall tested configuration. The QAMEBI AUROC difference from FedAvg is uncertain; this release does not claim external superiority, clinical readiness, causal explanations or formal privacy protection. Fixed-feature monotonicity is a structural property, not proof that BI-RADS predictions are accurate or that their independent AUROC contribution is established.

## License

The original project code is released under the [MIT License](LICENSE), copyright (c) 2026 ayuelei. Third-party code and dependencies retain their applicable licenses. Dataset/provider permissions are separate: the code license does not authorize redistribution of medical data, annotations or patient-level outputs. See [licensing details](docs/LICENSING.md).

## Release status and citation

This is a preparation-stage repository for a manuscript, not a claimed accepted publication. Author list, publication DOI and final citation will be added when confirmed. See [release checklist](docs/RELEASE_CHECKLIST.md), [source provenance](docs/PROVENANCE.md), and [licensing](docs/LICENSING.md). The repository owner's choice of visibility and license is independent of data-provider permissions.

## Reference implementations reviewed

The packaging survey is in [public repository comparison](docs/PUBLIC_REPOSITORY_SURVEY.md). These repositories informed organization only; their code has not been copied into this release.
