# Release checklist

## Included

- Preserved model, losses and federated training calculations.
- Paper-scoped CLI and a dry-run-first campaign launcher.
- Raw-data preparation, mask/BI-RADS attachment and split-fingerprint validation.
- Source calibration, explanation/clinical utility evaluation and statistical aggregation.
- Unit tests, environment pins, frozen protocol summaries and quantitative plotting source.
- Aggregate result tables, source hashes and the recovered candidate implementation snapshot.

## Excluded

- Images, masks, DICOM/RF archives, raw sidecars and clinical metadata.
- Patient/record split lists, individual predictions, checkpoints and pretrained weights.
- Private manuscripts, internal research diaries, development Git history and credentials.
- Artwork containing medical examples and the editorially selected framework image.

## Validation scope

The collection is tested in the existing development Python environment. Core tests, synthetic forward/monotonicity demo, CLI checks and original-data split reconstruction are recorded in `reproducibility/release_validation.json`. The raw-data path and generated manifests are **not** recorded there. All 15 content fingerprints matched the frozen paper splits during collection. These checks do not establish model performance on a fresh dataset, a clean environment installation or a new full multi-seed GPU campaign.

Existing tests that require optional CUDA conditions may be skipped on CPU. Test status must be reported with the skip count, not called universal hardware validation. Clean-environment installation and the full training/analysis/external workflow remain checks for downstream reproduction.

## Before public release

- MIT License selected by the owner on 2026-10-04 and included at the repository root (copyright 2026 ayuelei). Complete the third-party provenance/notice review before public release; MIT does not replace third-party or dataset terms.
- Confirm author names, manuscript title and eventual citation/DOI; do not invent an accepted publication.
- Review provider terms before sharing any future trained weights, examples or patient-level output.
- Review this GitHub code-only tree; repository visibility is currently private and was not changed during collection.
- Preserve the source commit and per-run implementation snapshots for future corrections.

There is no formal differential-privacy guarantee, no claim of prospective clinical readiness, and no claim that the uncertain QAMEBI difference establishes superiority to FedAvg.
