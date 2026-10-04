# Source and result provenance

The release was collected from the development project without copying its Git history, private laboratory notes, raw data, patient manifests, predictions or weights. `source_inventory.json` records the development commit and SHA-256 of each allowlisted copied file. Those hashes describe the source-copy stage; public wrappers and documentation added afterward are covered by the final release file inventory.

The model/trainer/runner computations were not refactored for this release. The historical Python package name `fedusdg_mvp` is retained. The public CLI limits method choices but delegates to the original runner.

## Historical candidate snapshot

`reproducibility/snapshots/344b61c9/` contains the recovered candidate implementation and its original runner, not a generic replacement of the whole release. The full implementation digest is:

```text
344b61c91934b005eabc2072ff15ce470cf5f2883f69c8bfcb5e501527efc38f
```

The original candidate seed-42 runs used an earlier snapshot. They were replayed under the recovered snapshot shared by seeds 43/44; the replay predictions were byte-identical to the original candidate predictions. This historical audit is distinct from release validation: this packaging task does **not** claim a new 135-run GPU experiment or bitwise reproduction on other hardware.

The aggregate reference results use the declared seed-42 replay policy for candidate and matched controls. R0 replay replacement altered some cells; its uncertainty and directional interpretation are preserved in `r0_replacement_sensitivity.csv`. Do not replace this with a blanket statement that all reruns were identical.

## Fresh campaigns

Use one clean release commit for all seeds/controls. `run_fedusdg_mvp.py` captures source bytes, implementation hash, Git status/diff and runtime/configuration metadata in each local output. `run_campaign.py` passes `--require-clean-git`; do not edit source halfway through training. `analyze_lodo.py` disables historical replacement folders for a fresh campaign. A historical audit table is not part of a new campaign unless the historical inputs are actually supplied.

## Dataset integrity without redistribution

`split_fingerprints.json` stores only aggregate row counts, field names and canonical split digests. It excludes filesystem roots and contains no patient-level rows. `verify_splits.py` recomputes the same signatures on locally generated manifests. The frozen exclusions are raw-image SHA-256 values, not diagnosis/patient metadata. These tools require legitimately acquired original data; hashes are neither anonymized medical records nor an authorization to redistribute provider data.

The source split is fixed independently of the model seed. Target labels and annotations are evaluation-only. Checkpoint selection, normalization, calibration and training label normalization use source data only.
