# Source and result provenance

The release was collected from the development project without copying its Git history, private laboratory notes, raw data, patient manifests, predictions or weights. `reproducibility/release_files.json` records SHA-256 values for the current code-only release files (excluding the inventory itself).

The paper-only cleanup starts from release commit `626e8cacb0b46db5bc596c3977d73d59e51fa002`. The package is now `fedevi_dg`; the runner is `scripts/train_fedevi.py`. Old physics/frequency/relay branches and their tests have been removed, while the nine paper methods remain. A local archive preserves the pre-cleanup tree. That archive and the old implementation snapshots are not distributed in the current tree; previous GitHub commits have not been rewritten.

`reproducibility/cleanup_equivalence.json` records a CPU synthetic regression against the pre-cleanup release: three source clients, heterogeneous annotations, three rounds, seed 42 and all nine retained methods. Selected model tensor hashes, prediction scores, source metrics, thresholds and selected rounds agree exactly. Two removed history columns only reported zero-valued relay diagnostics. This small check is evidence against accidental behavioral changes, not a rerun of the paper's complete GPU experiment or a guarantee for every possible input/hardware combination.

## Historical result audit

The reference results were audited against a recovered historical candidate snapshot. Its implementation digest was:

```text
344b61c91934b005eabc2072ff15ce470cf5f2883f69c8bfcb5e501527efc38f
```

The original candidate seed-42 runs used an earlier snapshot. During the historical audit they were replayed under the recovered snapshot shared by seeds 43/44; the candidate replay predictions were byte-identical to the original candidate predictions. These prior audit findings are distinct from the current cleanup validation: this packaging task does **not** claim a new 135-run GPU experiment or bitwise reproduction on other hardware. The historical source bytes remain in the owner's local archive and earlier repository commits, not the current paper-only tree.

The aggregate reference results use the declared seed-42 replay policy for candidate and matched controls. R0 replay replacement altered some cells; its uncertainty and directional interpretation are preserved in `r0_replacement_sensitivity.csv`. Do not replace this with a blanket statement that all reruns were identical.

## Fresh campaigns

Use one clean release commit for all seeds/controls. `train_fedevi.py` captures source bytes, implementation hash, Git status/diff and runtime/configuration metadata in each local output. `run_campaign.py` passes `--require-clean-git`; do not edit source halfway through training. `analyze_lodo.py` disables historical replacement folders for a fresh campaign. A historical audit table is not part of a new campaign unless the historical inputs are actually supplied.

## Dataset integrity without redistribution

`split_fingerprints.json` stores only aggregate row counts, field names and canonical split digests. It excludes filesystem roots and contains no patient-level rows. `verify_splits.py` recomputes the same signatures on locally generated manifests. The frozen exclusions are raw-image SHA-256 values, not diagnosis/patient metadata. These tools require legitimately acquired original data; hashes are neither anonymized medical records nor an authorization to redistribute provider data.

The source split is fixed independently of the model seed. Target labels and annotations are evaluation-only. Checkpoint selection, normalization, calibration and training label normalization use source data only.
