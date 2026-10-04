# Local data preparation

No dataset is included or redistributed. Obtain provider permission and the original files/sidecars yourself. This release does not grant rights to any images, masks, BI-RADS annotations or clinical metadata. SHA-256 exclusion rules and aggregate split digests are provided, not per-patient manifests. Generated manifests, audit reports, predictions and checkpoints must remain local.

## Source datasets

| Domain | Provider/reference | Required supervision |
|---|---|---|
| BUSBRA | [Original dataset paper](https://doi.org/10.1002/mp.16812) | Pathology, case ID, lesion mask, BI-RADS |
| BUSI_WHU | [Version 3 provider record](https://doi.org/10.17632/k6cpmwybk3.3) | Pathology, patient ID, lesion mask |
| GDPH / SYSUCC | [HoVer-Trans paper](https://doi.org/10.1109/TMI.2023.3236011) | Pathology in filenames and reader BI-RADS sidecar |
| UDIAT Dataset B | [Original study](https://doi.org/10.1117/1.JMI.6.1.011007) | Classification sidecar and lesion masks |

The parsers expect this arrangement (directory spacing matters):

```text
RAW_ROOT/
  BUSBRA/
    bus_data.csv
    Images/bus_*.png
    Masks/mask_*.png
  BUSI_WHU Breast Cancer Ultrasound Image  Dataset/
    Patient_infos_BUSI-WHU.xlsx
    img/*.bmp
    gt/*_anno.bmp
  GDPH&SYSUCC/
    GDPH/*.png
    SYSUCC/*.png
    BIRADS&FOLD.xlsx
  udiat/UDIAT_Dataset_B/UDIAT_Dataset_B/
    DatasetB.xlsx
    original/*.png
    GT/*.png
```

Required sidecars are not replaced with guessed labels. If an acquired distribution does not include these metadata, request them from its provider or build an equivalent licensed local sidecar. See the source parsers for required sheet/column names. A different dataset version or reconstructed patient IDs may not reproduce the frozen split.

## Build and verify

Run from the repository root, choosing a **new** local output directory:

```bash
python scripts/prepare_data.py --raw-root /path/to/provider_data --output-root splits/prepared
```

This reads provider originals, resolves exact duplicates, applies the frozen content-hash exclusions, attaches BI-RADS, audits images/patient groups/near duplicates, builds five domain-held-out bundles and verifies masks. It then compares all 15 split fingerprints against the paper's retained rows. It does not use stretched legacy derivatives. An unresolved near duplicate or a fingerprint mismatch stops the workflow: investigate the dataset version/metadata rather than suppressing the check. Partial files after a failed preparation are local diagnostic output, not a verified bundle; retry into a new directory.

```bash
python scripts/verify_splits.py --manifest-root splits/prepared/lodo
```

Each `target_DOMAIN/` contains `source_train.csv`, `source_val.csv`, `target_test.csv`, `metadata.json` and `COMPLETE`. CSVs include locally resolved paths and patient identifiers; **do not commit or publish them**. Source validation is 15%, split seed 20260728. BUSBRA and BUSI_WHU use reliable patient grouping. GDPH, SYSUCC and UDIAT record IDs must not be described as confirmed patient IDs. Missing auxiliary labels remain missing, not negative labels.

Within-domain BI-RADS ranking for BUSBRA and the shared reader scale for GDPH/SYSUCC are implemented in `attach_birads_to_metadata.py`. BI-RADS is auxiliary ordinal supervision, not a substitute for pathology. Training normalization is source-only; target auxiliary annotations are used only by post-training explanation evaluation.

## External datasets

QAMEBI: [Provider](https://qamebi.com/breast-ultrasound-images-database). BUS-CoT: [Dataset publication](https://doi.org/10.1038/s41597-026-06702-9). Obtain original archives/annotations separately. Their overlap audits, archive digests and screened selection must precede evaluation. See [external commands](REPRODUCTION.md#external-validation). Neither raw archives nor screened record manifests are redistributed.
