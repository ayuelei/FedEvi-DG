# Aggregate reference results

These are machine-readable **aggregate** tables from the manuscript analysis. They contain no original images, lesion masks, patient identifiers, raw BI-RADS tables or per-record predictions. They are included to check numerical reporting, not to substitute for rerunning experiments.

`lodo/` contains the final implementation-audited five-domain results, including non-significant and unfavorable comparisons and the original/replacement R0 sensitivity analysis. `qamebi/` contains the locked public external aggregate summary and per-seed metrics.

Expected headline values are five-domain three-seed macro AUROC 0.8522, ensemble macro AUROC 0.8601, QAMEBI ensemble AUROC 0.8540, and QAMEBI candidate-minus-FedAvg difference about 0.0024 with an interval crossing zero. These are different quantities and must not be mixed.

Individual predictions, raw split CSVs and model checkpoints are intentionally excluded. Regenerate them locally after obtaining the datasets. Fresh runs on a different environment are not guaranteed to match the original hardware bit for bit.
