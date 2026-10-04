import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
from scripts.verify_splits import frame_signature
from scripts import analyze_lodo

ROOT = Path(__file__).resolve().parents[1]


def test_portable_split_digest_ignores_machine_paths_and_order():
    first = pd.DataFrame({"dataset": ["A", "A"], "patient_id": ["a", "b"],
                          "label": [0, 1], "image_path": ["/machine1/a", "/machine1/b"]})
    second = first.iloc[::-1].copy()
    second["image_path"] = ["/machine2/b", "/machine2/a"]
    assert frame_signature(first) == frame_signature(second)
    second.loc[second.patient_id == "a", "label"] = 1
    assert frame_signature(first) != frame_signature(second)


def test_historical_snapshot_matches_recorded_digest():
    record = json.loads((ROOT / "reproducibility/historical_candidate.json").read_text())
    files = {name: hashlib.sha256((ROOT / record["location"] / name).read_bytes()).hexdigest()
             for name in record["files_sha256"]}
    assert files == record["files_sha256"]
    combined = hashlib.sha256(json.dumps(files, sort_keys=True,
                                         separators=(",", ":")).encode()).hexdigest()
    assert combined == record["implementation_sha256"]


def test_campaign_defaults_to_dry_run(tmp_path):
    completed = subprocess.run([
        sys.executable, str(ROOT / "scripts/run_campaign.py"),
        "--manifest-root", str(tmp_path / "nonexistent_splits"),
        "--data-root", str(tmp_path / "nonexistent_data"),
        "--targets", "UDIAT", "--seeds", "42", "--methods", "fedevi_monotonic",
    ], cwd=tmp_path, check=True, capture_output=True, text=True)
    assert "--require-clean-git" in completed.stdout
    assert "--rounds 30" in completed.stdout
    assert list(tmp_path.iterdir()) == []


def test_portable_analysis_keeps_both_clinical_endpoints(tmp_path, monkeypatch):
    analysis = analyze_lodo.analysis
    roots = {method: tmp_path / method for method in analysis.METHOD_ROOTS}
    monkeypatch.setattr(analysis, "METHOD_ROOTS", roots)
    monkeypatch.setattr(analysis, "SEED42_RERUN_ROOTS", {})
    for method, root in roots.items():
        for target in analysis.TARGETS:
            for seed in analysis.SEEDS:
                run = root / f"target_{target}" / method / f"seed_{seed}"
                run.mkdir(parents=True)
                (run / "COMPLETE").touch()
                pd.DataFrame({"manifest_row": range(6), "image_path": [f"synthetic/{i}" for i in range(6)],
                              "patient_id": [f"synthetic_patient_{i}" for i in range(6)],
                              "patient_grouping_reliable": [True] * 6,
                              "label": [0, 0, 0, 1, 1, 1],
                              "score": [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]}
                             ).to_csv(run / "target_predictions.csv", index=False)
    output = tmp_path / "analysis"
    monkeypatch.setattr(sys, "argv", ["analyze_lodo", "--output-root", str(output),
                                      "--bootstrap-iterations", "200"])
    monkeypatch.chdir(ROOT)  # Restore the caller's cwd after the analysis.
    analyze_lodo.main()
    endpoints = pd.read_csv(output / "clinical_operating_points.csv")
    assert len(endpoints) == 45
    assert endpoints["sensitivity_at_specificity_90"].eq(1).all()
    patient = pd.read_csv(output / "patient_aggregation_sensitivity.csv")
    assert len(patient) == 45
    assert patient["max_minus_mean_auc_delta"].eq(0).all()
