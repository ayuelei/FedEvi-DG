import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from fedevi_dg.manifest import apply_near_duplicate_dedup
from PIL import Image

import fedevi_dg.manifest as manifest_module
from fedevi_dg.config import ExperimentConfig
from fedevi_dg.manifest import (
    APPROVED_DOMAINS,
    audit_master_manifest,
    build_lodo_frames,
    find_cross_domain_near_duplicates,
    find_manifest_near_duplicates,
    validate_no_cross_domain_duplicates,
    write_lodo_bundle,
)
from fedevi_dg.provenance import canonical_hash


def rows():
    out = []
    for domain in ("BUSBRA", "BUSI_WHU", "GDPH", "SYSUCC", "UDIAT"):
        for label in (0, 1):
            for patient in range(10):
                out.append(
                    {
                        "dataset": domain,
                        "image_path": f"{domain}/{label}_{patient}.png",
                        "patient_id": f"{domain}:p{label}_{patient}",
                        "patient_id_source": "metadata",
                        "label": label,
                        "label_source": "pathology",
                        "is_original": True,
                        "sha256": f"{domain}-{label}-{patient}",
                        "phash": f"{patient:016x}",
                    }
                )
    return pd.DataFrame(out)


def safe_rows():
    frame = rows()
    for row_index in frame.index:
        identity = f"row-{row_index}".encode()
        frame.at[row_index, "sha256"] = hashlib.sha256(identity).hexdigest()
        frame.at[row_index, "phash"] = hashlib.sha256(b"phash-" + identity).hexdigest()[
            :16
        ]
    return frame


def audit_fixture(tmp_path, *, patients_per_class=1):
    data_root = tmp_path / "data"
    data_root.mkdir()
    records = []
    for domain in APPROVED_DOMAINS:
        for label in (0, 1):
            for patient in range(patients_per_class):
                relative_path = Path(domain) / f"{label}_{patient}.png"
                image_path = data_root / relative_path
                image_path.parent.mkdir(exist_ok=True)
                Image.new("L", (2, 2), color=(len(records) * 7) % 256).save(image_path)
                identity = f"{domain}-{label}-{patient}".encode()
                records.append(
                    {
                        "client": domain,
                        "file": str(relative_path),
                        "label": "benign" if label == 0 else "malignant",
                        "patient_id": f" {domain}-P{label}-{patient} ",
                        "patient_id_source": "metadata",
                        "label_source": "pathology",
                        "is_original": True,
                        "sha256": hashlib.sha256(identity).hexdigest(),
                        "phash": hashlib.sha256(b"phash-" + identity).hexdigest()[:16],
                    }
                )
    metadata_csv = tmp_path / "metadata.csv"
    pd.DataFrame(records).to_csv(metadata_csv, index=False)
    return data_root, metadata_csv


def test_master_audit_rejects_absolute_image_outside_data_root(tmp_path):
    data_root, metadata_csv = audit_fixture(tmp_path)
    outside = tmp_path / "outside.png"
    Image.new("L", (2, 2), color=123).save(outside)
    metadata = pd.read_csv(metadata_csv)
    metadata.loc[0, "file"] = str(outside.resolve())
    metadata.to_csv(metadata_csv, index=False)

    with pytest.raises(ValueError, match="image path escapes data root"):
        audit_master_manifest(metadata_csv, data_root=data_root)


def test_master_audit_rejects_symlink_image_escaping_data_root(tmp_path):
    data_root, metadata_csv = audit_fixture(tmp_path)
    metadata = pd.read_csv(metadata_csv)
    linked_path = data_root / metadata.loc[0, "file"]
    outside = tmp_path / "outside.png"
    Image.new("L", (2, 2), color=124).save(outside)
    linked_path.unlink()
    linked_path.symlink_to(outside)

    with pytest.raises(ValueError, match="image path escapes data root"):
        audit_master_manifest(metadata_csv, data_root=data_root)


def experiment_config(tmp_path, target_domain="UDIAT"):
    return ExperimentConfig(
        data_root=tmp_path / "data",
        metadata_csv=tmp_path / "metadata.csv",
        target_domain=target_domain,
    )


class ExplodingCsvCell:
    def __str__(self):
        raise RuntimeError("intentional CSV serialization failure")


def test_target_is_wholly_excluded_from_sources():
    f = build_lodo_frames(rows(), "UDIAT", 20260728)
    assert set(f["target_test"].dataset) == {"UDIAT"}
    assert "UDIAT" not in set(f["source_train"].dataset) | set(f["source_val"].dataset)


def test_patient_groups_do_not_cross_train_val():
    f = build_lodo_frames(rows(), "UDIAT", 20260728)
    train = set(zip(f["source_train"].dataset, f["source_train"].patient_id))
    val = set(zip(f["source_val"].dataset, f["source_val"].patient_id))
    assert train.isdisjoint(val)
    assert set(f["source_val"].label) == {0, 1}


def test_cross_domain_exact_duplicate_is_fatal():
    df = rows()
    df.loc[df.index[-1], "sha256"] = df.loc[df.index[0], "sha256"]
    with pytest.raises(ValueError, match="cross-domain exact duplicate"):
        validate_no_cross_domain_duplicates(df)


def test_same_domain_exact_duplicate_with_different_patients_is_fatal():
    df = safe_rows()
    same_domain = df.index[df.dataset == "BUSBRA"][:2]
    assert df.loc[same_domain[0], "patient_id"] != df.loc[same_domain[1], "patient_id"]
    df.loc[same_domain[1], "sha256"] = df.loc[same_domain[0], "sha256"]

    with pytest.raises(ValueError, match="exact duplicate"):
        validate_no_cross_domain_duplicates(df)


def test_cross_domain_near_duplicate_is_reported():
    df = rows().iloc[:2].copy()
    df.loc[df.index[0], ["dataset", "phash"]] = ["BUSBRA", "0000000000000000"]
    df.loc[df.index[1], ["dataset", "phash"]] = ["UDIAT", "0000000000000001"]
    report = find_cross_domain_near_duplicates(df, max_hamming_distance=4)
    assert len(report) == 1 and report.iloc[0].hamming_distance == 1


def test_manifest_near_duplicate_audit_includes_same_domain_pairs():
    df = safe_rows().iloc[:2].copy()
    df["dataset"] = "BUSBRA"
    df.loc[df.index[0], "phash"] = "0000000000000000"
    df.loc[df.index[1], "phash"] = "0000000000000001"

    assert find_cross_domain_near_duplicates(df).empty
    report = find_manifest_near_duplicates(df)

    assert len(report) == 1
    assert report.iloc[0].left_dataset == "BUSBRA"
    assert report.iloc[0].right_dataset == "BUSBRA"
    assert report.iloc[0].hamming_distance == 1


def test_arbitrary_near_duplicate_resolution_does_not_approve_pair():
    df = rows().iloc[:2].copy()
    df.loc[df.index[0], ["dataset", "phash", "near_duplicate_resolution"]] = [
        "BUSBRA",
        "0000000000000000",
        "unresolved",
    ]
    df.loc[df.index[1], ["dataset", "phash", "near_duplicate_resolution"]] = [
        "UDIAT",
        "0000000000000001",
        "reviewed",
    ]
    report = find_cross_domain_near_duplicates(df)
    assert len(report) == 1
    assert report.iloc[0].resolved == False  # noqa: E712
    assert len(report.iloc[0].pair_id) == 64


def test_pair_approval_is_bound_to_stable_pair_id_on_both_rows():
    df = rows().iloc[:2].copy()
    df.loc[df.index[0], ["dataset", "phash"]] = ["BUSBRA", "0000000000000000"]
    df.loc[df.index[1], ["dataset", "phash"]] = ["UDIAT", "0000000000000001"]
    first = find_cross_domain_near_duplicates(df)
    reversed_report = find_cross_domain_near_duplicates(df.iloc[::-1])
    pair_id = first.iloc[0].pair_id
    assert reversed_report.iloc[0].pair_id == pair_id
    df["near_duplicate_resolution"] = f"approved:{pair_id}"
    approved = find_cross_domain_near_duplicates(df)
    assert approved.iloc[0].resolved == True  # noqa: E712


def test_bundle_recomputes_near_duplicates_instead_of_trusting_empty_report(tmp_path):
    frame = safe_rows()
    frame.loc[frame.dataset == "BUSBRA", "phash"] = "0000000000000000"
    frame.loc[frame.dataset == "UDIAT", "phash"] = "0000000000000001"
    frames = build_lodo_frames(frame, "UDIAT", 20260728)
    supplied_empty = pd.DataFrame(columns=["resolved"])
    with pytest.raises(ValueError, match="unresolved manifest near duplicate"):
        write_lodo_bundle(
            frames,
            tmp_path / "bundle",
            experiment_config(tmp_path),
            near_duplicate_candidates=supplied_empty,
        )
    assert not (tmp_path / "bundle").exists()


def test_bundle_recomputes_near_duplicates_instead_of_trusting_stale_report(tmp_path):
    frame = safe_rows()
    frame.loc[frame.dataset == "BUSBRA", "phash"] = "0000000000000000"
    frame.loc[frame.dataset == "UDIAT", "phash"] = "0000000000000001"
    frames = build_lodo_frames(frame, "UDIAT", 20260728)
    stale = pd.DataFrame(
        [{"pair_id": "0" * 64, "resolved": True, "hamming_distance": 1}]
    )
    with pytest.raises(ValueError, match="unresolved manifest near duplicate"):
        write_lodo_bundle(
            frames,
            tmp_path / "bundle",
            experiment_config(tmp_path),
            near_duplicate_candidates=stale,
        )
    assert not (tmp_path / "bundle").exists()


def test_bundle_rejects_unresolved_same_domain_near_duplicate(tmp_path):
    frame = safe_rows()
    busbra = frame.index[frame.dataset == "BUSBRA"][:2]
    frame.loc[busbra[0], "phash"] = "0000000000000000"
    frame.loc[busbra[1], "phash"] = "0000000000000001"
    frames = build_lodo_frames(frame, "UDIAT", 20260728)

    with pytest.raises(ValueError, match="unresolved manifest near duplicate"):
        write_lodo_bundle(
            frames,
            tmp_path / "bundle",
            experiment_config(tmp_path),
        )

    assert not (tmp_path / "bundle").exists()


def test_bundle_rejects_logically_same_patient_with_different_id_formatting(tmp_path):
    frames = build_lodo_frames(safe_rows(), "UDIAT", 20260728)
    leaked = {name: value.copy() for name, value in frames.items()}
    duplicate = leaked["source_train"].iloc[[0]].copy()
    duplicate["patient_id"] = "  " + duplicate.iloc[0].patient_id.upper() + "  "
    leaked["source_val"] = pd.concat(
        [leaked["source_val"], duplicate], ignore_index=True
    )
    with pytest.raises(ValueError, match="patient leakage"):
        write_lodo_bundle(leaked, tmp_path / "bundle", experiment_config(tmp_path))
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_column", "patient_id_source"),
        ("empty", "missing patient_id_source"),
        ("unapproved", "unapproved patient_id_source"),
        ("conflict", "conflicting patient_id_source"),
    ],
)
def test_direct_bundle_validates_patient_provenance(tmp_path, mutation, message):
    frames = build_lodo_frames(safe_rows(), "UDIAT", 20260728)
    direct = {name: value.copy() for name, value in frames.items()}
    if mutation == "missing_column":
        for value in direct.values():
            value.drop(columns="patient_id_source", inplace=True)
    elif mutation == "empty":
        direct["source_train"].loc[0, "patient_id_source"] = "  "
    elif mutation == "unapproved":
        direct["source_train"].loc[0, "patient_id_source"] = "filename_guess"
    else:
        duplicate = direct["source_train"].iloc[[0]].copy()
        duplicate["patient_id_source"] = "center_id"
        direct["source_train"] = pd.concat(
            [direct["source_train"], duplicate], ignore_index=True
        )
    with pytest.raises(ValueError, match=message):
        write_lodo_bundle(direct, tmp_path / "bundle", experiment_config(tmp_path))
    assert not (tmp_path / "bundle").exists()


def test_direct_bundle_stores_canonical_patient_id_and_source(tmp_path):
    frames = build_lodo_frames(safe_rows(), "UDIAT", 20260728)
    direct = {name: value.copy() for name, value in frames.items()}
    original_id = direct["target_test"].loc[0, "patient_id"]
    direct["target_test"].loc[0, "patient_id"] = f"  {original_id.upper()}  "
    direct["target_test"].loc[0, "patient_id_source"] = " MetaData "
    output_dir = tmp_path / "bundle"
    write_lodo_bundle(direct, output_dir, experiment_config(tmp_path))
    stored = pd.read_csv(output_dir / "target_test.csv")
    assert stored.loc[0, "patient_id"] == original_id
    assert stored.loc[0, "patient_id_source"] == "metadata"


def test_audit_rejects_missing_patient_id_source_instead_of_inventing_it(tmp_path):
    data_root, metadata_csv = audit_fixture(tmp_path)
    frame = pd.read_csv(metadata_csv)
    frame = frame.drop(columns="patient_id_source")
    frame.to_csv(metadata_csv, index=False)
    with pytest.raises(ValueError, match="patient_id_source"):
        audit_master_manifest(metadata_csv, data_root)


def test_audit_rejects_unknown_patient_id_source(tmp_path):
    data_root, metadata_csv = audit_fixture(tmp_path)
    frame = pd.read_csv(metadata_csv)
    frame.loc[0, "patient_id_source"] = "guessed-from-filename"
    frame.to_csv(metadata_csv, index=False)
    with pytest.raises(ValueError, match="unapproved patient_id_source"):
        audit_master_manifest(metadata_csv, data_root)


def test_patient_ids_are_canonicalized_before_grouping(tmp_path):
    data_root, metadata_csv = audit_fixture(tmp_path)
    frame = pd.read_csv(metadata_csv)
    busbra = frame.index[frame.client == "BUSBRA"]
    frame.loc[busbra[0], "patient_id"] = " P001 "
    frame.loc[busbra[1], "patient_id"] = "p001"
    frame.to_csv(metadata_csv, index=False)
    audited = audit_master_manifest(metadata_csv, data_root)
    assert set(audited[audited.dataset == "BUSBRA"].patient_id) == {"p001"}
    with pytest.raises(ValueError, match="mixed-label patient group in BUSBRA"):
        build_lodo_frames(audited, "UDIAT", 20260728)


@pytest.mark.parametrize("missing_column", ["is_original", "label_source"])
def test_audit_requires_explicit_originality_and_label_provenance(
    tmp_path, missing_column
):
    data_root, metadata_csv = audit_fixture(tmp_path)
    frame = pd.read_csv(metadata_csv).drop(columns=missing_column)
    frame.to_csv(metadata_csv, index=False)
    with pytest.raises(ValueError, match=missing_column):
        audit_master_manifest(metadata_csv, data_root)


def test_original_filter_must_leave_all_five_domains(tmp_path):
    data_root, metadata_csv = audit_fixture(tmp_path)
    frame = pd.read_csv(metadata_csv)
    frame.loc[frame.client == "UDIAT", "is_original"] = False
    frame.to_csv(metadata_csv, index=False)
    with pytest.raises(ValueError, match="after original-file filtering.*UDIAT"):
        audit_master_manifest(metadata_csv, data_root)


def test_lodo_split_requires_exactly_five_approved_domains():
    frame = safe_rows()
    frame = frame[frame.dataset != "UDIAT"]
    with pytest.raises(ValueError, match="exactly five approved domains"):
        build_lodo_frames(frame, "SYSUCC", 20260728)


def test_bundle_hashes_and_force_overwrite_are_coherent(tmp_path):
    frame = safe_rows()
    frames = build_lodo_frames(frame, "UDIAT", 20260728)
    output_dir = tmp_path / "bundle"
    config = experiment_config(tmp_path)
    write_lodo_bundle(frames, output_dir, config)
    original_bytes = {
        path.name: path.read_bytes() for path in output_dir.iterdir() if path.is_file()
    }
    metadata = json.loads((output_dir / "metadata.json").read_text())
    assert metadata["split_seed"] == 20260728
    assert metadata["val_fraction"] == 0.15
    assert metadata["config_hash"] == canonical_hash(metadata["split_config"])
    assert metadata["split_config"]["target_domain"] == "UDIAT"
    assert (output_dir / "COMPLETE").is_file()
    for filename, details in metadata["files"].items():
        assert (
            hashlib.sha256((output_dir / filename).read_bytes()).hexdigest()
            == details["sha256"]
        )


def test_failed_force_publish_leaves_previous_complete_bundle_untouched(tmp_path):
    frame = safe_rows()
    frames = build_lodo_frames(frame, "UDIAT", 20260728)
    output_dir = tmp_path / "bundle"
    config = experiment_config(tmp_path)
    write_lodo_bundle(frames, output_dir, config)
    original_bytes = {
        path.name: path.read_bytes() for path in output_dir.iterdir() if path.is_file()
    }

    broken = {name: value.copy() for name, value in frames.items()}
    for value in broken.values():
        value["serialization_probe"] = "ok"
    broken["source_train"].loc[0, "image_path"] = "/new/train/image.png"
    broken["source_val"].at[0, "serialization_probe"] = ExplodingCsvCell()
    with pytest.raises(RuntimeError, match="intentional CSV serialization failure"):
        write_lodo_bundle(broken, output_dir, config, force=True)

    assert original_bytes == {
        path.name: path.read_bytes() for path in output_dir.iterdir() if path.is_file()
    }

    with pytest.raises(FileExistsError, match="complete LODO bundle"):
        write_lodo_bundle(frames, output_dir, config)
    assert original_bytes == {
        path.name: path.read_bytes() for path in output_dir.iterdir() if path.is_file()
    }

    changed = {name: value.copy() for name, value in frames.items()}
    changed["target_test"].loc[0, "image_path"] = "/replacement/image.png"
    write_lodo_bundle(changed, output_dir, config, force=True)
    replacement_metadata = json.loads((output_dir / "metadata.json").read_text())
    assert (output_dir / "COMPLETE").is_file()
    assert "/replacement/image.png" in (output_dir / "target_test.csv").read_text()
    for filename, details in replacement_metadata["files"].items():
        assert (
            hashlib.sha256((output_dir / filename).read_bytes()).hexdigest()
            == details["sha256"]
        )


@pytest.mark.parametrize("corruption", ["missing_csv", "hash_mismatch"])
def test_incomplete_or_corrupt_destination_requires_force_and_is_unchanged(
    tmp_path, corruption
):
    frames = build_lodo_frames(safe_rows(), "UDIAT", 20260728)
    output_dir = tmp_path / "bundle"
    config = experiment_config(tmp_path)
    write_lodo_bundle(frames, output_dir, config)
    if corruption == "missing_csv":
        (output_dir / "target_test.csv").unlink()
    else:
        with (output_dir / "target_test.csv").open("a") as handle:
            handle.write("corrupt\n")
    before = {
        path.name: path.read_bytes() for path in output_dir.iterdir() if path.is_file()
    }
    with pytest.raises(FileExistsError, match="incomplete or corrupt"):
        write_lodo_bundle(frames, output_dir, config)
    assert before == {
        path.name: path.read_bytes() for path in output_dir.iterdir() if path.is_file()
    }


def test_cli_preflights_all_five_targets_before_publishing_any_bundle(tmp_path):
    data_root, metadata_csv = audit_fixture(tmp_path, patients_per_class=4)
    output_root = tmp_path / "output"
    blocked_target = output_root / "target_UDIAT"
    blocked_target.mkdir(parents=True)
    (blocked_target / "sentinel").write_text("keep")
    script = Path(__file__).resolve().parents[2] / "scripts/build_lodo_manifests.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--data-root",
            str(data_root),
            "--metadata-csv",
            str(metadata_csv),
            "--output-root",
            str(output_root),
        ],
        cwd=script.parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "already exists" in result.stderr
    assert (blocked_target / "sentinel").read_text() == "keep"
    assert not any(
        (output_root / f"target_{domain}").exists() for domain in APPROVED_DOMAINS[:-1]
    )


def test_cli_success_publishes_five_complete_bundles(tmp_path):
    data_root, metadata_csv = audit_fixture(tmp_path, patients_per_class=4)
    output_root = tmp_path / "output"
    script = Path(__file__).resolve().parents[2] / "scripts/build_lodo_manifests.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--data-root",
            str(data_root),
            "--metadata-csv",
            str(metadata_csv),
            "--output-root",
            str(output_root),
        ],
        cwd=script.parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    for domain in APPROVED_DOMAINS:
        bundle = output_root / f"target_{domain}"
        assert {path.name for path in bundle.iterdir()} == {
            "source_train.csv",
            "source_val.csv",
            "target_test.csv",
            "metadata.json",
            "COMPLETE",
        }


def test_multi_bundle_publish_rolls_back_new_and_restores_old_on_late_race(tmp_path):
    frame = safe_rows()
    output_root = tmp_path / "output"
    first_domain = APPROVED_DOMAINS[0]
    first_config = experiment_config(tmp_path, first_domain)
    first_output = output_root / f"target_{first_domain}"
    first_frames = build_lodo_frames(frame, first_domain, 20260728)
    write_lodo_bundle(first_frames, first_output, first_config)
    old_first = {
        path.name: path.read_bytes()
        for path in first_output.iterdir()
        if path.is_file()
    }

    prepared = []
    for domain in APPROVED_DOMAINS:
        config = experiment_config(tmp_path, domain)
        frames = build_lodo_frames(frame, domain, 20260728)
        frames["target_test"].loc[0, "image_path"] += ".new"
        prepared.append(
            manifest_module._prepare_lodo_bundle(
                frames,
                output_root / f"target_{domain}",
                config,
                force=True,
            )
        )

    raced_output = output_root / f"target_{APPROVED_DOMAINS[2]}"
    raced_output.mkdir()
    (raced_output / "sentinel").write_text("concurrent owner")
    with pytest.raises(RuntimeError, match="changed after preflight"):
        manifest_module._publish_prepared_lodo_bundles(prepared)

    assert old_first == {
        path.name: path.read_bytes()
        for path in first_output.iterdir()
        if path.is_file()
    }
    assert not (output_root / f"target_{APPROVED_DOMAINS[1]}").exists()
    assert (raced_output / "sentinel").read_text() == "concurrent owner"
    assert not any(output_root.glob(".*.staging-*"))
    assert not any(output_root.glob(".*.backup-*"))


class TestApplyNearDuplicateDedup:
    @staticmethod
    def _frame(marks: dict[str, str] | None = None) -> pd.DataFrame:
        rows = [
            {"dataset": "A", "image_path": f"a_{index}.png", "label": 0}
            for index in range(4)
        ]
        frame = pd.DataFrame(rows)
        if marks is not None:
            frame["near_duplicate_resolution"] = frame["image_path"].map(
                lambda path: marks.get(path, "")
            )
        return frame

    @staticmethod
    def _mark(*pairs: tuple[str, str, str]) -> pd.DataFrame:
        marks: dict[str, str] = {}
        for left, right, pair_id in pairs:
            marks[left] = ";".join(
                token for token in (marks.get(left, ""), f"drop:{pair_id}") if token
            )
            marks[right] = ";".join(
                token for token in (marks.get(right, ""), f"drop:{pair_id}") if token
            )
        return TestApplyNearDuplicateDedup._frame(marks)

    def test_component_keeps_one_representative(self):
        pair_id = "a" * 64
        frame = self._mark(("a_0.png", "a_1.png", pair_id))
        reduced, decisions = apply_near_duplicate_dedup(frame)
        assert reduced["image_path"].tolist() == ["a_0.png", "a_2.png", "a_3.png"]
        assert len(decisions) == 1
        row = decisions.iloc[0]
        assert row.image_path == "a_1.png"
        assert row.representative_image_path == "a_0.png"
        assert row.reason == "exclude_same_lesion_near_duplicate"

    def test_chain_of_pairs_keeps_single_representative(self):
        frame = self._mark(
            ("a_0.png", "a_1.png", "a" * 64),
            ("a_1.png", "a_2.png", "b" * 64),
        )
        reduced, decisions = apply_near_duplicate_dedup(frame)
        assert reduced["image_path"].tolist() == ["a_0.png", "a_3.png"]
        assert set(decisions["image_path"]) == {"a_1.png", "a_2.png"}
        assert set(decisions["representative_image_path"]) == {"a_0.png"}

    def test_one_sided_mark_is_rejected(self):
        frame = self._frame()
        frame.loc[frame["image_path"] == "a_0.png", "near_duplicate_resolution"] = (
            f"drop:{'a' * 64}"
        )
        frame.loc[frame["image_path"] == "a_2.png", "near_duplicate_resolution"] = ""
        with pytest.raises(ValueError, match="both ends"):
            apply_near_duplicate_dedup(frame)

    def test_label_conflict_inside_group_excludes_all_members(self):
        frame = self._frame()
        frame.loc[frame["image_path"] == "a_0.png", "near_duplicate_resolution"] = (
            f"drop:{'a' * 64}"
        )
        frame.loc[frame["image_path"] == "a_1.png", "near_duplicate_resolution"] = (
            f"drop:{'a' * 64}"
        )
        frame.loc[frame["image_path"] == "a_1.png", "label"] = 1
        reduced, decisions = apply_near_duplicate_dedup(frame)
        assert reduced["image_path"].tolist() == ["a_2.png", "a_3.png"]
        assert set(decisions["image_path"]) == {"a_0.png", "a_1.png"}
        assert set(decisions["reason"]) == {"exclude_conflicting_label_near_duplicate"}

    def test_dataset_spanning_group_is_rejected(self):
        frame = self._frame()
        frame.loc[frame["image_path"] == "a_0.png", "dataset"] = "B"
        for path in ("a_0.png", "a_1.png"):
            frame.loc[frame["image_path"] == path, "near_duplicate_resolution"] = (
                f"drop:{'a' * 64}"
            )
        with pytest.raises(ValueError, match="spans datasets"):
            apply_near_duplicate_dedup(frame)

    def test_frame_without_marks_passes_through(self):
        frame = self._frame()
        reduced, decisions = apply_near_duplicate_dedup(frame)
        assert len(reduced) == 4
        assert decisions.empty

    def test_approved_marks_do_not_trigger_dedup(self):
        frame = self._frame()
        mark = f"approved:{'a' * 64}"
        for path in ("a_0.png", "a_1.png"):
            frame.loc[frame["image_path"] == path, "near_duplicate_resolution"] = mark
        reduced, decisions = apply_near_duplicate_dedup(frame)
        assert len(reduced) == 4
        assert decisions.empty
