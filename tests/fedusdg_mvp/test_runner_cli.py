import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pandas as pd
import pytest
import torch
from PIL import Image
from torch import nn

import scripts.run_fedusdg_mvp as runner
from fedusdg_mvp.manifest import split_protocol_config
from fedusdg_mvp.preprocessing import SourceNormalization
from fedusdg_mvp.provenance import canonical_hash
from scripts.run_fedusdg_mvp import (
    PrerequisiteError,
    SourceNormalizedCategoricalResNet18,
    SourceNormalizedResNet18,
    SourceNormalizedOrdinalResNet18,
    adapt_domain_loaders,
    assert_output_writable,
    build_parser,
    build_pretrained_model,
    build_source_signatures,
    build_target_predictions,
    load_and_validate_lodo_bundle,
    publish_staging_directory,
)


def test_parser_requires_run_identity():
    args = build_parser().parse_args(
        [
            "--manifest-root",
            "/splits",
            "--data-root",
            "/data",
            "--target",
            "UDIAT",
            "--method",
            "depth",
            "--seed",
            "42",
        ]
    )

    assert (args.target, args.method, args.seed) == ("UDIAT", "depth", 42)
    assert (args.rounds, args.device, args.output_root) == (
        30,
        "cuda:0",
        Path("outputs/fedusdg_mvp"),
    )


def test_parser_accepts_locked_qamebi_external_target():
    args = build_parser().parse_args(
        [
            "--manifest-root",
            "/splits",
            "--data-root",
            "/data",
            "--target",
            "QAMEBI",
            "--method",
            "fedevi_monotonic",
            "--seed",
            "42",
        ]
    )

    assert args.target == "QAMEBI"


def test_parser_accepts_segmentation_only_matched_control():
    args = build_parser().parse_args(
        [
            "--manifest-root",
            "/splits",
            "--data-root",
            "/data",
            "--target",
            "UDIAT",
            "--method",
            "fedevi_seg_only",
            "--seed",
            "42",
        ]
    )

    assert args.method == "fedevi_seg_only"


def test_runner_sets_cublas_workspace_before_importing_torch():
    repository_root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment.pop("CUBLAS_WORKSPACE_CONFIG", None)
    environment["PYTHONPATH"] = str(repository_root)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os; import scripts.run_fedusdg_mvp; "
                "print(os.environ['CUBLAS_WORKSPACE_CONFIG'])"
            ),
        ],
        cwd=repository_root,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == ":4096:8"


def test_runner_rejects_nonapproved_cuda_workspace(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    with pytest.raises(RuntimeError, match=":4096:8"):
        runner.set_global_seeds(42)


def test_complete_result_is_not_overwritten(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "COMPLETE").write_text("ok\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="complete run"):
        assert_output_writable(run, force=False)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle_fixture(tmp_path, target="UDIAT"):
    data_root = tmp_path / "data"
    data_root.mkdir()
    bundle = tmp_path / "splits" / f"target_{target}"
    bundle.mkdir(parents=True)
    sources = sorted(set(runner.APPROVED_DOMAINS) - {target})
    frames = {}
    for split in ("source_train", "source_val"):
        split_offset = 0 if split == "source_train" else 2
        records = []
        for domain_index, domain in enumerate(sources):
            for label in (0, 1):
                image_path = Path(domain) / f"{split}-{label}.png"
                absolute = data_root / image_path
                absolute.parent.mkdir(exist_ok=True)
                Image.new(
                    "L",
                    (12, 10),
                    20 + domain_index * 30 + label * 10 + split_offset,
                ).save(absolute)
                identity = absolute.read_bytes()
                records.append(
                    {
                        "dataset": domain,
                        "image_path": str(image_path),
                        "patient_id": f"{domain}-{split}-{label}",
                        "patient_id_source": "metadata",
                        "label": label,
                        "label_source": "pathology",
                        "is_original": True,
                        "sha256": hashlib.sha256(identity).hexdigest(),
                        "phash": hashlib.sha256(
                            f"phash:{image_path}".encode()
                        ).hexdigest()[:16],
                    }
                )
        frames[split] = pd.DataFrame(records)
    target_records = []
    for label in (0, 1):
        image_path = Path(target) / f"target-{label}.png"
        absolute = data_root / image_path
        absolute.parent.mkdir(exist_ok=True)
        Image.new("L", (12, 10), 180 + label * 20).save(absolute)
        target_records.append(
            {
                "dataset": target,
                "image_path": str(image_path),
                "patient_id": f"{target}-target-{label}",
                "patient_id_source": "metadata",
                "label": label,
                "label_source": "pathology",
                "is_original": True,
                "sha256": hashlib.sha256(absolute.read_bytes()).hexdigest(),
                "phash": hashlib.sha256(
                    f"phash:{image_path}".encode()
                ).hexdigest()[:16],
            }
        )
    frames["target_test"] = pd.DataFrame(target_records)

    files = {}
    for name, frame in frames.items():
        csv_path = bundle / f"{name}.csv"
        frame.to_csv(csv_path, index=False)
        files[csv_path.name] = {
            "rows": len(frame),
            "class_counts": {
                str(int(label)): int(count)
                for label, count in frame["label"].value_counts().sort_index().items()
            },
            "sha256": _sha256(csv_path),
        }
    split_config = (
        split_protocol_config(target)
        if target in runner.APPROVED_DOMAINS
        else runner._expected_split_protocol(target)
    )
    metadata = {
        "target_domain": target,
        "split_seed": 20260728,
        "val_fraction": 0.15,
        "split_config": split_config,
        "config_hash": canonical_hash(split_config),
        "files": files,
    }
    metadata_path = bundle / "metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    (bundle / "COMPLETE").write_text(
        json.dumps({"metadata_sha256": _sha256(metadata_path)}), encoding="utf-8"
    )
    return tmp_path / "splits", data_root, bundle, frames


def test_locked_external_bundle_requires_and_accepts_all_five_sources(tmp_path):
    manifest_root, data_root, _, _ = _bundle_fixture(tmp_path, target="QAMEBI")

    loaded = load_and_validate_lodo_bundle(manifest_root, "QAMEBI", data_root)

    assert set(loaded.frames["source_train"]["dataset"]) == set(
        runner.APPROVED_DOMAINS
    )
    assert set(loaded.frames["target_test"]["dataset"]) == {"QAMEBI"}
    assert (
        loaded.metadata["split_config"]["protocol"]
        == "five_source_locked_public_external_validation"
    )


def _rewrite_bundle_manifest(bundle, name, frame):
    path = bundle / f"{name}.csv"
    frame.to_csv(path, index=False)
    metadata_path = bundle / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["files"][path.name] = {
        "rows": len(frame),
        "class_counts": {
            str(int(label)): int(count)
            for label, count in frame["label"].value_counts().sort_index().items()
        },
        "sha256": _sha256(path),
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    (bundle / "COMPLETE").write_text(
        json.dumps({"metadata_sha256": _sha256(metadata_path)}), encoding="utf-8"
    )


def _runner_args(
    manifest_root,
    data_root,
    output_root,
    *,
    force=False,
    require_clean_git=False,
):
    arguments = [
        "--manifest-root",
        str(manifest_root),
        "--data-root",
        str(data_root),
        "--target",
        "UDIAT",
        "--method",
        "fedavg",
        "--seed",
        "42",
        "--device",
        "cpu",
        "--output-root",
        str(output_root),
        "--smoke",
    ]
    if force:
        arguments.append("--force")
    if require_clean_git:
        arguments.append("--require-clean-git")
    return build_parser().parse_args(arguments)


def _assert_no_run_or_staging(output_root):
    assert not (
        output_root / "target_UDIAT" / "fedavg" / "seed_42"
    ).exists()
    if output_root.exists():
        assert list(output_root.rglob(".seed_42.staging-*")) == []


def test_bundle_hash_failure_precedes_data_or_model_access(tmp_path, monkeypatch):
    manifest_root, data_root, bundle, _ = _bundle_fixture(tmp_path)
    (bundle / "target_test.csv").write_text("corrupt\n", encoding="utf-8")
    touched = []
    monkeypatch.setattr(runner, "build_domain_loaders", lambda *a, **k: touched.append("data"))
    monkeypatch.setattr(runner, "build_pretrained_model", lambda *a, **k: touched.append("model"))

    with pytest.raises(PrerequisiteError, match="SHA-256 mismatch for target_test.csv"):
        runner.run_experiment(
            build_parser().parse_args(
                [
                    "--manifest-root",
                    str(manifest_root),
                    "--data-root",
                    str(data_root),
                    "--target",
                    "UDIAT",
                    "--method",
                    "fedavg",
                    "--seed",
                    "42",
                    "--device",
                    "cpu",
                    "--output-root",
                    str(tmp_path / "outputs"),
                ]
            )
        )

    assert touched == []
    assert not (tmp_path / "outputs" / "target_UDIAT" / "fedavg" / "seed_42").exists()


def test_require_clean_git_rejects_dirty_worktree_before_data_access(
    tmp_path, monkeypatch
):
    manifest_root, data_root, _, _ = _bundle_fixture(tmp_path)
    touched = []
    monkeypatch.setattr(
        runner,
        "_git_provenance",
        lambda: (
            {
                "git_commit": "a" * 40,
                "git_dirty": True,
                "tracked_diff_sha256": "b" * 64,
                "git_status_sha256": "c" * 64,
            },
            b"diff",
            b" M file.py\n",
        ),
    )
    monkeypatch.setattr(
        runner, "build_domain_loaders", lambda *a, **k: touched.append("data")
    )

    with pytest.raises(PrerequisiteError, match="requires a clean"):
        runner.run_experiment(
            _runner_args(
                manifest_root,
                data_root,
                tmp_path / "outputs",
                require_clean_git=True,
            )
        )

    assert touched == []


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("split_seed", 7, "split_seed"),
        ("val_fraction", 0.2, "val_fraction"),
        ("config_hash", "b" * 64, "config_hash"),
        ("split_config", {"protocol": "tampered"}, "split_config"),
    ],
)
def test_bundle_rejects_tampered_split_protocol_before_loader_or_output(
    tmp_path, monkeypatch, key, value, message
):
    manifest_root, data_root, bundle, _ = _bundle_fixture(tmp_path)
    metadata_path = bundle / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata[key] = value
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    (bundle / "COMPLETE").write_text(
        json.dumps({"metadata_sha256": _sha256(metadata_path)}), encoding="utf-8"
    )
    monkeypatch.setattr(
        runner,
        "build_domain_loaders",
        lambda *args, **kwargs: pytest.fail("loader reached before split audit"),
    )
    output_root = tmp_path / "outputs"

    with pytest.raises(PrerequisiteError, match=message):
        runner.run_experiment(_runner_args(manifest_root, data_root, output_root))

    _assert_no_run_or_staging(output_root)


def test_bundle_rejects_target_leakage_after_hash_validation(tmp_path):
    manifest_root, data_root, bundle, _ = _bundle_fixture(tmp_path)
    path = bundle / "source_train.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "dataset"] = "UDIAT"
    frame.to_csv(path, index=False)
    metadata_path = bundle / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["files"][path.name]["sha256"] = _sha256(path)
    path_frame = pd.read_csv(path)
    metadata["files"][path.name]["class_counts"] = {
        str(int(label)): int(count)
        for label, count in path_frame.label.value_counts().sort_index().items()
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    (bundle / "COMPLETE").write_text(
        json.dumps({"metadata_sha256": _sha256(metadata_path)}), encoding="utf-8"
    )

    with pytest.raises(PrerequisiteError, match="source_train must contain exactly"):
        load_and_validate_lodo_bundle(manifest_root, "UDIAT", data_root)


def test_resigned_bundle_rejects_within_domain_exact_duplicate(tmp_path):
    manifest_root, data_root, bundle, frames = _bundle_fixture(tmp_path)
    source = frames["source_train"].iloc[0]
    victim = frames["source_val"].copy()
    victim_index = victim.index[victim.dataset == source.dataset][0]
    source_path = data_root / source.image_path
    victim_path = data_root / victim.loc[victim_index, "image_path"]
    victim_path.write_bytes(source_path.read_bytes())
    victim.loc[victim_index, "sha256"] = _sha256(source_path)
    _rewrite_bundle_manifest(bundle, "source_val", victim)

    with pytest.raises(PrerequisiteError, match="within-domain exact duplicate"):
        load_and_validate_lodo_bundle(manifest_root, "UDIAT", data_root)


def test_resigned_bundle_rejects_within_domain_near_duplicate(tmp_path):
    manifest_root, data_root, bundle, frames = _bundle_fixture(tmp_path)
    train = frames["source_train"].copy()
    val = frames["source_val"].copy()
    domain = str(train.iloc[0].dataset)
    train_index = train.index[train.dataset == domain][0]
    val_index = val.index[val.dataset == domain][0]
    train.loc[train_index, "phash"] = "0000000000000000"
    val.loc[val_index, "phash"] = "0000000000000001"
    _rewrite_bundle_manifest(bundle, "source_train", train)
    _rewrite_bundle_manifest(bundle, "source_val", val)

    with pytest.raises(
        PrerequisiteError, match="unresolved manifest near duplicate"
    ):
        load_and_validate_lodo_bundle(manifest_root, "UDIAT", data_root)


def test_replaced_image_bytes_fail_before_model_or_output(tmp_path, monkeypatch):
    manifest_root, data_root, _, frames = _bundle_fixture(tmp_path)
    image_path = data_root / frames["target_test"].loc[0, "image_path"]
    Image.new("L", (12, 10), 7).save(image_path)
    monkeypatch.setattr(
        runner,
        "build_domain_loaders",
        lambda *args, **kwargs: pytest.fail("loader reached before image audit"),
    )
    output_root = tmp_path / "outputs"

    with pytest.raises(
        PrerequisiteError,
        match=r"target_test\.csv row 0 image SHA-256 mismatch",
    ):
        runner.run_experiment(
            _runner_args(manifest_root, data_root, output_root)
        )

    _assert_no_run_or_staging(output_root)


def test_undecodable_image_fails_before_model_or_output(tmp_path, monkeypatch):
    manifest_root, data_root, bundle, frames = _bundle_fixture(tmp_path)
    image_path = data_root / frames["target_test"].loc[0, "image_path"]
    image_path.write_bytes(b"not a decodable image")
    target = frames["target_test"].copy()
    target.loc[0, "sha256"] = _sha256(image_path)
    _rewrite_bundle_manifest(bundle, "target_test", target)
    monkeypatch.setattr(
        runner,
        "build_domain_loaders",
        lambda *args, **kwargs: pytest.fail("loader reached before PIL audit"),
    )
    output_root = tmp_path / "outputs"

    with pytest.raises(
        PrerequisiteError,
        match=r"undecodable image referenced by target_test\.csv row 0",
    ):
        runner.run_experiment(
            _runner_args(manifest_root, data_root, output_root)
        )

    _assert_no_run_or_staging(output_root)


def test_image_audit_loads_pixels_after_verify_before_output(tmp_path, monkeypatch):
    manifest_root, data_root, _, frames = _bundle_fixture(tmp_path)
    audited_path = (data_root / frames["target_test"].loc[0, "image_path"]).resolve()
    real_open = runner.Image.open
    calls = []

    class VerifyPassLoadFails:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def verify(self):
            calls.append("verify")

        def convert(self, mode):
            assert mode == "L"
            calls.append("convert")
            return self

        def load(self):
            calls.append("load")
            raise OSError("pixel load boom")

    def controlled_open(path, *args, **kwargs):
        if Path(path).resolve() == audited_path:
            return VerifyPassLoadFails()
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(runner.Image, "open", controlled_open)
    monkeypatch.setattr(
        runner,
        "build_domain_loaders",
        lambda *args, **kwargs: pytest.fail("loader reached before pixel audit"),
    )
    output_root = tmp_path / "outputs"

    with pytest.raises(
        PrerequisiteError,
        match=r"undecodable image referenced by target_test\.csv row 0.*pixel load boom",
    ):
        runner.run_experiment(
            _runner_args(manifest_root, data_root, output_root)
        )

    assert "verify" in calls
    assert "load" in calls
    _assert_no_run_or_staging(output_root)


def test_manifest_string_identities_keep_leading_zeroes(tmp_path):
    manifest_root, data_root, bundle, frames = _bundle_fixture(tmp_path)
    for name in ("source_train", "source_val"):
        frame = frames[name].copy()
        frame["patient_id"] = [
            f"{index + (1 if name == 'source_train' else 101):03d}"
            for index in range(len(frame))
        ]
        frame["label_source"] = [f"{index + 1:03d}" for index in range(len(frame))]
        if name == "source_val":
            frame.loc[0, "patient_id"] = "1"
        _rewrite_bundle_manifest(bundle, name, frame)

    target = frames["target_test"].copy()
    target["patient_id"] = ["001", "002"]
    target["label_source"] = ["001", "002"]
    for index in target.index:
        old_path = data_root / target.loc[index, "image_path"]
        new_path = data_root / f"{index + 1:03d}"
        old_path.replace(new_path)
        target.loc[index, "image_path"] = new_path.name
    _rewrite_bundle_manifest(bundle, "target_test", target)

    loaded = load_and_validate_lodo_bundle(manifest_root, "UDIAT", data_root)
    predictions = build_target_predictions(
        loaded.frames["target_test"],
        {
            "labels": [0, 1],
            "scores": [0.2, 0.8],
            "predictions": [0, 1],
            "n": 2,
        },
    )

    assert loaded.frames["source_train"].loc[0, "patient_id"] == "001"
    assert loaded.frames["source_val"].loc[0, "patient_id"] == "1"
    assert loaded.frames["target_test"].loc[0, "image_path"] == "001"
    assert loaded.frames["target_test"].loc[0, "label_source"] == "001"
    assert predictions.loc[0, "patient_id"] == "001"
    assert predictions.loc[0, "image_path"] == "001"


class _RecordingBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 1000)
        self.seen = None

    def forward(self, image):
        self.seen = image.detach().clone()
        pooled = image.mean(dim=(-2, -1))
        return self.fc(pooled)


def test_model_wrapper_applies_source_normalization_and_three_channels():
    backbone = _RecordingBackbone()
    model = SourceNormalizedResNet18(
        backbone, SourceNormalization(mean=0.25, std=0.5)
    )

    output = model(torch.full((2, 1, 4, 4), 0.75))

    assert output.shape == (2, 2)
    assert backbone.seen.shape == (2, 3, 4, 4)
    assert torch.equal(backbone.seen, torch.ones_like(backbone.seen))


def test_model_wrapper_zeroes_normalized_letterbox_padding():
    backbone = _RecordingBackbone()
    model = SourceNormalizedResNet18(
        backbone, SourceNormalization(mean=0.25, std=0.5)
    )
    image = torch.full((1, 1, 4, 4), 0.75)
    mask = torch.zeros_like(image, dtype=torch.bool)
    mask[:, :, 1:3, :] = True
    image = image.masked_fill(~mask, 0.0)

    model(image, content_mask=mask)

    expected_content = mask.expand(-1, 3, -1, -1)
    assert torch.all(backbone.seen[expected_content] == 1)
    assert torch.all(backbone.seen[~expected_content] == 0)


def test_ordinal_model_has_independent_binary_and_risk_heads():
    backbone = _RecordingBackbone()
    model = SourceNormalizedOrdinalResNet18(
        backbone, SourceNormalization(mean=0.25, std=0.5)
    )

    binary, ordinal = model.forward_with_ordinal(
        torch.full((2, 1, 4, 4), 0.75)
    )

    assert binary.shape == (2, 2)
    assert ordinal.shape == (2,)
    assert model(torch.full((2, 1, 4, 4), 0.75)).shape == (2, 2)
    assert isinstance(model.backbone.fc, nn.Identity)


def test_categorical_model_has_independent_binary_and_unordered_heads():
    backbone = _RecordingBackbone()
    model = SourceNormalizedCategoricalResNet18(
        backbone, SourceNormalization(mean=0.25, std=0.5)
    )

    binary, categorical = model.forward_with_categorical(
        torch.full((2, 1, 4, 4), 0.75)
    )

    assert binary.shape == (2, 2)
    assert categorical.shape == (2, 6)
    assert model(torch.full((2, 1, 4, 4), 0.75)).shape == (2, 2)
    assert isinstance(model.backbone.fc, nn.Identity)


def test_task3_loader_tree_is_adapted_to_task8_shape():
    train = {"A": object(), "B": object()}
    val = {"A": object(), "B": object()}

    adapted = adapt_domain_loaders(
        {"source_train": train, "source_val": val, "target_test": object()}
    )

    assert adapted == {
        "A": {"train": train["A"], "val": val["A"]},
        "B": {"train": train["B"], "val": val["B"]},
    }


@pytest.mark.parametrize("method", ["fedavg", "fedprox"])
def test_non_spectral_arms_write_empty_signature_metadata(method, tmp_path):
    signatures, metadata = build_source_signatures(
        pd.DataFrame(), tmp_path, 16, method, 2, 42
    )

    assert signatures == {}
    assert metadata == {}


@pytest.mark.parametrize(
    ("method", "expected_windows"),
    [("whole", 1), ("depth", 3), ("fedus_dg_mvp", 3)],
)
def test_spectral_arms_compute_source_train_signatures_with_required_windows(
    method, expected_windows, tmp_path
):
    records = []
    for domain_index, domain in enumerate(["A", "B"]):
        for label in (0, 1):
            path = tmp_path / f"{domain}-{label}.png"
            Image.new("L", (16, 16), 40 + 60 * label + domain_index).save(path)
            records.append(
                {
                    "dataset": domain,
                    "image_path": path.name,
                    "patient_id": f"{domain}-{label}",
                    "label": label,
                }
            )

    signatures, metadata = build_source_signatures(
        pd.DataFrame(records), tmp_path, 16, method, 2, 42
    )

    assert set(signatures) == {"A", "B"}
    assert all(
        value.mean.shape == (expected_windows, 2)
        for value in signatures.values()
    )
    assert metadata["num_windows"] == expected_windows
    assert metadata["source_split"] == "source_train"
    assert metadata["target_excluded"] is True


def test_pretrained_constructor_keeps_loaded_body_weights(monkeypatch):
    backbone = _RecordingBackbone()
    backbone.register_parameter("pretrained_body", nn.Parameter(torch.tensor([7.0])))
    seen = []

    def fake_resnet18(*, weights):
        seen.append(weights)
        return backbone

    monkeypatch.setattr(runner, "resnet18", fake_resnet18)

    model = build_pretrained_model(SourceNormalization(mean=0.1, std=0.2))

    assert seen == [runner.ResNet18_Weights.IMAGENET1K_V1]
    assert model.backbone.pretrained_body.item() == 7.0
    assert model.backbone.fc.out_features == 2


def test_pretrained_constructor_builds_ordinal_variant(monkeypatch):
    backbone = _RecordingBackbone()
    monkeypatch.setattr(runner, "resnet18", lambda *, weights: backbone)

    model = build_pretrained_model(
        SourceNormalization(mean=0.1, std=0.2), with_ordinal_head=True
    )

    assert isinstance(model, SourceNormalizedOrdinalResNet18)
    assert model.binary_head.out_features == 2
    assert model.ordinal_head.out_features == 1


def test_pretrained_constructor_builds_categorical_variant(monkeypatch):
    backbone = _RecordingBackbone()
    monkeypatch.setattr(runner, "resnet18", lambda *, weights: backbone)

    model = build_pretrained_model(
        SourceNormalization(mean=0.1, std=0.2),
        with_categorical_head=True,
    )

    assert isinstance(model, SourceNormalizedCategoricalResNet18)
    assert model.binary_head.out_features == 2
    assert model.categorical_head.out_features == 6


def test_pretrained_constructor_rejects_two_auxiliary_heads():
    with pytest.raises(ValueError, match="mutually exclusive"):
        build_pretrained_model(
            SourceNormalization(mean=0.1, std=0.2),
            with_ordinal_head=True,
            with_categorical_head=True,
        )


def test_target_predictions_preserve_manifest_row_order():
    frame = pd.DataFrame(
        [
            {"dataset": "UDIAT", "image_path": "z.png", "patient_id": "p2", "patient_id_source": "metadata", "label": 1},
            {"dataset": "UDIAT", "image_path": "a.png", "patient_id": "p1", "patient_id_source": "original_source_id", "label": 0},
        ]
    )

    predictions = build_target_predictions(
        frame,
        {"labels": [1, 0], "scores": [0.8, 0.2], "predictions": [1, 0], "n": 2},
    )

    assert predictions["manifest_row"].tolist() == [0, 1]
    assert predictions["image_path"].tolist() == ["z.png", "a.png"]
    assert predictions["score"].tolist() == [0.8, 0.2]
    assert predictions["patient_id_source"].tolist() == [
        "metadata",
        "original_source_id",
    ]
    assert predictions["patient_grouping_reliable"].tolist() == [True, False]


def test_force_publish_failure_restores_old_complete_run(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    (run / "COMPLETE").write_text("old\n", encoding="utf-8")
    (run / "old.txt").write_text("preserve\n", encoding="utf-8")
    staging = tmp_path / ".run.staging"
    staging.mkdir()
    (staging / "COMPLETE").write_text("new\n", encoding="utf-8")
    expected = runner.destination_signature(run)
    real_replace = runner.os.replace
    calls = 0

    def fail_new_publish(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("publish boom")
        return real_replace(source, destination)

    monkeypatch.setattr(runner.os, "replace", fail_new_publish)

    with pytest.raises(OSError, match="publish boom"):
        publish_staging_directory(staging, run, force=True, expected_signature=expected)

    assert (run / "COMPLETE").read_text(encoding="utf-8") == "old\n"
    assert (run / "old.txt").read_text(encoding="utf-8") == "preserve\n"


def _old_and_staged_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "COMPLETE").write_text("old\n", encoding="utf-8")
    (run / "old.txt").write_text("preserve\n", encoding="utf-8")
    staging = tmp_path / ".run.staging"
    staging.mkdir()
    (staging / "COMPLETE").write_text("new\n", encoding="utf-8")
    (staging / "new.txt").write_text("candidate\n", encoding="utf-8")
    return run, staging


def test_post_publish_validation_failure_restores_old_run(tmp_path):
    run, staging = _old_and_staged_run(tmp_path)

    def reject_published(path):
        if path == run:
            raise ValueError("published validation boom")

    with pytest.raises(ValueError, match="published validation boom"):
        publish_staging_directory(
            staging,
            run,
            force=True,
            expected_signature=runner.destination_signature(run),
            validator=reject_published,
        )

    assert (run / "old.txt").read_text(encoding="utf-8") == "preserve\n"
    assert list(tmp_path.glob(".run.backup-*")) == []


def test_remove_failure_preserves_recoverable_backup(tmp_path, monkeypatch):
    run, staging = _old_and_staged_run(tmp_path)
    real_rmtree = runner.shutil.rmtree

    def fail_failed_result_removal(path, *args, **kwargs):
        if Path(path) == run:
            raise OSError("remove boom")
        return real_rmtree(path, *args, **kwargs)

    def reject_published(path):
        if path == run:
            raise ValueError("published validation boom")

    monkeypatch.setattr(runner.shutil, "rmtree", fail_failed_result_removal)

    with pytest.raises(RuntimeError, match="recovery incomplete") as captured:
        publish_staging_directory(
            staging,
            run,
            force=True,
            expected_signature=runner.destination_signature(run),
            validator=reject_published,
        )

    backups = list(tmp_path.glob(".run.backup-*"))
    assert len(backups) == 1
    assert (backups[0] / "old.txt").read_text(encoding="utf-8") == "preserve\n"
    assert str(backups[0].resolve()) in str(captured.value)
    assert "remove boom" in str(captured.value)


def test_restore_failure_preserves_recoverable_backup(tmp_path, monkeypatch):
    run, staging = _old_and_staged_run(tmp_path)
    real_replace = runner.os.replace

    def fail_backup_restore(source, destination):
        if Path(source).name.startswith(".run.backup-") and Path(destination) == run:
            raise OSError("restore boom")
        return real_replace(source, destination)

    def reject_published(path):
        if path == run:
            raise ValueError("published validation boom")

    monkeypatch.setattr(runner.os, "replace", fail_backup_restore)

    with pytest.raises(RuntimeError, match="recovery incomplete") as captured:
        publish_staging_directory(
            staging,
            run,
            force=True,
            expected_signature=runner.destination_signature(run),
            validator=reject_published,
        )

    backups = list(tmp_path.glob(".run.backup-*"))
    assert len(backups) == 1
    assert (backups[0] / "old.txt").read_text(encoding="utf-8") == "preserve\n"
    assert not run.exists()
    assert str(backups[0].resolve()) in str(captured.value)
    assert "restore boom" in str(captured.value)


def test_minimal_audited_run_writes_complete_artifacts_and_force_replaces(
    tmp_path, monkeypatch
):
    manifest_root, data_root, bundle, _ = _bundle_fixture(tmp_path)
    output_root = tmp_path / "outputs"
    training_calls = 0

    class LightweightTrainer:
        def __init__(
            self,
            model,
            method,
            source_loaders,
            target_loader,
            config,
            signatures,
            device,
        ):
            del model, signatures, device
            assert method == "fedavg"
            assert set(source_loaders) == set(runner.APPROVED_DOMAINS) - {"UDIAT"}
            assert all(set(loaders) == {"train", "val"} for loaders in source_loaders.values())
            self.source_loaders = source_loaders
            self.target_loader = target_loader
            self.config = config

        def train(self):
            nonlocal training_calls
            training_calls += 1
            for loaders in self.source_loaders.values():
                next(iter(loaders["train"]))
                next(iter(loaders["val"]))
            labels = []
            for batch in self.target_loader:
                labels.extend(torch.as_tensor(batch["label"]).tolist())
            scores = [0.2 if label == 0 else 0.8 for label in labels]
            predictions = [int(score >= 0.5) for score in scores]
            source_metrics = {
                domain: {
                    "auc": 1.0,
                    "loss": 0.2,
                    "labels": [0, 1],
                    "scores": [0.2, 0.8],
                    "predictions": [0, 1],
                    "n": 2,
                }
                for domain in sorted(self.source_loaders)
            }
            return runner.TrainingResult(
                best_round=training_calls,
                best_state_dict={"round": torch.tensor(training_calls)},
                source_val_metrics=source_metrics,
                source_threshold=0.5,
                target_metrics={
                    "auc": 1.0,
                    "loss": 0.2,
                    "labels": labels,
                    "scores": scores,
                    "predictions": predictions,
                    "n": len(labels),
                },
                target_evaluation_count=1,
                selected_checkpoint_kind="global",
                selection_domains=tuple(sorted(self.source_loaders)),
                target_domain=self.config.target_domain,
                training_history=[
                    {
                        "round": training_calls,
                        "client_train_loss": {
                            domain: 0.2 for domain in sorted(self.source_loaders)
                        },
                        "source_validation": {
                            domain: {
                                "auc": 1.0,
                                "balanced_accuracy": 1.0,
                                "sensitivity": 1.0,
                                "specificity": 1.0,
                                "macro_f1": 1.0,
                                "loss": 0.2,
                            }
                            for domain in sorted(self.source_loaders)
                        },
                        "macro_source_auc": 1.0,
                        "macro_source_loss": 0.2,
                        "selected_checkpoint": True,
                    }
                ],
            )

    monkeypatch.setattr(runner, "FederatedMVPTrainer", LightweightTrainer)
    monkeypatch.setattr(
        runner,
        "build_pretrained_model",
        lambda normalization: nn.Identity(),
    )

    first = runner.run_experiment(
        _runner_args(manifest_root, data_root, output_root)
    )
    expected_files = set(runner._RESULT_FILES) | {"COMPLETE"}
    assert {path.name for path in first.iterdir()} == expected_files
    marker = json.loads((first / "COMPLETE").read_text(encoding="utf-8"))
    assert set(marker["artifact_sha256"]) == set(runner._RESULT_FILES)
    for filename, expected_hash in marker["artifact_sha256"].items():
        assert _sha256(first / filename) == expected_hash

    resolved = json.loads((first / "resolved_config.json").read_text(encoding="utf-8"))
    assert resolved["metadata_csv"] is None
    assert resolved["metadata_csv_status"] == "not_applicable_prebuilt_lodo"
    assert resolved["lodo_bundle_metadata"] == str((bundle / "metadata.json").resolve())
    assert resolved["lodo_manifests"] == {
        f"{name}.csv": str((bundle / f"{name}.csv").resolve())
        for name in runner._FRAME_NAMES
    }
    assert resolved["split_protocol"] == split_protocol_config("UDIAT")
    assert resolved["split_protocol_hash"] == canonical_hash(
        split_protocol_config("UDIAT")
    )
    assert len(resolved["implementation_sha256"]) == 64
    assert resolved["implementation_files"]["scripts/run_fedusdg_mvp.py"]
    assert resolved["runtime_versions"]["torch"]
    assert resolved["cublas_workspace_config"] == ":4096:8"
    metadata = json.loads((first / "run_metadata.json").read_text(encoding="utf-8"))
    assert isinstance(metadata["git_dirty"], bool)
    assert len(metadata["tracked_diff_sha256"]) == 64
    assert metadata["split_protocol"] == split_protocol_config("UDIAT")
    assert metadata["split_protocol_hash"] == canonical_hash(
        split_protocol_config("UDIAT")
    )
    assert metadata["implementation_sha256"] == resolved["implementation_sha256"]
    assert metadata["runtime_versions"] == resolved["runtime_versions"]
    assert _sha256(first / "tracked_diff.patch") == metadata["tracked_diff_sha256"]
    assert _sha256(first / "git_status.txt") == metadata["git_status_sha256"]
    with zipfile.ZipFile(first / "implementation_snapshot.zip") as archive:
        assert set(archive.namelist()) == set(resolved["implementation_files"])
        for name, expected_hash in resolved["implementation_files"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected_hash
    predictions = pd.read_csv(
        first / "target_predictions.csv",
        dtype={"patient_id": "string", "image_path": "string"},
    )
    assert predictions["manifest_row"].tolist() == [0, 1]
    assert predictions["image_path"].tolist() == [
        "UDIAT/target-0.png",
        "UDIAT/target-1.png",
    ]
    assert predictions["patient_id_source"].tolist() == ["metadata", "metadata"]
    assert predictions["patient_grouping_reliable"].tolist() == [True, True]

    (first / "old-sentinel.txt").write_text("old\n", encoding="utf-8")
    second = runner.run_experiment(
        _runner_args(manifest_root, data_root, output_root, force=True)
    )
    assert second == first
    assert not (second / "old-sentinel.txt").exists()
    second_metadata = json.loads(
        (second / "run_metadata.json").read_text(encoding="utf-8")
    )
    assert second_metadata["best_round"] == 2
    assert list(second.parent.glob(".seed_42.staging-*")) == []
    assert list(second.parent.glob(".seed_42.backup-*")) == []
