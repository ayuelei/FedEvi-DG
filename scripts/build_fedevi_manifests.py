#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

import pandas as pd
from PIL import Image

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from fedusdg_mvp.manifest import APPROVED_DOMAINS
from fedusdg_mvp.provenance import atomic_write_json


FRAME_NAMES = ("source_train", "source_val", "target_test")
MASK_DATASETS = frozenset({"BUSBRA", "BUSI_WHU", "UDIAT"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_mask_path(dataset: str, image_path: Path) -> Path | None:
    if dataset == "BUSBRA":
        return (
            image_path.parent.parent
            / "Masks"
            / f"mask_{image_path.stem.removeprefix('bus_')}{image_path.suffix}"
        )
    if dataset == "BUSI_WHU":
        return (
            image_path.parent.parent
            / "gt"
            / f"{image_path.stem}_anno{image_path.suffix}"
        )
    if dataset == "UDIAT":
        return image_path.parent.parent / "GT" / image_path.name
    return None


def _attach_masks(
    frame: pd.DataFrame,
    hash_cache: dict[Path, str],
    validated_pairs: set[tuple[Path, Path]],
) -> pd.DataFrame:
    result = frame.copy()
    mask_paths: list[str] = []
    mask_hashes: list[str] = []
    mask_sources: list[str] = []
    for row in result.itertuples(index=False):
        dataset = str(row.dataset)
        image_path = Path(str(row.image_path)).resolve()
        mask_path = resolve_mask_path(dataset, image_path)
        if mask_path is None:
            mask_paths.append("")
            mask_hashes.append("")
            mask_sources.append("")
            continue
        mask_path = mask_path.resolve()
        if not mask_path.is_file():
            raise FileNotFoundError(f"missing {dataset} lesion mask: {mask_path}")
        pair = (image_path, mask_path)
        if pair not in validated_pairs:
            with Image.open(image_path) as image, Image.open(mask_path) as mask:
                if image.size != mask.size:
                    raise ValueError(
                        f"mask/image size mismatch: {mask_path} vs {image_path}"
                    )
                extrema = mask.convert("L").getextrema()
                if extrema is None or extrema[1] <= 0:
                    raise ValueError(f"empty lesion mask: {mask_path}")
            validated_pairs.add(pair)
        if mask_path not in hash_cache:
            hash_cache[mask_path] = _sha256(mask_path)
        mask_paths.append(str(mask_path))
        mask_hashes.append(hash_cache[mask_path])
        mask_sources.append("source_pixel_annotation")
    result["mask_path"] = mask_paths
    result["mask_sha256"] = mask_hashes
    result["mask_source"] = mask_sources
    return result


def _class_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame["label"].astype(int).value_counts().sort_index()
    return {str(label): int(count) for label, count in counts.items()}


def build_bundle(
    source: Path,
    destination: Path,
    hash_cache: dict[Path, str] | None = None,
    validated_pairs: set[tuple[Path, Path]] | None = None,
) -> None:
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    hash_cache = {} if hash_cache is None else hash_cache
    validated_pairs = set() if validated_pairs is None else validated_pairs
    try:
        evidence_counts: dict[str, dict[str, int]] = {}
        for frame_name in FRAME_NAMES:
            frame = pd.read_csv(source / f"{frame_name}.csv")
            frame = _attach_masks(frame, hash_cache, validated_pairs)
            csv_path = staging / f"{frame_name}.csv"
            frame.to_csv(csv_path, index=False)
            metadata["files"][csv_path.name] = {
                "rows": len(frame),
                "class_counts": _class_counts(frame),
                "sha256": _sha256(csv_path),
            }
            evidence_counts[frame_name] = {
                str(domain): int(group["mask_path"].fillna("").ne("").sum())
                for domain, group in frame.groupby("dataset", sort=True)
            }
        metadata["evidence_schema_version"] = 1
        metadata["lesion_mask_datasets"] = sorted(MASK_DATASETS)
        metadata["lesion_mask_counts"] = evidence_counts
        metadata["lesion_mask_integrity"] = {
            "path_exists": True,
            "image_size_match": True,
            "non_empty": True,
            "mask_sha256_in_manifest": True,
        }
        atomic_write_json(staging / "metadata.json", metadata)
        atomic_write_json(
            staging / "COMPLETE",
            {"metadata_sha256": _sha256(staging / "metadata.json")},
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Attach verified lesion masks to frozen FedEvi-DG LODO bundles."
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    if args.output_root.exists():
        raise FileExistsError(f"output root already exists: {args.output_root}")
    args.output_root.mkdir()
    hash_cache: dict[Path, str] = {}
    validated_pairs: set[tuple[Path, Path]] = set()
    try:
        for target in APPROVED_DOMAINS:
            build_bundle(
                args.source_root / f"target_{target}",
                args.output_root / f"target_{target}",
                hash_cache,
                validated_pairs,
            )
    except Exception:
        shutil.rmtree(args.output_root)
        raise


if __name__ == "__main__":
    main()
