#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from fedusdg_mvp.config import ExperimentConfig
from fedusdg_mvp.manifest import (
    APPROVED_DOMAINS,
    apply_near_duplicate_dedup,
    audit_master_manifest,
    build_lodo_frames,
    find_manifest_near_duplicates,
    preflight_lodo_bundle,
    write_lodo_bundles,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build audited, leakage-safe five-domain LODO manifests."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument(
        "--output-root", type=Path, default=Path("splits/fedusdg_mvp_lodo")
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit_config = ExperimentConfig(
        data_root=args.data_root,
        metadata_csv=args.metadata_csv,
        target_domain=APPROVED_DOMAINS[0],
    )
    master = audit_master_manifest(audit_config)
    master, dedup_decisions = apply_near_duplicate_dedup(master)
    candidate_path = args.output_root / "near_duplicate_candidates.csv"
    candidates = find_manifest_near_duplicates(
        master, max_hamming_distance=4
    )
    unresolved = candidates[~candidates["resolved"].astype(bool)]
    if not unresolved.empty:
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        candidates.to_csv(candidate_path, index=False)
        raise ValueError(
            "unresolved manifest near duplicate candidates; inspect "
            f"{candidate_path}:\n{unresolved.to_csv(index=False)}"
        )

    bundles = []
    for target_domain in APPROVED_DOMAINS:
        config = ExperimentConfig(
            data_root=args.data_root,
            metadata_csv=args.metadata_csv,
            target_domain=target_domain,
        )
        frames = build_lodo_frames(
            master,
            target_domain=target_domain,
            split_seed=config.split_seed,
            val_fraction=config.val_fraction,
        )
        output_dir = args.output_root / f"target_{target_domain}"
        bundles.append((frames, output_dir, config))

    # Every split and every destination must pass before the first bundle is
    # published, so a late target cannot leave an apparently valid prefix.
    for frames, output_dir, config in bundles:
        preflight_lodo_bundle(frames, output_dir, config, force=args.force)

    args.output_root.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(candidate_path, index=False)
    dedup_decisions.to_csv(
        args.output_root / "near_duplicate_dedup_decisions.csv", index=False
    )
    write_lodo_bundles(bundles, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
