#!/usr/bin/env python3
"""Print or execute the paper LODO run matrix (dry-run is the default)."""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_fedevi_final import METHOD_ROOTS, TARGETS, SEEDS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=list(METHOD_ROOTS),
                        default=["fedevi_monotonic", "fedavg"])
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=list(TARGETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--calibrate", action="store_true",
                        help="Fit source-validation temperature after main/FedAvg runs.")
    args = parser.parse_args()
    manifest = args.manifest_root.resolve()
    data = args.data_root.resolve()
    for method in args.methods:
        for target in args.targets:
            for seed in args.seeds:
                out = METHOD_ROOTS[method]
                run_dir = ROOT / out / f"target_{target}" / method / f"seed_{seed}"
                command = [sys.executable, "scripts/run_fedevi.py",
                           "--manifest-root", str(manifest), "--data-root", str(data),
                           "--target", target, "--method", method, "--seed", str(seed),
                           "--rounds", "30", "--device", args.device,
                           "--output-root", str(out), "--require-clean-git"]
                print(shlex.join(command), flush=True)
                if args.execute and not (run_dir / "COMPLETE").is_file():
                    subprocess.run(command, cwd=ROOT, check=True)
                if args.calibrate and method in {"fedevi_monotonic", "fedavg"}:
                    name = ("fedevi_monotonic_gate_calibration" if method ==
                            "fedevi_monotonic" else "fedavg_gate_calibration")
                    calibration_dir = ROOT / "outputs" / name / f"target_{target}" / method / f"seed_{seed}"
                    calibration = [sys.executable, "scripts/calibrate_source_only.py",
                                   "--run-dir", str(run_dir), "--output-dir", str(calibration_dir)]
                    print(shlex.join(calibration), flush=True)
                    if args.execute and not (calibration_dir / "calibration.json").is_file():
                        subprocess.run(calibration, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
