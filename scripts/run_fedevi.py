#!/usr/bin/env python3
"""Paper-scoped CLI delegating to the unchanged federated runner."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import train_fedevi as runner

PAPER_METHODS = (
    "fedevi_monotonic",
    "fedavg",
    "fedevi_mtl",
    "fedevi_seg_only",
    "fedavg_ord",
    "fedevi_shuffle",
    "fedevi_mask_shift",
    "fedevi_dg",
    "fedevi_no_taskagg",
)


def main(argv=None) -> int:
    parser = runner.build_parser()
    parser.description = "Run one FedEvi-DG paper experiment (LODO or locked external)."
    for action in parser._actions:
        if action.dest == "method":
            action.choices = PAPER_METHODS
    argv = sys.argv[1:] if argv is None else argv
    parser.parse_args(argv)
    return runner.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
