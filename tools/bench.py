#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import statistics
import subprocess
import sys
from pathlib import Path


SCORE_RE = re.compile(r"Score\s*=\s*(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the local AHC025 tester on multiple inputs and summarize scores."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("tools/in"),
        help="Directory containing input files (default: tools/in)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N inputs after sorting by filename",
    )
    parser.add_argument(
        "--solver",
        type=Path,
        default=Path("target/release/ahc025"),
        help="Solver binary path (default: target/release/ahc025)",
    )
    parser.add_argument(
        "--tester",
        type=Path,
        default=Path("tools/target/release/tester"),
        help="Tester binary path (default: tools/target/release/tester)",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip cargo build steps and use existing binaries",
    )
    return parser.parse_args()


def run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def ensure_binaries(root: Path, solver: Path, tester: Path, skip_build: bool) -> None:
    if skip_build and solver.exists() and tester.exists():
        return

    run(["cargo", "build", "--release", "--quiet"], cwd=root)
    run(["cargo", "build", "--release", "--quiet"], cwd=root / "tools")


def parse_score(stderr: str, input_path: Path) -> int:
    match = SCORE_RE.search(stderr)
    if match is None:
        raise RuntimeError(f"failed to parse score for {input_path.name}:\n{stderr}")
    return int(match.group(1))


def collect_inputs(input_dir: Path, limit: int | None) -> list[Path]:
    inputs = sorted(input_dir.glob("*.txt"))
    if limit is not None:
        inputs = inputs[:limit]
    if not inputs:
        raise RuntimeError(f"no input files found in {input_dir}")
    return inputs


def run_case(tester: Path, solver: Path, input_path: Path) -> int:
    with input_path.open("rb") as infile:
        completed = subprocess.run(
            [str(tester), str(solver)],
            stdin=infile,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"tester failed for {input_path.name} with exit code {completed.returncode}:\n"
            f"{completed.stderr}"
        )
    return parse_score(completed.stderr, input_path)


def summarize(scores: list[int]) -> dict[str, float]:
    ordered = sorted(scores)
    return {
        "count": float(len(scores)),
        "best": float(min(scores)),
        "median": float(statistics.median(ordered)),
        "mean": float(sum(scores) / len(scores)),
        "worst": float(max(scores)),
    }


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    input_dir = (root / args.input_dir).resolve()
    solver = (root / args.solver).resolve()
    tester = (root / args.tester).resolve()

    ensure_binaries(root, solver, tester, args.skip_build)
    inputs = collect_inputs(input_dir, args.limit)

    print("lower score is better")
    print(f"inputs={len(inputs)} solver={solver.relative_to(root)}")

    scores: list[int] = []
    for input_path in inputs:
        score = run_case(tester, solver, input_path)
        scores.append(score)
        print(f"{input_path.stem}\t{score}")

    summary = summarize(scores)
    print("---")
    print(f"count  : {int(summary['count'])}")
    print(f"best   : {int(summary['best'])}")
    print(f"median : {summary['median']:.1f}")
    print(f"mean   : {summary['mean']:.2f}")
    print(f"worst  : {int(summary['worst'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
