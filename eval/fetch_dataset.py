"""
Fetch the sample-resolved dataset using the prophet CLI.

The dataset is 26 events with known ground-truth outcomes — exactly what
we need to compute a Brier score and validate our agent before submission.

Usage:
    python -m eval.fetch_dataset
    python -m eval.fetch_dataset --dataset sample-sports  # different dataset
    python -m eval.fetch_dataset --output eval/data/custom.json

Requires the `prophet` CLI to be installed:
    pip install ai-prophet
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_DATASET = "sample-resolved"
DEFAULT_OUTPUT = "eval/data/events_resolved.json"


def fetch_dataset(dataset: str, output_path: str, include_resolved: bool = True) -> int:
    """
    Run `prophet forecast retrieve` to fetch a dataset to local JSON.

    Returns the number of events fetched, or raises on error.
    """
    # Ensure the prophet CLI is available
    if shutil.which("prophet") is None:
        raise RuntimeError(
            "The `prophet` CLI is not installed or not on PATH.\n"
            "Install it with: pip install ai-prophet"
        )

    # Ensure output directory exists
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "prophet", "forecast", "retrieve",
        "--dataset", dataset,
        "-o", str(out),
    ]
    if include_resolved:
        cmd.append("--include-resolved")

    # Force UTF-8 I/O so Windows cp1252 doesn't crash when the CLI prints
    # non-ASCII characters (e.g. team names with accented letters).
    cli_env = os.environ.copy()
    cli_env["PYTHONIOENCODING"] = "utf-8"

    print(f"[fetch] running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=cli_env,
    )

    if result.returncode != 0:
        # Surface the CLI's stderr so the user can see what went wrong
        print(f"[fetch] CLI exited with code {result.returncode}")
        if result.stdout:
            print(f"  stdout: {result.stdout}")
        if result.stderr:
            print(f"  stderr: {result.stderr}")
        # On Windows the CLI sometimes crashes after writing the file
        # (UnicodeEncodeError on the summary print).  If the output file
        # exists and is valid JSON we treat that as success with a warning.
        if out.exists():
            print("[fetch] WARNING: CLI exited non-zero but output file exists — treating as success.")
        else:
            raise RuntimeError("prophet CLI invocation failed; see output above.")

    # Read the output and tell the caller how many events we got
    with open(out, "r", encoding="utf-8") as f:
        events = json.load(f)

    if not isinstance(events, list):
        raise RuntimeError(f"Unexpected output format: top level is {type(events).__name__}")

    print(f"[fetch] wrote {len(events)} events to {out}")

    # Light sanity check: report how many have resolved outcomes
    n_resolved = sum(1 for e in events if e.get("resolved_outcome"))
    print(f"[fetch] {n_resolved} of {len(events)} events have resolved_outcome (needed for scoring)")

    if n_resolved == 0:
        print(
            "[fetch] WARNING: no resolved outcomes. Did you forget --include-resolved? "
            "Or is this an unresolved dataset?"
        )

    return len(events)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch a Prophet Arena dataset to local JSON.")
    parser.add_argument(
        "--dataset", default=DEFAULT_DATASET,
        help=f"Dataset name (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--output", "-o", default=DEFAULT_OUTPUT,
        help=f"Output JSON file path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--no-include-resolved", action="store_true",
        help="Do NOT pass --include-resolved to the CLI (only fetch open events).",
    )
    args = parser.parse_args()

    try:
        fetch_dataset(
            dataset=args.dataset,
            output_path=args.output,
            include_resolved=not args.no_include_resolved,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[fetch] ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
