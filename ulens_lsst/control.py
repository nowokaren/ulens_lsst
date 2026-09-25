"""Command-line controls for independent image-processing runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from .image_processing_run import process_work_unit
from .run_state import finalize, pending_work_units, status_summary


def _print_status(run_dir: Path) -> None:
    summary = status_summary(run_dir)
    print(f"Run: {run_dir}")
    print("Work units")
    for key in (
        "total_work_units",
        "completed",
        "running",
        "pending",
        "failed",
        "stale",
        "invalid",
        "empty_success",
    ):
        print(f"  {key:20s} {summary[key]}")
    print(f"Images: {summary['images_completed']}/{summary['images_expected']}")
    print(f"Photometry rows: {summary['photometry_rows']}")
    if summary["mean_work_unit_seconds"] is not None:
        print(f"Mean work-unit seconds: {summary['mean_work_unit_seconds']:.2f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Control an ulens_lsst image-processing run.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    process_parser = subparsers.add_parser("process", help="process one work unit")
    process_parser.add_argument("run_dir", type=Path)
    process_parser.add_argument("work_unit_id")
    process_parser.add_argument("--force", action="store_true")

    status_parser = subparsers.add_parser("status", help="summarize operational state")
    status_parser.add_argument("run_dir", type=Path)

    restart_parser = subparsers.add_parser("restart-plan", help="list work units needing processing")
    restart_parser.add_argument("run_dir", type=Path)

    finalize_parser = subparsers.add_parser("finalize", help="validate campaign completeness")
    finalize_parser.add_argument("run_dir", type=Path)

    args = parser.parse_args(argv)
    if args.command == "process":
        process_work_unit(args.run_dir, args.work_unit_id, force=args.force)
        return 0
    if args.command == "status":
        _print_status(args.run_dir)
        return 0
    if args.command == "restart-plan":
        for work_unit_id in pending_work_units(args.run_dir):
            print(work_unit_id)
        return 0
    result = finalize(args.run_dir)
    _print_status(args.run_dir)
    return 0 if result["complete"] else 1
