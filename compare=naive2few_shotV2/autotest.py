#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run a predefined list of terminal commands sequentially.

Usage:
1) Edit COMMANDS below (each item is a full command string as typed in terminal).
2) Run: python autotest.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON_EXE = sys.executable
TEXTUAL_SCRIPT = SCRIPT_DIR / "V2Compare=naive2few_shot.py"
MULTIMODAL_SCRIPT = SCRIPT_DIR / "MultimodalV2Compare=naive2few_shot.py"

# Edit this list directly: each item should be a full terminal command string.
# These defaults are examples; replace/add/remove as needed.
COMMANDS: list[str] = [
    #f'"{PYTHON_EXE}" "{TEXTUAL_SCRIPT}" --scenario besides_edges --variant no_cot --repeats 20',
    #f'"{PYTHON_EXE}" "{TEXTUAL_SCRIPT}" --scenario plain_grounds --variant no_cot --repeats 20',
    #f'"{PYTHON_EXE}" "{TEXTUAL_SCRIPT}" --scenario besides_edges --variant naive --repeats 50',
    #f'"{PYTHON_EXE}" "{TEXTUAL_SCRIPT}" --scenario besides_edges --variant fewshot_balanced --repeats 50',
    #f'"{PYTHON_EXE}" "{TEXTUAL_SCRIPT}" --scenario besides_edges --variant fewshot_narrow --repeats 50',
    #f'"{PYTHON_EXE}" "{TEXTUAL_SCRIPT}" --scenario besides_edges --variant fewshot_walk_bias --repeats 50',
    #f'"{PYTHON_EXE}" "{TEXTUAL_SCRIPT}" --scenario besides_edges --variant fewshot_jump_bias --repeats 50',
    #f'"{PYTHON_EXE}" "{TEXTUAL_SCRIPT}" --scenario plain_grounds --variant naive --repeats 50',
    #f'"{PYTHON_EXE}" "{TEXTUAL_SCRIPT}" --scenario plain_grounds --variant fewshot_balanced --repeats 50',
    #f'"{PYTHON_EXE}" "{TEXTUAL_SCRIPT}" --scenario plain_grounds --variant fewshot_narrow --repeats 50',
    #f'"{PYTHON_EXE}" "{TEXTUAL_SCRIPT}" --scenario plain_grounds --variant fewshot_walk_bias --repeats 50',
    f'"{PYTHON_EXE}" "{TEXTUAL_SCRIPT}" --scenario plain_grounds --variant fewshot_jump_bias --repeats 50',


    f'"{PYTHON_EXE}" "{MULTIMODAL_SCRIPT}" --scenario besides_edges --variant multifewshot --fewshot-base narrow --repeats 50',
    f'"{PYTHON_EXE}" "{MULTIMODAL_SCRIPT}" --scenario plain_grounds --variant multifewshot --fewshot-base narrow --repeats 50',
    f'"{PYTHON_EXE}" "{MULTIMODAL_SCRIPT}" --scenario besides_edges --variant multifewshot --fewshot-base balanced --repeats 50',
    f'"{PYTHON_EXE}" "{MULTIMODAL_SCRIPT}" --scenario plain_grounds --variant multifewshot --fewshot-base balanced --repeats 50',
    f'"{PYTHON_EXE}" "{MULTIMODAL_SCRIPT}" --scenario besides_edges --variant multifewshot --fewshot-base walk_bias --repeats 50',
    f'"{PYTHON_EXE}" "{MULTIMODAL_SCRIPT}" --scenario plain_grounds --variant multifewshot --fewshot-base walk_bias --repeats 50',
    f'"{PYTHON_EXE}" "{MULTIMODAL_SCRIPT}" --scenario besides_edges --variant multifewshot --fewshot-base jump_bias --repeats 50',
    f'"{PYTHON_EXE}" "{MULTIMODAL_SCRIPT}" --scenario plain_grounds --variant multifewshot --fewshot-base jump_bias --repeats 50',
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run command strings in COMMANDS sequentially."
    )
    parser.add_argument(
        "--workdir",
        default=str(SCRIPT_DIR),
        help="Working directory for executing commands (default: script directory).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Delay in seconds between invocations.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately when one invocation fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing.",
    )
    args = parser.parse_args()

    if args.delay < 0:
        print("--delay must be >= 0", file=sys.stderr)
        return 2

    commands = [cmd.strip() for cmd in COMMANDS if cmd.strip()]
    if not commands:
        print("COMMANDS is empty. Edit autotest.py and add at least one command string.")
        return 1

    workdir = Path(args.workdir).resolve()
    if not workdir.is_dir():
        print(f"--workdir is not a directory: {workdir}", file=sys.stderr)
        return 2

    total = len(commands)
    print(f"Planned commands: {total}")
    print(f"Workdir: {workdir}")
    failures = 0

    for idx, command in enumerate(commands, start=1):
        print(f"[{idx}/{total}] {command}")

        if args.dry_run:
            continue

        result = subprocess.run(command, cwd=str(workdir), shell=True, check=False)
        if result.returncode != 0:
            failures += 1
            print(f"  !! failed with exit code {result.returncode}")
            if args.stop_on_error:
                print("Stopping due to --stop-on-error.")
                return result.returncode
        if args.delay > 0 and idx < total:
            time.sleep(args.delay)

    if failures:
        print(f"Completed with failures: {failures}/{total}")
        return 1
    print(f"All commands completed successfully: {total}/{total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
