# -*- coding: utf-8 -*-

import argparse
import json
import re
import sys
from collections import defaultdict
from io import StringIO
from pathlib import Path
import time

import ollama

SCRIPT_DIR = Path(__file__).resolve().parent
################mixed cases##################
#INFERENCE_RESULTS_DIR = SCRIPT_DIR / "inference_results/naive"
#INFERENCE_RESULTS_DIR = SCRIPT_DIR / "inference_results/fewshot"

################besides edge##################
INFERENCE_RESULTS_DIR = SCRIPT_DIR / "generalization_results/besides_edge/naive"
#INFERENCE_RESULTS_DIR = SCRIPT_DIR / "generalization_results/besides_edge/fewshot"

#################plain ground#################
#INFERENCE_RESULTS_DIR = SCRIPT_DIR / "generalization_results/plain_ground/naive"
#INFERENCE_RESULTS_DIR = SCRIPT_DIR / "generalization_results/plain_ground/fewshot"

#################cleared edge#################
#INFERENCE_RESULTS_DIR = SCRIPT_DIR / "generalization_results/cleared_edge/naive"
#INFERENCE_RESULTS_DIR = SCRIPT_DIR / "generalization_results/cleared_edge/fewshot"

MODEL = "qwen3-vl:8b-instruct"
SCREENSHOT_PATH = Path("compare=naive2few_shot/besides_edges/")
GROUND_TRUTH_FILE = SCREENSHOT_PATH / "ground_truth.txt"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

PROMPT = """
You are playing a 2D side-scrolling platformer game independently, your objective is to move to the right as far as possible, without falling into any gaps.

Your character is a white cat centered in the screen.
Use keys "a" to move left and "d" to move right, 
"space" to jump (can be combined with "a" or "d" to jump left or right).

You will be given game screenshots showing the current game state, then reason step by step for a reasonable and efficient action to take:
1) Evaluate the given game screenshot, use a few sentences to describe the position of yourself and relative position to any gaps or obstacles.

2) After evaluating the given game screenshot, decide what kind of action from the following, in strict JSON format:
{
    "action": "hold_press|no_action",
    "keys": ["key1", "key2"],
    "reason": "reason"
}

hold_press: specify two keys, hold both keys for a period of time.
no_action: do nothing.
"""


#  !!!These examples can come from the model's former successful memories.
REASONING_EXAMPLES = """
=======Reasoning Examples=======
a) The scene shows the cat at the left-side edge of a gap, I should jump right by holding "d" and "space" at the same time to get over the gap.
c) The cat is on a plain platform and no gaps or obstacles in front, I should press "d" key to move right and explore further into the level.
d) The scene is not clear, I should not take any action and wait for a logical scene.
"""


def load_ground_truth(path: Path) -> dict[str, dict]:
    if not path.is_file():
        print(f"Ground truth file not found: {path.resolve()}")
        return {}
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Ground truth root must be a JSON object, got {type(data)}")
    return {str(k): v for k, v in data.items() if isinstance(v, dict)}


def parse_json_from_model_text(raw: str) -> dict | None:
    raw = raw.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    text = m.group(1).strip() if m else raw
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _norm_key(k: str) -> str:
    return str(k).strip().lower()


def _norm_keys_list(keys) -> tuple[str, ...] | None:
    if keys is None:
        return None
    if not isinstance(keys, list):
        return None
    out = [_norm_key(x) for x in keys]
    return tuple(sorted(out))


def action_matches(expected: dict, actual: dict) -> bool:
    ea = str(expected.get("action", "")).strip().lower()
    aa = str(actual.get("action", "")).strip().lower()
    if ea != aa:
        return False
    if ea == "no_action":
        return True
    ek = _norm_keys_list(expected.get("keys"))
    ak = _norm_keys_list(actual.get("keys"))
    if ek is None or ak is None:
        return False
    if ek != ak:
        return False
    return True


def collect_image_paths() -> list[Path]:
    if not SCREENSHOT_PATH.is_dir():
        print(f"Not a directory: {SCREENSHOT_PATH.resolve()}")
        return []
    return sorted(
        p
        for p in SCREENSHOT_PATH.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def next_inference_results_path() -> Path:
    INFERENCE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    max_n = 0
    for p in INFERENCE_RESULTS_DIR.iterdir():
        if p.is_file() and p.suffix.lower() == ".txt" and p.stem.isdigit():
            max_n = max(max_n, int(p.stem))
    return INFERENCE_RESULTS_DIR / f"{max_n + 1}.txt"


def main():
    parser = argparse.ArgumentParser(description="Vision model screenshot comparison / accuracy test.")
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        metavar="N",
        help="Run the full image set N times (default: 1). Used to estimate average accuracy.",
    )
    parser.add_argument(
        "--answers",
        type=Path,
        default=GROUND_TRUTH_FILE,
        help=f"TXT file containing JSON map: filename -> expected action object (default: {GROUND_TRUTH_FILE})",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        print("--repeats must be >= 1", file=sys.stderr)
        sys.exit(1)

    paths = collect_image_paths()
    if not paths:
        print(f"No image files found in {SCREENSHOT_PATH.resolve()}")
        return

    try:
        ground_truth = load_ground_truth(args.answers)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Invalid ground truth file {args.answers}: {e}", file=sys.stderr)
        sys.exit(1)

    out_path = next_inference_results_path()
    body_chunks: list[str] = []

    def log_print(*args, sep=" ", end="\n", flush=False):
        print(*args, sep=sep, end=end, flush=flush)
        buf = StringIO()
        print(*args, sep=sep, end=end, file=buf)
        body_chunks.append(buf.getvalue())

    print(f"Run log file: {out_path.resolve()}", file=sys.stderr)

    graded = 0
    correct = 0
    per_file_correct: dict[str, int] = defaultdict(int)
    per_file_total: dict[str, int] = defaultdict(int)

    for rep in range(args.repeats):
        for path in paths:
            name = path.name
            expected = ground_truth.get(name)

            resp = ollama.chat(
                model=MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": PROMPT,
                        "images": [str(path)],
                    }
                ],
            )
            raw = (resp["message"]["content"] or "").strip()
            log_print("=" * 10 + f"Screenshot {name}" + (f" (repeat {rep + 1}/{args.repeats})" if args.repeats > 1 else "") + "=" * 10)
            log_print(raw)

            if expected is None:
                log_print("(no ground truth entry for this file — skipped in accuracy)")
            else:
                actual = parse_json_from_model_text(raw)
                if actual is None:
                    ok = False
                    log_print("GRADE: could not parse JSON from model output -> incorrect")
                else:
                    ok = action_matches(expected, actual)
                    log_print("GRADE:", "correct" if ok else "incorrect")
                    if not ok:
                        log_print("expected:", json.dumps(expected, ensure_ascii=False))
                        log_print("parsed:  ", json.dumps(actual, ensure_ascii=False))
                graded += 1
                per_file_total[name] += 1
                if ok:
                    correct += 1
                    per_file_correct[name] += 1

            time.sleep(0.05)

    summary_lines: list[str] = []
    if graded:
        pct = 100.0 * correct / graded
        summary_lines.append("Per-image accuracy (this run):")
        for name in sorted(per_file_total.keys()):
            t = per_file_total[name]
            c = per_file_correct[name]
            summary_lines.append(f"  {name}: {c}/{t} correct ({100.0 * c / t:.1f}%)")
        summary_lines.append("")
        summary_lines.append(f"Run accuracy (all graded): {correct}/{graded} correct ({pct:.1f}%)")
    elif ground_truth:
        summary_lines.append(
            "No images were graded (add matching keys in ground truth for your filenames)."
        )

    summary_text = ("\n".join(summary_lines) + "\n") if summary_lines else ""
    print()
    print(summary_text, end="")
    out_path.write_text(summary_text + "".join(body_chunks), encoding="utf-8")


if __name__ == "__main__":
    main()
