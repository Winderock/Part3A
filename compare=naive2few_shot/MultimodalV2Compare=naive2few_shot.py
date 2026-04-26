# -*- coding: utf-8 -*-
"""
V2 compare runner (conversation-era behavior):
- single query image per call, content=PROMPT only (REASONING_EXAMPLES string kept for reference)
- per-image + run accuracy, summary block at top of saved log file

Scenarios are discovered under train_cases/ (default) or test_cases/ (with --test).
Logs are written to train_cases/train_output/<scenario>/<variant>/ or
test_cases/test_output/<scenario>/<variant>/.
"""

from __future__ import annotations

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
TRAIN_CASES_ROOT = SCRIPT_DIR / "train_cases"
TEST_CASES_ROOT = SCRIPT_DIR / "test_cases"
TRAIN_OUTPUT_ROOT = TRAIN_CASES_ROOT / "train_output"
TEST_OUTPUT_ROOT = TEST_CASES_ROOT / "test_output"
SKIP_SCENARIO_DIRS = {"train_output", "test_output"}

MODEL = "qwen3-vl:8b-instruct"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

PROMPT = """
You are playing a 2D side-scrolling platformer game independently, and you will need to provide a sequence of actions based on your observations of the game state.
====================game rules====================
Your objective is to move to the right side of the level as far as possible, jump over any gaps you encounter.
Your character is a white cat centered in the screen, the background is light blue, and the ground's texture is dark blue bricks, use the difference in texture to identify between solid ground and gaps.

==============Action Schema==============
use keys "a" to move left and "d" to move right, 
"space" to jump 
("space" can be combined with "a" or "d" to jump left or right, this is useful when you are about to cross a gap).

You will be given game screenshots showing the current game state, then reason step by step for a reasonable and efficient sequence of actions to take:

1) Evaluate the given game screenshot, in "observation" field, use a few sentences to describe key information such as position of yourself and any obstacles.
2) After evaluating the given game screenshot, decide a sequence of actions to take from the following and explain the reason in "reason" field, return the result in strict JSON format:
{
    "observation": "your observation of the game state, include spatial information of your position relative to any obstacles.",
    "sequence": [
        {
            "action": "hold|no_action",
            "keys": ["a", "d", "space"]
        }
    ],
    "reason": "your reasoning for the sequence of actions"
}

sequence: list of actions that you will take in order, one by one, each action last for a very short period of time, but do not plan ahead too much, include the currently immediate action actions only.

allowed actions:
    "hold": specify one or many keys, hold all keys at the same time, for a short period of time.
    "no_action": do nothing.
"""


#a
JUMP1 = """
{
    "observation": "The character, a white cat, is positioned on the left platform. There is a visible gap between the left platform and the right platform, which needs to be jumped over to proceed further right.",
    "sequence": [
        {
            "action": "hold",
            "keys": ["d", "space"]
            
        }
    ],
    "reason": "To cross the gap between the platforms, the character needs to jump forward. Pressing 'd' to move right while simultaneously pressing 'space' to jump will allow the character to leap across the gap efficiently."
}
"""

#b
JUMP2 = """
{
    "observation": "The character is positioned at the edge of a platform on the left, with a gap to the right. There is another platform on the right side that is at the same height. The goal is to move right and cross the gap to reach the right platform.",
    "sequence": [
        {
            "action": "hold",
            "keys": ["d", "space"]
            
        }
    ],
    "reason": "To cross the gap and reach the right platform, I need to jump to the right. Holding 'd' to move right while pressing 'space' to jump will allow me to propel myself across the gap and land on the right platform. This is the most direct and efficient way to progress."
}
"""

#c
JUMP3 = """
{
    "observation": "The character is positioned on the left side of the screen, standing on a solid platform made of dark blue bricks. To the right, there is a gap (empty blue space) that needs to be jumped over to reach the next platform on the right side.",
    "sequence": [
        {
            "action": "hold",
            "keys": ["d", "space"]
            
        }
    ],
    "reason": "To cross the gap, I need to jump to the right. Holding 'd' will move the character right, and pressing 'space' will initiate a jump. Combining them allows the character to jump while moving right, which is necessary to clear the gap and continue to the right side of the level."
}
"""

#d
JUMP4 = """
{
    "observation": "The character is positioned on the left platform, facing right. There is a visible gap between the left and right platforms. The goal is to cross this gap by jumping to the right platform.",
    "sequence": [
        {
            "action": "hold",
            "keys": ["d", "space"]
            
        }
    ],
    "reason": "To cross the gap, I need to jump to the right platform. Pressing 'd' will move the character right, and pressing 'space' will initiate a jump. Combining both keys will allow the character to jump to the right platform effectively."
}
"""

#a
WALK1 = """
{
    "observation": "The character is positioned on the right edge of the visible ground platform, facing right. The ground is dark blue brick texture, and the background is light blue sky. There are no visible gaps or obstacles immediately ahead, so moving right should be straightforward.",
    "sequence": [
        {
            "action": "hold",
            "keys": ["d"]
            
        }
    ],
    "reason": "Since the character is already at the right edge of the current platform and there are no gaps ahead, the most efficient action is to move right to continue progressing toward the right side of the level."
}
"""

#b
WALK2 = """
{
    "observation": "The player character, a white cat, is standing on a platform made of dark blue bricks, the solid ground extends to the right. There are no visible gaps to the right.",
    "sequence": [
        {
            "action": "hold",
            "keys": ["d"]
            
        }
    ],
    "reason": "The character is positioned on the platform and needs to move right to reach the goal. Since no gaps are to the right, the immediate action is to hold 'd' to move right along the platform."
}
"""

#c
WALK3 = """
{
    "observation": "The player character, a white cat, is standing on the right side of a gap, indicating that the character has just jumped over the gap, the dark blue bricks extends further to the right, which is walkable solid ground.",
    "sequence": [
        {
            "action": "hold",
            "keys": ["d"]
            
        }
    ],
    "reason": "A gap is on the left of the cat, but it is ignorable since I just jumped over it, and there are no immediate obstacles to the right, I should hold 'd' key to move right to continue progressing toward the right side of the level."
}
"""

#d
WALK4 = """
{
    "observation": "The player character, a white cat, is positioned near the center of the screen on a solid brick platform. To the right, there is a refrigerator on the platform, and beyond that, there appears to be no visible obstacles. The goal is to move right as far as possible, so I need to move right, and potientially approach the refrigerator.",
    "sequence": [
        {
            "action": "hold",
            "keys": ["d"]
            
        }
    ],
    "reason": "Since no obstacles or gaps are to my right, I should hold 'd' key to move right to continue progressing toward the right side of the level."
}
"""

#a
FAR_EDGE1 = """
{
    "observation": "The player character is positioned on a solid platform made of dark blue bricks, centered on the screen. There is no visible gap directly ahead, but the right edge of the screen shows a slight drop-off, indicating the platform ends. The objective is to move right as far as possible, which means continuing to move right until reaching the end of the platform.",
    "sequence": [
        {
            "action": "hold",
            "keys": ["d"]
        }
    ],
    "reason": "Since there is no gap to jump over immediately, the most direct way to progress is to hold the 'd' key to move right across the platform. This will allow the character to reach the rightmost point of the current platform."
}
"""

#b
FAR_EDGE2 = """
{
    "observation": "The player character, a white cat, is standing on a platform made of dark blue bricks with no gaps nearby. A gap is visible to the far right of the screen.",
    "sequence": [
        {
            "action": "hold",
            "keys": ["d"]
            
        }
    ],
    "reason": "Although there is a visible gap to the far right, it is not immediately ahead, I should hold 'd' key to move right to continue progressing toward the right side of the level instead of jumping."
}
"""

#c
FAR_EDGE3 = """
{
    "observation": "The player character is on a platform made of dark blue bricks, the dark blue bricks ends several steps ahead of the character, indicating a gap to the further right.",
    "sequence": [
        {
            "action": "hold",
            "keys": ["d"]
        }
    ],
    "reason": "Since there is no gap to jump over immediately, the most direct way to progress is to hold the 'd' key to move right across the platform. Jumping is not needed until the character reach the rightmost point of the current platform."
}
"""
#d
FAR_EDGE4 = """
{
    "observation": "The player character is positioned on a solid platform made of dark blue bricks, centered on the screen, the dark blue bricks ends several steps ahead of the character, indicating a gap to the further right. ",
    "sequence": [
        {
            "action": "hold",
            "keys": ["d"]
        }
    ],
    "reason": "There is no gap to jump over immediately, the most direct way to progress is to hold the 'd' key to carefully move right to the end of the platform, getting ready for the jump over the gap."
}
"""

NO_ACTION = """
{
    "sequence": [
        {
            "action": "no_action",
            "keys": []
            
        }
    ],
    "reason": "The scene is not clear, I probably died and the level is restarting, I should not take any action and wait for a logical scene."
}
"""

def discover_scenarios(cases_root: Path) -> list[str]:
    if not cases_root.is_dir():
        return []
    scenarios: list[str] = []
    for p in sorted(cases_root.iterdir()):
        if not p.is_dir() or p.name in SKIP_SCENARIO_DIRS:
            continue
        if (p / "ground_truth.txt").is_file():
            scenarios.append(p.name)
    return scenarios


MULTIMODAL_NARROW = [
    {
        "role": "system",
        "content": PROMPT,
    },
    {
        "role": "user",
        "images": [str(TEST_CASES_ROOT / "besides_edges" / "a2.png")],
    },
    {
        "role": "assistant",
        "content": JUMP1,
    },
    {
        "role": "user",
        "images": [str(TEST_CASES_ROOT / "plain_grounds" / "a2.png")],
    },
    {
        "role": "assistant",
        "content": WALK1,
    },
    {
        "role": "user",
        "images": [str(TEST_CASES_ROOT / "far_edge" / "a2.png")],
    },
    {
        "role": "assistant",
        "content": FAR_EDGE1,
    },
]

#eeff
MULTIMODAL_BALANCED = [
    {
        "role": "system",
        "content": PROMPT,
    },
    {
        "role": "user",
        "images": [str(TEST_CASES_ROOT / "besides_edges" / "a2.png")],
    },
    {
        "role": "assistant",
        "content": JUMP1,
    },
    {
        "role": "user",
        "images": [str(TEST_CASES_ROOT / "plain_grounds" / "a2.png")],
    },
    {
        "role": "assistant",
        "content": WALK1,
    },
    {
        "role": "user",
        "images": [str(TEST_CASES_ROOT / "far_edge" / "a2.png")],
    },
    {
        "role": "assistant",
        "content": FAR_EDGE1,
    },
    {
        "role": "user",
        "images": [str(TEST_CASES_ROOT / "besides_edges" / "b2.png")],
    },
    {
        "role": "assistant",
        "content": JUMP2,
    },
    {
        "role": "user",
        "images": [str(TEST_CASES_ROOT / "plain_grounds" / "b2.png")],
    },
    {
        "role": "assistant",
        "content": WALK2,
    },
    {
        "role": "user",
        "images": [str(TEST_CASES_ROOT / "far_edge" / "b2.png")],
    },
    {
        "role": "assistant",
        "content": FAR_EDGE2,
    },
    {
        "role": "user",
        "images": [str(TEST_CASES_ROOT / "besides_edges" / "c2.png")],
    },
    {
        "role": "assistant",
        "content": JUMP3,
    },
    {
        "role": "user",
        "images": [str(TEST_CASES_ROOT / "plain_grounds" / "c2.png")],
    },
    {
        "role": "assistant",
        "content": WALK3,
    },
    {
        "role": "user",
        "images": [str(TEST_CASES_ROOT / "far_edge" / "c2.png")],
    },
    {
        "role": "assistant",
        "content": FAR_EDGE3,
    },
    {
        "role": "user",
        "images": [str(TEST_CASES_ROOT / "besides_edges" / "d2.png")],
    },
    {
        "role": "assistant",
        "content": JUMP4,
    },
    {
        "role": "user",
        "images": [str(TEST_CASES_ROOT / "plain_grounds" / "d2.png")],
    },
    {
        "role": "assistant",
        "content": WALK4,
    },
    {
        "role": "user",
        "images": [str(TEST_CASES_ROOT / "far_edge" / "d2.png")],
    },
    {
        "role": "assistant",
        "content": FAR_EDGE4,
    },
]

#eefh
MULTIMODAL_WALK_BIAS = [
    {
        "role": "system",
        "content": PROMPT,
    },
    {
        "role": "user",
        "images": [str(TEST_CASES_ROOT / "plain_grounds" / "a2.png")],
    },
    {
        "role": "assistant",
        "content": WALK1,
    },
    {
        "role": "user",
        "images": [str(TEST_CASES_ROOT / "plain_grounds" / "b2.png")],
    },
    {
        "role": "assistant",
        "content": WALK2,
    },
    {
        "role": "user",
        "images": [str(TEST_CASES_ROOT / "plain_grounds" / "c2.png")],
    },
    {
        "role": "assistant",
        "content": WALK3,
    },
    {
        "role": "user",
        "images": [str(TEST_CASES_ROOT / "plain_grounds" / "d2.png")],
    },
    {
        "role": "assistant",
        "content": WALK4,
    },
]

#efge
MULTIMODAL_JUMP_BIAS = [

    {
        "role": "system",
        "content": PROMPT,
    },
    {
        "role": "user",
        "images": [str(TEST_CASES_ROOT / "besides_edges" / "a2.png")],
    },
    {
        "role": "assistant",
        "content": JUMP1,
    },
    {
        "role": "user",
        "images": [str(TEST_CASES_ROOT / "besides_edges" / "b2.png")],
    },
    {
        "role": "assistant",
        "content": JUMP2,
    },
    {
        "role": "user",
        "images": [str(TEST_CASES_ROOT / "besides_edges" / "c2.png")],
    },
    {
        "role": "assistant",
        "content": JUMP3,
    },
    {
        "role": "user",
        "images": [str(TEST_CASES_ROOT / "besides_edges" / "d2.png")],
    },
    {
        "role": "assistant",
        "content": JUMP4,
    },
]

MULTIMODAL_FAR_EDGE_BIAS = [
    {
        "role": "system",
        "content": PROMPT,
    },
    {
        "role": "user",
        "images": [str(TEST_CASES_ROOT / "far_edge" / "a2.png")],
    },
    {
        "role": "assistant",
        "content": FAR_EDGE1,
    },
    {
        "role": "user",
        "images": [str(TEST_CASES_ROOT / "far_edge" / "b2.png")],
    },
    {
        "role": "assistant",
        "content": FAR_EDGE2,
    },
    {
        "role": "user",
        "images": [str(TEST_CASES_ROOT / "far_edge" / "c2.png")],
    },
    {
        "role": "assistant",
        "content": FAR_EDGE3,
    },
    {
        "role": "user",
        "images": [str(TEST_CASES_ROOT / "far_edge" / "d2.png")],
    },
    {
        "role": "assistant",
        "content": FAR_EDGE4,
    },
]
FEWSHOT_BASES = {
    "narrow": MULTIMODAL_NARROW,
    "balanced": MULTIMODAL_BALANCED,
    "walk_bias": MULTIMODAL_WALK_BIAS,
    "jump_bias": MULTIMODAL_JUMP_BIAS,
    #"far_edge_bias": MULTIMODAL_FAR_EDGE_BIAS,
}


def scenario_paths(
    cases_root: Path, output_root: Path, scenario: str, variant: str
) -> tuple[Path, Path, Path]:
    """
    Returns (image_dir, default_ground_truth_file, results_dir_for_numbered_logs).
    variant: "naive" | "fewshot" | "multifewshot".
    """
    image_dir = cases_root / scenario
    answers = image_dir / "ground_truth.txt"
    results_dir = output_root / scenario / variant
    return image_dir, answers, results_dir


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


def _light_json_cleanup(text: str) -> str:
    """
    Apply lightweight cleanup before JSON parse.
    Currently handles trailing commas before '}' or ']'.
    """
    return re.sub(r",\s*([}\]])", r"\1", text)


def parse_json_from_model_text(raw: str) -> dict | None:
    raw = raw.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    text = m.group(1).strip() if m else raw
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        cleaned = _light_json_cleanup(text)
        if cleaned == text:
            return None
        try:
            obj = json.loads(cleaned)
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


def _normalize_sequence(payload: dict) -> list[dict] | None:
    """
    Normalize action payload to a sequence list.
    Supports both:
      - new format: {"sequence": [ ...actions... ]}
      - legacy format: {"action": ..., "keys": ...}
    """
    if not isinstance(payload, dict):
        return None
    seq = payload.get("sequence")
    if seq is None:
        return [payload]
    if not isinstance(seq, list):
        return None
    if not all(isinstance(item, dict) for item in seq):
        return None
    return seq


def _single_action_matches(expected: dict, actual: dict) -> bool:
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
    return ek == ak


def action_matches(expected: dict, actual: dict) -> bool:
    exp_seq = _normalize_sequence(expected)
    act_seq = _normalize_sequence(actual)
    if exp_seq is None or act_seq is None:
        return False
    if len(exp_seq) != len(act_seq):
        return False
    return all(_single_action_matches(e, a) for e, a in zip(exp_seq, act_seq))


def collect_image_paths(screenshot_path: Path) -> list[Path]:
    if not screenshot_path.is_dir():
        print(f"Not a directory: {screenshot_path.resolve()}")
        return []
    return sorted(
        p
        for p in screenshot_path.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def next_inference_results_path(results_dir: Path, prompt_variant: str) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    max_n = 0
    for p in results_dir.iterdir():
        if not p.is_file() or p.suffix.lower() != ".txt":
            continue
        for token in p.stem.split("_"):
            if token.isdigit():
                max_n = max(max_n, int(token))
                break
    safe_variant = re.sub(r"[^a-zA-Z0-9_-]+", "_", prompt_variant).strip("_") or "variant"
    return results_dir / f"multimodal_{max_n + 1}_{safe_variant}.txt"


def main():
    parser = argparse.ArgumentParser(
        description="V2 multimodal compare: run train_cases by default, or test_cases with --test."
    )
    parser.add_argument(
        "--scenario",
        default=None,
        help="Scenario folder under selected cases root (default: first discovered scenario).",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Use test_cases images and write logs to test_cases/test_output.",
    )
    parser.add_argument(
        "--variant",
        choices=("naive", "fewshot", "multifewshot"),
        default="naive",
        help="Inference mode and output subdirectory name (default: naive).",
    )
    parser.add_argument(
        "--fewshot-base",
        choices=tuple(FEWSHOT_BASES.keys()),
        default="balanced",
        help="Select which handwritten multimodal few-shot demonstration to use.",
    )
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
        default=None,
        help="Override ground-truth JSON txt (default: <scenario_dir>/ground_truth.txt).",
    )
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="Print available scenarios for selected root and log dirs, then exit.",
    )
    args = parser.parse_args()

    cases_root = TEST_CASES_ROOT if args.test else TRAIN_CASES_ROOT
    output_root = TEST_OUTPUT_ROOT if args.test else TRAIN_OUTPUT_ROOT
    available_scenarios = discover_scenarios(cases_root)
    if not available_scenarios:
        print(f"No scenarios found under {cases_root.resolve()}", file=sys.stderr)
        sys.exit(1)
    if args.scenario is None:
        args.scenario = available_scenarios[0]
    if args.scenario not in available_scenarios:
        print(
            f"Unknown scenario {args.scenario!r} under {cases_root.resolve()}",
            file=sys.stderr,
        )
        print(f"Available: {', '.join(available_scenarios)}", file=sys.stderr)
        sys.exit(1)

    if args.list_scenarios:
        for sid in available_scenarios:
            img, _, rdir = scenario_paths(cases_root, output_root, sid, args.variant)
            print(f"{sid}:")
            print(f"  images:  {img.resolve()}")
            print(f"  logs:    {rdir.resolve()}")
        return

    if args.repeats < 1:
        print("--repeats must be >= 1", file=sys.stderr)
        sys.exit(1)

    screenshot_path, default_answers, results_dir = scenario_paths(
        cases_root, output_root, args.scenario, args.variant
    )
    answers_path = Path(args.answers) if args.answers is not None else default_answers
    fewshot_messages = FEWSHOT_BASES[args.fewshot_base]

    paths = collect_image_paths(screenshot_path)
    if not paths:
        print(f"No image files found in {screenshot_path.resolve()}")
        return

    try:
        ground_truth = load_ground_truth(answers_path)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Invalid ground truth file {answers_path}: {e}", file=sys.stderr)
        sys.exit(1)

    out_path = next_inference_results_path(results_dir, args.variant)
    body_chunks: list[str] = []

    def log_print(*args, sep=" ", end="\n", flush=False):
        print(*args, sep=sep, end=end, flush=flush)
        buf = StringIO()
        print(*args, sep=sep, end=end, file=buf)
        body_chunks.append(buf.getvalue())

    print(
        f"scenario={args.scenario} variant={args.variant} fewshot_base={args.fewshot_base}",
        file=sys.stderr,
    )
    print(f"Run log file: {out_path.resolve()}", file=sys.stderr)

    graded = 0
    correct = 0
    per_file_correct: dict[str, int] = defaultdict(int)
    per_file_total: dict[str, int] = defaultdict(int)

    for rep in range(args.repeats):
        for path in paths:
            name = path.name
            expected = ground_truth.get(name)

            messages = fewshot_messages + [{"role": "user", "images": [str(path)]}]

            resp = ollama.chat(
                model=MODEL,
                messages=messages,
                #keep_alive="30m",
            )
            raw = (resp["message"]["content"] or "").strip()
            log_print(
                "=" * 10
                + f"Screenshot {name}"
                + (f" (repeat {rep + 1}/{args.repeats})" if args.repeats > 1 else "")
                + "=" * 10
            )
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
    summary_lines.append(
        f"scenario: {args.scenario}  variant: {args.variant}  fewshot_base: {args.fewshot_base}"
    )
    summary_lines.append(f"images: {screenshot_path.resolve()}")
    summary_lines.append(f"answers: {answers_path.resolve()}")
    summary_lines.append("")
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

    summary_text = "\n".join(summary_lines) + "\n"
    print()
    print(summary_text, end="")
    out_path.write_text(summary_text + "".join(body_chunks), encoding="utf-8")
    
    #unload the model from memory

#    from ollama import generate
#
#    generate(
#        model=MODEL,
#        prompt="",
#        keep_alive=0
#    )


if __name__ == "__main__":
    main()
