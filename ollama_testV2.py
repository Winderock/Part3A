import argparse
import importlib.util
import os
import time
from copy import deepcopy
from pathlib import Path

import mss
import ollama
import pydirectinput
import pygetwindow as gw
from PIL import Image


MODEL_NAME = "qwen3-vl:8b-instruct"
WINDOW_TITLE = "cw1"
CAPTURE_DIR = "captures"
PAUSE_KEY = "p"
ACTION_HOLD_SECONDS = 0.5

SCRIPT_DIR = Path(__file__).resolve().parent
COMPARE_FILE = (
    SCRIPT_DIR / "compare=naive2few_shot" / "MultimodalV2Compare=naive2few_shot.py"
)


def load_compare_module(compare_file: Path):
    if not compare_file.is_file():
        raise FileNotFoundError(f"Compare file not found: {compare_file}")
    spec = importlib.util.spec_from_file_location("compare_multimodal_v2", compare_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module spec from: {compare_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_fewshot_messages(compare_module):
    prompt = compare_module.PROMPT
    base = deepcopy(compare_module.MULTIMODAL_BALANCED)
    if not base:
        raise ValueError("MULTIMODAL_BALANCED is empty")
    first = base[0]
    if first.get("role") != "system":
        base.insert(0, {"role": "system", "content": prompt})
    else:
        first["content"] = prompt
    return base


def get_window_rect(title_keyword: str):
    windows = gw.getWindowsWithTitle(title_keyword)
    if not windows:
        raise RuntimeError(f"Window not found: {title_keyword}")
    win = windows[0]
    return win.left, win.top, win.width, win.height


def capture_window(title_keyword: str, image_path: str):
    left, top, width, height = get_window_rect(title_keyword)
    with mss.mss() as sct:
        monitor = {"left": left, "top": top, "width": width, "height": height}
        shot = sct.grab(monitor)
        img = Image.frombytes("RGB", shot.size, shot.rgb)
        img.save(image_path)
    return left, top, width, height


def normalize_action(payload: dict):
    """
    Normalize model output to runtime action schema:
      {"action": "hold_press|no_action", "keys": [...], "reason": "..."}
    Supports compare script style:
      {"sequence":[{"action":"hold|no_action","keys":[...]}], ...}
    """
    if not isinstance(payload, dict):
        raise ValueError("Model output is not a JSON object")

    if isinstance(payload.get("sequence"), list):
        if not payload["sequence"]:
            return {"action": "no_action", "keys": [], "reason": payload.get("reason", "")}
        step = payload["sequence"][0]
        if not isinstance(step, dict):
            raise ValueError("sequence[0] must be an object")
        raw_action = str(step.get("action", "")).strip().lower()
        raw_keys = step.get("keys") or []
        keys = [str(k).strip().lower() for k in raw_keys if str(k).strip()]
        if raw_action == "hold":
            return {"action": "hold_press", "keys": keys, "reason": payload.get("reason", "")}
        if raw_action == "no_action":
            return {"action": "no_action", "keys": [], "reason": payload.get("reason", "")}
        raise ValueError(f"Unknown sequence action: {raw_action}")

    raw_action = str(payload.get("action", "")).strip().lower()
    raw_keys = payload.get("keys") or []
    keys = [str(k).strip().lower() for k in raw_keys if str(k).strip()]
    if raw_action in {"hold_press", "hold"}:
        return {"action": "hold_press", "keys": keys, "reason": payload.get("reason", "")}
    if raw_action == "no_action":
        return {"action": "no_action", "keys": [], "reason": payload.get("reason", "")}
    raise ValueError(f"Unknown action: {raw_action}")


def apply_action(action: dict):
    act = action["action"]
    if act == "hold_press":
        keys = action.get("keys") or []
        if not keys:
            time.sleep(ACTION_HOLD_SECONDS)
            return
        for k in keys:
            pydirectinput.keyDown(str(k))
        time.sleep(ACTION_HOLD_SECONDS)
        for k in reversed(keys):
            pydirectinput.keyUp(str(k))
        return
    if act == "no_action":
        time.sleep(ACTION_HOLD_SECONDS)
        return
    raise ValueError(f"Unknown action: {act}")


def ask_model(model_name: str, base_messages: list[dict], image_path: str):
    messages = deepcopy(base_messages)
    messages.append({"role": "user", "images": [image_path]})
    response = ollama.chat(
        model=model_name,
        messages=messages,
    )
    return (response["message"]["content"] or "").strip()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-title", default=WINDOW_TITLE)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--compare-file", default=str(COMPARE_FILE))
    return parser.parse_args()


def main():
    args = parse_args()
    compare_module = load_compare_module(Path(args.compare_file))
    parse_json_from_model_text = compare_module.parse_json_from_model_text
    fewshot_messages = build_fewshot_messages(compare_module)

    os.makedirs(CAPTURE_DIR, exist_ok=True)
    frame = 0
    game_paused_by_us = False
    action = {"action": "no_action", "keys": [], "reason": "bootstrap"}

    def ensure_running():
        nonlocal game_paused_by_us
        if game_paused_by_us:
            pydirectinput.press(PAUSE_KEY)
            game_paused_by_us = False

    def ensure_paused():
        nonlocal game_paused_by_us
        if not game_paused_by_us:
            pydirectinput.press(PAUSE_KEY)
            game_paused_by_us = True

    print("starts after 5 seconds")
    time.sleep(5)
    print("mode: single-frame inference with MULTIMODAL_BALANCED few-shot")

    while True:
        frame += 1
        tag = f"[frame {frame:06d}]"

        ensure_running()
        print(f"{tag} unpaused ({PAUSE_KEY}) before apply_action")
        _ = get_window_rect(args.window_title)
        apply_action(action)

        ensure_paused()
        print(f"{tag} paused ({PAUSE_KEY}) after action")

        ts = time.time_ns()
        image_path = os.path.join(CAPTURE_DIR, f"frame_{ts}.png")
        _ = capture_window(args.window_title, image_path)
        print(f"{tag} captured 1 frame: {image_path}")

        raw_output = ask_model(args.model, fewshot_messages, image_path)
        print(f"{tag} model output: {raw_output}")

        parsed = parse_json_from_model_text(raw_output)
        if parsed is None:
            print(f"{tag} JSON parse failed; fallback to no_action")
            action = {"action": "no_action", "keys": [], "reason": "parse_failed"}
            continue

        try:
            action = normalize_action(parsed)
            print(f"{tag} normalized action: {action}")
        except Exception as e:
            print(f"{tag} action normalize failed ({e}); fallback to no_action")
            action = {"action": "no_action", "keys": [], "reason": "normalize_failed"}


if __name__ == "__main__":
    main()
