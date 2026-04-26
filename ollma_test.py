import json
import os
import re
import time
import argparse

import mss
import ollama
import pyautogui
import pydirectinput
import pygetwindow as gw
from PIL import Image

#multiframe capture tools
import threading
from collections import deque


MODEL_NAME = "qwen3-vl:8b-instruct"
WINDOW_TITLE = "cw1"
CAPTURE_DIR = "captures"
PAUSE_KEY = "p"
# ==============================


PROMPT = """
You are playing a 2D side-scrolling platformer game independently, your objective is to move to the right as far as possible, without falling into any gaps.

Your character is a white cat centered in the screen, the background is light blue, and the ground's texture is dark blue bricks, use the difference in texture to identify solid ground and gaps.
Use keys "a" to move left and "d" to move right, 
"space" to jump (can be combined with "a" or "d" to jump left or right).

You will be given game screenshots showing the current game state, then reason step by step for a reasonable and efficient action to take:
1) Evaluate the given game screenshot, use a few sentences to describe the position of yourself and relative position to any gaps or obstacles.

2) After evaluating the given game screenshot, decide what kind of action from the following, in strict JSON format:
{
    "action": "hold_press",
    "keys": ["key1", "key2"],
    "reason": "reason"
}

hold_press: specify two keys, hold both keys for a period of time.
"""


#  !!!These examples can come from the model's former successful memories.
REASONING_EXAMPLES = """
=======Reasoning Examples=======
a) The scene shows the cat at the left-side edge of a gap, I should jump right by holding "d" and "space" at the same time to get over the gap.
c) The cat is on a plain platform and no gaps or obstacles in front, I should press "d" key to move right and explore further into the level.
"""

def get_window_rect(title_keyword: str):
    windows = gw.getWindowsWithTitle(title_keyword)
    if not windows:
        raise RuntimeError(f"Window not found: {title_keyword}")
    win = windows[0]
    return win.left, win.top, win.width, win.height


def capture_window(title_keyword: str, image_path: str):
    left, top, width, height = get_window_rect(title_keyword)

    with mss.mss() as sct:
        monitor = {
            "left": left,
            "top": top,
            "width": width,
            "height": height
        }
        shot = sct.grab(monitor)
        img = Image.frombytes("RGB", shot.size, shot.rgb)
        img.save(image_path)

    return (left, top, width, height)

def capture_burst(title_keyword: str, frame_count=4, interval=0.1): #0.1 * 4 frames = 0.4 seconds, which is the time for one hold_press action
    paths = []
    window_rect = None

    for i in range(frame_count):
        ts = time.time_ns()
        image_path = os.path.join(CAPTURE_DIR, f"burst_{ts}_{i}.png")
        window_rect = capture_window(title_keyword, image_path)
        paths.append(image_path)
        if i < frame_count - 1:
            time.sleep(interval)

    return paths, window_rect

recent_frames = deque(maxlen=80)
latest_window_rect = None
lock = threading.Lock()
capture_allowed = threading.Event()

def capture_worker():
    global latest_window_rect
    while True:
        if not capture_allowed.is_set():
            time.sleep(0.05)
            continue
        ts = time.time_ns()
        image_path = os.path.join(CAPTURE_DIR, f"frame_{ts}.png")
        rect = capture_window(WINDOW_TITLE, image_path)
        with lock:
            recent_frames.append(image_path)
            latest_window_rect = rect
        time.sleep(0.1)

def get_recent_frames(n=4):
    with lock:
        if len(recent_frames) < n:
            return None, latest_window_rect
        return list(recent_frames)[-n:], latest_window_rect

def ask_model(image_paths: list[str]):
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": PROMPT + REASONING_EXAMPLES,
                "images": image_paths
            }
        ]
    )
    return response["message"]["content"]


def parse_json(text: str):
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError("No valid JSON found")
        return json.loads(match.group(0))


def apply_action(action: dict, window_rect):
    left, top, width, height = window_rect
    act = action["action"]

    if act == "hold_press":
        keys = action.get("keys") or []
        if not keys:
            return
        for k in keys:
            pydirectinput.keyDown(str(k))
        time.sleep(0.5)
        for k in reversed(keys):
            pydirectinput.keyUp(str(k))

    elif act == "no_action":
        time.sleep(0.5)

    else:
        raise ValueError(f"Unknown action: {act}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--latest-frame-only",
        action="store_true",
        help="Only send the latest captured frame to the model instead of a sequence of frames.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(CAPTURE_DIR, exist_ok=True)
    frame = 0
    game_paused_by_us = False

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

    t = threading.Thread(target=capture_worker, daemon=True)
    t.start()
    
    print("starts after 5 seconds")
    time.sleep(5)
    print(
        "inference mode:",
        "latest-frame-only" if args.latest_frame_only else "multi-frame-sequence",
    )

    # First iteration: take no action, but still collect frames during that "action window"
    action = {"action": "no_action", "keys": [], "reason": "bootstrap"}

    while True:
        frame += 1
        tag = f"[frame {frame:06d}]"

        # Unpause BEFORE apply_action, and only capture during the action window.
        ensure_running()
        print(f"{tag} unpaused ({PAUSE_KEY}) before apply_action")
        with lock:
            recent_frames.clear()
        capture_allowed.set()
        window_rect = get_window_rect(WINDOW_TITLE)
        apply_action(action, window_rect)
        capture_allowed.clear()

        # Pause immediately after the action completes (model thinks while paused).
        ensure_paused()
        print(f"{tag} paused ({PAUSE_KEY}) after action")

        # Collect all frames captured during the action window and ask model for next action.
        time.sleep(0.02)  # allow capture thread to finish its last append
        with lock:
            image_paths = list(recent_frames)
            window_rect = latest_window_rect

        if not image_paths:
            time.sleep(0.05)
            with lock:
                image_paths = list(recent_frames)
                window_rect = latest_window_rect

        if not image_paths:
            # If the action window was too short to grab frames, keep last action.
            print(f"{tag} no frames captured during action window; keeping last action")
            continue

        model_images = image_paths[-1:] if args.latest_frame_only else image_paths
        print(f"{tag} captured {len(image_paths)} frames during action window")
        print(f"{tag} sending {len(model_images)} frame(s) to model")
        raw_output = ask_model(model_images)
        print(f"{tag} model output: {raw_output}")
        action = parse_json(raw_output)


if __name__ == "__main__":
    main()
