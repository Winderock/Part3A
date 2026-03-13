# -*- coding: utf-8 -*-

import base64
import os
import re
import json
import time
from pathlib import Path
from typing import Tuple

import pyautogui
import pygetwindow as gw
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv(
    "OPENAI_API_BASE", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
)
API_KEY = os.getenv("DASHSCOPE_API_KEY")
MODEL = os.getenv("VISION_MODEL", "qwen-vl-plus")

SCREENSHOT_PATH = "screenshot.png"
WINDOW_TITLE = "localhost:8000"
WEBPAGE_INSET = (0, 0, 0, 0)  # Use entire window; model coords + offset = screen coords


################### Research notes

# click rarely triggers?? Model seems to keep moving mouse around without confirming click

# Direct move to coordinates seems inaccurate? Try moving by fixed distance up/down/left/right?

# Sometimes json from model causes: json.decoder.JSONDecodeError: Expecting ',' delimiter: line 5 column 20 (char 78). Model response may not be valid format.

###################


PROMPT = """You are playing a simple clicking videogame. There is a green sphere on the screen. To gain score you need to move your mouse precisely onto the sphere and click it.

Check the given game screenshot and tell me about the exact pixel coordinates of the cursor and the sphere, then choose the next action: 
    "click" (when cursor is already on the sphere);
    or "move" (in correct direction and distance, move the cursor to the sphere center then click);


Output STRICT pure JSON only, no other text, no comments:
{
    "actions": [
        {
            "action": "click",
            "reason": "reason for this action",
            "score": 0,
            "confidence": 0.95
        },
        {
            "action": "move",
            "x": 100,
            "y": 100,
            "reason": "reason for this action",
            "score": 0,
            "confidence": 0.95
        }
    ]
}

Return ONE action in the actions array - either click or move. For move, provide the exact pixel coordinates (x, y) of the green sphere center.
"""


def get_window_region(title: str) -> Tuple[Tuple[int, int, int, int], Tuple[int, int]]:
    """
    Get screenshot region by window title. Returns (region, offset).
    region: (left, top, width, height) for pyautogui.screenshot
    offset: (left, top) window top-left on screen; model coords (0,0)=screenshot top-left, screen = offset + (x,y)
    """
    wins = gw.getWindowsWithTitle(title)
    if not wins:
        raise RuntimeError(
            f"No window found with title containing '{title}'. Use gw.getAllTitles() to list all window titles."
        )
    w = wins[0]
    w.activate()  # Bring to front, ensure correct content is captured
    time.sleep(0.1)
    left_inset, top_inset, right_inset, bottom_inset = WEBPAGE_INSET
    content_left = w.left + left_inset
    content_top = w.top + top_inset
    content_width = w.width - left_inset - right_inset
    content_height = w.height - top_inset - bottom_inset
    region = (content_left, content_top, content_width, content_height)
    offset = (content_left, content_top)
    return region, offset


def take_screenshot(region=None, window_title=None, step_num: int = 0):
    """
    Capture screen/region/specified window.
    - region: (left, top, width, height), mutually exclusive with window_title
    - window_title: window title (partial match), e.g. "game", "Chrome"
    Returns (abs_path, offset). With window_title, crops by WEBPAGE_INSET; offset is screen coords of webpage content top-left
    """
    offset = (0, 0)
    if window_title:
        region, offset = get_window_region(window_title)
    elif region is None:
        region = (0, 0, 1920, 1080)
    screenshot = pyautogui.screenshot(region=region)
    abs_path = str(Path(f"screenshot_{step_num}.png").resolve())
    screenshot.save(abs_path)
    return abs_path, offset


def call_vision_api(filepath: str) -> str:
    """Call vision model (OpenAI-compatible API), returns JSON text"""
    abs_path = Path(filepath).resolve()
    if not abs_path.exists():
        raise FileNotFoundError(f"Screenshot file does not exist: {abs_path}")

    with open(abs_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    image_url = f"data:image/png;base64,{b64}"

    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": PROMPT},
                ],
            },
        ],
    )

    content = response.choices[0].message.content
    return (content or "").strip()


def parse_json_from_response(text: str) -> dict:
    """Parse JSON from model output (may be wrapped in ```json ... ```)"""
    # Try to extract content from ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        text = match.group(1).strip()
    # Parse directly
    return json.loads(text)


def execute_actions(actions: list, offset: Tuple[int, int] = (0, 0)) -> None:
    """
    Execute pyautogui actions from JSON action list.
    offset: (ox, oy) window/screenshot top-left on screen; model (x,y) -> screen (ox+x, oy+y)
    """
    ox, oy = offset
    for item in actions:
        action_type = item.get("action", "").lower()
        reason = item.get("reason", "")
        confidence = item.get("confidence", 0)
        print(
            f"  Executing: {action_type} | Reason: {reason} | Confidence: {confidence:.2f}"
        )

        if action_type == "click":
            pyautogui.click()
        elif action_type == "move":
            # Model coords are relative to screenshot; add offset to get screen coords
            screen_x = int(item.get("x", 0)) + ox
            screen_y = int(item.get("y", 0)) + oy
            pyautogui.moveTo(screen_x, screen_y)


def step(step_num: int):
    """Single step: screenshot -> call API -> parse JSON -> execute actions"""
    filepath, offset = take_screenshot(window_title=WINDOW_TITLE, step_num=step_num)
    # print(f"[Screenshot saved] {filepath}" + (f" (window offset {offset})" if offset != (0, 0) else ""))

    # call model
    raw_response = call_vision_api(filepath)
    print(f"[Model output]\n{raw_response}")

    # Parse JSON
    try:
        data = parse_json_from_response(raw_response)
    except json.JSONDecodeError as e:
        print(f"[Warning] JSON parse error, skipping: {e}")
        return
    actions = data.get("actions", [])

    if not actions:
        print("[Warning] No valid actions parsed, skipping")
        return

    # Execute actions
    execute_actions(actions, offset)


def main():
    """Main loop: continuously screenshot, recognize, execute"""
    print(f"API: {API_BASE_URL} | Model: {MODEL}")
    print()
    step_num = 0

    try:
        while True:
            step(step_num)
            step_num += 1
            time.sleep(1.5)  # Interval between steps, avoid running too fast
    except KeyboardInterrupt:
        print("\nExited")


if __name__ == "__main__":
    main()
