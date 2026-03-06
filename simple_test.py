# -*- coding: utf-8 -*-

import base64
import json
import os
import re
import sys
from pathlib import Path
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

SCREENSHOT_PATH = Path("test_folder/fulltut/")
API_BASE_URL = os.getenv("OPENAI_API_BASE", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
API_KEY = os.getenv("DASHSCOPE_API_KEY")
MODEL = os.getenv("VISION_MODEL", "qwen-vl-plus")

PROMPT = """
You are playing a 2D side-scrolling platformer game independently, the right side is always the end of the level.

You will be given game screenshots showing the current game state, then reason step by step for a reasonable and efficient action to take:
1) Evaluating the given game screenshot, take note of the position of yourself and any obstacles or points of interest.

2) After evaluating the given game screenshot, decide what kind of action (including a no action option) to take to progress in the game.

3) Output one option from the following actions in the following strict JSON format:
{
    "actions": [
        {"action": "click_mouse", "x": 100, "y": 200, "reason": "reason"},
        {"action": "press_key", "key": "space", "reason": "reason"},
        {"action": "press_keys", "keys": ["d", "j"], "hold_ms": 150, "reason": "reason"},
        {"action": "no_action", "reason": "reason"},
    ]
}

press_key: press a single key.
press_keys: two keys, hold first key for 150ms, then press the second key.
click_mouse: move mouse to the given x,y coordinates and click.
no_action: do nothing.

*For each action, provide a reason for the action.
"""


#  !!!These examples can come from the model's former successful memories.
REASONING_EXAMPLES = """
=======Reasoning Examples=======
a) The scene shows the cat at the edge of a cliff, I should hold "d" key to move right and also press space to jump over the cliff.
b) The cat is on what seems to be a moving platform, nothing seems reachable from the right side, so I should wait for the platform to move, so no action is needed.
c) The cat is on a plain platform, I should press "d" key to move right and explore further into the level.
d) The screenshot shows a main menu, I should click the mouse on the first available level button to start the game.
e) The scene is not clear, I should not take any action and wait for the next screenshot.
"""


def main():

    for i in range(1, 9):
        path = SCREENSHOT_PATH / f"{i}.jpg"

        with open(path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

        client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                    {"type": "text", "text": PROMPT},
                ],
            }],
        )
        raw = (resp.choices[0].message.content or "").strip()
        print("=" * 10 + f"Screenshot {i}" + "=" * 10)

        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
        text = m.group(1).strip() if m else raw
        try:
            data = json.loads(text)
            print(json.dumps(data, ensure_ascii=False, indent=2))
        except json.JSONDecodeError:
            pass

        time.sleep(0.5)


if __name__ == "__main__":
    main()
