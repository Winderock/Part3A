FIXED_PROMPT_KEYS = """
You are playing a 2D platformer game, your character is a white cat and moving towards the right is the goal.

Look at the screenshot from the game window, evaluate any obstacles then output exactly one action you should take to reach the goal as JSON only.

Allowed actions:
- hold_keys
- wait

allowed keys:
- "a": move left
- "d": move right
- "j": jump (combine with a or d)

JSON format:
{
  "reason": "reason for this action",
  "action": "hold_keys|wait",
  "hold_keys": ["key1", "key2"]
}

Heads up:
- game pauses after each action finishes, and unpauses when you give the next action.
- "hold_keys" action will hold specified keys for 0.4 seconds.
- "wait" action will do nothing for 0.4 seconds.
- Do not output anything except JSON.
""".strip()

FIXED_PROMPT_ACTIONS = """
You are playing a 2D platformer game, your character is a white cat and moving towards the right is the goal.

Look at the screenshot from the game window, evaluate any obstacles then output exactly one action you should take to reach the goal as JSON only.

allowed actions:
- move_left
- move_right
- jump_left
- jump_right

JSON format:
{
  "reason": "reason for this action",
  "action": "move_left|move_right|jump_left|jump_right"
}

Heads up:
- game pauses after each action finishes, and unpauses when you give the next action.
- Do not output anything except JSON.
""".strip()

FEWSHOT_PROMPTS_ACTIONS = f"""
=====================
Below are some examples responses you should make:
A: 
  "reason": "The cat is right in front of a gap, to proceed I should jump over the gap.",
  "action": "jump_right"

B: 
  "reason": "There is no immediate obstacle in front of the cat, I should hold "d" key to move right.",
  "action": "move_right"

C: 
  "reason": "There is a gap to the far right, I should move right carefully to the gap to get ready for a jump.",
  "action": "move_right"
""".strip()

FEWSHOT_PROMPTS_KEYS = f"""
=====================
Below are some examples responses you should make:
A: 
  "reason": "The cat is on a platform, there is a gap in front of it, to proceed I should hold both "d" and "space" keys to jump over the gap.",
  "action": "hold_keys",
  "hold_keys": ["d", "space"]

B: 
  "reason": "No obstacles in front of the cat, I should hold "d" key to move right.",
  "action": "hold_keys",
  "hold_keys": ["d"]

C: 
  "reason": "Nothing sensible is on the screen, I should wait for more information to appear.",
  "action": "wait"


""".strip()

PROMPT = """
You are playing a 2D side-scrolling platformer game independently, your objective is to move to the right side of the level.

use keys "a" to move left and "d" to move right, 
"space" to jump (can be combined with "a" or "d" to jump left or right).

You will be given game screenshots showing the current game state, then reason step by step for a reasonable and efficient action to take:
1) Evaluate the given game screenshot, use a few sentences to describe key information such as position of yourself and any obstacles.

2) After evaluating the given game screenshot, decide what kind of action from the following, in strict JSON format:
{
    "action": "hold_press|no_action",
    "keys": ["key1", "key2"],
    "reason": "reason"
}

hold_press: specify two keys, hold both keys for a period of time.
no_action: do nothing.
"""

REASONING_EXAMPLES = """
=======Reasoning Examples=======
a) The scene shows the cat at the edge of a gap, I should jump over the gap to progress, by holding "d" key to move right while pressing "space" at the same time.
c) The cat is on a plain platform and no obstacles in front, I should press "d" key to move right and explore further into the level.
d) The scene is not clear, I should not take any action and wait for a logical scene.
"""
