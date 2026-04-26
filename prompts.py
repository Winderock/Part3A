PROMPT = """
You are playing a 2D side-scrolling platformer game independently, your objective is to move to the right as much as possible, falling into gaps will reset your progress, but you can jump over them.

Your character is a white cat centered in the screen.
Use keys "a" to move left and "d" to move right, 
"space" to jump (can be combined with "a" or "d" to jump left or right).

You will be given consecutive game frames from oldest to newest.
Base your action on the latest frame, but use earlier frames to infer motion and situation, then reason step by step for a reasonable action to progress:
1) By looking into the latest frame, use a few sentences to describe your character's position and relative position to any gaps or obstacles.
2) By looking into the earlier frames, use another few sentences to infer the motion of yourself and any obstacles.
3) After evaluating the situation, decide what kind of action from the following, in strict JSON format:
{
    "action": "hold_press",
    "keys": ["key1", "key2"],
    "earlier_frames_description": "your description of the motion and situation from the earlier frames",
    "reason": "your reason for the action"
}

hold_press: specify two keys, hold both keys for a period of time.
"""


#  !!!These examples can come from the model's former successful memories.
REASONING_EXAMPLES = """
=======Reasoning Examples=======
a) The scene shows the cat at the left-side edge of a gap, I should jump right by holding "d" and "space" at the same time to get over the gap.
c) The cat is on a plain platform and no gaps or obstacles in front, I should press "d" key to move right and explore further into the level.
"""