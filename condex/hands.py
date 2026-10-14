"""Hand and taxel naming shared by the sensor, the record, the renderer and the prompts.

Per hand: 4 taxels on each fingertip link (2x2 grid) and 1 on the palm link -> 21 sites.
site_id = hand_index * 21 + finger_index * 4 + cell (palm: hand_index * 21 + 20).
"""

from __future__ import annotations

HANDS = ("left", "right")
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
PALM = "palm"
CELLS = 4
SITES_PER_HAND = len(FINGERS) * CELLS + 1


def site_id(hand: str, finger: str, cell: int = 0) -> int:
    h = HANDS.index(hand) * SITES_PER_HAND
    if finger == PALM:
        return h + len(FINGERS) * CELLS
    return h + FINGERS.index(finger) * CELLS + cell


def site_name(hand: str, finger: str, cell: int = 0) -> str:
    return f"{hand}/{finger}/{cell}"
