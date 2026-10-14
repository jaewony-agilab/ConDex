"""Judge packets (PLAN.md §5): the same message for every backend, for training and for the loop.

Conditions, built from the same record:
  plain    camera frames only
  text     camera frames + per-fingertip contact as text
  overlay  frames with the contact overlay drawn on + a one-line legend
All images go through preprocess.to_model_input first, so every backend sees the same pixels.
Inputs: current frames from the configured cameras, the previous boundary's frames and its
verdict, subtask text, boundary type, retry count.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from condex import render as R
from condex.episode import EpisodeData
from condex.hands import FINGERS, PALM
from condex.judge.schema import Verdict
from condex.preprocess import to_model_input

CONDITIONS = ("plain", "text", "overlay")

SYSTEM = (
    "You check one hand action of a humanoid robot with two dexterous hands doing a tabletop task. "
    "The images were taken right after one hand finished closing (grasp) or opening (release).\n"
    "Decide:\n"
    "- outcome: for a grasp, \"success\" if the hand holds the object firmly enough to lift and carry it; "
    "for a release, \"success\" if the object was let go and rests in its target. \"fail\" otherwise. "
    "\"unclear\" only if the images do not show enough to tell.\n"
    "- plan_still_valid: \"yes\" if retrying the same subtask on the same object is worthwhile; "
    "\"no\" if the hand holds the wrong object, the object is out of reach or off the table, "
    "or the object is stuck to the hand.\n"
    "- reason: one short sentence on what you see.\n"
    "Answer with one JSON object with exactly these keys: outcome, plan_still_valid, reason."
)

CAMERA_WORDS = {"ego_view": "head camera", "eye_in_left_hand": "left wrist camera", "eye_in_right_hand": "right wrist camera"}


@dataclass
class Item:
    kind: str  # text | image
    text: str = ""
    image: np.ndarray | None = None
    path: str | None = None  # set when the image was saved (datasets)


@dataclass
class Packet:
    system: str
    content: list[Item]
    meta: dict = field(default_factory=dict)

    def images(self) -> list[np.ndarray]:
        return [i.image for i in self.content if i.kind == "image"]

    def text(self) -> str:
        """Text-only view, for logs."""
        return "\n".join(i.text if i.kind == "text" else f"<image {i.path or ''}>" for i in self.content)


def _direction_word(d: tuple[float, float] | None) -> str:
    if d is None:
        return ""
    ang = math.degrees(math.atan2(-d[1], d[0])) % 360  # image y points down
    words = ["right", "up-right", "up", "up-left", "left", "down-left", "down", "down-right"]
    return words[int((ang + 22.5) // 45) % 8]


def contact_text(data: EpisodeData, step: int, camera: str, style: R.Style) -> str:
    """Condition (b): what the overlay shows, as text. Directions are in the `camera` image."""
    marks = R.make_marks(data, step, camera, style)
    lines = []
    for hand in ("left", "right"):
        ms = {m.finger: m for m in marks if m.hand == hand}
        if not ms:
            continue
        parts = []
        for f in (*FINGERS, PALM):
            m = ms.get(f)
            if m is None:
                continue
            if m.contact:
                w = _direction_word(m.direction) if f != PALM else ""
                parts.append(f"{f} {m.magnitude:.1f} N" + (f" pushing {w}" if w else ""))
            else:
                parts.append(f"{f} none" + (" (lost since previous check)" if m.lost else ""))
        lines.append(f"{hand} hand contact: " + ", ".join(parts) + ".")
    return f"Contact sensors (directions as seen in the {CAMERA_WORDS.get(camera, camera)}): " + " ".join(lines)


ImageSink = Callable[[str, np.ndarray], str]  # (name, image) -> saved path


def _frames(data, step, cameras, condition, style, sink, tag) -> list[Item]:
    items = []
    for cam in cameras:
        if not data.frame_path(cam, step).exists():
            continue
        img = R.render_frame(data, step, cam, style) if condition == "overlay" else data.frame(cam, step)
        img = to_model_input(img, style.judge)
        path = sink(f"{tag}_{cam}_{step:05d}", img) if sink else None
        items += [Item("text", f"[{tag}] {CAMERA_WORDS.get(cam, cam)}:"), Item("image", image=img, path=path)]
    return items


def build_packet(data: EpisodeData, b_index: int, condition: str, style: R.Style, cameras: list[str],
                 prev_verdict: Verdict | None = None, sink: ImageSink | None = None,
                 instruction: str | None = None, prev_frames: bool = True) -> Packet:
    if condition not in CONDITIONS:
        raise ValueError(f"condition {condition!r} not in {CONDITIONS}")
    b = data.boundaries[b_index]
    step = int(b["settled_step"])
    instruction = instruction if instruction is not None else data.meta.get("instruction", "")
    head = (f"Task: {instruction}\nCurrent subtask: {b.get('subtask') or instruction}\n"
            f"Check: {b['hand']} hand {b['type']}, attempt {int(b.get('retry', 0)) + 1}.")
    content = [Item("text", head)]
    if b_index > 0:
        p = data.boundaries[b_index - 1]
        pv = (f"outcome={prev_verdict.outcome}, plan_still_valid={prev_verdict.plan_still_valid}, reason: {prev_verdict.reason}"
              if prev_verdict else "not available")
        content.append(Item("text", f"Previous check: {p['hand']} hand {p['type']}; verdict {pv}."))
        if prev_frames:
            content += _frames(data, int(p["settled_step"]), cameras, condition, style, sink, "previous")
    else:
        content.append(Item("text", "Previous check: none (first check of the episode)."))
    content.append(Item("text", "Current images:"))
    content += _frames(data, step, cameras, condition, style, sink, "current")
    if condition == "overlay":
        content.append(Item("text", R.legend_text(style)))
    elif condition == "text":
        content.append(Item("text", contact_text(data, step, cameras[0], style)))
    content.append(Item("text", "Your JSON verdict for the current check:"))
    meta = {"episode": str(data.root), "boundary_index": b_index, "step": step, "hand": b["hand"], "type": b["type"],
            "condition": condition, "style_hash": style.hash(), "cameras": list(cameras)}
    return Packet(SYSTEM, content, meta)
