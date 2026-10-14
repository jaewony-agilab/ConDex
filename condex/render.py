"""Contact overlay, rendered offline from an episode record (PLAN.md §4).

`render_overlay(episode_dir, step, camera, style)` is a pure function of the stored record and a
style config: changing the style never touches rollouts. The closed loop calls `render_frame` on
the live `EpisodeWriter` (same data interface), so offline and online overlays are identical.

Encoding (default style):
  * one arrow per fingertip, from the projected fingertip pad center (the mean taxel position on
    the link; `anchor: link_origin` uses the link origin, see DECISIONS); direction = projected sum
    of that fingertip's taxel force vectors; length grows with |F|, clipped at `force_clip_n`;
    drawn only while |F| >= contact threshold
  * palm: filled disc, radius = magnitude bin
  * fill color = finger (same on both hands, 5 fingers + palm = six colors); outline = hand
    (left white, right black)
  * dashed outline = anchor occluded in this view (depth test)
  * small X at the anchor = that fingertip had contact at the previous boundary and has none now
  * all sizes are set in model-input pixels (preprocess.smart_resize) and scaled to the render
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

from condex import episode as E
from condex.hands import FINGERS, HANDS, PALM
from condex.preprocess import JudgeInput, model_scale

OKABE_ITO = {
    "thumb": [213, 94, 0],  # vermillion
    "index": [230, 159, 0],  # orange
    "middle": [240, 228, 66],  # yellow
    "ring": [0, 158, 115],  # bluish green
    "pinky": [86, 180, 233],  # sky blue
    "palm": [204, 121, 167],  # reddish purple
}
COLOR_WORDS = {"thumb": "vermillion", "index": "orange", "middle": "yellow", "ring": "green",
               "pinky": "sky blue", "palm": "pink"}


@dataclass(frozen=True)
class Style:
    name: str = "default"
    force_clip_n: float = 10.0  # CALIBRATE: 95th percentile of fingertip |F| in contact (Phase 1 step 4)
    contact_threshold_n: float = 0.5
    taxel_aggregation: str = "sum"  # sum | max (the cell with the largest |F|)
    anchor: str = "pad_center"  # pad_center (mean taxel position on the link) | link_origin
    self_contact: bool = True  # False: drop the share from other robot links (taxel rows' force_self_world)
    shaft_frac: float = 0.015  # shaft width / shorter side, in model-input pixels
    head_ratio: float = 3.0  # arrowhead width and length / shaft width
    outline_px: float = 1.0  # model-input pixels
    arrow_min_frac: float = 0.05  # arrow length at the contact threshold / shorter side
    arrow_max_frac: float = 0.16  # arrow length at force_clip_n / shorter side
    probe_m: float = 0.02  # 3D length used to find the projected force direction
    palm_bins_n: tuple = (0.5, 3.0, 8.0)  # |F| bin edges
    palm_radius_frac: tuple = (0.012, 0.02, 0.028)  # disc radius per bin / shorter side
    occlusion_tol_m: float = 0.01
    dash_px: float = 3.0  # model-input pixels
    mark_lost: bool = True
    colors: dict = field(default_factory=lambda: dict(OKABE_ITO))
    hand_outline: dict = field(default_factory=lambda: {"left": [255, 255, 255], "right": [0, 0, 0]})
    judge_input: dict = field(default_factory=lambda: asdict(JudgeInput()))

    @property
    def judge(self) -> JudgeInput:
        return JudgeInput(**self.judge_input)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["palm_bins_n"], d["palm_radius_frac"] = list(self.palm_bins_n), list(self.palm_radius_frac)
        return d

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:12]


def load_style(path_or_style: str | Path | Style | dict | None) -> Style:
    if isinstance(path_or_style, Style):
        return path_or_style
    if path_or_style is None:
        return Style()
    d = dict(path_or_style) if isinstance(path_or_style, dict) else yaml.safe_load(Path(path_or_style).read_text())
    for k in ("palm_bins_n", "palm_radius_frac"):
        if k in d:
            d[k] = tuple(d[k])
    return Style(**d)


# --- what to draw ---------------------------------------------------------------------------


@dataclass
class Mark:
    hand: str
    finger: str  # thumb..pinky or palm
    uv: tuple[float, float]
    in_view: bool
    occluded: bool
    magnitude: float
    direction: tuple[float, float] | None  # unit image-space direction, None if no contact/degenerate
    contact: bool
    lost: bool


def fingertip_forces(rows: list[dict], aggregation: str = "sum", self_contact: bool = True) -> dict[tuple[str, str], np.ndarray]:
    """(hand, finger) -> world-frame force from that link's taxel rows."""
    by_key: dict[tuple[str, str], list[np.ndarray]] = {}
    for r in rows:
        f = np.asarray(r["force_world"], float)
        if not self_contact and r.get("force_self_world") is not None:
            f = f - np.asarray(r["force_self_world"], float)
        by_key.setdefault((r["hand"], r["finger"]), []).append(f)
    out = {}
    for k, fs in by_key.items():
        fs = np.stack(fs)
        out[k] = fs.sum(0) if aggregation == "sum" else fs[np.argmax(np.linalg.norm(fs, axis=1))]
    return out


def links_of(rows: list[dict]) -> dict[tuple[str, str], str]:
    return {(r["hand"], r["finger"]): r["link"] for r in rows}


def quat_to_mat(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def anchor_point(data: E.EpisodeData, rows: list[dict], key: tuple[str, str], link: str, step: int, mode: str) -> np.ndarray:
    pos, quat = data.link_pose(link, step)
    if mode == "link_origin":
        return pos
    local = np.mean([r["local_pos"] for r in rows if (r["hand"], r["finger"]) == key], axis=0)
    return pos + quat_to_mat(quat) @ local


def previous_boundary_step(data: E.EpisodeData, step: int) -> int | None:
    prev = [int(b["settled_step"]) for b in data.boundaries if int(b["settled_step"]) < step]
    return max(prev) if prev else None


def make_marks(data: E.EpisodeData, step: int, camera: str, style: Style) -> list[Mark]:
    cam = data.camera(camera, step)
    rows = data.taxel_rows(step)
    forces = fingertip_forces(rows, style.taxel_aggregation, style.self_contact)
    links = links_of(rows)
    depth = data.depth(camera, step)
    prev = previous_boundary_step(data, step)
    prev_forces = fingertip_forces(data.taxel_rows(prev), style.taxel_aggregation, style.self_contact) if prev is not None else {}
    marks = []
    for (hand, finger), f in forces.items():
        anchor = anchor_point(data, rows, (hand, finger), links[(hand, finger)], step, style.anchor)
        uv, d = cam.project(anchor)
        in_view = bool(d > 0 and 0 <= uv[0] < cam.width and 0 <= uv[1] < cam.height)
        occluded = False
        if in_view and depth is not None:
            x, y = int(uv[0]), int(uv[1])
            occluded = bool(float(depth[y, x]) < float(d) - style.occlusion_tol_m)
        mag = float(np.linalg.norm(f))
        contact = mag >= style.contact_threshold_n
        direction = None
        if contact and d > 0:
            tip, td = cam.project(anchor + style.probe_m * f / mag)
            v = tip - uv
            n = float(np.linalg.norm(v))
            if td > 0 and n > 1e-6:
                direction = (float(v[0] / n), float(v[1] / n))
        had = prev is not None and float(np.linalg.norm(prev_forces.get((hand, finger), np.zeros(3)))) >= style.contact_threshold_n
        marks.append(Mark(hand, finger, (float(uv[0]), float(uv[1])), in_view, occluded, mag, direction, contact,
                          lost=bool(style.mark_lost and finger != PALM and had and not contact)))
    order = {(h, f): i for i, (h, f) in enumerate((h, f) for h in HANDS for f in (*FINGERS, PALM))}
    return sorted(marks, key=lambda m: order.get((m.hand, m.finger), 99))


# --- drawing --------------------------------------------------------------------------------


def _arrow_polygon(start, direction, length, shaft, head_w, head_l):
    ux, uy = direction
    nx, ny = -uy, ux
    sx, sy = start
    body = max(length - head_l, 0.0)
    bx, by = sx + ux * body, sy + uy * body
    ex, ey = sx + ux * length, sy + uy * length
    h, H = shaft / 2, head_w / 2
    return [(sx + nx * h, sy + ny * h), (bx + nx * h, by + ny * h), (bx + nx * H, by + ny * H), (ex, ey),
            (bx - nx * H, by - ny * H), (bx - nx * h, by - ny * h), (sx - nx * h, sy - ny * h)]


def _dashed_polyline(draw, pts, color, width, dash):
    for a, b in zip(pts, pts[1:] + pts[:1]):
        seg = math.dist(a, b)
        n = max(1, int(seg / dash))
        for i in range(0, n, 2):
            t0, t1 = i / n, min((i + 1) / n, 1.0)
            draw.line([(a[0] + (b[0] - a[0]) * t0, a[1] + (b[1] - a[1]) * t0),
                       (a[0] + (b[0] - a[0]) * t1, a[1] + (b[1] - a[1]) * t1)], fill=color, width=width)


def _circle_pts(c, r, n=24):
    return [(c[0] + r * math.cos(2 * math.pi * i / n), c[1] + r * math.sin(2 * math.pi * i / n)) for i in range(n)]


def draw_marks(image: np.ndarray, marks: list[Mark], style: Style, mask: bool = False) -> np.ndarray:
    """Draw marks on a copy of `image`. mask=True draws every mark in white on black (survival check)."""
    h, w = image.shape[:2]
    short = min(h, w)
    s = model_scale(h, w, style.judge)  # render px per model px
    shaft = max(1.0, style.shaft_frac * short)  # shaft_frac is defined on the model input; the ratio is scale-free
    head = style.head_ratio * shaft
    ow = max(1, round(style.outline_px * s))
    dash = max(2.0, style.dash_px * s)
    lmin, lmax = style.arrow_min_frac * short, style.arrow_max_frac * short
    img = Image.new("RGB", (w, h)) if mask else Image.fromarray(np.ascontiguousarray(image)).convert("RGB")
    draw = ImageDraw.Draw(img)

    def colors(m):
        if mask:
            return (255, 255, 255), (255, 255, 255)
        return tuple(style.colors[m.finger]), tuple(style.hand_outline[m.hand])

    def outline(pts, color, occluded):
        if occluded:
            _dashed_polyline(draw, pts, color, ow, dash)
        else:
            draw.line(pts + pts[:1], fill=color, width=ow, joint="curve")

    for m in marks:
        if not m.in_view:
            continue
        fill, line = colors(m)
        if m.finger == PALM:
            if not m.contact:
                continue
            k = int(np.searchsorted(np.asarray(style.palm_bins_n), m.magnitude, side="right")) - 1
            r = style.palm_radius_frac[min(max(k, 0), len(style.palm_radius_frac) - 1)] * short
            pts = _circle_pts(m.uv, r)
            draw.polygon(pts, fill=fill)
            outline(pts, line, m.occluded)
        elif m.direction is not None:
            frac = min(max((m.magnitude - style.contact_threshold_n) / max(style.force_clip_n - style.contact_threshold_n, 1e-6), 0.0), 1.0)
            length = max(lmin + (lmax - lmin) * frac, head * 1.2)
            pts = _arrow_polygon(m.uv, m.direction, length, shaft, head, head)
            draw.polygon(pts, fill=fill)
            outline(pts, line, m.occluded)
        if m.lost:
            k = head / 2
            x, y = m.uv
            for a, b in (((x - k, y - k), (x + k, y + k)), ((x - k, y + k), (x + k, y - k))):
                draw.line([a, b], fill=line, width=max(1, round(shaft)) + 2 * ow)
                draw.line([a, b], fill=fill, width=max(1, round(shaft)))
    return np.asarray(img)


def render_frame(data: E.EpisodeData, step: int, camera: str, style: Style | str | Path | None = None) -> np.ndarray:
    style = load_style(style)
    return draw_marks(data.frame(camera, step), make_marks(data, step, camera, style), style)


def render_overlay(episode_dir: str | Path, step: int, camera: str, style_cfg: Style | str | Path | None = None) -> np.ndarray:
    """PLAN.md §3: image = f(stored record, style config)."""
    return render_frame(E.load(str(Path(episode_dir).resolve())), step, camera, style_cfg)


LEGEND = ("Overlay legend: colored arrows show contact force on each fingertip, drawn from the fingertip "
          "in the direction of the force; a longer arrow means a stronger force (full length = {clip:.0f} N or more); "
          "no arrow = no contact. Colors: {colors}. Outline: white = left hand, black = right hand. "
          "A colored disc on the palm = palm contact (bigger = stronger). Dashed outline = that part is hidden "
          "behind something in this view. A small X = that fingertip touched something at the previous check and has now lost contact.")


def legend_text(style: Style) -> str:
    colors = ", ".join(f"{f} {COLOR_WORDS[f]}" for f in (*FINGERS, PALM))
    return LEGEND.format(clip=style.force_clip_n, colors=colors)
