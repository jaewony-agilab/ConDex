"""Preprocessing survival (PLAN.md Phase 1 step 3): does every drawn marker stay visible after the
judge's exact preprocessing?

For each drawn mark: draw it alone as a white-on-black mask at render resolution, resize the mask
like the image, and count the footprint pixels where the preprocessed overlay differs clearly from
the preprocessed plain frame. A mark survives if enough of its footprint differs and its shaft is
at least `min_shaft_px` model-input pixels wide.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from condex import render as R
from condex.preprocess import smart_resize, to_model_input


def mark_survival(plain: np.ndarray, marks: list[R.Mark], style: R.Style, diff_thresh: int = 40,
                  min_visible: float = 0.6, min_shaft_px: float = 2.0) -> list[dict]:
    h, w = plain.shape[:2]
    mh, mw = smart_resize(h, w, style.judge)
    over_m = to_model_input(R.draw_marks(plain, marks, style), style.judge).astype(np.int16)
    plain_m = to_model_input(plain, style.judge).astype(np.int16)
    diff = np.abs(over_m - plain_m).max(axis=2)
    shaft_px = style.shaft_frac * min(mh, mw)
    out = []
    for m in marks:
        drawn = m.in_view and ((m.finger == "palm" and m.contact) or m.direction is not None or m.lost)
        if not drawn:
            continue
        mask = R.draw_marks(plain, [m], style, mask=True)[..., 0]
        foot = np.asarray(Image.fromarray(mask).resize((mw, mh), Image.BILINEAR)) >= 128
        n = int(foot.sum())
        vis = float((diff[foot] > diff_thresh).mean()) if n else 0.0
        out.append({"hand": m.hand, "finger": m.finger, "footprint_px": n, "visible_frac": round(vis, 3),
                    "shaft_model_px": round(shaft_px, 2), "occluded": m.occluded,
                    "ok": bool(n > 0 and vis >= min_visible and shaft_px >= min_shaft_px)})
    return out
