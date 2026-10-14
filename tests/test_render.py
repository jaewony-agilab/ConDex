import numpy as np
import pytest
from conftest import ANCHORS, CAM, make_episode

from condex import episode as E
from condex.camera import Camera
from condex import render as R
from condex.preprocess import JudgeInput, smart_resize, to_model_input
from condex.survival import mark_survival


def test_record_round_trip(episode):
    d = E.EpisodeReader(episode)
    assert d.meta["task"] == "SyntheticTask"
    assert d.frame_steps("ego_view")[:3] == [0, 10, 20]
    assert len(d.taxel_rows(60)) == 21
    assert d.boundaries[0]["type"] == "grasp" and d.boundaries[1]["settled_step"] == 100
    cam = d.camera("ego_view", 60)
    assert np.allclose(cam.project(ANCHORS["middle"])[0], CAM.project(ANCHORS["middle"])[0])
    pos, quat = d.link_pose("R_index_intermediate_link", 60)
    assert np.allclose(pos, ANCHORS["index"]) and quat[0] == 1.0


def test_projection_top_down():
    uv, depth = CAM.project(np.array([0.0, 0.0, 0.5]))
    assert np.allclose(uv, [128, 128]) and depth == pytest.approx(1.0)
    uv_x, _ = CAM.project(np.array([0.1, 0.0, 0.5]))
    uv_y, _ = CAM.project(np.array([0.0, 0.1, 0.5]))
    assert uv_x[0] > 128 and uv_y[1] < 128  # world +x -> image right, world +y -> image up


def test_padded_resized_camera_matches_pad_and_resize():
    c = Camera.from_fovy(50.0, 1280, 800, [0, 0, 2], np.eye(3))
    p = np.array([0.3, -0.2, 0.5])
    uv_raw, _ = c.project(p)
    uv_sq, _ = c.padded_resized(256).project(p)
    assert np.allclose(uv_sq, (uv_raw + [0, 240]) * 256 / 1280)


def test_marks_contact_direction_and_lost(episode):
    style = R.Style()
    d = E.load(str(episode))
    marks = {m.finger: m for m in R.make_marks(d, 60, "ego_view", style)}
    assert marks["index"].contact and marks["index"].direction[0] > 0.99  # +x force -> arrow right
    assert marks["thumb"].direction[1] < -0.99  # +y force -> arrow up
    assert not marks["middle"].contact and marks["middle"].direction is None
    assert marks["palm"].contact  # palm: disc only, direction unused
    # at step 100 the previous boundary (settled 50) had index contact; still in contact -> not lost
    assert not any(m.lost for m in R.make_marks(d, 100, "ego_view", style))


def test_lost_contact_x(tmp_path):
    bounds = [{"step": 15, "hand": "right", "type": "grasp", "settled_step": 20, "subtask": "", "retry": 0},
              {"step": 55, "hand": "right", "type": "release", "settled_step": 60, "subtask": "", "retry": 0}]
    ep = make_episode(tmp_path / "e", forces_at=lambda s: {"index": [4.0, 0, 0]} if 20 <= s < 40 else {}, boundaries=bounds)
    marks = {m.finger: m for m in R.make_marks(E.load(str(ep)), 60, "ego_view", R.Style())}
    assert marks["index"].lost and not marks["middle"].lost


def test_render_is_pure_and_draws_only_with_contact(episode):
    style = R.load_style("configs/style/default.yaml")
    a = R.render_overlay(episode, 60, "ego_view", style)
    b = R.render_overlay(episode, 60, "ego_view", "configs/style/default.yaml")
    assert np.array_equal(a, b)
    plain = E.load(str(episode)).frame("ego_view", 60)
    diff = np.abs(a.astype(int) - plain.astype(int)).max(2) > 0
    u, v = CAM.project(ANCHORS["index"])[0].astype(int)
    assert diff[v, u + 5:u + 20].any()  # arrow to the right of the index anchor
    before = R.render_overlay(episode, 40, "ego_view", style)
    assert np.array_equal(before, E.load(str(episode)).frame("ego_view", 40))  # no contact -> untouched


def test_occluded_anchor_is_marked(tmp_path):
    ep = make_episode(tmp_path / "occ", forces_at=lambda s: {"index": [5.0, 0, 0]}, depth_value=0.5)
    m = {m.finger: m for m in R.make_marks(E.load(str(ep)), 10, "ego_view", R.Style())}
    assert m["index"].occluded  # depth buffer at 0.5 m, anchor at 1.0 m


def test_style_hash():
    a = R.load_style("configs/style/default.yaml")
    assert a.hash() == R.load_style("configs/style/default.yaml").hash()
    assert a.hash() != R.Style(shaft_frac=0.02).hash()


def test_smart_resize():
    assert smart_resize(256, 256) == (256, 256)
    h, w = smart_resize(800, 1280)
    assert h % 32 == 0 and w % 32 == 0 and h * w <= JudgeInput().max_pixels
    assert to_model_input(np.zeros((300, 300, 3), np.uint8)).shape == (288, 288, 3)


def test_survival_default_style_passes_thin_fails(episode):
    d = E.load(str(episode))
    plain = d.frame("ego_view", 60)
    style = R.Style()
    res = mark_survival(plain, R.make_marks(d, 60, "ego_view", style), style)
    assert {r["finger"] for r in res} == {"index", "thumb", "palm"}
    assert all(r["ok"] for r in res), res
    thin = R.Style(shaft_frac=0.003)
    res = mark_survival(plain, R.make_marks(d, 60, "ego_view", thin), thin)
    assert not all(r["ok"] for r in res)


def test_self_contact_can_be_dropped_offline(tmp_path):
    ep = make_episode(tmp_path / "self", forces_at=lambda s: {"index": [4.0, 0, 0]})
    d = E.EpisodeReader(ep)
    for r in d.taxel_rows(10):  # pretend the index force came from another finger
        r["force_self_world"] = r["force_world"]
    on = {m.finger: m for m in R.make_marks(d, 10, "ego_view", R.Style())}
    off = {m.finger: m for m in R.make_marks(d, 10, "ego_view", R.Style(self_contact=False))}
    assert on["index"].contact and not off["index"].contact
