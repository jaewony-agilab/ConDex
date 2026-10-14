"""Taxel sensor on a real MuJoCo model: force sign, 2x2 cell binning, self-contact exclusion."""

import mujoco
import numpy as np
import pytest

from condex.sim.taxels import TaxelSensor

XML = """<mujoco><option timestep="0.002"/><worldbody>
<body name="robot0_tip" pos="0 0 0.1"><geom type="box" size="0.04 0.02 0.005"/></body>
<body name="robot0_palm" pos="0.3 0 0.1"><geom type="box" size="0.04 0.04 0.005"/></body>
<body name="robot0_knuckle" pos="0.3 0 0.116"><freejoint/><geom type="box" size="0.01 0.01 0.01" mass="0.05"/></body>
<body name="ball" pos="{x} {y} 0.13"><freejoint/><geom type="sphere" size="0.01" mass="0.1"/></body>
</worldbody></mujoco>"""


def settle(x, y, exclude=False):
    m = mujoco.MjModel.from_xml_string(XML.format(x=x, y=y))
    d = mujoco.MjData(m)
    for _ in range(1500):
        mujoco.mj_step(m, d)
    s = TaxelSensor(m, d, {"right": {"index": "robot0_tip", "palm": "robot0_palm"}}, ["robot0_"], exclude_robot_contacts=exclude)
    return {r["site"]: r for r in s.read()}, s


@pytest.mark.parametrize("x,y,cell", [(0.02, 0.01, 3), (-0.02, 0.01, 1), (0.02, -0.01, 2), (-0.02, -0.01, 0)])
def test_resting_weight_lands_in_its_cell(x, y, cell):
    rows, s = settle(x, y)
    w = 0.1 * 9.81
    f = np.array(rows[f"right/index/{cell}"]["force_world"])
    assert f == pytest.approx([0, 0, -w], abs=0.02)  # the ball pushes the pad down
    others = [np.linalg.norm(rows[f"right/index/{c}"]["force_world"]) for c in range(4) if c != cell]
    assert max(others) < 1e-6
    assert np.allclose(rows[f"right/index/{cell}"]["force_link"], f, atol=1e-6)  # link frame = world here
    # palm: another robot link (the knuckle) resting on it registers, like fingertips pressed together
    assert rows["right/palm/0"]["force_world"] == pytest.approx([0, 0, -0.05 * 9.81], abs=0.01)
    assert rows["right/palm/0"]["force_self_world"] == pytest.approx(rows["right/palm/0"]["force_world"])
    assert np.linalg.norm(rows[f"right/index/{cell}"]["force_self_world"]) < 1e-9  # the ball is not robot
    assert len(rows) == 5 and rows["right/index/0"]["site_id"] < rows["right/palm/0"]["site_id"]
    poses = s.link_poses()
    assert poses["link.robot0_tip.pos"] == pytest.approx([0, 0, 0.1])


def test_robot_contacts_can_be_excluded():
    rows, _ = settle(0.02, 0.01, exclude=True)
    assert np.linalg.norm(rows["right/palm/0"]["force_world"]) < 1e-6
    assert np.linalg.norm(rows["right/index/3"]["force_world"]) > 0.9


def test_unknown_link_raises():
    m = mujoco.MjModel.from_xml_string(XML.format(x=0, y=0))
    with pytest.raises(KeyError):
        TaxelSensor(m, mujoco.MjData(m), {"right": {"index": "nope"}}, ["robot0_"])
    with pytest.raises(ValueError):
        TaxelSensor(m, mujoco.MjData(m), {"right": {"index": "robot0_tip"}}, ["gripper0_"])


def test_projection_matches_mujoco_render():
    """condex.camera (from cam_xpos / cam_xmat / fovy, rows top-down) vs MuJoCo's own render."""
    from condex.camera import Camera

    xml = """<mujoco><visual><global offwidth="320" offheight="240"/></visual><worldbody><light pos="0 0 3"/>
    <camera name="c" pos="0.6 -0.5 0.7" xyaxes="0.64 0.77 0 -0.4 0.33 0.85" fovy="50"/>
    <body name="s" pos="0.12 0.05 0.03"><geom type="sphere" size="0.02" rgba="1 0 0 1"/></body></worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    try:
        r = mujoco.Renderer(m, 240, 320)
    except Exception as e:  # noqa: BLE001 - no GL on this machine
        pytest.skip(f"no offscreen GL: {e}")
    r.update_scene(d, "c")
    img = r.render()
    ys, xs = np.nonzero((img[..., 0] > 100) & (img[..., 1] < 60))
    cam = Camera.from_fovy(50.0, 320, 240, d.cam_xpos[0], d.cam_xmat[0].reshape(3, 3))
    uv, depth = cam.project(d.xpos[1])
    assert depth > 0 and abs(uv[0] - xs.mean()) < 1.5 and abs(uv[1] - ys.mean()) < 1.5
