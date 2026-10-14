# Decisions

Each entry gives the choice, the reason, and its basis: **run** (observed in this project) or **prior** (from code, docs or a paper). Newest last.

## 2026-10-14: build of the PLAN.md scaffold

**D1. GR-1 Tabletop runs on MuJoCo, not Isaac.** PLAN.md §2 says "Isaac; Isaac Lab contact-sensor API". RLDX-1's GR-1 eval at commit `cf67c31` (`run_scripts/eval/gr1_tabletop/eval_gr1.sh`, `rldx/eval/sim/robocasa-gr1-tabletop-tasks/setup_RoboCasaGR1TabletopTasks.sh`) installs `robosuite==1.5.1` plus `robocasa-gr1-tabletop-tasks` and runs `gym.make("gr1_unified/<task>")`. *prior (RLDX-1 source, read in third_party/).* Consequences:
- Taxels are read from MuJoCo contacts (`condex/sim/taxels.py`), not from Isaac Lab sensors. Reading `mjData` is passive, so the benchmark, its dynamics and RLDX-1 stay unmodified (§7).
- "When Isaac is available locally" (§8) becomes "when the robocasa sim venv is installed locally". MuJoCo already runs on the 5080 (unit tests, D7).
- The RoboCasa fallback is also MuJoCo, so the sensor code carries over.

**D2. Taxel geometry.** The fingertip links are `R/L_thumb_distal_link` and `R/L_{index,middle,ring,pinky}_intermediate_link`; the palm is `r_palm`/`l_palm`. Each fingertip link has 4 taxels: a 2×2 grid over the bounding box of its collision geometry in the link frame, split along the box's two longest axes (`taxels.grid_axes` overrides). Each palm has 1 taxel. A contact adds its force to the cell that contains the contact point. Each taxel row stores the cell centre (link frame) and the force in the link and world frames, which is the force *on the link*. *prior (robosuite 1.5.1 `fourier_{left,right}_hand.xml`: collision geoms are meshes) + run (`tests/test_taxels.py`: a 0.1 kg ball resting on each quadrant of a pad gives (0, 0, −0.981) N in exactly that cell).*

**D3. Contacts with other robot links count, and are also stored separately.** A tactile skin feels a fingertip pressing on the thumb, and "hand closed with nothing between the fingertips" is exactly that pattern. Contacts with the link's kinematic neighbours are dropped using MuJoCo's own parent-filter rule (through weld bodies, world excepted). Each taxel row also stores `force_self_world`, the share of the force that comes from robot links, so a style with `self_contact: false` can drop it offline (§3: any style re-renders from the record). *run: on the real Fourier right-hand MJCF closed around a ball, the index pad read 4.9 N per contact from the thumb's proximal link. The standalone hand file, being welded to the world, also produced 1 cm palm↔proximal interpenetrations; MuJoCo filters these on the arm, and so does our rule.*

**D4. Harness steps one action at a time and replays the 16-step chunk itself.** It uses RLDX-1's `MultiStepWrapper` with `n_action_steps=1`, so every step is visible to the recorder and the boundary detector. Recovery and replan flush the rest of the chunk. Seeding mirrors RLDX-1's `VideoRecordingWrapper` (episode k with base seed s resets with `s·100000 + k`). *prior (RLDX-1 source); to be confirmed by the Phase 0 sweep matching published SR.*

**D5. Phase 0 steps 1 and 5 share one sweep.** Both ask for executor-alone SR on all 24 tasks, 20 episodes, 1 seed. One recorded sweep gives the reproduction numbers (gate: within 10pp) and the task-selection SR, and it provides the failures to label. *design.*

**D6. Stored cameras.**
- `ego_view` is the policy image itself, the 256×256 obs that RLDX-1 consumes: egoview 1280×800, padded to square and resized. Its intrinsics are stored for that 256 image, with the pad/resize folded in (`Camera.padded_resized`), and recorded every step.
- The wrist cameras `eye_in_{left,right}_hand` are rendered by us at 256², for the judge only. They are recorded every 5 steps and at every boundary.
- Every camera is stored as moving (per-step extrinsics), because the head and waist move. Depth is metric float16; the ego depth is rendered at half the raw resolution and padded/resized with nearest-neighbour sampling.

*prior (robocasa obs key `video.ego_view_pad_res256_freq20`, robosuite GR-1 MJCF cameras) + design for strides. Disk per episode (§9) is measured by `scripts/dataset_stats.py`.*

**D7. Projection convention.** The pinhole model comes from `cam_xpos`, `cam_xmat` and `fovy` (the camera looks along −z with +y up), with stored rows top-down. *run: `tests/test_taxels.py::test_projection_matches_mujoco_render` agrees with MuJoCo's offscreen render within 1.5 px under EGL on the 5080.*

**D8. Hand command order and closure.** The 6-D hand command is (pinky, ring, middle, index, thumb-bend, thumb-yaw), mapped to 11 joint drives by `FourierHand.format_action`; the drive ranges are 0 to 1.57–1.74 rad. Dims 1–4 are flexion in this order and in its reverse, so `flexion_dims: [1,2,3,4]`. `open_value: 0.0` and `closed_value: 1.5` remain CALIBRATE until `probe_sensors.py` reports command percentiles. *prior (robosuite MJCF actuators + `format_action`).*

**D9. Overlay anchor = fingertip pad centre (deviation from §4 "link origin").** GR-1's fingertip links are intermediate links, so their origin is the middle knuckle and an arrow drawn there starts mid-finger. The default style uses `anchor: pad_center`, the mean taxel position on the link; `anchor: link_origin` is still available. *run: real-hand overlay render, see D3.*

**D10. Overlay colours and markers.** Okabe-Ito colours, one per finger plus the palm: six colours in total. The outline gives the hand: white = left, black = right. The 1 px outline is drawn in the hand colour, so no seventh colour is needed. The lost-contact X uses the finger colour with the hand outline. The arrow shows the force on the finger (as measured by the taxels), so it points away from the object. *design.*

**D11. The judge's input resolution is set by us.** Images are resized once with Qwen's `smart_resize` rule (multiples of 32 for Qwen3-VL: patch 16 × merge 2; `max_pixels` = 512²) before any backend sees them. vLLM's processor then has nothing to change, and plain, overlay, Claude and LoRA all see the same pixels. At 256² the shaft is 1.5% × 256 ≈ 3.8 model px. `scripts/check_preprocess.py` verifies this against the real HF processor on the cluster. *design; pending run.*

**D12. Label definitions (Phase 2).**
- grasp success = the hand holds the target, lifts it ≥ 3 cm within 60 steps and keeps holding it for 30 steps (or until its own release or task success).
- release success = within 40 steps the hand no longer holds the target and the target sits in the receptacle, still (< 2 cm/s) for 10 steps; or the task succeeded inside that window.
- `unclear` is used only when the episode ends before the window can decide.
- `plan_still_valid=no` when the hand holds a different object, the target is > `reach_limit_m` from the robot base (CALIBRATE) or has fallen ≥ 10 cm, or the target is stuck to the hand after a release.
- Pushes that close a door can be labelled "hand closed with nothing between the fingertips"; review them in Phase 1 before scoring.

*design.*

**D13. Training data comes from executor-alone rollouts only.** Splits are by a hash of (task, episode seed), 70/15/15, so they are condition-independent. The previous verdict shown in a training or offline-eval packet is the previous boundary's *label* (teacher forcing). In the closed loop it is the judge's own previous verdict. *design.*

**D14. Gate details §5 leaves open.**
- A single `unclear` continues; two in a row replan.
- Retries are counted per hand and reset on that hand's success.
- The recovery's own open/close is not a task boundary (the detector is paused during recovery and re-armed open).
- Without a planner, a replan resumes the executor.

*design.*

**D15. Scripted recovery works in joint space.** The hand's flexion dims ramp to `open_value` over 10 steps. The arm replays its last 20 commanded targets in reverse ("lift along approach", no IK) and holds for 5 steps. The executor then resumes with its memory reset (`reset_memory_on_resume`). *design.*

**D16. The planner never rewrites the executor's instruction.** RLDX-1-FT-GR1 was fine-tuned on the benchmark's own instruction strings. At start the planner lists the expected hand events; their text becomes the boundary `subtask`. On replan it chooses resume | recover_then_resume | abort. *design.*

**D17. Episodes stop at first success.** RLDX-1 scores success at any step, so SR is unchanged and judge and planner calls are saved. The labels treat task success inside a window as decisive. *prior.*

**D18. Judges.**
- Qwen3-VL-8B-Instruct served by vLLM (0.11.0 is the first pin and is unverified on the cluster), temperature 0, JSON-schema guided. A LoRA is served with `--enable-lora` under the name `judge_lora`.
- The reference judge is Claude Haiku 5.5 (`claude-haiku-5-5`) at effort `low`, with `output_config.format` JSON schema; a refusal counts as `unclear`.
- Planner and Claude judge refuse to run without `configs/costcaps.yaml` (run and global caps, prices, every call logged with its purpose).

*prior (API docs) + design.*

**D19. The oracle (condition X) reads only the current state at the boundary.** It uses whether the target is held, in the receptacle, held by another hand, or out of reach. Labels look ahead (lift/hold windows), which no online judge can. *design.*

**D20. LoRA recipe defaults.** r=16, α=32, dropout 0.05, applied to `language_model.*.{q,k,v,o,gate,up,down}_proj` only. A runtime check refuses if any vision parameter is trainable. lr 1e-4 cosine with 3% warmup, 2 epochs, batch 1 × grad-accum 16, bf16, gradient checkpointing. Loss is on the assistant JSON tokens only. Training uses `train_oversampled.jsonl` (minority classes up to 0.5× the largest, at most ×5). *design.*

**D21. Self-contained, not built on TDexAgent.** PLAN.md calls this a standalone project. TDexAgent (sibling repo) informed what to check: RLDX-1 commit, env construction, and the robosuite naming suffixes. Every reused fact above was re-checked against the RLDX-1 or robosuite source here, and `condex` imports nothing from `tdex`. *design.*
