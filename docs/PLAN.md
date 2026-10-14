# Contact-Annotated Judge for a Frozen Dexterous VLA — Plan

Status: draft, 2026-10-14. Owner: Jaewon. Standalone project.

## 1. Claim

A small open-weight VLM, fine-tuned as a boundary-level **judge** on RGB frames with per-fingertip contact forces drawn onto them, catches and recovers grasp/release failures of a frozen dexterous-hand VLA better than the same judge trained on plain frames. The executor is never fine-tuned.

Pre-registered comparison: executor + judge(overlay) vs executor + judge(plain), same executor, same seeds, same tasks. Executor-alone is the floor; a privileged oracle judge is the ceiling.

Follow-up (same data, deferred): fine-tune the executor itself on overlay frames and compare to its native tactile stream.

Primary references: Visible Touch (contact rendering from replayed sim contacts; fixed-taxel real-world rendering); RLDX-1 technical report and repo (model, GR-1 Tabletop checkpoint and eval); Set-of-Mark-style visual prompting (marker reading by VLMs).

## 2. Platform

| Component | Choice | Notes |
|---|---|---|
| Benchmark | GR-1 Tabletop (Fourier GR-1 humanoid, dexterous hands, Isaac) | First-party RLDX-1 checkpoint, reported 58.7% |
| Executor | RLDX-1-FT-GR1, frozen, served by its own inference server | Touch-blind as released |
| Sensor | Contact sensors on fingertip and palm links, both hands; 4 taxels per fingertip (2×2), 1 per palm | Isaac Lab contact-sensor API; force vectors in link frame |
| Judge | Qwen3-VL-8B, LoRA; frozen first | Multi-image input, Apache-2.0, same backbone family as the executor |
| Planner | Haiku 5.5, called at init and on replan only | Cost caps; sees the judge's reason text |
| Compute | Local 5080 for development and smoke tests; Slurm A100s for rollouts at scale and the judge LoRA | Executor fine-tunes are out of scope for this phase |
| Fallback | RoboCasa Kitchen + RLDX-1-FT-ROBOCASA (MuJoCo, Panda gripper) | Only if Phase 0 gates fail |

## 3. Data format: raw trajectories, overlays rendered later

Overlays are **never** baked into the stored data. Every rollout is saved so that any overlay style can be re-rendered offline from the record.

Per episode, one directory with:
- `frames/<camera>/<step>.png` — raw RGB from every camera the executor consumes, at render resolution.
- `depth/<camera>/<step>.npz` — depth buffer per frame (for occlusion tests).
- `camera.json` — intrinsics and extrinsics per camera (per step if any camera moves, e.g. wrist cameras).
- `proprio.parquet` — per step: joint positions, finger joint positions, hand-closure command, link poses for every fingertip and palm (world frame).
- `taxels.parquet` — per step, per taxel: site id, link, local position, force vector (link frame) and world frame.
- `events.parquet` — recorder output (privileged): object poses, contact pairs, and derived events with step indices: reach, first_close, grasp_ok, lift, drop, place, release, success.
- `boundaries.parquet` — per boundary: step index, hand, type (grasp/release), settled step index, subtask text, retry count.
- `meta.json` — task, seed, executor checkpoint hash, commit hash, config path, job id.

`render_overlay(episode_dir, step, camera, style_cfg) -> image` is a pure function of the stored data and a style config (arrow thickness, colors, clip, occlusion marking, taxel aggregation). Changing the style never touches the rollouts; a new style = a new config file and a re-render job.

## 4. Overlay design (default style config, frozen at the end of Phase 1)

- One marker per fingertip, never per taxel: arrow from the projected fingertip link origin; direction = projected sum of the fingertip's taxel vectors; length = magnitude clipped at the 95th percentile measured on Phase 1 data.
- Palm: one filled disc, radius = magnitude bin.
- Colors: five finger colors, identical across hands; left/right by marker outline. Six colors total.
- Occlusion: dashed outline when the anchor is behind the depth buffer.
- Contact lost since the previous boundary: small X at the anchor.
- Thickness defined in **model-input pixels**: shaft ≈ 1.5% of the shorter side at the judge's input resolution, scaled up to render resolution; arrowhead ≈ 3× shaft; filled, 1 px dark outline. Verified by pushing frames through the judge's exact preprocessing and inspecting the result.
- No text in the image. A one-line legend in the prompt, overlay condition only.

## 5. Judge specification

Inputs per boundary: current frames from all cameras (plain or overlay by condition), the previous boundary's frames and its verdict, subtask text, boundary type, retry count, legend (overlay condition).

Output, one JSON object:
- `outcome`: success | fail | unclear
- `plan_still_valid`: yes | no (same subtask, same target, retry is worthwhile)
- `reason`: one sentence

Gate rule (code, not prompt):

| judge | gate |
|---|---|
| success | continue |
| fail, valid | scripted recovery: open hand, lift along approach, resume executor on the same subtask; cap N retries |
| fail, invalid / unclear twice / cap hit | replan; planner receives `reason` |
| budget cap | abort |

## 6. Phases

### Phase 0 — Bring-up (screening; local smoke tests, cluster for numbers)

1. Install RLDX-1 (uv, pinned commit) and its GR-1 Tabletop eval; reproduce the published SR on all tasks, 20 episodes each, 1 seed. **Gate:** within 10pp of reported.
2. Add contact sensors to the hand assets. Probe script: fingertips pressed together, grasp of a known-mass object, open in air; report per-taxel readings and step-time overhead.
3. Harness: episode loop around the inference server with the recorder (Section 3 format) in the observation path. Isolation test: the executor process cannot import the recorder.
4. Boundary detector: hand-closure command crossing with hysteresis, then a settle window until finger joints stop; per hand. Verify on 10 episodes by eye that boundaries land after the fingers settle.
5. Task selection: keep tasks with 30–70% executor-alone SR and at least one grasp. Label 5 failures per task from video: slip / drop / reach-miss / wrong-object / other. **Gate:** ≥3 tasks in band with mostly slip/drop failures; otherwise switch to the fallback and stop.

### Phase 1 — Rollout dataset and renderer (cluster for rollouts; local for renderer)

1. Collect executor-alone rollouts on the selected tasks: a few hundred episodes, fixed seed list, Section 3 format. Target several thousand boundaries.
2. Implement `render_overlay` and the default style config. Render 20 random boundaries per task; inspect: arrows only during contact, colors stable, anchors on the fingertips, no drift.
3. Preprocessing survival: run rendered frames through the judge's exact preprocessing; every arrow must remain clearly visible. Adjust thickness, repeat, then freeze the style config.
4. Dataset statistics: force distribution (sets the clip), contact rates per finger, boundary counts per type, label balance.

### Phase 2 — Labels

From `events.parquet`, per boundary:
- `outcome`: grasp → object lifted and held for a fixed window after the boundary; release → object seated and stationary after the hand opens.
- `plan_still_valid` (state-based, first version): correct object still within reach and subtask unchanged → yes; wrong object held, target out of reach, object stuck to the hand → no.
- `reason`: templated from events ("object slipped from thumb and index after lift", "hand closed with nothing between the fingertips", "released before the object was seated").
- Behavioral `plan_still_valid` (second version, once recovery exists): run scripted recovery on every failed boundary during a dedicated collection pass; label yes if it converted.

Split by episode seed into train / val / held-out. Oversample minority labels (fail-invalid, unclear) for training.

### Phase 3 — Judge, frozen then fine-tuned (1 A100)

1. Frozen Qwen3-VL-8B on the held-out boundaries, three input conditions from the same records: (a) plain, (b) plain + contact as text, (c) overlay. Score `outcome` and `plan_still_valid` accuracy. Also run Haiku 5.5 as a reference judge on the same set.
   **Gate:** if (c) > (a) frozen, the fine-tune is optional; if (c) ≤ (a), fine-tune.
2. LoRA fine-tune Qwen3-VL-8B: LoRA on the language model's attention and MLP projections, vision encoder frozen; input = images + text as in Section 5; target = the JSON; loss = next-token cross-entropy on the target tokens only. One LoRA per input condition (a / b / c), same recipe, same splits, 2 seeds.
3. Report held-out accuracy per field per condition, with confusion matrices for `outcome` and `plan_still_valid`.

### Phase 4 — Closed loop — the pre-registered comparison (cluster)

Conditions, same tasks, same seed list:
- (A) executor alone
- (P) executor + judge(plain) + gate + recovery
- (O) executor + judge(overlay) + gate + recovery
- (X) executor + oracle judge (privileged) + gate + recovery

Screening first (1 seed, 20 episodes/condition) to confirm plumbing and recovery conversion rate. Then the one full protocol: (P) vs (O), 3 seeds × 50 episodes × selected tasks. Report SR, verdict accuracy vs recorder, recovery conversions, replans and planner cost per episode.

Ablations at screening scale only: no recovery (recover→abort); judge without the previous-boundary frames; text (b) in the loop.

### Phase 5 — Deferred: executor fine-tune

Fine-tune RLDX-1-FT-GR1 on the Phase 1 demos/rollouts with overlay frames vs plain vs native physics stream. Same rendered data; multi-GPU; only after Phase 4 lands.

## 7. Working rules

- Screening = 1 seed, 20 episodes. Full protocol = 3 seeds × 50 episodes, once, on (P) vs (O) only.
- Benchmark, Isaac Lab and RLDX-1 model code are never modified; sensors, recorder, renderer, judge and harness live in our package.
- Privileged state only in the recorder and the oracle; isolation test on from day one.
- Overlay style frozen after Phase 1 step 3; every rendered dataset names its style config hash.
- Every choice in DECISIONS.md with its reason; every number marked run (observed) or prior (from a paper).
- Planner: per-run and global dollar caps; every call logged by purpose.

## 8. Cluster workflow (no coding agents on the cluster)

- The cluster is reached only by ssh and git. The coding agent works in the local checkout; the cluster runs committed scripts via `git pull` and `sbatch`.
- Runs where: Phase 0 bring-up and Phase 1 renderer are developed and smoke-tested locally (5080) when Isaac is available locally, then re-run on the cluster for numbers; Phase 1 rollouts, Phase 3 LoRA and Phase 4 full protocol are cluster-only.
- `scripts/install_*.sh` run unattended from a clean clone (venvs, pins, checkpoint downloads to the cluster cache).
- One `*.sbatch` per job; job id, commit hash, config path and style-config hash written into every output.
- Results come back small: per-job JSON/CSV summaries, label tables, a handful of rendered frames, in `results/<phase>/<job>/`, committed or rsynced. Raw episode directories, LoRA weights and videos stay on cluster storage.
- Analysis, tables and plots are generated locally from `results/`.

## 9. Open questions for Phase 0

- Exact sim stack behind GR-1 Tabletop in RLDX-1's eval; contact-sensor availability on the hand links.
- Cameras the GR-1 checkpoint consumes, and the judge's input resolution (sets arrow thickness).
- Whether the benchmark exposes object poses and contact pairs cleanly enough for the recorder.
- How to interrupt and resume the executor mid-episode for recovery (inference-server API).
- Disk per episode in Section 3 format; compression for frames and depth if needed.

## 10. Immediate next actions

1. Install RLDX-1 + GR-1 eval via script; reproduce the published SR (1–2 days).
2. Contact sensors + probe script (1–2 days).
3. Recorder in the Section 3 format + isolation test (1–2 days).
4. Boundary detector + task-selection sweep (1 job).
5. `render_overlay` + default style config against recorded episodes; preprocessing survival check.
