# ConDex: contact-annotated judge for a frozen dexterous VLA

A small open VLM judge (Qwen3-VL-8B, LoRA) checks every grasp and release of a frozen RLDX-1 on GR-1 Tabletop. Its input is camera frames with per-fingertip contact forces drawn on them. The pre-registered comparison is executor + judge(overlay) vs executor + judge(plain). Plan: [docs/PLAN.md](docs/PLAN.md). Every choice and deviation: [DECISIONS.md](DECISIONS.md). Read D1 first: the benchmark runs in MuJoCo, not Isaac.

## Layout

```
condex/
  episode.py            §3 record: EpisodeWriter (live) / EpisodeReader, same read interface
  camera.py             pinhole in MuJoCo convention; pad/resize folded into intrinsics
  render.py             render_overlay(episode_dir, step, camera, style) - pure; Style + hash
  preprocess.py         judge input resize (Qwen smart_resize, factor 32); HF processor check
  survival.py           per-marker visibility after preprocessing (Phase 1 step 3)
  boundary.py           grasp/release from the hand-closure command, hysteresis + settle window
  judge/                packets per condition (plain | text | overlay), schema, vLLM + Claude backends
  gate.py, recovery.py  gate table (§5); joint-space scripted recovery
  planner.py, cost.py   Haiku 5.5 planner (init + replan); dollar caps, per-call ledger
  harness.py            episode loop (sim-agnostic, injected env/executor/sensor/cameras/judge)
  metrics.py            accuracy per field, confusion matrices
  sim/                  MuJoCo side: taxels (2x2 per fingertip + palm), env, executor client, cameras
  privileged/           recorder, oracle, derived events, Phase 2 labels - imported only by scripts/
  train/                judge datasets (JSONL + rendered images), LoRA fine-tune
configs/gr1.yaml        tasks, hands, taxel links, closure (CALIBRATE fields), cameras, labels, judge
configs/style/          overlay style configs (one file per style; hash recorded in every dataset)
scripts/                install_*.sh and one script per phase step (below)
slurm/                  common.sh + one sbatch per job
results/<phase>/<job>/  small outputs (committed); raw episodes, datasets, adapters stay on cluster storage
```

Isolation (§7) is enforced by `tests/test_isolation.py`. Only `scripts/` and `condex/privileged/` import `condex.privileged`; nothing else reads the record's privileged events; importing the executor, harness, judge or renderer loads no privileged module. The executor itself runs in RLDX-1's own venv, where condex is not installed.

## Local (5080 or CPU)

```bash
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest -q        # 45 tests: record, renderer, survival, labels, gate, judge packets,
                                     # MuJoCo taxels + projection vs render, fake-env closed loop, scripts end to end
python scripts/summarize.py sr results/phase0/sweep      # tables from committed results
```

## Cluster (ssh + git only)

One-time, from a clean clone:

```bash
srun --cpus-per-task=8 --mem=32G -t 4:00:00 bash scripts/install_rldx.sh   # RLDX-1 @ cf67c31, sim venv, condex, checkpoint
bash scripts/install_judge.sh                                              # vLLM + peft + Qwen3-VL-8B
cp configs/costcaps.example.yaml configs/costcaps.yaml                     # set caps; ANTHROPIC_API_KEY in the env
```

Env: `CONDEX_RAW` (raw episodes, default `~/condex_raw`), `CONDEX_SCRATCH` (datasets, adapters, logs). Pass partition and account as `sbatch -p … -A …`.

| Phase | Job | Then |
|---|---|---|
| 0.2 | `sbatch slurm/phase0_probe.sbatch` | fill CALIBRATE fields in `configs/gr1.yaml` from `results/phase0/probe/*/probe.json`; look at the `mostclosed_*` PNGs |
| 0.1+0.5 | `sbatch slurm/phase0_sweep.sbatch` (24 tasks × 20 eps) | `summarize.py sr results/phase0/sweep` (gate: within 10pp) |
| 0.4–0.5 | `sbatch slurm/phase0_review.sbatch` | inspect boundary sheets; fill `failure_labels.csv`; set `selected_tasks` |
| 1.1 | `sbatch --array=0-<n-1> slurm/phase1_collect.sbatch` | |
| 1.2–1.4 | `sbatch slurm/phase1_render.sbatch` (`STYLE=…`) | set `force_clip_n` from `stats.json`; iterate the style; freeze it |
| 2 | `sbatch slurm/phase2_labels.sbatch` | |
| 3 | `LABELS=… NAME=ds1 sbatch --export=ALL slurm/phase3_dataset.sbatch` | |
| 3.1 | `DATASET=ds1 CONDITION=overlay JUDGE=vlm sbatch --export=ALL slurm/phase3_eval.sbatch` (× plain/text/overlay, × claude) | `summarize.py judge results/phase3/eval` |
| 3.2 | `DATASET=ds1 CONDITION=overlay SEED=0 sbatch --export=ALL slurm/phase3_train.sbatch` (× 3 conditions × 2 seeds) | eval with `ADAPTER=…` |
| 4 | `CONDITION=overlay ADAPTER=… EPISODES=20 sbatch --export=ALL --array=0-<n-1> slurm/phase4_loop.sbatch` | A / P / O / X; then `STAGE=full EPISODES=50`, one job per `SEED` in 42, 43, 44, for P vs O only |

Every job writes its job id, commit, config path and style hash into its outputs.
