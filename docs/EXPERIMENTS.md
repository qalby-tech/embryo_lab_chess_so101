# Experiment log

A running record of what was tried, what it measured, and what it showed. Kept in the
repository so the numbers survive the machine: several runs were lost to WSL restarts and
the raw logs under `outputs/` are not backed up.

Entries are chronological. Every number here was measured, not estimated; where something
is an estimate or an inference it says so.

---

## 1. Platform

| | |
| --- | --- |
| GPU | RTX 5090, 32 GB |
| Host | WSL2 on Windows, 32 GB system RAM, `.wslconfig` `memory=24GB swap=72GB` |
| Simulator | MuJoCo 3.12, `MUJOCO_GL=egl` with `GALLIUM_DRIVER=d3d12` |
| Policy stack | LeRobot (editable checkout at `~/vla/lerobot_src`), Python 3.12, torch 2.8.0+cu128 |
| Policy | MolmoAct2, `allenai/MolmoAct2-SO100_101`, 5.6 B parameters |

Rendering note: MuJoCo's EGL picks Mesa llvmpipe (CPU, ~300 ms/frame) unless
`GALLIUM_DRIVER=d3d12` routes it to the GPU (22 ms with shadows, 12 ms without). The NVIDIA
EGL vendor is unusable here — no `PLATFORM_DEVICE`.

---

## 2. Scene and task

Board is procedural: **28 mm squares**, 224 mm playing field, 15 mm border, 12 mm slab, table
top at 430 mm. The mini set is forced by the arm — the SO-101 reaches about 330 mm.

The overhead camera sits on a printed mast 160 mm to the arm's right, 600 mm above the base
plate, `fovy=24°` looking down at the board. The wrist camera is `fovy=75°`. Only these two
cameras appear in datasets, matching the real rig; a third "external" camera exists for demo
video only.

Instructions are deliberately spatial, so executing one needs no chess knowledge:

    pick up the piece on e2 and place it on e4

Success is the same rule the scripted expert is held to: the piece ends within **11 mm** of
the target square centre, still upright, and nothing else displaced.

---

## 3. Datasets

| name | episodes | frames | fps | instructions | note |
| --- | --- | --- | --- | --- | --- |
| v1 | 600 | — | 30 | many | first attempt |
| v2 | 4,956 | 1,197,528 | 30 | 1,252 | published to the Hub, 26 GB |
| v3 | 4,956 | 399,749 | 10 | 1,252 | same recordings, `--stride 3` |
| narrow (rung 0) | 300 | 21,630 | 10 | 1 | `e2e4` only |
| rung 1 | 800 | 59,424 | 10 | 4 | `e2e4`, `g2g3`, `a7a6`, `d7d5` |

v2 is at `https://huggingface.co/datasets/XvKuoMing/so101_chess`, filed in the
`so101-datasets` collection.

Collection throughput: **928 episodes/hour** with 4 worker processes after the memory fix
(§7.1); 4.4–4.5 s/episode. Expert success on collection is essentially total — 800/800 on
rung 1, 300/300 on the narrow set.

---

## 4. Runs

### 4.1 Full dataset, 30 Hz (v2)

| step | successes | median error | loss |
| --- | --- | --- | --- |
| 2,000 | 0/6 | 33.8 mm | — |
| 6,020 | 0/6 | 57.2 mm | 0.628 |
| 10,020 | 0/6 | 102.7 mm | 0.539 |

Loss fell 30% while placement error tripled. An earlier v1 run (600 episodes, 4,000 steps,
loss 2.025 → 0.667) scored 0/20 at a median of 33.8 mm.

### 4.2 Full dataset, 10 Hz (v3)

| step | successes | median error |
| --- | --- | --- |
| 4,000 | 0/6 | 48.7 mm |
| 6,000 | 0/12 | 39.6 mm |

### 4.3 Rung 0 — one move, 300 episodes

Trained from the base checkpoint, 12,000 steps = 4.44 epochs, final loss 0.154
(action-flow component 0.003). Evaluated on unseen positions of the same move.

| step | epochs | successes | toppled | successful placements |
| --- | --- | --- | --- | --- |
| 3,000 | 1.1 | 3/8 | 5 | 6.0, 1.6, 1.6 mm |
| 6,000 | 2.2 | 4/8 | 4 | 1.8, 1.7, 3.6, 2.2 mm |
| 9,000 | 3.3 | 3/8 | 4 | 4.8, 1.2, 1.3 mm |
| 12,000 | 4.4 | 12/16 | 4 | median 5.2 mm |
| 12,000, interpolated | 4.4 | **16/16** | 0 | **median 2.1 mm** |

### 4.4 Rung 1 — four moves, 800 episodes

Step 8,000 (1.08 epochs), interpolated execution, 16 episodes, 4 per move:

**13/16 successes, correct named piece in 14/16, median error 3.6 mm.**

| move | board region | success | right piece |
| --- | --- | --- | --- |
| e2→e4 | near centre | 4/4 | 4/4 |
| g2→g3 | near right | 3/4 | 3/4 |
| a7→a6 | far left | 3/4 | 4/4 |
| d7→d5 | far centre | 3/4 | 3/4 |

Two of the three failures picked the right piece and placed it accurately (4.1 mm, 3.8 mm)
but brushed a neighbouring piece; one genuinely missed (54.1 mm).

The run was stopped here at the user's request; it is resumable from step 6,000.

---

## 5. Findings

### 5.1 The action label was a copy of the next observed state

Recorded at 30 Hz the servo tracks its commanded target within a degree, so:

| measurement | value |
| --- | --- |
| mean \|action[i] − state[i]\| | 0.078–0.817° per joint |
| mean \|action[i] − state[i+1]\| | **0.008–0.383°** per joint |
| joint ranges over an episode | 16.9–111.0° |

A policy could score well by echoing its own input and never grounding the instruction.
Two further measurements sharpened this:

- **The starting pose is identical in every episode** — standard deviation 0.000° on all six
  joints. At frame 0 the state carries no information, so the entire decision must come from
  the image and the text.
- **The decisive wrist-roll choice is made inside the first 30 frames and then held**, and it
  is multi-modal: −84.7°, −78.0°, −55.8°, +27.5°, +28.1°, +154.5° across sampled episodes.

That is the failure the diagnostics showed: agreement with the expert to 3.5° mid-trajectory
but 11.6° on the first chunk, with wrist roll off by up to 80°.

### 5.2 Reducing the command rate — validated before use

`--stride N` keeps every Nth recorded frame. Chosen by replaying the **expert's own actions**
through the physics at the reduced rate — if a known-good trajectory cannot execute, no policy
trained on it could either. 80 episodes per condition:

| encoding | successes | median error |
| --- | --- | --- |
| stride 1 (control) | **80/80** | 0.9 mm |
| stride 2 → 15 Hz | 73/80 | — |
| **stride 3 → 10 Hz** | **72/80** | 1.6 mm |
| stride 5 → 6 Hz | 11/80 | 10.3 mm (pieces toppled) |
| lookahead 10 | 80/80 | 1.0 mm |

Stride 2 and stride 3 cost the same ~10%, so the loss is inherent to lowering the rate rather
than to lowering it further; stride 3 was chosen because it buys more. Stride 5 is unusable.

What stride 3 actually changes, measured on the exported data:

| | v2 (30 Hz) | v3 (10 Hz) |
| --- | --- | --- |
| mean \|action − state\| | 0.563° | 0.595° |
| mean per-step motion | 0.589° | **1.734°** |
| frames per episode | 216 | 72 |

Note the first row: **striding does not remove the copy shortcut for the first element of a
chunk**. What it does is triple the per-step motion, so echoing the current state across a
30-step chunk costs three times as much, the chunk spans three times as much of the
trajectory, and the decisive opening frames make up three times as much of the data.

### 5.3 Executing sparse targets as steps is what topples pieces

A policy trained at 10 Hz emits one target per three control periods. Holding each as a step
asks the servo for the whole jump at once. Ramping between them instead — same checkpoint,
same 16 positions, same seed:

| execution | successes | median error |
| --- | --- | --- |
| targets held | 12/16 | 5.2 mm |
| **targets interpolated** | **16/16** | **2.1 mm** |

Every toppling failure converted. This is an execution-side fix costing no retraining, and it
should be applied on the real robot too.

### 5.4 Steps predict performance, not epochs

| run | step | epochs | success |
| --- | --- | --- | --- |
| rung 0, one move | 3,000 | 1.1 | 3/8 (37%) |
| rung 1, four moves | 8,000 | 1.08 | 13/16 (81%) |

At the same epoch count and four times the task variety, the run with more gradient steps did
far better. This reframes the full-dataset failures in §4.1–4.2: they were stopped at
6,000–10,000 steps, which on this evidence is where a model of this size is still warming up.

### 5.5 Two rungs of capability, both passed

- **Rung 0** (one move, constant instruction) proves the pipeline end to end: action mapping,
  normalization, cameras, chunking, scoring. It cannot distinguish "follows instructions" from
  "learned one reflex" — by design, since the text never varies.
- **Rung 1** (four moves, instruction is the only difference) proves the policy reads the
  instruction: 14/16 correct piece across four board regions.

Neither settles the full task: the complete dataset has **1,252 distinct instructions**, and
four is not 1,252.

---

## 6. Metrics — and a correction

**Median placement error was misleading and results reported with it should be re-read.**

The error of an episode where the policy never touched the piece equals the distance between
the two squares. With 28 mm squares that is 28.0 mm for an adjacent move and 39.6 mm for a
diagonal — so a "median error of 39.6 mm" was largely reporting board geometry, not policy
quality. Checked against the geometry, 6 of 12 episodes in the v3 step-6,000 evaluation had
errors matching the untouched distance to within 0.1 mm.

Conclusions drawn from median error alone (including reading §4.1's 33.8 → 57.2 → 102.7 as
evidence for the copycat mechanism) are withdrawn. The trend was real worsening — untouched
pieces giving way to pieces knocked further away — but it was not the evidence it was
presented as.

Metrics now used instead:

- **untouched / knocked / carried** — comparing final error against the move distance
- **right piece** — which piece actually moved, versus the one named in the instruction;
  separates a grounding failure from a clumsy grasp of the correct piece
- **success** — the 11 mm / upright / undisturbed rule, unchanged

Re-scored with the grasp-aware metric, all four full-dataset evaluations look alike:

| run | untouched | knocked | carried |
| --- | --- | --- | --- |
| v2 30 Hz @ 6,020 | 3/6 | 3 | 0 |
| v2 30 Hz @ 10,020 | 1/6 | 4 | 1 |
| v3 10 Hz @ 4,000 | 3/6 | 2 | 1 |
| v3 10 Hz @ 6,000 | 6/12 | 5 | 1 |

**Sample size:** 8-episode evaluations proved too noisy to support conclusions — rung 0's
3/8, 4/8, 3/8 was read as a plateau and then became 12/16 in the next block. Evaluations are
now 16 episodes.

---

## 7. Infrastructure

### 7.1 Collection throughput

Every scene rebuild leaks about a gigabyte of driver memory. Four long-lived workers on a
23 GB box ended up 45 GB into swap with the CPU 65% idle. Fixed by building one scene per
process and recycling processes per chunk (`maxtasksperchild=1`):

| | before | after |
| --- | --- | --- |
| rate | 650 eps/hour | **928 eps/hour** |
| per-worker RSS | 5–6.7 GB | 2 GB |
| swap | 45 GB | 0 |

### 7.2 Export

LeRobot's default encoder (libsvtav1 with staged PNGs) took about a minute per episode.
`vcodec="h264"` with `streaming_encoding=True` brought it to ~3 s — about 20× faster. The
codec name must come from `VALID_VIDEO_CODECS` (`"h264"`, not `"libx264"`).

### 7.3 Training

Batch 8 with gradient checkpointing (mandatory — batch 4 without it exhausts 32 GB),
LoRA on the VLM plus a trainable action expert: 737 M trainable of 5.6 B, 29.7 GB VRAM,
**1.9–2.1 s/step**. A 4,000-step block plus a 16-episode evaluation is about 3 hours.

### 7.4 Failure modes worth remembering

- **WSL restarts.** Three in twelve hours, each killing a run. Windows itself stayed up, no
  sleep events, no Linux OOM — Hyper-V logged an orderly VM teardown, and the last thing WSL
  logged before dying was `systemd-journald: Under memory pressure`. `memory=24GB` on a 32 GB
  host leaves Windows ~8 GB. Unproven but the leading explanation; suggested change is
  `memory=20GB`.
- **Partial checkpoints.** A run killed mid-save leaves a directory whose files exist but whose
  optimizer state is missing and whose `training_step.json` is empty — and in one case three
  processor configs were empty too, making the checkpoint unloadable. Existence checks are not
  enough: the step must be read back and the optimizer state confirmed non-empty. Processor
  configs are identical across a run's checkpoints, so a truncated one can be restored from a
  sibling.
- **Transient ffmpeg failures.** An export of 800 recordings died on one video whose metadata
  ffmpeg could not read; the same file exported cleanly on the next run. Skipping unreadable
  episodes silently would thin the data unnoticed, so each gets three attempts before being
  dropped and counted.
- **A corrupt GPU driver library, mistaken for a memory problem.** Training began failing
  with CUDA out-of-memory on a job that had run the day before unchanged. The numbers never
  made sense - 4.60 GiB requested with 12.34 GiB free, later 22 MiB requested with 8.73 GiB
  free - and the failure always landed in the optimizer step. Eliminated by direct test, in
  this order: the dataset (the previous one failed identically), edits to the training script
  (reverting to the committed code failed too), batch size and gradient accumulation (failed
  at micro-batch 2), LoRA rank (failed at 32), `expandable_segments` (verified present in the
  process, no effect), a WSL restart, and a full Windows reboot. A container reached 30 GiB
  where the host capped at 27, so the run was moved into one - where it then *hung* twice at
  different steps, the trainer spinning at 106% CPU in `futex_wait` while the GPU held 30 GB
  at 2% utilization.

  The answer was in `dmesg` the whole time: `file /usr/lib/wsl/lib/libnvidia-gpucomp.so is
  truncated`, alongside 108 instances of `misc dxg: dxgk: dxgkio_escape: Ioctl failed: -22`.
  WSL mounts that compute library from the Windows driver; a truncated copy explains the
  impossible allocation failures, the hangs, and why the container did better (the NVIDIA
  container runtime injects its own copies rather than using the host's). Remedy is a clean
  reinstall of the Windows driver, not any change to the training configuration.

  The lesson is cheap to state and was expensive to learn: when a job that worked yesterday
  fails today with numbers that do not add up, read the kernel log before tuning anything.

- **GPU resets that look like hangs.** With the driver library repaired, training still froze
  at random steps - 259, 908, 1,085, 2,430 - and on the old four-move dataset at 185. Every
  freeze had the same shape: the trainer at ~100% CPU in `futex_wait`, the GPU at idle clocks
  (247 MHz, 29 W, 0% utilization) while still holding its memory. None of batch 8 vs 4 with
  accumulation, 4 dataloader workers vs 0, host vs container, or the full vs the old dataset
  changed it. The Windows System log had the answer: `nvlddmkm` event 153 (level Error) six
  times in twelve hours, at the times runs died - the driver resetting the GPU under sustained
  load, which leaves the CUDA context waiting on work the device no longer has. Nothing appears
  in `dmesg` under WSL; look in the Windows event log. Mitigations adopted: checkpoints every
  500 steps instead of 2,000, and a stall detector that alerts when the step counter stops for
  8 minutes, because a process that is alive and holding memory is not evidence of training.

- **An environment that checks out but cannot train.** After a disk cleanup deleted the
  virtualenv, Miniforge and the Hugging Face cache, the rebuilt environment passed every
  hardware check - torch with CUDA, 31 GiB allocatable, a clean kernel log - and the trainer
  died on its first import: LeRobot had been reinstalled without its `[dataset]` extra.
  Install `lerobot[molmoact2,dataset,training]`; `training/preflight.sh` now imports the
  dataset stack and the policy, so this is caught before a run starts.

- **A learning-rate schedule sized for someone else's run.** The full run's evaluations went
  3/16 at step 13,000, 6/16 at 23,000, then 5/16 at 33,000 and 5/16 at 43,000, with the loss
  flat at 0.49-0.55 from step 27,000. MolmoAct2's default scheduler is a cosine decay over a
  fixed `scheduler_decay_steps: 24000`, independent of `--steps`. The action-expert learning
  rate reached its floor of 5.0e-06 at step ~24,000 and stayed there: 54% decayed at the
  13,000 evaluation, 96% at 23,000, 100% at every one after. Both runs that worked finished well
  above the floor - rung 0 at 50% of the decay, rung 1 at 58% - so no successful run had ever
  trained at it. A second factor halved the data: the resumed run inherited micro-batch 4 with
  accumulation 2, and LeRobot counts micro-batches as steps, so step 42,000 was only 0.28 epochs.
  Size `scheduler_decay_steps` to the run, and budget in samples as well as steps.

- **Killing processes by pattern.** `pkill -f <pattern>` repeatedly matched and killed the
  wrapper shells doing the killing. List PIDs first, then kill explicit numeric PIDs.

---

## 8. Open questions

1. **Does grounding scale from 4 instructions to 1,252?** Rung 1 passed with four. The full
   set is three orders of magnitude larger, and nothing yet tests that.
2. **What step budget does the full dataset need?** §5.4 says count steps, not epochs. The
   failed runs stopped at 6,000–10,000; rung 1 was already working at 8,000 with four
   instructions.
3. **Piece identity.** Every instruction so far names a square. Recovering a piece that is off
   the board requires naming the piece instead — a recognition capability never yet exercised.
4. **Sim-to-real geometry.** Board square size, camera height and field of view must match the
   physical rig, or none of this transfers.
