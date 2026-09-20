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

### 4.5 Full run - moves, captures and restores, 7,343 episodes

MolmoAct2 from the base checkpoint on all three families (4,956 moves, 1,196 captures, 1,191
restores). After step 3,000 the run resumed from a checkpoint saved at micro-batch 4 with
accumulation 2, so steps count micro-batches of 4. Interpolated execution throughout. Every
10,000 steps it was evaluated on the same 16 random move positions (seed 100):

| step | successes | median error | untouched | knocked / fell | learning-rate decay |
| --- | --- | --- | --- | --- | --- |
| 13,000 | 3/16 | 53.9 mm | 4 | 9 | 54% |
| 23,000 | 6/16 | 20.5 mm | 5 | 3 | 96% |
| 33,000 | 5/16 | 20.2 mm | 5 | 3 | floor |
| 43,000 | 5/16 | 28.6 mm | 5 | 4 | floor |
| 53,000 | 7/16 | 24.6 mm | 3 | 5 | floor |

Stopped at step ~57,400 (0.28 epochs). Final evaluation of step 57,000, seed 100, 300 control
steps per episode:

| family | episodes | successes | named piece engaged | median error |
| --- | --- | --- | --- | --- |
| moves | 32 | **15/32 (47%)** | not measured | 12.9 mm |
| captures | 16 | **9/16 (56%)** | 10/16 | 7.4 mm |
| restores | 16 | **1/16 (6%)** | 2/16 | 157.5 mm |

- On the 16 comparison positions step 57,000 scored 6/16, level with every checkpoint since
  23,000: the steps trained at the floor learning rate bought nothing measurable. The 16 new
  positions scored 9/16, so the comparison set is harder than average.
- Captures that engaged the piece nearly all succeeded, placing it 2.0-8.0 mm from its tray slot
  (tolerance 30 mm, looser than a move's 11 mm, so the rates are not comparable). Six of the seven
  failures were the named piece never being moved.
- Restores fail the same way, far more often: the loose piece was moved in 2 of 16 episodes.
  Two explanations were measured and ruled out. The horizon: restore expert episodes run a
  median 256 control steps with 8% over the 300 allowed, the same as moves (11%). Visibility:
  loose pieces project to u = 23-127 px in the 640 px overhead image, none clipped, and the tray
  where captures succeed sits at u = 79, inside that range. Restores are the one family whose
  instruction does not name the source square; the policy did not learn to find the piece.
- Across all three families the dominant failure is the policy never engaging the named piece,
  not mishandling it once grasped.

---

### 4.6 Moves and captures, re-collected - 6,095 episodes, 70,000 steps

MolmoAct2 from the base checkpoint on the two families that survived the 2026-09-14 review
(4,903 moves, 1,192 captures; 497,565 frames at 10 Hz). Everything the earlier run got wrong
was changed at once: contact fixed so pieces cannot sink through the board, restores dropped,
the board shifted up to 10 mm per axis and the arm started up to 0.1 rad off its parked pose in
every episode, batch 8 with no accumulation, and `scheduler_decay_steps` sized to the run rather
than LeRobot's fixed 24,000. 70,000 steps is 1.13 epochs. Evaluation every 10,000 steps on 32
moves and 16 captures (seed 100), 450 control steps per episode, interpolated, with the same
layout randomization the training data has - a harder test than §4.5's fixed board and 300 steps.

| step | total | moves | captures | named piece picked | median error | learning-rate decay |
| --- | --- | --- | --- | --- | --- | --- |
| 10,000 | 5/48 | 2/32 | 3/16 | 8/16 | 109.0 mm | 14% |
| 20,000 | 11/48 | 7/32 | 4/16 | 12/16 | 92.4 mm | 29% |
| 30,000 | 19/48 | 10/32 | 9/16 | 12/16 | 20.3 mm | 43% |
| 40,000 | 27/48 | 18/32 | 9/16 | 14/16 | 9.1 mm | 57% |
| 50,000 | 32/48 | 20/32 | 12/16 | 14/16 | 7.5 mm | 71% |
| 62,500 | 27/48 | 18/32 | 9/16 | 14/16 | 9.7 mm | 89% |
| **70,000** | **35/48** | **25/32 (78%)** | **10/16 (63%)** | **16/16** | **5.5 mm** | floor |

Final checkpoint `outputs/molmoact2_mc/checkpoints/070000`: moves median 4.8 mm, captures median
7.0 mm. Of 31 move episodes shown in the log, 24 succeeded, 2 ended with the piece fallen and 3
upright just outside the 11 mm tolerance. Five of the six capture failures happened after a
correct grasp - three fell, one disturbed neighbours - and the named piece was engaged in every
one of the 16.

**Confirmation run on 96 episodes** (seed 100, same checkpoint, recorded): moves **54/64 (84%)**,
captures **25/32 (78%)**, named piece engaged in 30/32 captures, both medians 5.3 mm. Of the 10 move
failures, 2 dropped the piece, 4 ended upright within 25 mm of the square and 4 beyond it, and 2
disturbed a neighbour; of the 7 capture failures, 3 dropped the piece. Reels:
`sim/out/vla_video/reel_mc70k_{moves,captures}.mp4`.

- **A score on 48 episodes carries a spread of several successes.** The 96-episode run repeats the
  training-time evaluation's positions before going beyond them, and on the 29 move episodes that
  line up it scored 24 against 22 - with **8 outcomes flipped** (5 fail->OK, 3 OK->fail). Same
  weights, same board, different rollout: the policy samples its actions. Differences of this size
  between checkpoints (50,000 at 32/48 against 62,500 at 27/48) are therefore not evidence of
  anything; compare checkpoints on 96 episodes or more, or on the same rollout seed.
- **Sizing the decay to the run changed the shape of the curve.** In §4.5 the score stopped
  moving once the learning rate hit its floor at 24,000 (6/16, 5/16, 5/16, 7/16 from 23,000 to
  53,000). Here it rose at every evaluation but one and was still rising at the end, which
  suggests the budget, not the data, was the limit. The single dip (62,500 at 27/48 between
  32/48 and 35/48) is within the noise of 48 episodes.
- **Grounding is solved for these instructions; handling is what is left.** The final policy
  engaged the named piece in 16/16 captures against 10/16 in §4.5, and the old run's dominant
  failure - never touching the piece - is gone. What remains is dropping a piece or setting it
  down a few millimetres out, and captures trail moves because the piece is carried further.
- **The randomization cost nothing measurable.** Despite the shifted board, the varied start
  pose and a 50% longer horizon, moves went from 15/32 to 25/32 and captures from 9/16 to 10/16.
- **Run cost.** 6.3 h to record the moves, 1.7 h the captures, 3.2 h to export, and 47 h of
  training and evaluation (2026-09-14 16:55 to 2026-09-16 15:34) at ~2.0 s/step on the 5090. An
  accidental reboot at 04:17 on the last day cost about 500 steps: the user service restarted,
  resumed from its last checkpoint, and shifted the 60,000 evaluation to 62,500.

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

### 5.6 The policy was acting three seconds blind

MolmoAct2 predicts a 30-action chunk and `select_action` pops one action per call, regenerating
only when the queue empties. With targets held for three control steps, one forward pass covered
90 control steps - **three seconds** - and about five forwards ran in a 450-step episode. Executing
only the first five actions of each chunk and re-planning scored **82% against 75%** on the same
128 positions and halved dropped pieces (8 against 19, paired McNemar **p = 0.043**), with the
median placement error unchanged at 5.1 mm. A 300-position re-run (§5.7) puts the overall gap at
four points rather than seven, and locates all of it in captures. No retraining: one config field, and five seconds more
per episode.

The curve is not monotone, and the far end is catastrophic:

| actions per model call | open loop | success (128 positions) | dropped |
| --- | --- | --- | --- |
| 30 (trained default) | 3.0 s | 96/128 (75%) | 19 |
| 10 | 1.0 s | 100/128 (78%) | 15 |
| **5** | **0.5 s** | **105/128 (82%)** | **8** |
| 3 | 0.3 s | 84/128 (66%) | 20 |
| 1 | 0.1 s | 3/128 (2%) | 33 |

Each chunk is drawn from fresh noise, so re-planning often replaces one committed motion with many
disagreeing ones; at one action per call the arm never commits to a grasp at all. A fifth of the
chunk is the best of the settings tried. Sampling noise is the other half of this story - the same
weights on the same 128 positions disagree with themselves on 45 of them - and more
flow-matching steps do not settle it (§5.7).

Protocol note: these comparisons replay identical positions in every arm and are read with paired
tests, because at 80% on 128 episodes the unpaired interval is ±7 points - wider than every
difference in the table.

### 5.7 The horizon is a capture fix, not a general one - and flow steps buy nothing

§5.6 read a 128-position run as "five actions per call beats thirty, 82% against 75%".
Re-run on 300 positions per arm, every arm facing the identical board, shift and start
pose, that gap shrinks to 80% against 76% and stops being significant on its own
(paired 52 to 38, p = 0.17). Split by what the arm was asked to do, the reason is plain:

| | five per call | thirty per call | paired |
| --- | --- | --- | --- |
| moves (200 positions) | 156/200 (78%) | 159/200 (80%) | 29 to 32, p = 0.80 |
| captures (100 positions) | **85/100 (85%)** | 68/100 (68%) | 23 to 6, **p = 0.002** |

The horizon does nothing for a move and a great deal for a capture. The failure counts
say why: "still on the board" - the piece never reaches the tray - happens 24 times at
thirty actions per call and 10 times at five. A capture is the longer trajectory, out
past the board edge and down into a slot, and three seconds of open-loop motion is
where it dies. A move is short enough that one chunk covers it.

It is not free: five actions per call means 30 model calls in a 450-step episode against
5, and at 0.8-3.3 s per call (measured on a 5090) that is the difference between a loop
that can keep up with a 30 Hz arm and one that cannot. In simulation nothing is waiting,
so five is the better setting; on hardware the arm does not pause for inference.

Also measured, and worth the negative result:

| flow-matching steps per chunk | success (100 positions) | paired against 10 |
| --- | --- | --- |
| 10 (default) | 80/100 (80%) | - |
| 20 | 82/100 (82%) | 10 against 12, p = 0.83 |
| 40 | 82/100 (82%) | 11 against 13, p = 0.84 |

Two episodes either way: integrating the flow more finely costs two to four times the
inference and returns nothing. `tools/sweep_inference.py` produced all of this - it scores
every arm on the same positions in one process, because loading the checkpoint costs more
than the episodes do.

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

- **The far side of the board fails, and the obvious reasons are not why.** Across five
  evaluations of the full run (steps 13K-53K, the same 16 positions each time), five positions
  never succeeded once: h7->h6, g6->g7, h7->h8, a7->c6, a5->a6. Their source squares average
  276 mm from the arm base, against 174 mm for the six positions that succeeded at least 3 of 5
  times; move length is the same in both groups (35 vs 34 mm). Four explanations were measured
  and ruled out. Coverage: the never-solved squares appear in 182 training episodes on average
  against 202 for solved ones. Expert quality: the scripted expert succeeds on 97-100% of
  recordings from every rank, median error 0.8-1.0 mm. Perception: a rank-7 square is 42.0 px
  wide in the overhead camera against 44.7 px on rank 2 (0.94x). Kinematic amplification: a
  1-degree error in the worst joint moves the fingertip 4.7 mm near the base and 4.8 mm far
  away (1.01x). Not ruled out: the base joints' own sensitivity roughly triples with reach
  (pan 1.7 -> 4.9 mm/deg, lift 2.8 -> 5.0) while the elbow's falls (4.7 -> 3.6), so a policy
  whose errors sit in pan and lift would still degrade with distance. Next test: compare the
  policy's joint commands with the expert's, joint by joint, on these five positions.
  Correction after the final evaluation: 16 further positions at step 57,000 included far-side
  successes - b8->c6, its source 313 mm from the base, placed at 4.8 mm, and e7->d7 at 277 mm -
  so distance alone does not predict failure. Individual positions also swing widely between
  checkpoints (b6->c7 succeeded three evaluations running, then missed by 618 mm). The five
  positions above still failed, but the conclusion drawn from them was too strong.

- **Killing processes by pattern.** `pkill -f <pattern>` repeatedly matched and killed the
  wrapper shells doing the killing. List PIDs first, then kill explicit numeric PIDs.

- **Pieces passed through the board.** A recording from the final evaluation showed a piece
  pushed down into the board and out of sight. Measured on the scene as it stood: a 10 N
  push on an upright pawn sank it 41 mm - through the 12 mm board and into the table - and
  the real gripper driven 10 mm onto a rook's top sank it 18 mm. The mechanism is MuJoCo's
  soft contact: with impedance `d` the constraint cancels only a `d` share of the pushing
  acceleration, and the rest has to be balanced by the penetration-proportional term. At the
  default 0.9-0.95 that leaves 5-10% of the push unopposed, harmless for a kilogram and fatal
  for a 1 g piece, which is 1,000 g of acceleration under 10 N. Three fixes were measured on
  the same push, press and 24 identical scripted episodes per family:
  explicit stiffness (`solref = (-1e5, -1e3)`) flung every piece in every episode, idle
  pieces drifting 640 mm; a 5 ms time constant on every geom held 5 N to 0.8 mm but
  tunnelled at 10 N (-20 mm); near-unit impedance `(0.999, 0.9999)` on the board and table
  only, given `priority=1` so their setting wins against the pieces, holds 5-20 N to
  0.3-0.7 mm and the gripper press to 0.6 mm with no change to piece-piece or pad-piece
  contact. Making pieces 5x heavier changed nothing at any setting. An upright push of 30 N
  still tunnels - at 1 g that is 60 mm of travel in one 2 ms step, before any contact can
  act - but no servo in the arm can deliver it: the press test peaked at 15-40 N *total*
  contact force including the jaw squeeze.
- **Stiffer contact exposed a burial bug in `displace()`.** Every stiff variant failed 1-4
  of 24 scripted restores that the old contact passed, all with the piece hundreds of mm
  from its square. Tracing one: the loose piece was never picked up because it was lying
  on its side before the arm arrived. `displace()` set the piece's body origin - its centre,
  21 mm above its base - at table height, so every loose piece started 21 mm inside the
  table. Soft contact oozed it out over five steps; near-rigid contact ejected it and it
  toppled. `reset()` had always placed pieces through `_place()`, which subtracts the base
  offset; `displace()` now does too. Consequence for the data: all 1,191 restore episodes in
  the published dataset open with the loose piece rising out of the table, which the policy
  saw at every restore start. With both fixes: 24/24 moves, 24/24 captures, 24/24 restores
  on the identical episodes, medians 0.9 / 1.4 / 1.1 mm against 0.9 / 1.3 / 0.9 mm before.

- **The board and the arm's start now vary between episodes.** Every earlier recording had
  the board centred exactly and the arm leaving one parked pose, so a policy could map an
  instruction to joint angles without looking at the board; a real board is set down wherever
  it lands. From 2026-09-14 `reset(rng=...)` shifts the board up to 10 mm on each axis (a third
  of a square) and starts each joint up to 0.1 rad off its parked pose, and collection and
  evaluation both draw it. The board had to become a mocap body: moving a static world geom at
  runtime moved its image but not its collision surface, because MuJoCo fixes static collision
  bounds at compile time, and a piece set on the strip the board newly covered fell to the
  table. Scripted expert on fresh seeds: moves 162/168 (96%) randomized against 166/168 (99%)
  nominal, captures 24/24 in both, median error 1.2-1.4 mm against 0.9-1.1 mm. The extra
  failures sit at the edges of the workspace - a destination shifted to 91 mm from the arm
  base, the far corner at 324 mm with a jittered start - and cost only collection time, since
  only verified successes are exported.

---

## 8. Open questions

1. **Does grounding scale from 4 instructions to 1,252?** Rung 1 passed with four. The full
   set is three orders of magnitude larger, and nothing yet tests that.
2. **What step budget does the full dataset need?** §5.4 says count steps, not epochs. The
   failed runs stopped at 6,000–10,000; rung 1 was already working at 8,000 with four
   instructions.
3. **Piece identity.** Every instruction names a square. The restore family (`put the loose
   piece on e4`) was the one exception, dropped on 2026-09-14 after scoring 1/16: a visual search
   the other families are not, and a magnetic board removes most of the need. Recovering a
   knocked-over piece properly would need the piece named by type - recognition never yet
   exercised - and a grasp for a piece lying on its side.
4. **Sim-to-real geometry.** Board square size, camera height and field of view must match the
   physical rig, or none of this transfers.
