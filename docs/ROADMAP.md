# Roadmap

Where the project stands on 2026-09-28 and what comes next, in order, with what each step
costs and what has to be true before the next one starts. The experiment record
(`EXPERIMENTS.md`) says why each decision was made; this file only says what is being done.

## Where things stand

| published | what | 300 positions, shipped horizon |
| --- | --- | --- |
| `XvKuoMing/so101_chess_molmoact2` | MolmoAct2 + LoRA, 70,000 steps on 6,095 demonstrations | 227 (moves 159, captures 68) |
| `XvKuoMing/so101_chess_molmoact2_dagger` | the same, continued on 1,000 expert corrections | 233 (moves 155, captures 78) |
| `XvKuoMing/so101_chess` (dataset) | the 6,095 demonstrations | |

Settled by measurement, and not to be re-run: more clean demonstrations do not help; a
warm start must run at a tenth of the schedule; the 48-episode in-training checks only
pick a block; the paired 300-position test is the only score that counts. The framework
trains and scores any LeRobot architecture through one config (`TrainConfig` + `RECIPES`).

## Phase 1 - reinforcement learning on the corrected checkpoint

Advantage-weighted fine-tuning: the policy plays, the simulator scores, the better-than-
average rollouts are folded back into training. Corrections from the expert continue as
the failure branch of the same loop, so there is no separate DAgger round.

| step | what | effort |
| --- | --- | --- |
| 1.1 | `ChessGymEnv`: reset(task) / step(action) with the shaped reward and the success rule | half a day |
| 1.2 | batched inference: one policy instance serving several simulator processes | one day |
| 1.3 | collector: K = 4 rollouts per position on training seeds, return = success + shaping, advantage against the position's mean; expert hand-over on a trigger, as now | half a day |
| 1.4 | iteration 1: collect 1,200 rollouts (~2 h batched), keep positive-advantage episodes weighted through `merge_datasets.py --repeat`, continue at a tenth of the rate for 8,000 steps (~4.5 h), paired sweep (1.2 h) | one day |
| 1.5 | iterations 2 and 3 from the new checkpoint | two days |

Exit: a paired win on the 300 positions at the shipped horizon - captures up with moves
held, or overall p < 0.05. Publish as `so101_chess_molmoact2_dagger_rl` with the user's
approval. Expected: +5 to +15 on captures, a few on moves from placement precision.
Six days.

## Phase 2 - the rig, on the user's side, in parallel

Bought parts, no printing: a webcam on an overhead boom arm, the wrist camera already in
hand, a magnetic travel set with 28 mm squares, the arm clamped to the table. Four
measurements go into the config - camera height, camera field of view, arm height above
the table, tray position and size - and the simulation, the reach calibration and every
later run follow them. `hardware.md` has the parts; it is being rewritten for bought
parts and any 28 mm board.

## Phase 3 - a wide training distribution

So that a user can buy any board and any Staunton set. The EmbodiedGen toolchain is
being reinstalled for this.

| step | what | effort |
| --- | --- | --- |
| 3.1 | 20-30 piece sets from text-to-3D, king 40-55 mm, neck 10-16 mm, nothing wider than 22 mm; a few held out of training | two days, GPU between runs |
| 3.2 | board styles: borders with and without labels, tournament colours to wood to black and white, grain, gloss; table textures; camera tilt, height and roll jitter; square size 26-32 mm | one day |
| 3.3 | regenerate the demonstrations at scale on the wide distribution, plus corrections and RL rollouts from the phase-1 checkpoint | one to two days at four workers |

Exit: a dataset an order of magnitude more varied than the current one, with held-out
sets and styles. Publish as `so101_chess_v2` (dataset) with approval.

## Phase 4 - the long run, and the architectures against it

| step | what | effort |
| --- | --- | --- |
| 4.1 | MolmoAct2 from the base on the wide set, two to three epochs (the current model saw one), schedule sized to the run | four days |
| 4.2 | held-out evaluation: unseen piece sets and board styles, the first real number for "any board" | half a day |
| 4.3 | the same data and budget on SmolVLA (recipe ready), then pi0.5 and GR00T, each at its own shipped horizon and at three seconds | one to two days each |
| 4.4 | a leaderboard in `EXPERIMENTS.md` and on the cards; each model published as `so101_chess_<architecture>` | |

Exit: one architecture chosen for hardware by the held-out score. Two weeks of GPU.

## Phase 5 - the real arm

The winning checkpoint on the rig at 10 Hz with the interpolation multiplier; real-time
chunking if the arm pauses between calls; the board read by the camera as the success
rule; corrections recorded from hardware failures the same way as in simulation; a
showcase of one instruction executed in simulation and on the arm. Real recordings
published beside the simulated ones.

## Calendar

| when | what |
| --- | --- |
| 2026-09-29 to 10-04 | phase 1, RL iterations and the `_dagger_rl` verdict |
| 2026-09-29 onward | phase 2 as parts arrive; phase 3.1 generation whenever the GPU is idle |
| 2026-10-05 to 10-08 | phase 3.2-3.3, the wide dataset |
| 2026-10-09 to 10-22 | phase 4, the long run and the architectures |
| after | phase 5 on the rig |

Deferred, and not on this calendar: naming pieces by type, a grasp for a toppled piece,
LingBot-VA (pose-space, needs kinematics in the loop), reward models for hardware
collection.
