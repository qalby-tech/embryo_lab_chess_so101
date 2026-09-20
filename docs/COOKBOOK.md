# Cookbook

Worked examples of the `chess_sim` API: designing an environment, collecting
demonstrations, training, running a policy, scoring it, and reinforcement
learning. Everything here runs against the installed package - the scripts in
`examples/` are these same pieces with argument parsing around them.

```bash
export MUJOCO_GL=egl PYTHONPATH=~/chess_so101
export GALLIUM_DRIVER=d3d12      # WSL2 only: render on the GPU
```

## 1. Designing an environment

Every tunable number is a field on a frozen pydantic model. Compose what you
need and leave the rest at the measured defaults.

```python
import random
import numpy as np
from chess_sim import (AppearanceConfig, BoardConfig, Camera, ChessSimEnv, ClearanceConfig,
                       ControlConfig, EnvConfig, RandomizationConfig, ToleranceConfig, START_FEN)

config = EnvConfig(
    board=BoardConfig(square=0.028, arm_gap=0.08, tray_slots=4),
    control=ControlConfig(hz=30, cameras=(Camera.TOP, Camera.WRIST), image_size=(320, 240)),
    randomization=RandomizationConfig(board_shift=0.010, arm_joint_jitter=0.10),
    tolerances=ToleranceConfig(placement=0.011, tray=0.030, disturbance=0.010, upright_min=0.85),
    clearance=ClearanceConfig(moving_jaw_room=0.020))

env = ChessSimEnv(config)
rng, looks = random.Random(0), np.random.default_rng(0)

observation = env.reset(START_FEN, rng=rng)   # rng: board shifted, arm start jittered
env.recolor(AppearanceConfig.sample(looks))   # colors and lighting only, no recompile
env.export_xml("scene.xml")                   # open with: python -m mujoco.viewer --mjcf scene.xml
```

An observation carries the joint state and one frame per configured camera,
indexed by the camera itself (`observation.images[Camera.TOP]`, HxWx3 uint8,
shape checked on the way in) - and no frames at all when you ask for none with
`env.step(action, images=False)`.

`env.reset(..., rng=None)` puts everything exactly at nominal - use that when
you want two runs to differ only in what you are testing. What was drawn is kept
in `env.layout`; `env.board` is the live geometry (its `origin` moves with the
shift) and `env.position` is the python-chess mirror of what stands on the board.

Piece size and the board texture are baked into the compiled scene, so changing
them means a new `ChessSimEnv`; colors and lighting change on a live one.

## 2. Tasks and the scripted expert

A task carries the instruction a policy would be given, so the text used in
training and the text used in scoring cannot drift apart.

```python
from chess_sim import CaptureTask, FailureReason, MoveTask, Square

env.reset("8/8/4k3/8/3P4/8/8/4K3")
task = MoveTask(from_square=Square.D4, to_square=Square.D5)   # "d4" validates into Square.D4 too
task.instruction          # 'pick up the piece on d4 and place it on d5'
task.label                # 'd4d5'

result = env.execute(task)                         # scripted expert, then scored
result.success, result.placement_error, result.steps
result.reason is FailureReason.PLACEMENT           # None when it succeeded
result.disturbed                                   # [Square.C5, ...]

env.execute(CaptureTask(square=Square.D5))         # take it off the board instead
```

Squares are an enum, so a typo is a validation error rather than a move that
silently never matches; failure reasons are one too, so branching on an outcome
does not mean comparing prose.

The expert plans against simulator state - it is a demonstrator and a yardstick,
not something you can deploy. `env.executable_moves()` and
`env.executable_captures()` say what it can actually do in the current position.

## 3. Collecting demonstrations

A sampler resets the env to a fresh random position and returns a task the arm
can attempt there. The recorder writes one directory per episode with its verdict.

```python
from chess_sim import EpisodeRecorder, MoveSampler, PositionConfig, RecorderConfig, demonstrate

sampler = MoveSampler(position=PositionConfig(min_extra_pieces=2, max_extra_pieces=8))
recorder = EpisodeRecorder(env, RecorderConfig(root="datasets/chess_moves"))

results = demonstrate(env, sampler, episodes=500, rng=random.Random(0),
                      recorder=recorder, record_appearance=True)
print(sum(r.success for r in results), "verified successes")
```

Each episode holds `data.npz` (`observation_state`, `action` at the control
rate), one mp4 per robot camera and `meta.json`; `manifest.jsonl` indexes them.
Train on the successes only - the exporter does that by default:

```bash
python examples/collect_demonstrations.py --episodes 5000 --workers 4 --randomize \
    --task move --out datasets/chess_moves
python examples/export_dataset.py --in datasets/chess_moves datasets/chess_captures \
    --repo-id XvKuoMing/so101_chess --root datasets/lerobot/chess_mc --stride 3
```

`--stride 3` matters: at the full 30 Hz the action label is a copy of the next
observed state, and a policy scores well by echoing its input.

## 4. Training

`MolmoAct2TrainConfig` holds the fine-tuning flags - each one with the reason it
is there - and builds the command:

```python
import subprocess
from chess_sim.training import MolmoAct2TrainConfig

train = MolmoAct2TrainConfig(dataset_root="datasets/lerobot/chess_mc",
                             output_dir="outputs/molmoact2_mc",
                             total_steps=70_000, batch_size=8)

subprocess.run(train.command(steps=10_000), check=True)          # first block
subprocess.run(train.resume_command(steps=20_000), check=True)   # continue

train.complete_checkpoints()          # ['010000', '020000'] - partial ones are deleted
train.checkpoint_path("020000")       # outputs/.../checkpoints/020000/pretrained_model
train.keep_only({"020000"})           # a full checkpoint is ~12 GB
```

Block-wise training with an evaluation between blocks, keeping the best
checkpoint, is `examples/train_policy.py`; `training/train.sh` wraps it with
every setting in `training/settings.env`.

## 5. Running a trained policy

`LeRobotPolicy` reads the checkpoint's own type and input features, so any
policy in LeRobot's registry loads the same way and gets exactly the cameras and
state it was trained on (`policy.cameras` tells you what to render). It has been
exercised here on molmoact2.

```python
from chess_sim import LeRobotPolicy, LeRobotPolicyConfig, MoveSampler, RolloutConfig, run_episode
from chess_sim.hub import MODEL_REPO       # 'XvKuoMing/so101_chess'

policy = LeRobotPolicy.load(LeRobotPolicyConfig.for_dataset(
    checkpoint=MODEL_REPO,                        # or a local checkpoint directory
    dataset_root="datasets/lerobot/chess_mc",     # read for the rate it emits targets at
    n_action_steps=5,                             # +17 points on captures, nothing on moves
    interpolate=True))

task = MoveSampler().sample(env, random.Random(100))
result = run_episode(env, policy, task, RolloutConfig(max_steps=450))
```

Two knobs decide how a chunked policy behaves, and both live on the config.
`interpolate` ramps to each emitted target instead of stepping to it, worth
12/16 -> 16/16 on the same checkpoint. `n_action_steps` is how much of a
30-action chunk to execute before looking again: on 300 paired positions, 5
instead of 30 is worth 17 points on captures (85% against 68%, p = 0.002) and
nothing at all on moves (78% against 80%) - a capture is the longer trajectory,
and three seconds of open-loop motion is where it dies. It costs six times the
model calls, so keep the checkpoint's 30 if the loop has to keep up with a
moving arm. The published `config.json` ships 30; the API defaults to 5.

What a call costs, measured on the 70,000-step checkpoint (RTX 5090, another
job sharing the machine, so read these as upper bounds): loading the checkpoint
4-8 minutes, a network call 0.8-3.3 s, a queued action 68 ms, `env.step` 53 ms
with GPU rendering. With `n_action_steps=5` and targets held for 3 control
steps, the network runs once every 15 control steps - 3 times in the 45 steps
timed, 30 times in a full 450-step episode.

Driving the loop yourself is the same three calls:

```python
policy.reset()
observation = env.observe()
for _ in range(450):
    action = policy.select_action(observation, task)       # 6 joint targets, radians
    observation = env.step(action, images=policy.needs_images())
outcome = env.evaluate(task, before, steps=450)            # before = env.piece_snapshot()
```

## 6. Writing your own policy

Anything with these three methods is a policy; `run_episode` and `evaluate` take
it as-is.

```python
import numpy as np
from chess_sim import Observation, square_index
from chess_sim.tasks import Task

class MyPolicy:
    def reset(self) -> None:
        """Forget anything carried over from the last episode."""

    def needs_images(self) -> bool:
        return False          # rendering is most of the cost of a step - skip it when unused

    def select_action(self, observation: Observation, task: Task) -> np.ndarray:
        action = observation.joint_pos.copy()
        file = square_index(task.source) % 8                 # a1 = 0 ... h8 = 63
        action[0] += 0.01 * np.sign(file - 3.5)              # lean toward it
        return action
```

## 7. Scoring

`evaluate` runs a schedule of `(sampler, episodes)` pairs and scores every
attempt by the same rule the expert is held to.

```python
from chess_sim import (CaptureSampler, MoveSampler, RolloutConfig, TaskFamily, evaluate,
                       wilson_interval)

report = evaluate(env, policy, [(MoveSampler(), 64), (CaptureSampler(), 32)],
                  RolloutConfig(max_steps=450, seed=100))
print(report.summary())
report.success_rate, report.interval, report.median_error
report.by_family()[TaskFamily.CAPTURE].right_piece      # engaged the named piece
```

Rollout sampling alone flips about a quarter of identical positions, so compare
checkpoints on the same seed and read the interval, not the point estimate:

```python
a = evaluate(env, policy_a, schedule, RolloutConfig(seed=100))
b = evaluate(env, policy_b, schedule, RolloutConfig(seed=100))   # same positions
flipped = [(x.task.label, x.success, y.success)
           for x, y in zip(a.results, b.results) if x.success != y.success]
```

A report is a pydantic model, so `report.model_dump_json()` stores it and reads
back as the same objects, tasks included.

## 8. Reinforcement learning

`EpisodeRunner` puts a gym-shaped loop around one task: shaping reward per step,
outcome terms at the end, the same verdict at the close.

```python
from chess_sim import EpisodeRunner, RewardConfig, RolloutConfig

runner = EpisodeRunner(env, RolloutConfig(max_steps=450),
                       RewardConfig(success=1.0, right_piece=0.2, progress=2.0,
                                    disturbance=-0.5, drop=-0.5, step_cost=-0.002))
task = MoveSampler().sample(env, rng)
observation, total = runner.begin(task), 0.0
while True:
    step = runner.step(policy.select_action(observation, task))
    observation, total = step.observation, total + step.reward
    if step.done:
        print(total, step.result.success, step.result.reason)
        break
```

Shaping has two terms, and both matter: `reach_reward` for bringing the tool to
the piece and `progress_reward` for carrying the piece to its target. Progress
alone gives a policy that has not grasped anything a flat zero to climb.

`examples/train_rl.py` puts a search around that loop - a residual of per-joint
offsets on top of a frozen base policy, fitted with the cross-entropy method, no
gradients through the base (which matters when the base costs seconds per call):

```bash
python examples/train_rl.py --iterations 5 --population 8        # analytic base, minutes
python examples/train_rl.py --checkpoint outputs/molmoact2_mc/checkpoints/070000/pretrained_model
```

Every candidate is scored on the same pool of replayed positions, and the
verdict comes from a held-out pool the search never saw. That held-out pool is
the point of the whole arrangement - measured on the analytic base:

| positions | training pool | held-out base | held-out residual |
| --- | --- | --- | --- |
| 2-8 extra pieces, 3 episodes | -2.43 -> +0.19 | +0.185 | +0.070 |
| 2-8 extra pieces, 6 episodes | -1.12 -> +0.19 | +0.184 | -0.267 |
| 0-2 extra pieces, 6 episodes | +0.18 -> +0.19 | +0.184 | **+0.194** |
| 0-2 pieces, 6 cm/4 cm base error | +0.14 -> +0.15 | +0.142 | **+0.154** |

On a crowded board the return is dominated by which pieces a candidate happens
to knock over, and the search tunes itself to those particular boards - it beats
its training pool and loses on fresh ones. Thin the board out with
`--pieces 0 2` and the shaping signal is what is left to climb, so the learned
offsets transfer. Read the held-out line, never the training pool.

For a reward of your own, the pieces are public: `env.piece_snapshot()` for
where things were, `env.evaluate(task, before)` for the verdict, and
`reach_reward` / `progress_reward` / `terminal_reward` for the arithmetic.

**DAgger.** The expert plans from whatever pose the arm is in, so letting the
policy run and then calling `env.execute(task)` finishes the same episode by
hand - a corrected trajectory with no extra machinery:

```python
policy.reset()
observation = env.observe(images=False)
for _ in range(prefix_steps):
    observation = env.step(policy.select_action(observation, task), images=False)
corrected = env.execute(task)      # expert suffix, recorded like any demonstration
```

## 9. Published artifacts

```python
from chess_sim import hub
hub.MODEL_REPO        # XvKuoMing/so101_chess - MolmoAct2 + LoRA, 70,000 steps
hub.DATASET_REPO      # XvKuoMing/so101_chess - 6,095 episodes, 497,565 frames
hub.SHOWCASE          # {'moves': ..., 'captures': ...} - the reels on the model page
```
