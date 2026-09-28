# Reinforcement learning on the corrected policy - design

Status: designed, not started. The decision to run it is in `ROADMAP.md` phase 1; this
file is the design it will be built to, so that the build is a day of code and not a week
of choices.

## The idea in one paragraph

The policy already produces good and bad rollouts of the same position: the flow-matching
sampler makes each attempt different, and half of the captures it failed on a 300-position
test it managed on a second try. Reinforcement learning here means collecting many attempts
per position, scoring each one with the simulator, and training on the better-than-average
ones with a weight that says how much better. Advantage-weighted fine-tuning, iterated.
Nothing is estimated: the reward is the simulator's own success rule plus the dense shaping
already in `chess_sim/rewards.py`.

Why not online policy gradients (PPO, GRPO): a flow-matching head gives an action, not the
probability of the action, which the gradient needs; and rollouts (17 GB) and training
(29 GB) cannot share the card, so an update per batch of episodes means reloading a 5.6B
model thousands of times. Iterated offline weighting gets the same signal in three large
steps instead of a hundred thousand small ones.

## Components

### `ChessGymEnv` (`chess_sim/gym.py`)

A thin wrapper over `ChessSimEnv` with the interface reinforcement-learning code expects:

```python
env = ChessGymEnv(EnvConfig(), RewardConfig(), sampler=CaptureSampler(), max_steps=450)
obs = env.reset(seed=7)                 # samples a task; obs carries images, joints, task
obs, reward, terminated, truncated, info = env.step(action)   # one control step
```

- `reward`: `shaping_reward` per step plus the terminal `success` bonus from `RewardConfig`.
- `terminated`: the success rule fired, or a failure that cannot be undone (piece toppled,
  neighbour disturbed) - the same reasons `evaluate` reports.
- `truncated`: the step budget.
- `info`: the `TaskResult` on the last step, the task, and the running return.
- `env.task`, `env.result` for the collector; `env.expert()` runs the scripted expert from
  the current state and records through a given recorder - the DAgger branch.

A vectorised form is not needed: the collector below runs one policy against several
processes each holding one env.

### Batched inference (`chess_sim/policies.py`)

`LeRobotPolicy.select_actions(observations, tasks)` - the plural form - builds one batch
from N observations and runs the model once. The single-observation `select_action` calls
it with N = 1, so nothing else changes. Measured on the sweep, one call is 0.3-1 s and the
simulator step is 53 ms, so serving four environments from one call roughly quadruples
throughput. Each environment keeps its own action queue and hold counter; the policy
object holds no per-episode state beyond those.

### The collector (`examples/collect_rollouts.py`)

```
--checkpoint <policy>  --positions 300 --samples 4 --seed 7000 --workers 4
--task capture|move|both  --out datasets/chess_rl/iter1
```

For each position (seeded, disjoint from the 2000-series evaluation seeds):

1. Sample the task once; K worker processes each reset to the identical board, shift and
   start pose.
2. Run the policy at its shipped horizon (30) through the batched call until every
   environment terminates or truncates.
3. Return per episode: `success` (1/0) times the terminal bonus, plus the summed shaping.
   Advantage = return minus the mean return of the K attempts at this position.
4. Record every episode through `EpisodeRecorder` with `advantage`, `return`, `sample`
   and `position` in `meta.json`. The frames are the policy's own actions, clean.
5. If every attempt failed and the last one is in a recoverable state, hand over to the
   expert from the best attempt's end state and record the correction with the `recovery`
   tag, as `collect_recoveries.py` does now.

Output per iteration: 300 positions x 4 samples = 1,200 rollouts; at the corrected
checkpoint's success rate about 900 successes. About two hours batched, five unbatched.

### Weighting and training (existing tools)

`examples/export_dataset.py` gains `--min-advantage`: exports only episodes at or above
it. Three exports from one collection give three tiers:

| tier | filter | weight in the merge |
| --- | --- | --- |
| clearly better | advantage >= +0.5 | x3 |
| better | 0 < advantage < 0.5 | x1 |
| corrections | `recovery` tag | x3 |

`tools/merge_datasets.py --repeat` composes them with the base set, exactly as the
corrections run was built. Training is `training/settings.rl.env`: `INIT_FROM` the
corrected checkpoint, `LR_SCALE=0.1`, 8,000 steps, one block. Then the paired
300-position sweep at horizons 30 and 5 against `dagger.jsonl`.

Iteration 2 starts from the iteration-1 checkpoint and collects fresh rollouts with it.

## Reward

From `RewardConfig`, unchanged:

| term | when | value |
| --- | --- | --- |
| reach | tool approaching the piece | up to 0.3, scaled by distance |
| progress | piece lifted, carried toward the target | per-step shaping |
| success | the evaluation rule | terminal bonus |
| failure | toppled, disturbed, dropped | terminal penalty |

Two rules the shaping must keep: it never outweighs success (a slow success beats a fast
failure), and it is computed from the simulator's state, never from images.

## Budget

| phase | VRAM | RAM | wall time per iteration |
| --- | --- | --- | --- |
| rollouts, batched over 4 environments | ~20 GB | ~8 GB | ~2 h |
| export + merge | - | ~4 GB | ~20 min |
| training, batch 8, 8,000 steps | 29 GB | 12-16 GB | ~4.5 h |
| paired sweep, two arms | 17 GB | 6 GB | ~2.5 h |

Three iterations in six days including the code. Disk: ~1 MB per rollout.

## What would count as a result

The paired 300-position test at the shipped horizon against the corrected checkpoint
(233; captures 78, moves 155): captures up with moves held, or overall p < 0.05. The
five-action arm must not fall. Publish as `so101_chess_molmoact2_dagger_rl` on a win,
with approval.

## Risks, and what is done about them

- **Narrowing.** Training only on successes can collapse the sampler onto a few
  trajectories. The tenth-rate schedule and keeping the full base set in the merge are the
  brake; the five-action arm is the canary.
- **Positions the policy never solves.** No positive advantage exists there; the expert
  branch covers them.
- **Reward hacking.** Not possible against the simulator's success rule, which checks the
  final board, but the shaping could be gamed by hovering. Capped per step and dominated by
  the terminal terms.
- **Evaluation leakage.** Collection seeds are disjoint from the 2000-series; the sweep is
  never used for collection.
