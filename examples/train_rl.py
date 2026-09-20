"""Train a residual policy on top of a frozen base policy, in the simulator.

    python examples/train_rl.py --iterations 5 --population 8        # analytic base, minutes
    python examples/train_rl.py --checkpoint outputs/molmoact2_mc/checkpoints/070000/pretrained_model

The learned part is a per-joint offset added to whatever the base policy
commands. Offsets are searched with the cross-entropy method: sample a
population, keep the best, refit - no gradients through the base policy, which
matters when the base is a 5B VLA that costs seconds per call.

Reward comes from `chess_sim.rewards`: reaching the named piece, carrying it
toward its target, and the outcome terms at the end (see `RewardConfig`).

The default base is an analytic reacher with a deliberate calibration error, so
the loop can be exercised end to end in minutes; `--checkpoint` puts the same
loop on a trained policy, which is the real use and far slower.

`--episodes` is how many positions each candidate is scored on, and it is the
setting that decides whether anything transfers: at 3 the search happily finds
offsets that suit those three positions and lose to the base policy on held-out
ones. Raise it until the held-out return stops disagreeing with the training
pool.
"""
import argparse
import json
import random

import numpy as np

from chess_sim import (DOF, CaptureSampler, ChessSimEnv, ControlConfig, EnvConfig, EpisodeRunner,
                       LeRobotPolicy, LeRobotPolicyConfig, MoveSampler, Observation, PositionConfig,
                       RewardConfig, RolloutConfig, TaskFamily)
from chess_sim.env import GRASP_HEIGHT
from chess_sim.tasks import Task

ELITE_FRACTION = 0.25       # of each population kept to refit the search distribution
STD_FLOOR = 0.002           # rad: keeps the search from collapsing onto one point


class IkReachPolicy:
    """Analytic baseline: servo the tool toward the named piece.

    `bias` is a deliberate calibration error in the target it aims at - the kind
    of systematic offset a residual is supposed to absorb.
    """

    def __init__(self, env: ChessSimEnv, bias=(0.03, 0.0, 0.02), jaw: float = 0.6):
        self.env = env
        self.bias = np.asarray(bias, dtype=float)
        self.jaw = jaw            # gripper angle: the IK solves the five arm joints only

    def reset(self) -> None:
        pass

    def needs_images(self) -> bool:
        return False

    def select_action(self, observation: Observation, task: Task) -> np.ndarray:
        slot = self.env.slot_at(task.source)
        if slot is None:
            return observation.joint_pos
        target = self.env.piece_position(slot) + np.array([0.0, 0.0, GRASP_HEIGHT]) + self.bias
        arm = self.env.ik.solve(self.env.data, target).q      # five joints: the IK has no gripper
        return np.append(arm, self.jaw)


class ResidualPolicy:
    """A frozen base policy plus a learned per-joint offset (radians)."""

    def __init__(self, base, offsets: np.ndarray):
        self.base = base
        self.offsets = np.asarray(offsets, dtype=float)

    def reset(self) -> None:
        self.base.reset()

    def needs_images(self) -> bool:
        return self.base.needs_images()

    def select_action(self, observation: Observation, task: Task) -> np.ndarray:
        return self.base.select_action(observation, task) + self.offsets


def draw_episodes(env, sampler, count: int, rng: random.Random) -> list[tuple[Task, str]]:
    """A fixed pool of episodes, each as (task, position).

    The sampler places the board nominally, so replaying the FEN puts every
    piece and the arm back exactly where they were - which is what makes the
    returns of two candidates comparable at all.
    """
    episodes = []
    for index in range(count):
        task = sampler.sample(env, rng, index)
        episodes.append((task, env.position.board_fen()))
    return episodes


def pool_return(env, policy, episodes, rollout: RolloutConfig, reward: RewardConfig) -> float:
    """Mean reward over a fixed pool of episodes."""
    return float(np.mean([episode_return(env, policy, task, fen, rollout, reward)
                          for task, fen in episodes]))


def episode_return(env, policy, task, fen: str, rollout: RolloutConfig,
                   reward: RewardConfig) -> float:
    """Replay one episode under `policy` and sum its reward."""
    env.reset(fen)
    runner = EpisodeRunner(env, rollout, reward)
    observation, total = runner.begin(task), 0.0
    policy.reset()
    while True:
        action = policy.select_action(observation, task)
        # a base policy that ignores the cameras should not pay for rendering them
        step = runner.step(action, images=policy.needs_images())
        observation, total = step.observation, total + step.reward
        if step.done:
            return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="base policy to improve on; omitted, an analytic reacher is used")
    ap.add_argument("--dataset-root", default="datasets/lerobot/chess_mc")
    ap.add_argument("--task", type=TaskFamily, choices=list(TaskFamily), default=TaskFamily.MOVE)
    ap.add_argument("--iterations", type=int, default=5, help="cross-entropy refits")
    ap.add_argument("--population", type=int, default=8, help="candidates per iteration")
    ap.add_argument("--episodes", type=int, default=3, help="training episodes per candidate")
    ap.add_argument("--held-out", type=int, default=4, help="episodes for the final comparison")
    ap.add_argument("--max-steps", type=int, default=60, help="control steps per episode")
    ap.add_argument("--sigma", type=float, default=0.05, help="initial offset spread, radians")
    ap.add_argument("--bias", type=float, nargs=3, default=(0.03, 0.0, 0.02), metavar=("X", "Y", "Z"),
                    help="calibration error built into the analytic base, metres - the thing "
                         "the residual has to cancel")
    ap.add_argument("--pieces", type=int, nargs=2, default=(2, 8), metavar=("MIN", "MAX"),
                    help="extra pieces besides the kings; crowded boards make the outcome "
                         "terms swamp the shaping signal")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None, help="write the learned offsets here as JSON")
    args = ap.parse_args()

    env = ChessSimEnv(EnvConfig(control=ControlConfig()))
    rollout = RolloutConfig(max_steps=args.max_steps, seed=args.seed)
    reward = RewardConfig()
    if args.checkpoint:
        base = LeRobotPolicy.load(LeRobotPolicyConfig.for_dataset(
            args.checkpoint, args.dataset_root, device=args.device, control_hz=env.control_hz))
        print(f"base: {base.policy_type} from {args.checkpoint}")
    else:
        base = IkReachPolicy(env, bias=tuple(args.bias))
        print(f"base: analytic reacher with a {tuple(round(v * 100, 1) for v in args.bias)} cm "
              f"calibration error")

    # nominal layout: the pool has to be replayable, position for position
    position = PositionConfig(min_extra_pieces=args.pieces[0], max_extra_pieces=args.pieces[1])
    sampler = (CaptureSampler(position=position, randomize_layout=False)
               if args.task is TaskFamily.CAPTURE
               else MoveSampler(position=position, randomize_layout=False))
    rng = np.random.default_rng(args.seed)
    mean, std = np.zeros(DOF), np.full(DOF, args.sigma)
    elite_count = max(2, round(args.population * ELITE_FRACTION))

    # One pool for training, a disjoint one for the verdict. Every candidate in
    # every iteration faces the same training pool, so the returns are comparable
    # and the search is optimizing something that stands still.
    task_rng = random.Random(args.seed)
    training = draw_episodes(env, sampler, args.episodes, task_rng)
    held_out = draw_episodes(env, sampler, args.held_out, random.Random(args.seed + 99))
    start = pool_return(env, ResidualPolicy(base, np.zeros(DOF)), training, rollout, reward)
    print(f"base return on the training pool: {start:+.3f} "
          f"({args.episodes} episodes, {args.max_steps} steps each)", flush=True)

    for iteration in range(args.iterations):
        population = rng.normal(mean, std, size=(args.population, DOF))
        returns = np.array([pool_return(env, ResidualPolicy(base, offsets), training, rollout, reward)
                            for offsets in population])
        elite = population[np.argsort(returns)[-elite_count:]]
        mean, std = elite.mean(axis=0), np.maximum(elite.std(axis=0), STD_FLOOR)
        print(f"iteration {iteration}: best {returns.max():+.3f}, mean {returns.mean():+.3f}, "
              f"offsets {np.round(mean, 3)}", flush=True)

    # the verdict: the learned offsets against doing nothing, on positions the
    # search never saw
    for label, offsets in (("base", np.zeros(DOF)), ("residual", mean)):
        score = pool_return(env, ResidualPolicy(base, offsets), held_out, rollout, reward)
        print(f"{label:9s} return {score:+.3f} on {len(held_out)} held-out episodes")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"offsets": mean.tolist(), "std": std.tolist(),
                       "iterations": args.iterations, "population": args.population}, f, indent=2)
        print("wrote", args.out)
    env.close()


if __name__ == "__main__":
    main()
