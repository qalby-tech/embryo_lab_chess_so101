"""Running policies: one episode, a scored batch of them, or a manual RL loop.

    report = evaluate(env, policy, [(MoveSampler(), 64), (CaptureSampler(), 32)],
                      RolloutConfig(max_steps=450))
    print(report.summary())

Every episode is scored by `env.evaluate` - the rule the scripted expert is held
to - so policy numbers and expert numbers mean the same thing.
"""
from __future__ import annotations

import random
import statistics
from typing import Callable, Sequence

from .config import Config, RecoveryTrigger
from .env import ChessSimEnv, Observation, OnStep, PieceSnapshot, Record, TaskResult
from .policies import Policy
from .rewards import RewardConfig, shaping_reward, terminal_reward
from .tasks import AnyTask, Task, TaskFamily, TaskSampler

CONFIDENCE_Z = 1.96     # 95%


class RolloutConfig(Config):
    max_steps: int = 450    # control steps per episode: about 15 s at 30 Hz
    seed: int = 100


class FamilyScore(Record):
    family: TaskFamily
    successes: int
    episodes: int
    right_piece: int
    median_error: float

    @property
    def success_rate(self) -> float:
        return self.successes / self.episodes if self.episodes else 0.0

    @property
    def interval(self) -> tuple[float, float]:
        return wilson_interval(self.successes, self.episodes)


class EvaluationReport(Record):
    results: list[TaskResult]

    @property
    def episodes(self) -> int:
        return len(self.results)

    @property
    def successes(self) -> int:
        return sum(r.success for r in self.results)

    @property
    def success_rate(self) -> float:
        return self.successes / self.episodes if self.episodes else 0.0

    @property
    def median_error(self) -> float:
        return statistics.median([r.placement_error for r in self.results]) if self.results else 0.0

    @property
    def interval(self) -> tuple[float, float]:
        return wilson_interval(self.successes, self.episodes)

    def by_family(self) -> dict[TaskFamily, FamilyScore]:
        scores = {}
        for family in {r.task.family for r in self.results}:
            group = [r for r in self.results if r.task.family is family]
            scores[family] = FamilyScore(
                family=family, successes=sum(r.success for r in group), episodes=len(group),
                right_piece=sum(r.right_piece for r in group),
                median_error=statistics.median([r.placement_error for r in group]))
        return scores

    def summary(self) -> str:
        low, high = self.interval
        lines = [f"{self.successes}/{self.episodes} successes "
                 f"({self.success_rate * 100:.0f}%, 95% CI {low * 100:.0f}-{high * 100:.0f}%); "
                 f"median placement error {self.median_error * 1000:.1f} mm"]
        scores = self.by_family()
        if len(scores) > 1:
            for family, score in sorted(scores.items()):
                lines.append(f"  {family}: {score.successes}/{score.episodes}, "
                             f"named piece {score.right_piece}/{score.episodes}, "
                             f"median {score.median_error * 1000:.1f} mm")
        return "\n".join(lines)


def wilson_interval(successes: int, episodes: int, z: float = CONFIDENCE_Z) -> tuple[float, float]:
    """Confidence interval for a success rate. A score of 54/64 is 84% give or
    take nine points; without the interval two checkpoints look different when
    they are not."""
    if episodes == 0:
        return (0.0, 0.0)
    p = successes / episodes
    denominator = 1 + z * z / episodes
    center = p + z * z / (2 * episodes)
    margin = z * ((p * (1 - p) + z * z / (4 * episodes)) / episodes) ** 0.5
    return ((center - margin) / denominator, (center + margin) / denominator)


def run_episode(env: ChessSimEnv, policy: Policy, task: Task, config: RolloutConfig = RolloutConfig(),
                on_step: OnStep | None = None) -> TaskResult:
    """One instructed attempt under policy control, from the position the env is
    already in (a sampler puts it there). Runs the full budget: a policy has no
    way to say it is finished."""
    before = env.piece_snapshot()
    policy.reset()
    observation = env.observe()
    for _ in range(config.max_steps):
        action = policy.select_action(observation, task)
        # rendering is most of the cost of a step, so only render what gets looked at
        observation = env.step(action, on_step, images=policy.needs_images())
    return env.evaluate(task, before, steps=config.max_steps)


def evaluate(env: ChessSimEnv, policy: Policy, schedule: Sequence[tuple[TaskSampler, int]],
             config: RolloutConfig = RolloutConfig(), on_step: OnStep | None = None,
             on_episode: Callable[[TaskResult], None] | None = None) -> EvaluationReport:
    """Score a policy over several task families: `[(sampler, episodes), ...]`."""
    rng = random.Random(config.seed)
    results: list[TaskResult] = []
    for sampler, episodes in schedule:
        for index in range(episodes):
            task = sampler.sample(env, rng, index)
            result = run_episode(env, policy, task, config, on_step)
            results.append(result)
            if on_episode is not None:
                on_episode(result)
    return EvaluationReport(results=results)


def demonstrate(env: ChessSimEnv, sampler: TaskSampler, episodes: int, rng: random.Random,
                recorder=None, record_appearance: bool = False, on_step: OnStep | None = None,
                on_episode: Callable[[TaskResult], None] | None = None,
                on_episode_start: Callable[[int], None] | None = None) -> list[TaskResult]:
    """Scripted-expert demonstrations - the data-collection counterpart of
    `evaluate`. With an `EpisodeRecorder` each attempt is written to disk,
    verdict included, so the exporter can keep the successes."""
    results = []
    for index in range(episodes):
        if on_episode_start is not None:
            on_episode_start(index)         # e.g. a new look for this episode
        task = sampler.sample(env, rng, index)
        if recorder is not None:
            recorder.begin(task, record_appearance=record_appearance)
        result = env.execute(task, on_step=_chain(None if recorder is None else recorder.on_step,
                                                  on_step))
        if recorder is not None:
            recorder.end(result)
        results.append(result)
        if on_episode is not None:
            on_episode(result)
    return results


def _chain(*hooks: OnStep | None) -> OnStep | None:
    """One per-step callback out of several - recording and a video writer, say."""
    hooks = [hook for hook in hooks if hook is not None]
    if len(hooks) <= 1:
        return hooks[0] if hooks else None

    def call(action) -> None:
        for hook in hooks:
            hook(action)
    return call


class DaggerConfig(Config):
    """When the expert should take over from a policy.

    A fixed hand-over point mostly re-records ordinary demonstrations from a
    random pose. Waiting until the policy is demonstrably going wrong collects
    corrections for the states it actually gets itself into, which is the only
    reason to run DAgger at all.
    """

    min_prefix: int = 30          # control steps before anything counts as going wrong
    check_every: int = 15         # how often to look, in control steps
    stall_fraction: float = 0.4   # of the budget: by here the named piece must be off the board
    lift: float = 0.035           # metres above the board surface that counts as lifted


class DaggerResult(Record):
    """One correction attempt."""

    task: AnyTask
    trigger: RecoveryTrigger | None      # None: the policy was doing fine, nothing recorded
    prefix_steps: int                    # how long the policy drove before the hand-over
    policy_success: bool                 # did the policy finish it before any trigger fired
    result: TaskResult | None = None     # the expert's attempt, when it took over


def _trigger(env: ChessSimEnv, task: Task, before: PieceSnapshot, steps: int,
             rollout: "RolloutConfig", dagger: DaggerConfig) -> RecoveryTrigger | None:
    """Is the policy's attempt already wrong? Scored with the same rule as everything else."""
    verdict = env.evaluate(task, before, steps=steps)
    if verdict.success:
        return None
    if verdict.disturbed:
        return RecoveryTrigger.DISTURBED
    if verdict.picked is not None and verdict.picked != task.source:
        return RecoveryTrigger.WRONG_PIECE
    if steps >= dagger.stall_fraction * rollout.max_steps and not _lifted(env, task, dagger):
        return RecoveryTrigger.STALLED
    return None


def _lifted(env: ChessSimEnv, task: Task, dagger: DaggerConfig) -> bool:
    """Is the named piece in the air? `evaluate` only knows that it moved, and a piece
    nudged a centimetre across its square has moved without ever being held - the
    commonest way a grasp fails, and one that must still hand over to the expert."""
    slot = env.slot_at(task.source)
    return slot is not None and env.piece_position(slot)[2] - env.board.top > dagger.lift


def recover(env: ChessSimEnv, policy: Policy, task: Task, recorder=None,
            config: RolloutConfig = RolloutConfig(),
            dagger: DaggerConfig = DaggerConfig()) -> DaggerResult:
    """Let the policy drive until it goes wrong, then have the expert finish.

        result = recover(env, policy, task, recorder)

    Only the expert's half is recorded - the policy's own bad behaviour must not
    end up in the training set - and the episode's metadata carries what went
    wrong, so the corrections can be read back by failure mode.
    """
    before = env.piece_snapshot()
    policy.reset()
    observation = env.observe()
    trigger, steps = None, 0
    for step in range(config.max_steps):
        observation = env.step(policy.select_action(observation, task),
                               images=policy.needs_images())
        steps = step + 1
        if steps < dagger.min_prefix or steps % dagger.check_every:
            continue
        finished = env.evaluate(task, before, steps=steps)
        if finished.success:
            break                   # nothing to correct; no need to run out the budget
        trigger = _trigger(env, task, before, steps, config, dagger)
        if trigger is not None:
            break
    if trigger is None:
        finished = env.evaluate(task, before, steps=steps)
        return DaggerResult(task=task, trigger=None, prefix_steps=steps,
                            policy_success=finished.success)

    if recorder is not None:
        recorder.begin(task, recovery=trigger)
    result = env.execute(task, on_step=None if recorder is None else recorder.on_step)
    if recorder is not None:
        recorder.end(result)
    return DaggerResult(task=task, trigger=trigger, prefix_steps=steps, policy_success=False,
                        result=result)


class StepResult(Record):
    observation: Observation
    reward: float
    done: bool
    result: TaskResult | None = None    # the verdict, on the last step


class EpisodeRunner:
    """A gym-shaped loop for writing RL scripts by hand.

        runner = EpisodeRunner(env, RolloutConfig(), RewardConfig())
        task = MoveSampler().sample(env, rng)
        observation = runner.begin(task)
        while True:
            step = runner.step(my_policy(observation))
            if step.done:
                break

    The reward is the shaping progress made this step, plus the outcome terms
    once the episode ends (see `chess_sim.rewards`).
    """

    def __init__(self, env: ChessSimEnv, config: RolloutConfig = RolloutConfig(),
                 reward: RewardConfig = RewardConfig()):
        self.env = env
        self.config = config
        self.reward = reward
        self.task: Task | None = None
        self._before: PieceSnapshot | None = None
        self._shaping = 0.0
        self._steps = 0

    def begin(self, task: Task) -> Observation:
        """Start scoring a task in the position the env is already in."""
        self.task = task
        self._before = self.env.piece_snapshot()
        self._shaping = 0.0
        self._steps = 0
        return self.env.observe()

    def step(self, action, on_step: OnStep | None = None, images: bool = True) -> StepResult:
        """One control period. Pass `images=False` when the policy driving this
        loop does not look at them - rendering is most of the cost of a step."""
        if self.task is None:
            raise RuntimeError("call begin() with a task first")
        observation = self.env.step(action, on_step, images=images)
        self._steps += 1
        shaping = shaping_reward(self.env, self.task, self._before, self.reward)
        reward, self._shaping = shaping - self._shaping, shaping
        done = self._steps >= self.config.max_steps
        result = self.result() if done else None
        if result is not None:
            reward += terminal_reward(result, self.reward)
        return StepResult(observation=observation, reward=reward, done=done, result=result)

    def result(self) -> TaskResult:
        """Score the attempt as it stands."""
        if self.task is None:
            raise RuntimeError("call begin() with a task first")
        return self.env.evaluate(self.task, self._before, steps=self._steps)
