"""An SO-101 arm playing chess in MuJoCo: simulator, scripted expert, datasets, policies.

    from chess_sim import ChessSimEnv, EnvConfig, MoveTask

    env = ChessSimEnv(EnvConfig())
    env.reset()
    result = env.execute(MoveTask(from_square="g1", to_square="f3"))

Trained policy and demonstrations: see `chess_sim.hub`.
"""
from .config import (DOF, JOINTS, ROBOT_CAMERAS, SQUARES, START_FEN, AppearanceConfig, BoardConfig,
                     Camera, ClearanceConfig, Config, ControlConfig, EnvConfig, FailureReason,
                     JointName, JointPose, RandomizationConfig, RecoveryTrigger, Square,
                     ToleranceConfig, square_at, square_index)
from .conventions import RADIANS, SO101_DEGREES, ConventionName, JointConvention
from .env import CameraImages, ChessSimEnv, Layout, Observation, PieceSnapshot, TaskResult
from .policies import LeRobotPolicy, LeRobotPolicyConfig, Policy
from .positions import PositionConfig
from .recorder import EpisodeRecorder, EpisodeMeta, RecorderConfig
from .rewards import (RewardConfig, progress_reward, reach_reward, shaping_reward,
                      terminal_reward)
from .rollout import (DaggerConfig, DaggerResult, EpisodeRunner, EvaluationReport, FamilyScore,
                      RolloutConfig, StepResult, demonstrate, evaluate, recover, run_episode,
                      wilson_interval)
from .tasks import CaptureSampler, CaptureTask, MoveSampler, MoveTask, Task, TaskFamily, TaskSampler

__all__ = [
    # simulator
    "ChessSimEnv", "Observation", "CameraImages", "TaskResult", "Layout", "PieceSnapshot",
    # configuration
    "Config", "EnvConfig", "BoardConfig", "AppearanceConfig", "ControlConfig",
    "RandomizationConfig", "ToleranceConfig", "ClearanceConfig", "PositionConfig",
    "Camera", "JointName", "JointPose", "JOINTS", "DOF", "ROBOT_CAMERAS",
    "Square", "SQUARES", "square_at", "square_index", "FailureReason", "START_FEN",
    # tasks
    "Task", "MoveTask", "CaptureTask", "TaskFamily", "TaskSampler", "MoveSampler", "CaptureSampler",
    # policies and rollouts
    "Policy", "LeRobotPolicy", "LeRobotPolicyConfig", "RolloutConfig", "run_episode", "evaluate",
    "demonstrate", "EvaluationReport", "FamilyScore", "wilson_interval",
    "recover", "DaggerConfig", "DaggerResult", "RecoveryTrigger",
    "EpisodeRunner", "StepResult",
    # data
    "EpisodeRecorder", "RecorderConfig", "EpisodeMeta",
    "JointConvention", "ConventionName", "SO101_DEGREES", "RADIANS",
    # reinforcement learning
    "RewardConfig", "terminal_reward", "shaping_reward", "reach_reward", "progress_reward",
]
