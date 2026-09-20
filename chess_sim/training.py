"""Fine-tuning MolmoAct2 on recorded demonstrations.

The flags below are the ones a run needs on this hardware, each with the reason
it is there. Build a command and run it:

    config = MolmoAct2TrainConfig(dataset_root="datasets/lerobot/chess_mc",
                                  output_dir="outputs/molmoact2_mc", total_steps=70_000)
    subprocess.run(config.command(steps=10_000), check=True)     # first block
    subprocess.run(config.resume_command(steps=20_000), check=True)

`examples/train_policy.py` drives that loop and scores the checkpoints in the
simulator between blocks.
"""
from __future__ import annotations

import json
import os
import shutil

from .config import ROBOT_CAMERAS, Camera, Config
from .hub import DATASET_REPO
from .rollout import EvaluationReport
from .tasks import TaskFamily

ACCELERATE = "~/vla/venv/bin/accelerate"
TRAIN_MODULE = "lerobot.scripts.lerobot_train"
BASE_CHECKPOINT = "allenai/MolmoAct2-SO100_101"   # pretrained on this exact arm
SETUP_TYPE = "single so100/so101 robotic arm in molmoact2"
CONTROL_MODE = "absolute joint pose"
MM_PER_M = 1000.0
CHECKPOINT_DIR = "checkpoints"
LAST = "last"
PRETRAINED = "pretrained_model"
TRAINING_STATE = "training_state"


class FamilyOutcome(Config):
    successes: int
    episodes: int
    median_mm: float


class CheckpointScore(Config):
    """What one checkpoint scored, as kept in `best.json` between blocks."""

    successes: int
    episodes: int
    median_mm: float
    families: dict[TaskFamily, FamilyOutcome] = {}
    step: int = 0
    minutes: float = 0.0

    @classmethod
    def from_report(cls, report: EvaluationReport, step: int, minutes: float) -> "CheckpointScore":
        return cls(successes=report.successes, episodes=report.episodes,
                   median_mm=report.median_error * MM_PER_M,
                   families={family: FamilyOutcome(successes=score.successes,
                                                   episodes=score.episodes,
                                                   median_mm=score.median_error * MM_PER_M)
                             for family, score in report.by_family().items()},
                   step=step, minutes=minutes)

    def better_than(self, other: "CheckpointScore | None") -> bool:
        """More successes wins; a tie goes to the tighter placement."""
        if other is None:
            return True
        if self.successes != other.successes:
            return self.successes > other.successes
        return self.median_mm < other.median_mm

    @classmethod
    def load(cls, path: str) -> "CheckpointScore | None":
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return cls.model_validate_json(f.read())

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.model_dump_json(indent=2))

    def summary(self) -> str:
        families = ", ".join(f"{family} {score.successes}/{score.episodes}"
                             for family, score in self.families.items())
        return (f"step {self.step}: {self.successes}/{self.episodes} successes"
                + (f" ({families})" if families else "")
                + f", median {self.median_mm:.1f} mm")


class MolmoAct2TrainConfig(Config):
    dataset_root: str
    output_dir: str
    repo_id: str = DATASET_REPO
    base_checkpoint: str = BASE_CHECKPOINT
    total_steps: int = 70_000
    chunk_size: int = 30
    n_action_steps: int = 30
    # 8 with gradient checkpointing peaks at ~30 GB of a 32 GB card
    batch_size: int = 8
    # same effective batch at a fraction of the activation memory; raise it with a
    # smaller batch on a smaller card
    grad_accum: int = 1
    # loading in worker processes deadlocked the run at random steps: the trainer
    # spinning in futex, the workers polling, the GPU holding memory with nothing
    # executing. 0 loads in the training process itself.
    num_workers: int = 4
    train_mode_vlm: str = "lora"        # 737M trainable of 5.6B at MolmoAct2's default rank
    cameras: tuple[Camera, ...] = ROBOT_CAMERAS
    # the GPU resets under load (nvlddmkm event 153) at random steps; saving often
    # keeps each reset cheap
    save_every: int = 500
    accelerate: str = ACCELERATE
    device: str = "cuda"

    @property
    def checkpoints_dir(self) -> str:
        return os.path.join(self.output_dir, CHECKPOINT_DIR)

    @property
    def last_train_config(self) -> str:
        """The saved config a resumed run reads everything else from."""
        return os.path.join(self.checkpoints_dir, LAST, PRETRAINED, "train_config.json")

    def command(self, steps: int) -> list[str]:
        """Start a run that trains up to `steps` total."""
        image_keys = json.dumps([f"observation.images.{c}" for c in self.cameras])
        return self._launcher() + [
            f"--dataset.repo_id={self.repo_id}", f"--dataset.root={self.dataset_root}",
            "--dataset.video_backend=pyav", "--dataset.image_transforms.enable=true",
            "--policy.type=molmoact2", f"--policy.device={self.device}", "--policy.action_mode=both",
            f"--policy.train_mode_vlm={self.train_mode_vlm}",
            f"--policy.chunk_size={self.chunk_size}", f"--policy.n_action_steps={self.n_action_steps}",
            f"--policy.setup_type={SETUP_TYPE}", f"--policy.control_mode={CONTROL_MODE}",
            f"--policy.image_keys={image_keys}",
            # mandatory: without it even batch 4 exhausts the card
            "--policy.gradient_checkpointing=true",
            "--policy.normalize_gripper=true", "--policy.push_to_hub=false", "--wandb.enable=false",
            f"--batch_size={self.batch_size}", f"--steps={steps}",
            # MolmoAct2 decays its learning rate over a fixed 24,000 steps whatever
            # --steps says; the first full run spent everything past that at the floor
            f"--policy.scheduler_decay_steps={self.total_steps}",
            f"--accelerator.gradient_accumulation.steps={self.grad_accum}",
            f"--num_workers={self.num_workers}",
            f"--save_freq={self.save_every}", "--env_eval_freq=-1",
            f"--output_dir={self.output_dir}",
            f"--policy.checkpoint_path={self.base_checkpoint}",
        ]

    def resume_command(self, steps: int) -> list[str]:
        """Continue the run to `steps` total; everything else comes from the saved config."""
        return self._launcher() + [f"--config_path={self.last_train_config}", "--resume=true",
                                   f"--steps={steps}", f"--save_freq={self.save_every}"]

    def _launcher(self) -> list[str]:
        return [os.path.expanduser(self.accelerate), "launch", "--num_processes=1",
                "--mixed_precision=bf16", "-m", TRAIN_MODULE]

    def complete_checkpoints(self) -> list[str]:
        """Step directories a run can resume from, newest last.

        A session killed mid-save leaves a partial one behind: the files exist but
        the optimizer state is missing and the step is never written, so existence
        alone does not show progress. Those are deleted here.
        """
        if not os.path.isdir(self.checkpoints_dir):
            return []
        complete = []
        for tag in sorted(d for d in os.listdir(self.checkpoints_dir) if d.isdigit()):
            path = os.path.join(self.checkpoints_dir, tag)
            needed = [os.path.join(path, PRETRAINED, "model.safetensors"),
                      os.path.join(path, TRAINING_STATE, "optimizer_state.safetensors"),
                      os.path.join(path, TRAINING_STATE, "training_step.json")]
            try:
                if not all(os.path.getsize(f) for f in needed):
                    raise OSError("empty file")
                with open(needed[-1]) as f:
                    json.load(f)["step"]
            except (OSError, ValueError, KeyError):
                print(f"discarding partial checkpoint {tag}", flush=True)
                shutil.rmtree(path, ignore_errors=True)
                continue
            complete.append(tag)
        return complete

    def checkpoint_path(self, tag: str) -> str:
        return os.path.join(self.checkpoints_dir, tag, PRETRAINED)

    def keep_only(self, tags: set[str]) -> None:
        """Delete every checkpoint but these - a full one is ~12 GB."""
        for tag in self.complete_checkpoints():
            if tag not in tags:
                shutil.rmtree(os.path.join(self.checkpoints_dir, tag), ignore_errors=True)
