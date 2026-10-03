"""Fine-tuning a LeRobot policy on recorded demonstrations.

The flags a run needs on this hardware, each with the reason it is there. What
is common to every policy - the dataset, the step budget, the checkpoint
cadence, a warm start - lives in `TrainConfig`; what a particular architecture
needs lives in its `PolicyRecipe`. Build a command and run it:

    config = TrainConfig(recipe=RECIPES["smolvla"], dataset_root="datasets/lerobot/chess_mc",
                         output_dir="outputs/smolvla_mc", total_steps=20_000)
    subprocess.run(config.command(steps=5_000), check=True)     # first block
    subprocess.run(config.resume_command(steps=10_000), check=True)

`examples/train_policy.py` drives that loop and scores the checkpoints in the
simulator between blocks, so every architecture is measured the same way.
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
MM_PER_M = 1000.0
CHECKPOINT_DIR = "checkpoints"


class PolicyRecipe(Config):
    """What one architecture needs from `lerobot-train` beyond the shared flags.

    `learning_rates` are the model's own peak rates by flag name; `lr_scale`
    on the run multiplies all of them together, floor included, so a warm
    start can continue a converged policy at a fraction of its schedule.
    """

    policy_type: str
    base: str                                   # the pretrained weights a fresh run starts from
    chunk_size: int
    n_action_steps: int
    learning_rates: dict[str, float]
    install_extra: str                          # `pip install lerobot[<this>]`

    def fresh_start(self, base: str) -> list[str]:
        """How a fresh run names its weights. Most policies open them by path."""
        return [f"--policy.path={base}"]

    def model_flags(self, run: "TrainConfig") -> list[str]:
        return []


class MolmoAct2Recipe(PolicyRecipe):
    policy_type: str = "molmoact2"
    base: str = "allenai/MolmoAct2-SO100_101"   # pretrained on this exact arm
    chunk_size: int = 30
    n_action_steps: int = 30
    # MolmoAct2's own rates (lerobot configuration_molmoact2.py), one per parameter
    # group. The scheduler decays each group to decay/language of its peak. With the
    # language model trained through LoRA - train_mode_vlm below - the model ignores
    # the first three and runs the LoRA, the vision encoder and the connector at a
    # fixed 5e-5 (modeling_molmoact2.get_optim_params): the action expert's rate is
    # the only one a run really sets, and the only one `lr_scale` moves.
    learning_rates: dict[str, float] = {
        "optimizer_lr": 1e-5,                     # language model (LoRA)
        "optimizer_vit_lr": 5e-6,                 # vision encoder
        "optimizer_connector_lr": 5e-6,           # vision-to-language connector
        "optimizer_action_expert_lr": 5e-5,       # action expert - the part that moves the arm
        "scheduler_decay_lr": 1e-6,               # floor, relative to optimizer_lr
    }
    install_extra: str = "molmoact2"
    train_mode_vlm: str = "lora"                  # 737M trainable of 5.6B at MolmoAct2's default rank
    setup_type: str = "single so100/so101 robotic arm in molmoact2"
    control_mode: str = "absolute joint pose"

    def fresh_start(self, base: str) -> list[str]:
        # MolmoAct2 builds itself from the upstream HF weights and loads trained
        # weights on top, so the base goes through checkpoint_path, never path
        return ["--policy.type=molmoact2", f"--policy.checkpoint_path={base}"]

    def model_flags(self, run: "TrainConfig") -> list[str]:
        image_keys = json.dumps([f"observation.images.{c}" for c in run.cameras])
        return ["--policy.action_mode=both", f"--policy.train_mode_vlm={self.train_mode_vlm}",
                f"--policy.setup_type={self.setup_type}", f"--policy.control_mode={self.control_mode}",
                f"--policy.image_keys={image_keys}",
                # mandatory: without it even batch 4 exhausts the card
                "--policy.gradient_checkpointing=true", "--policy.normalize_gripper=true"]


class SmolVLARecipe(PolicyRecipe):
    """SmolVLA: a 450M-parameter VLM with a flow-matching action expert.

    The base ships a 50-step chunk recorded at 30 Hz - under two seconds. Our
    data is 10 Hz, so the same count would be five blind seconds; 30 keeps the
    three-second chunk every other result here is measured at."""
    policy_type: str = "smolvla"
    base: str = "lerobot/smolvla_base"
    chunk_size: int = 30
    n_action_steps: int = 30
    learning_rates: dict[str, float] = {"optimizer_lr": 1e-4, "scheduler_decay_lr": 2.5e-6}
    install_extra: str = "smolvla"
    # the base was trained on three cameras named camera1-3; ours map onto the
    # first two and the third stays an empty slot, which keeps the token layout
    # the base saw. The map is saved with the checkpoint, and `LeRobotPolicy`
    # reads it back so the environment still renders `top` and `wrist`.
    camera_names: dict[Camera, str] = {Camera.TOP: "camera1", Camera.WRIST: "camera2"}
    empty_cameras: int = 1

    def model_flags(self, run: "TrainConfig") -> list[str]:
        rename = {f"observation.images.{c}": f"observation.images.{name}"
                  for c, name in self.camera_names.items() if c in run.cameras}
        return [f"--rename_map={json.dumps(rename)}", f"--policy.empty_cameras={self.empty_cameras}",
                f"--policy.scheduler_warmup_steps={min(1000, run.total_steps // 20)}"]


RECIPES: dict[str, PolicyRecipe] = {"molmoact2": MolmoAct2Recipe(), "smolvla": SmolVLARecipe()}
BASE_CHECKPOINT = RECIPES["molmoact2"].base
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


class TrainConfig(Config):
    dataset_root: str
    output_dir: str
    recipe: PolicyRecipe = RECIPES["molmoact2"]
    repo_id: str = DATASET_REPO
    base_checkpoint: str | None = None  # overrides the recipe's pretrained weights
    total_steps: int = 70_000
    chunk_size: int | None = None       # overrides the recipe's chunk, and the horizon with it
    # 8 with gradient checkpointing peaks at ~30 GB of a 32 GB card for MolmoAct2
    batch_size: int = 8
    # same effective batch at a fraction of the activation memory; raise it with a
    # smaller batch on a smaller card
    grad_accum: int = 1
    # loading in worker processes deadlocked the run at random steps: the trainer
    # spinning in futex, the workers polling, the GPU holding memory with nothing
    # executing. 0 loads in the training process itself.
    num_workers: int = 4
    cameras: tuple[Camera, ...] = ROBOT_CAMERAS
    # the GPU resets under load (nvlddmkm event 153) at random steps; saving often
    # keeps each reset cheap
    save_every: int = 500
    accelerate: str = ACCELERATE
    device: str = "cuda"
    # A checkpoint of our own to continue from, on new data; it carries its
    # architecture and flags, so the recipe only supplies the learning rates.
    init_from: str | None = None
    # Scales the recipe's learning rates and the floor together. A warm start at the
    # full peak re-heats a converged policy to ten times the rate it finished at. For
    # MolmoAct2 with LoRA only the action expert's rate responds - see the recipe -
    # and that is the rate whose restart makes a continued policy drop pieces.
    lr_scale: float = 1.0
    # Anything else for `lerobot-train`, verbatim - a recipe's default worth
    # varying for one run, such as --policy.train_expert_only=false
    extra_flags: tuple[str, ...] = ()

    @property
    def base(self) -> str:
        return self.base_checkpoint or self.recipe.base

    @property
    def chunk(self) -> int:
        return self.chunk_size or self.recipe.chunk_size

    @property
    def checkpoints_dir(self) -> str:
        return os.path.join(self.output_dir, CHECKPOINT_DIR)

    @property
    def last_train_config(self) -> str:
        """The saved config a resumed run reads everything else from."""
        return os.path.join(self.checkpoints_dir, LAST, PRETRAINED, "train_config.json")

    def command(self, steps: int) -> list[str]:
        """Start a run that trains up to `steps` total."""
        # a fresh run names its pretrained weights; a warm start takes the
        # architecture and every flag from the checkpoint it continues
        policy = ([f"--policy.path={self.init_from}"] if self.init_from
                  else self.recipe.fresh_start(self.base))
        return self._launcher() + [
            f"--dataset.repo_id={self.repo_id}", f"--dataset.root={self.dataset_root}",
            "--dataset.video_backend=pyav", "--dataset.image_transforms.enable=true",
            *policy, f"--policy.device={self.device}",
            f"--policy.chunk_size={self.chunk}", f"--policy.n_action_steps={self.chunk}",
            *self.recipe.model_flags(self),
            "--policy.push_to_hub=false", "--wandb.enable=false",
            f"--batch_size={self.batch_size}", f"--steps={steps}",
            # every LeRobot scheduler decays over its own fixed count whatever
            # --steps says; the first full run spent everything past 24,000 at the floor
            f"--policy.scheduler_decay_steps={self.total_steps}",
            f"--accelerator.gradient_accumulation.steps={self.grad_accum}",
            f"--num_workers={self.num_workers}",
            f"--save_freq={self.save_every}", "--env_eval_freq=-1",
            f"--output_dir={self.output_dir}",
            *self._learning_rates(),
            *self.extra_flags,
        ]

    def _learning_rates(self) -> list[str]:
        if self.lr_scale == 1.0:
            return []
        return [f"--policy.{name}={rate * self.lr_scale:g}"
                for name, rate in self.recipe.learning_rates.items()]

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


# the name the first runs were written against
MolmoAct2TrainConfig = TrainConfig
