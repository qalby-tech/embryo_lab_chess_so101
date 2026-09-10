"""Fine-tune MolmoAct2 in blocks, scoring the policy in the simulator between
blocks and keeping only the best checkpoint.

    python examples/train_select_molmoact2.py --root datasets/lerobot/chess_v2 \
        --out outputs/molmoact2_chess --total-steps 40000 --block 500 --eval-episodes 5

Each block trains `--block` steps (resuming the previous state), then runs the
closed-loop evaluation on the checkpoint just written. Two checkpoints are
kept: the best seen (success rate first, median placement error second) and
the most recent. Everything else is deleted as the run goes. Training resumes
from the most recent, so re-running this with a larger `--total-steps`
continues where it left off rather than starting over; the best is kept for
use, and both carry their `training_state`.
Training and evaluation run one after the other because a 5B policy plus its
optimizer already fills the GPU.

Runs in the VLA environment; the simulator package must be importable
(PYTHONPATH=~/chess_so101:~/chess_so101/examples) and rendering needs
MUJOCO_GL=egl with GALLIUM_DRIVER=d3d12 on WSL2.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

BEST = "best.json"
SAVE_EVERY = 2000   # checkpoint within a block: a killed session then loses an hour, not the block


def train_block(args, resume: bool, steps_target: int) -> None:
    if resume:
        # resuming needs the saved config; everything else comes from it
        config = os.path.join(args.out, "checkpoints", "last", "pretrained_model", "train_config.json")
        subprocess.run([
            os.path.expanduser("~/vla/venv/bin/accelerate"), "launch", "--num_processes=1",
            "--mixed_precision=bf16", "-m", "lerobot.scripts.lerobot_train",
            f"--config_path={config}", "--resume=true",
            f"--steps={steps_target}", f"--save_freq={min(SAVE_EVERY, args.block)}",
        ], check=True)
        return
    cmd = [
        os.path.expanduser("~/vla/venv/bin/accelerate"), "launch", "--num_processes=1",
        "--mixed_precision=bf16", "-m", "lerobot.scripts.lerobot_train",
        f"--dataset.repo_id={args.repo_id}", f"--dataset.root={args.root}",
        "--dataset.video_backend=pyav", "--dataset.image_transforms.enable=true",
        "--policy.type=molmoact2", "--policy.device=cuda", "--policy.action_mode=both",
        f"--policy.train_mode_vlm={args.train_mode_vlm}",
        "--policy.chunk_size=30", "--policy.n_action_steps=30",
        "--policy.setup_type=single so100/so101 robotic arm in molmoact2",
        "--policy.control_mode=absolute joint pose",
        '--policy.image_keys=["observation.images.top","observation.images.wrist"]',
        # mandatory: without it even batch 4 exhausts the 32 GB card
        "--policy.gradient_checkpointing=true",
        "--policy.normalize_gripper=true", "--policy.push_to_hub=false", "--wandb.enable=false",
        f"--batch_size={args.batch_size}", f"--steps={steps_target}",
        f"--save_freq={min(SAVE_EVERY, args.block)}", "--env_eval_freq=-1",
        f"--output_dir={args.out}",
    ]
    cmd += [f"--policy.checkpoint_path={args.base_checkpoint}"]
    subprocess.run(cmd, check=True)


def evaluate(args, checkpoint: str) -> dict:
    """Closed-loop success rate of `checkpoint` on held-out positions."""
    out = subprocess.run(
        [os.path.expanduser("~/vla/venv/bin/python"), "examples/eval_molmoact2.py",
         "--checkpoint", checkpoint, "--episodes", str(args.eval_episodes),
         "--seed", str(args.eval_seed), "--max-steps", str(args.eval_max_steps),
         "--dataset-root", args.root]
        + (["--moves", args.moves] if args.moves else [])
        + (["--interpolate"] if args.interpolate else []),
        capture_output=True, text=True)
    sys.stdout.write(out.stdout[-2000:])
    match = re.search(r"(\d+)/(\d+) successes; median placement error ([\d.]+) mm", out.stdout)
    if not match:
        print("evaluation produced no score:", out.stderr[-600:])
        return {"successes": -1, "episodes": args.eval_episodes, "median_mm": float("inf")}
    return {"successes": int(match.group(1)), "episodes": int(match.group(2)),
            "median_mm": float(match.group(3))}


def better(candidate: dict, best: dict | None) -> bool:
    if best is None:
        return True
    if candidate["successes"] != best["successes"]:
        return candidate["successes"] > best["successes"]
    return candidate["median_mm"] < best["median_mm"]


def checkpoints(out: str) -> list[str]:
    """Step directories holding a checkpoint a run can resume from. A session
    killed mid-save leaves a partial one behind: the files exist but the
    optimizer state is missing and the step is never written, so existence
    alone does not show progress and must not be resumed from."""
    root = os.path.join(out, "checkpoints")
    if not os.path.isdir(root):
        return []
    complete = []
    for d in sorted(x for x in os.listdir(root) if x.isdigit()):
        state = os.path.join(root, d, "training_state")
        needed = [os.path.join(root, d, "pretrained_model", "model.safetensors"),
                  os.path.join(state, "optimizer_state.safetensors"),
                  os.path.join(state, "training_step.json")]
        try:
            if not all(os.path.getsize(f) for f in needed):
                raise OSError("empty file")
            with open(needed[-1]) as f:
                json.load(f)["step"]
        except (OSError, ValueError, KeyError):
            print(f"discarding partial checkpoint {d}", flush=True)
            shutil.rmtree(os.path.join(root, d), ignore_errors=True)
            continue
        complete.append(d)
    return complete


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/lerobot/chess_v2")
    ap.add_argument("--repo-id", default="XvKuoMing/so101_chess")
    ap.add_argument("--out", default="outputs/molmoact2_chess")
    ap.add_argument("--base-checkpoint", default="allenai/MolmoAct2-SO100_101")
    ap.add_argument("--total-steps", type=int, default=40000)
    # a cycle costs ~18 min outside training (two model loads plus the rollouts),
    # measured; 4000 steps keeps that overhead near 13% of the run
    ap.add_argument("--block", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--train-mode-vlm", default="lora")
    ap.add_argument("--eval-episodes", type=int, default=6)
    ap.add_argument("--moves", default=None,
                    help="score only these UCI moves, for a narrow-task run")
    ap.add_argument("--interpolate", action="store_true",
                    help="ramp between the policy's targets when scoring")
    ap.add_argument("--eval-seed", type=int, default=100)
    ap.add_argument("--eval-max-steps", type=int, default=300)
    args = ap.parse_args()

    best_path = os.path.join(args.out, BEST)
    best = json.load(open(best_path)) if os.path.exists(best_path) else None
    done = max((int(c) for c in checkpoints(args.out)), default=0)

    while done < args.total_steps:
        target = min(done + args.block, args.total_steps)
        print(f"\n=== training to step {target} ===", flush=True)
        started = time.time()
        train_block(args, resume=done > 0, steps_target=target)
        tags = checkpoints(args.out)
        if not tags:
            print("no checkpoint written; stopping"); break
        tag = tags[-1]
        checkpoint = os.path.join(args.out, "checkpoints", tag, "pretrained_model")
        score = evaluate(args, checkpoint)
        score.update(step=int(tag), minutes=round((time.time() - started) / 60, 1))
        print(f"step {tag}: {score['successes']}/{score['episodes']} successes, "
              f"median {score['median_mm']:.1f} mm ({score['minutes']} min)", flush=True)

        if better(score, best):
            best = score
            json.dump(best, open(best_path, "w"), indent=2)
            keep = tag
            print(f"new best at step {target}", flush=True)
        else:
            keep = f"{best['step']:06d}"
        # keep the newest as well: it carries the training state a later run
        # resumes from, and only these two survive
        for tag_other in checkpoints(args.out):
            if tag_other not in (keep, tag):
                shutil.rmtree(os.path.join(args.out, "checkpoints", tag_other), ignore_errors=True)
        done = int(tag)

    print(f"\nbest checkpoint: step {best['step']} with {best['successes']}/{best['episodes']} "
          f"successes, median {best['median_mm']:.1f} mm" if best else "no successful evaluation")
    keep_final = {f"{done:06d}"} | ({f"{best['step']:06d}"} if best else set())
    for tag_other in checkpoints(args.out):
        if tag_other not in keep_final:
            shutil.rmtree(os.path.join(args.out, "checkpoints", tag_other), ignore_errors=True)
    print("checkpoints remaining:", checkpoints(args.out),
          "(the highest is where a later run resumes)")


if __name__ == "__main__":
    main()
