"""Fine-tune MolmoAct2 in blocks, scoring each checkpoint in the simulator and
keeping only the best.

    python examples/train_policy.py --root datasets/lerobot/chess_mc \
        --out outputs/molmoact2_mc --total-steps 70000 --block 10000

Each block trains `--block` steps (resuming the previous state), then scores the
checkpoint just written. Two checkpoints survive: the best seen (success rate
first, median placement error second) and the most recent, which carries the
training state a later run resumes from - so re-running with a larger
`--total-steps` continues rather than starting over. Training and evaluation run
one after the other because a 5B policy plus its optimizer already fills the GPU.

Runs in the VLA environment; the simulator package must be importable
(PYTHONPATH=~/chess_so101) and rendering needs MUJOCO_GL=egl, with
GALLIUM_DRIVER=d3d12 on WSL2.
"""
import argparse
import os
import subprocess
import sys
import tempfile
import time

from chess_sim.hub import DATASET_REPO
from chess_sim.rollout import EvaluationReport
from chess_sim.training import CheckpointScore, MolmoAct2TrainConfig

BEST = "best.json"
PYTHON = os.path.expanduser("~/vla/venv/bin/python")
EVALUATOR = "examples/evaluate_policy.py"
TAIL = 2000     # of the evaluator's output to echo


def score_checkpoint(args, checkpoint: str, step: int, minutes: float) -> CheckpointScore | None:
    """Closed-loop success of `checkpoint`, moves and captures in one process:
    two copies of the policy do not fit in memory at once. None if it produced
    no report - a crashed evaluation must not read as a bad checkpoint."""
    with tempfile.TemporaryDirectory() as tmp:
        report_path = os.path.join(tmp, "report.json")
        command = [PYTHON, EVALUATOR, "--checkpoint", checkpoint,
                   "--moves", str(args.eval_episodes), "--captures", str(args.eval_captures),
                   "--seed", str(args.eval_seed), "--max-steps", str(args.eval_max_steps),
                   "--dataset-root", args.root, "--report", report_path]
        if args.moves:
            command += ["--only-moves", args.moves]
        if args.no_interpolate:
            command += ["--no-interpolate"]
        out = subprocess.run(command, capture_output=True, text=True)
        sys.stdout.write(out.stdout[-TAIL:])
        if not os.path.exists(report_path):
            print("evaluation produced no report:", out.stderr[-600:])
            return None
        with open(report_path) as f:
            report = EvaluationReport.model_validate_json(f.read())
    return CheckpointScore.from_report(report, step=step, minutes=minutes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/lerobot/chess_mc")
    ap.add_argument("--repo-id", default=DATASET_REPO)
    ap.add_argument("--out", default="outputs/molmoact2_mc")
    ap.add_argument("--base-checkpoint", default=None,
                    help="weights to start from; a local checkpoint warm-restarts a run on new "
                         "data instead of beginning again from the pretrained arm policy")
    ap.add_argument("--total-steps", type=int, default=70000)
    ap.add_argument("--block", type=int, default=10000, help="steps between evaluations")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=1, help="micro-batches per optimizer step")
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--eval-episodes", type=int, default=32, help="move episodes per evaluation")
    ap.add_argument("--eval-captures", type=int, default=16, help="capture episodes per evaluation")
    ap.add_argument("--moves", default=None, help="score only these UCI moves, for a narrow-task run")
    ap.add_argument("--no-interpolate", action="store_true",
                    help="step to each of the policy's targets instead of ramping between them")
    ap.add_argument("--eval-seed", type=int, default=100)
    ap.add_argument("--eval-max-steps", type=int, default=450)
    args = ap.parse_args()

    settings = {"dataset_root": args.root, "repo_id": args.repo_id, "output_dir": args.out,
                "total_steps": args.total_steps, "batch_size": args.batch_size,
                "grad_accum": args.grad_accum, "num_workers": args.num_workers,
                "save_every": min(args.save_every, args.block)}
    if args.base_checkpoint:
        settings["base_checkpoint"] = args.base_checkpoint
    config = MolmoAct2TrainConfig(**settings)
    best_path = os.path.join(args.out, BEST)
    best = CheckpointScore.load(best_path)
    done = max((int(tag) for tag in config.complete_checkpoints()), default=0)

    while done < args.total_steps:
        target = min(done + args.block, args.total_steps)
        print(f"\n=== training to step {target} ===", flush=True)
        started = time.time()
        subprocess.run(config.resume_command(target) if done else config.command(target), check=True)
        tags = config.complete_checkpoints()
        if not tags:
            print("no checkpoint written; stopping")
            break
        tag = tags[-1]
        score = score_checkpoint(args, config.checkpoint_path(tag), int(tag),
                                 round((time.time() - started) / 60, 1))
        if score is not None:
            print(f"{score.summary()} ({score.minutes} min)", flush=True)
            if score.better_than(best):
                best = score
                best.save(best_path)
                print(f"new best at step {tag}", flush=True)
        # keep the newest as well: it carries the training state a later run resumes from
        config.keep_only({tag} | ({f"{best.step:06d}"} if best else set()))
        done = int(tag)

    print(f"\nbest checkpoint: {best.summary()}" if best else "no successful evaluation")
    config.keep_only({f"{done:06d}"} | ({f"{best.step:06d}"} if best else set()))
    print("checkpoints remaining:", config.complete_checkpoints(),
          "(the highest is where a later run resumes)")


if __name__ == "__main__":
    main()
