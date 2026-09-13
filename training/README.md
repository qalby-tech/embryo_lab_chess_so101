# Training

Everything needed to train the SO-101 chess policy on another machine. The
settings here were chosen by measurement; `settings.env` carries the reasoning
inline and `../docs/EXPERIMENTS.md` has the numbers behind each one.

## What is being trained

A VLA policy (MolmoAct2, fine-tuned from `allenai/MolmoAct2-SO100_101`) that
executes one instruction per episode against a simulated board:

| family | instruction | share of the dataset |
| --- | --- | --- |
| move | `pick up the piece on e2 and place it on e4` | 5,000 episodes (81%) |
| capture | `take the piece on d5 off the board` | 1,200 (19%) |

Instructions are deliberately spatial: executing one needs no chess knowledge,
so a chess engine can choose the move and the policy only has to carry it out.
Every instruction names its source square, so nothing has to be searched for.

Success is the rule the scripted expert is held to: the piece ends within
**11 mm** of the target square, upright, nothing else displaced.

## Run it

```bash
# 0. will this machine actually train? (one minute, saves a day)
training/preflight.sh

# 1. build the dataset - skips collection if the recordings are already present
training/build_dataset.sh          # ~3 h collection + ~4 h export for 7,343 episodes

# 2. train
training/train.sh                  # on the host, in an environment with lerobot
docker build -t chess-train:cu128 -f training/Dockerfile .
training/train_docker.sh           # or self-contained in a container
```

`train.sh` resumes from the last complete checkpoint, so re-running it after any
interruption continues rather than restarts. Checkpoints land every 2,000 steps
in `outputs/molmoact2_full/checkpoints/`; the loop keeps the best and the newest
and deletes the rest.

Or skip collection entirely and pull the published dataset:
`https://huggingface.co/datasets/XvKuoMing/so101_chess` (7,343 episodes,
609,097 frames, 1,380 instructions, 10 fps, 18 GB).

## Results so far, as sanity references

If a fresh environment reproduces these, it is set up correctly.

| run | data | step | result |
| --- | --- | --- | --- |
| one move (`e2e4`) | 300 eps | 12,000 | **16/16**, median 2.1 mm |
| four moves | 800 eps | 8,000 | **13/16**, right piece 14/16 |
| all instructions | 7,343 eps | — | not yet run to completion |

Two findings matter more than any hyperparameter:

**Steps predict performance, not epochs.** Four moves at 8,000 steps and 1.08
epochs scored 81%; one move at 3,000 steps and 1.1 epochs scored 37%. Budget in
steps.

**Interpolating the policy's targets is free accuracy.** A 10 Hz policy emits
one target per three control periods; holding each as a step topples pieces.
Ramping between them took the same checkpoint from 12/16 to 16/16. This applies
on the real robot too.

## Failures worth recognising

Each of these cost hours at least once.

| symptom | cause | fix |
| --- | --- | --- |
| CUDA OOM with gigabytes free; hangs with the GPU idle holding memory | a truncated NVIDIA library — check `dmesg` for `is truncated` and `dxgkio_escape` | reinstall the GPU driver; **no training setting can compensate** |
| hang, GPU at 2% still holding 30 GB, workers polling | dataloader deadlock in a container | `--ipc=host` (already in `train_docker.sh`) |
| `OSError: [Errno 24] Too many open files` | video dataloading exceeds 1024 descriptors | `ulimit -n 65536` (already in `train.sh`) |
| a checkpoint that exists but will not resume | killed mid-save: files present, optimizer state absent, `training_step.json` empty | the loop validates and discards these; processor configs are identical across a run's checkpoints, so a truncated one can be copied from a sibling |
| evaluation scores look flat, then jump | 8 episodes is too noisy — 3/8, 4/8, 3/8 read as a plateau, the next block gave 12/16 | evaluate on 16 |
| placement error looks like policy quality | with 28 mm squares, an untouched piece scores exactly the move distance (28.0 mm adjacent, 39.6 mm diagonal) | score `right piece` and untouched/knocked/carried, not the median error |

## Environment

Pinned in `Dockerfile`; the host these ran on had Python 3.12.3, torch
2.8.0+cu128, LeRobot 0.6.2 at commit `2774d9b`, transformers 5.5.4, accelerate
1.14.0, peft 0.20.0, on an RTX 5090 (32 GB). Training peaks near **30 GB** at
batch 8 with gradient checkpointing, so a smaller card needs a smaller
micro-batch with gradient accumulation to keep the effective batch at 8.

Evaluation renders the simulator, so the training environment also needs the
`chess_sim` package importable and a working EGL. On WSL that means
`GALLIUM_DRIVER=d3d12` with `/dev/dxg` and `/usr/lib/wsl` available to the
container; elsewhere the driver's own EGL is enough.

## What is not settled

- Whether grounding scales from the 4 instructions proven in the rung runs to
  the full 1,380. The full run has not completed; the evaluation metric to watch
  is **right-piece rate on unseen instructions**, which every evaluation
  measures for free because positions are drawn at random.
- **Loose pieces** — a piece knocked over or off its square during play. A
  third family, `put the loose piece on e4`, was collected and trained once
  (1,191 episodes; 1/16 at evaluation against 15/32 moves and 9/16 captures)
  and then dropped: it is a visual search task the other two are not, and the
  scripted expert cannot pick up a piece lying on its side (standing one up
  needs the IK tilt cone relaxed and a regrasp). The plan is a magnetic board
  so pieces do not get knocked over in the first place; if the task returns,
  the instruction has to name the piece type, since several may be loose at
  once, and that needs actual recognition.
