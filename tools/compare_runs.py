"""Compare two evaluation runs that scored the same positions.

    python tools/compare_runs.py outputs/sweeps/before.jsonl outputs/sweeps/after.jsonl \
        --arm horizon30

Both files come from `tools/sweep_inference.py` run with the same `--seed`, so
position i is the same board, the same shift and the same arm start pose in
both. That makes the comparison paired: only the positions where the two runs
disagree carry information, and a few dozen of those settle a difference that
hundreds of unpaired episodes would leave ambiguous.
"""
import argparse
import collections
import json
import sys

sys.path.insert(0, "tools")
from sweep_inference import mcnemar                      # noqa: E402

from chess_sim import wilson_interval                    # noqa: E402


def load(path: str, arm: str | None) -> dict[int, dict]:
    rows = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if arm is None or record["arm"] == arm:
                rows[record["position"]] = record
    return rows


def report(name: str, rows: dict[int, dict]) -> None:
    ok = sum(r["success"] for r in rows.values())
    low, high = wilson_interval(ok, len(rows))
    print(f"  {name:28s} {ok}/{len(rows)} ({100 * ok / max(len(rows), 1):.0f}%, "
          f"95% CI {low * 100:.0f}-{high * 100:.0f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--arm", default=None, help="only this arm from each file")
    args = ap.parse_args()

    before, after = load(args.before, args.arm), load(args.after, args.arm)
    shared = sorted(set(before) & set(after))
    if not shared:
        raise SystemExit("the two runs share no positions - were they run with the same --seed?")
    print(f"{len(shared)} shared positions" + (f", arm {args.arm}" if args.arm else ""))
    report(args.before, {i: before[i] for i in shared})
    report(args.after, {i: after[i] for i in shared})

    families = collections.defaultdict(list)
    for i in shared:
        families[before[i]["family"]].append(i)
    for family, positions in sorted(families.items()):
        b, c, p = mcnemar([(before[i]["success"], after[i]["success"]) for i in positions])
        gain = sum(after[i]["success"] for i in positions) - sum(before[i]["success"] for i in positions)
        print(f"  {family:10s} {len(positions):3d} positions: {gain:+d} episodes "
              f"(before won {b}, after won {c}, p={p:.3f})")
    b, c, p = mcnemar([(before[i]["success"], after[i]["success"]) for i in shared])
    print(f"  {'overall':10s} {len(shared):3d} positions: before won {b}, after won {c}, p={p:.3f}")

    # what changed underneath the score
    for label, rows in (("before", before), ("after", after)):
        reasons = collections.Counter(rows[i]["reason"] or "success" for i in shared)
        print(f"  {label} failures: "
              + ", ".join(f"{k} {v}" for k, v in reasons.most_common() if k != "success"))


if __name__ == "__main__":
    main()
