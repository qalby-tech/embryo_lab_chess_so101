"""Cut policy evaluation recordings into captioned per-episode clips and a highlight reel.

    python tools/make_vla_video.py moves.mp4:moves.log:move captures.mp4:captures.log:capture \
        --out sim/out/vla_video/reel.mp4 --per-group 3

The evaluator writes every episode back to back into one video and prints one result line per
episode, so the frames of episode i are the i-th equal share of the recording. Each clip is
captioned with its instruction and outcome; the reel shows successes, then failures, per family.
"""
import argparse, os, re, subprocess, tempfile

LINE = re.compile(r"^\[(\d+)\] (.+): (OK|FAIL) \(([\d.]+) mm([^)]*)\)\s*$", re.M)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_SIZE = 22          # at the full size a caption is drawn at
MARGIN = 14             # px of caption inset from the left edge
ADVANCE = 0.62          # DejaVu Sans Bold: average glyph width as a fraction of the size

def frames(path):
    """Frame count, frame rate and width of a recording."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,nb_frames,r_frame_rate", "-of", "csv=p=0", path],
                         capture_output=True, text=True, check=True).stdout.strip().split(",")
    width, rate, count = out
    num, den = rate.split("/")
    return int(count), float(num) / float(den), int(width)

def font_size(text, width):
    """The largest size up to FONT_SIZE at which `text` fits across the frame."""
    return max(12, min(FONT_SIZE, int((width - 2 * MARGIN) / (ADVANCE * max(len(text), 1)))))

def clip(src, start_f, n_f, fps, caption, outcome, ok, dst, tmp, width):
    cap, res = os.path.join(tmp, "cap.txt"), os.path.join(tmp, "res.txt")
    open(cap, "w").write(caption); open(res, "w").write(outcome)   # textfile avoids escaping colons
    color = "0x33dd55" if ok else "0xff4444"
    size = font_size(caption, width)       # a narrow view needs a smaller caption to fit
    vf = (f"trim=start_frame={start_f}:end_frame={start_f + n_f},setpts=PTS-STARTPTS,"
          f"drawbox=x=0:y=0:w=iw:h=66:color=black@0.6:t=fill,"
          f"drawtext=fontfile={FONT}:textfile={cap}:x={MARGIN}:y=10:fontsize={size}:fontcolor=white,"
          f"drawtext=fontfile={FONT}:textfile={res}:x={MARGIN}:y=38:fontsize={FONT_SIZE}:fontcolor={color}")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-vf", vf, "-an",
                    "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", "-r", str(round(fps)), dst],
                   check=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+", help="video:log:family")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-group", type=int, default=3, help="successes and failures shown per family")
    ap.add_argument("--limit", type=int, default=None, help="only cut the first N episodes (testing)")
    ap.add_argument("--reel-only", action="store_true",
                    help="cut only the clips the reel uses, not one per episode")
    args = ap.parse_args()
    # one folder per reel, so cutting a second view does not overwrite the first one's clips
    stem = os.path.splitext(os.path.basename(args.out))[0]
    clip_dir = os.path.join(os.path.dirname(args.out), "clips", stem); os.makedirs(clip_dir, exist_ok=True)
    reel = []
    with tempfile.TemporaryDirectory() as tmp:
        for spec in args.specs:
            video, log, family = spec.rsplit(":", 2)
            eps = LINE.findall(open(log, errors="ignore").read())
            total, fps, width = frames(video)
            per = total // max(len(eps), 1)
            made = {"OK": [], "FAIL": []}
            for i, instr, k, err, extra in eps[: args.limit]:
                i = int(i); ok = k == "OK"
                if args.reel_only and len(made[k]) >= args.per_group:
                    continue
                outcome = f"SUCCESS  {float(err):.1f} mm" if ok else f"FAIL  {float(err):.1f} mm{extra}"
                dst = os.path.join(clip_dir, f"{family}_{i:02d}_{'ok' if ok else 'fail'}.mp4")
                clip(video, i * per, per, fps, f"{family.upper()}  |  {instr}", outcome, ok, dst, tmp, width)
                made[k].append(dst)
            print(f"{family}: {len(eps)} episodes, {per} frames each, {len(made['OK'])} successes, {len(made['FAIL'])} failures")
            reel += made["OK"][: args.per_group] + made["FAIL"][: args.per_group]
        listing = os.path.join(tmp, "list.txt")
        open(listing, "w").write("".join(f"file '{os.path.abspath(c)}'\n" for c in reel))
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", listing,
                        "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", args.out], check=True)
    print(f"reel: {len(reel)} clips -> {args.out}")

if __name__ == "__main__":
    main()
