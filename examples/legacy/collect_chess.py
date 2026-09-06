"""Scripted-expert episode collector: SO-101 plays chess moves in MuJoCo.

Each episode: sample a sparse random legal position + random legal quiet move
(python-chess), pick the piece with the SO-101 (weld attach on grasp) and place
it on the target square. Records LeRobot-style data at 30 Hz control rate:
  observation.state (6 joint pos), action (6 target pos), RGB from 2 cameras,
  language instruction, FEN + move metadata. Episodes auto-verified; failures
  are recorded but flagged so training uses successes only.

Usage:
  python collect_chess.py --episodes 3 [--render_video] [--out ~/chess_so101/datasets/chess]
"""
import argparse
import json
import os
import random

import chess
import imageio.v2 as imageio
import mujoco
import numpy as np

import build_chess_scene as bc
from so101_ik import So101Ik

CTRL_HZ = 30
SUBSTEPS = 16  # 30Hz * 16 * ~2ms = real time
HOVER = 0.055
GRASP_H = 0.014  # fallback pinch height above board top

def jaw_angle_for_gap(gap_m):
    """Gripper joint angle giving a fingertip gap (measured: 11.8mm @ -0.05, 7.6mm/0.1rad)."""
    return max(-0.17, -0.05 + (gap_m - 0.0118) / 0.076 * 0.1)
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
GRIP_OPEN, GRIP_CLOSED = 1.2, -0.05

PIECE_LETTERS = "PRNBQKprnbqk"
NAMES = {"p": "pawn", "r": "rook", "n": "knight", "b": "bishop", "q": "queen", "k": "king"}


def sparse_random_board(rng, n_extra=4):
    """Kings + a few random pieces, legal position, white to move."""
    while True:
        board = chess.Board(None)
        squares = rng.sample(range(64), 2 + n_extra)
        board.set_piece_at(squares[0], chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(squares[1], chess.Piece(chess.KING, chess.BLACK))
        pool = "PRNBQ" + "prnbq"
        for sq in squares[2:]:
            letter = rng.choice(pool)
            piece = chess.Piece.from_symbol(letter)
            if piece.piece_type == chess.PAWN and chess.square_rank(sq) in (0, 7):
                continue
            board.set_piece_at(sq, piece)
        board.turn = chess.WHITE
        if board.is_valid():
            return board


def fen_letter_and_square(board):
    out = []
    for sq, piece in board.piece_map().items():
        out.append((piece.symbol(), chess.square_file(sq), chess.square_rank(sq)))
    return out


class ChessEnv:
    def __init__(self, board: chess.Board):
        placements = fen_letter_and_square(board)
        fen_field = board.board_fen()
        self.spec = bc.build(fen_field)
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        self.board_top = bc.TABLE_TOP + bc.BOARD_T
        self.ik = So101Ik(self.model, prefix="so101:")
        # map square -> piece body name (pieces added in parse_fen_board order)
        self.sq2body = {}
        self.sq2geom = {}
        for idx, (letter, f, r) in enumerate(bc.parse_fen_board(fen_field)):
            adir = bc.ASSET_DIRS[letter]
            name = os.path.basename(os.path.dirname(adir))
            self.sq2body[(f, r)] = f"pc{idx}_{name}"
            b = bc.asset_bounds(adir, name)
            cur_h = float(b[1][2] - b[0][2])
            cur_w = float(max(b[1][0] - b[0][0], b[1][1] - b[0][1]))
            target_h = bc.PIECE_H[letter.lower()] * (bc.SQUARE / 0.035)
            sc = min(target_h / max(cur_h, 1e-6), 0.88 * bc.SQUARE / max(cur_w, 1e-6))
            self.sq2geom[(f, r)] = {"height": cur_h * sc, "width": cur_w * sc}
        self.jids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "so101:" + n)
            for n in JOINTS
        ]
        self.qadr = np.array([self.model.jnt_qposadr[j] for j in self.jids])
        self.aids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "so101:" + n)
            for n in JOINTS
        ]
        self.weld = None  # (body_id, eq index) while grasped
        self.renderer = None
        self.cams = {}

    def add_cameras(self):
        # external + top-down; created via renderer-defined free cameras
        ext = mujoco.MjvCamera()
        ext.lookat = [0, 0.02, self.board_top + 0.04]
        ext.distance, ext.azimuth, ext.elevation = 0.42, 210, -32
        top = mujoco.MjvCamera()
        top.lookat = [0, 0, self.board_top]
        top.distance, top.azimuth, top.elevation = 0.34, 90, -89
        self.cams = {"external": ext, "top": top}
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)

    def settle(self, seconds=1.0):
        rest = {"shoulder_pan": 0.0, "shoulder_lift": -1.0, "elbow_flex": 1.3,
                "wrist_flex": 0.4, "wrist_roll": 0.0, "gripper": GRIP_OPEN}
        for n, v in rest.items():
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "so101:" + n)
            self.data.ctrl[aid] = v
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "so101:" + n)
            self.data.qpos[self.model.jnt_qposadr[jid]] = v
        mujoco.mj_forward(self.model, self.data)
        for _ in range(int(seconds / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)

    def square_xy(self, f, r):
        return bc.square_center(f, r)

    def piece_pos(self, body_name):
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        return self.data.xpos[bid].copy(), bid

    def attach(self, body_name):
        """Weld piece to gripper by snapping its freejoint each step (kinematic attach)."""
        pos, bid = self.piece_pos(body_name)
        site = self.ik.site
        smat = self.data.site_xmat[site].reshape(3, 3)
        spos = self.data.site_xpos[site]
        from so101_ik import TIP_LEN
        tip = spos + smat[:, 0] * TIP_LEN
        pmat = self.data.xmat[bid].reshape(3, 3).copy()
        self.weld = {
            "bid": bid,
            "jadr": self.model.jnt_qposadr[self.model.body_jntadr[bid]],
            "rel_pos": smat.T @ (pos - spos),
            # carry the piece upright: keep only the yaw of its current pose
            # (a bumped/tilted piece gets straightened rather than placed tilted)
            "world_quat": np.array([
                np.cos(np.arctan2(pmat[1, 0], pmat[0, 0]) / 2), 0.0, 0.0,
                np.sin(np.arctan2(pmat[1, 0], pmat[0, 0]) / 2),
            ]),
            "xy_offset": (pos - tip)[:2].copy(),
        }

    def release(self):
        self.weld = None

    def step_ctrl(self, targets):
        for aid, t in zip(self.aids, targets):
            self.data.ctrl[aid] = t
        for _ in range(SUBSTEPS):
            mujoco.mj_step(self.model, self.data)
            if self.weld is not None:
                site = self.ik.site
                smat = self.data.site_xmat[site].reshape(3, 3)
                spos = self.data.site_xpos[site]
                new_pos = spos + smat @ self.weld["rel_pos"]
                # keep the piece's WORLD orientation from grasp time: the tool may
                # tilt up to ~45deg in transit and a gripper-relative pin would lay
                # the piece down; world-pinned it stays upright for placement
                quat = self.weld["world_quat"]
                j = self.weld["jadr"]
                self.data.qpos[j:j + 3] = new_pos
                self.data.qpos[j + 3:j + 7] = quat
                self.data.qvel[
                    self.model.jnt_dofadr[self.model.body_jntadr[self.weld["bid"]]]:
                    self.model.jnt_dofadr[self.model.body_jntadr[self.weld["bid"]]] + 6
                ] = 0

    def render_all(self):
        out = {}
        for name, cam in self.cams.items():
            self.renderer.update_scene(self.data, camera=cam)
            out[name] = self.renderer.render()
        return out


def interp(q0, q1, n):
    for i in range(1, n + 1):
        yield q0 + (q1 - q0) * (i / n)


_REACH = {}


def global_reach_mask():
    """Board-independent reachability (IK + tilt + executed accuracy + near-field)."""
    if _REACH:
        return _REACH
    env = ChessEnv(chess.Board("8/8/8/8/8/8/8/8 w - - 0 1"))
    env.settle()
    arm_y = -(bc.BOARD_W / 2 + bc.ARM_OFFSET_Y)
    scratch = mujoco.MjData(env.model)
    exe_path = os.path.join(bc.OUT, "executed_reach.json")
    executed = json.load(open(exe_path)) if os.path.exists(exe_path) else {}
    for f in range(8):
        for r in range(8):
            x, y = env.square_xy(f, r)
            q_sol, err = env.ik.solve(env.data, np.array([x, y, env.board_top + GRASP_H]))
            scratch.qpos[:] = env.data.qpos
            scratch.qpos[env.qadr[:5]] = q_sol
            mujoco.mj_kinematics(env.model, scratch)
            xt = scratch.site_xmat[env.ik.site].reshape(3, 3)[:, 0]
            tilt_ok = -xt[2] > np.cos(np.radians(30))
            exe_ok = executed.get(f"{f},{r}", 0.0) < 0.005
            near = np.hypot(x, y - arm_y) < 0.14
            _REACH[(f, r)] = bool(err < 0.008 and tilt_ok and exe_ok and not near)
    print(f"reach mask: {sum(_REACH.values())}/64 squares eligible")
    return _REACH


def candidate_moves(board, reach):
    arm_y = -(bc.BOARD_W / 2 + bc.ARM_OFFSET_Y)
    occ = {(chess.square_file(sq), chess.square_rank(sq)) for sq in board.piece_map()}

    def armside_clear(f, r, vacated=None):
        x, y = bc.square_center(f, r)
        v = np.array([0.0 - x, arm_y - y])
        v = v / (np.linalg.norm(v) + 1e-9)
        nf, nr = f + int(round(v[0])), r + int(round(v[1]))
        if not (0 <= nf < 8 and 0 <= nr < 8):
            return True
        return not ((nf, nr) in occ and (nf, nr) != vacated)

    out = []
    for mv in board.legal_moves:
        if board.is_capture(mv) or mv.promotion or board.is_castling(mv):
            continue
        f0, r0 = chess.square_file(mv.from_square), chess.square_rank(mv.from_square)
        f1, r1 = chess.square_file(mv.to_square), chess.square_rank(mv.to_square)
        if (reach[(f0, r0)] and reach[(f1, r1)]
                and armside_clear(f0, r0) and armside_clear(f1, r1, vacated=(f0, r0))):
            out.append(mv)
    return out


def run_episode(ep_idx, rng, out_dir, render_video=False):
    reach = global_reach_mask()
    for _ in range(200):
        board = sparse_random_board(rng, n_extra=rng.randint(2, 8))
        moves = candidate_moves(board, reach)
        if moves:
            break
    else:
        return None
    mv = rng.choice(moves)
    env = ChessEnv(board)
    env.settle()
    env.add_cameras()
    start_pose = {}
    for sq, bname in env.sq2body.items():
        p, bid = env.piece_pos(bname)
        start_pose[bname] = p

    f0, r0 = chess.square_file(mv.from_square), chess.square_rank(mv.from_square)
    f1, r1 = chess.square_file(mv.to_square), chess.square_rank(mv.to_square)
    piece = board.piece_at(mv.from_square)
    body = env.sq2body[(f0, r0)]
    instruction = (
        f"move the {'white' if piece.color else 'black'} {NAMES[piece.symbol().lower()]} "
        f"from {chess.square_name(mv.from_square)} to {chess.square_name(mv.to_square)}"
    )

    x0, y0 = env.square_xy(f0, r0)
    x1, y1 = env.square_xy(f1, r1)
    geom = env.sq2geom[(f0, r0)]
    # pinch at the piece waist: ~55% of its height, but at least 8mm above board
    zg = env.board_top + max(0.008, 0.55 * geom["height"])
    grip_hold = jaw_angle_for_gap(0.55 * geom["width"])
    grip_release = jaw_angle_for_gap(geom["width"] + 0.014)  # just clears the piece
    # approach with jaws opened only slightly wider than the piece: a fully open
    # jaw spans ~10cm (3+ squares) and bulldozes the target and its neighbors
    grip_approach = jaw_angle_for_gap(geom["width"] + 0.016)
    zh = env.board_top + HOVER

    # expert waypoints: (xyz target, gripper, n_ctrl_steps, attach/release action)
    q_now = env.data.qpos[env.qadr][:5].copy()
    plan = [
        (np.array([x0, y0, zh]), GRIP_OPEN, 25, None),
        (np.array([x0, y0, zg]), GRIP_OPEN, 18, None),
        (np.array([x0, y0, zg]), grip_hold, 10, "close_then_attach"),
        (np.array([x0, y0, zh]), grip_hold, 15, None),
        (np.array([x1, y1, zh]), grip_hold, 25, None),
        (np.array([x1, y1, zg]), grip_hold, 18, None),
        (np.array([x1, y1, zg]), grip_release, 8, "release"),
        (np.array([x1, y1, zh]), grip_release, 15, None),
        (np.array([x1, y1, zh]), GRIP_OPEN, 6, None),
    ]

    obs_state, actions, frames = [], [], {"external": [], "top": []}
    ok = True

    def exec_waypoint(target, grip, n_steps, precise=False, z_off=0.0):
        """Interp to IK goal, then closed-loop hold with servo-bias correction."""
        nonlocal q_now, ok
        q_goal, err = env.ik.solve(env.data, target, z_off=z_off)
        if err > 0.010 and precise:
            ok = False
        for q in interp(q_now, q_goal, n_steps):
            full = np.concatenate([q, [grip]])
            obs_state.append(env.data.qpos[env.qadr].copy())
            actions.append(full.copy())
            env.step_ctrl(full)
            imgs = env.render_all()
            for k in frames:
                frames[k].append(imgs[k])
        # hold with integral bias against the EXECUTED tip error. The bias is
        # updated only every 8 control steps: per-step updates with a position
        # servo (multi-step response) limit-cycle and never converge
        from so101_ik import TIP_LEN as _TL
        loc = np.array([_TL, 0.0, z_off])
        bias = np.zeros(5)
        for round_i in range(11 if precise else 2):
            _sm = env.data.site_xmat[env.ik.site].reshape(3, 3)
            _tip = env.data.site_xpos[env.ik.site] + _sm @ loc
            if np.linalg.norm(_tip - target) < 0.004:
                break
            resid = q_goal - env.data.qpos[env.qadr][:5]
            bias = np.clip(bias + 0.6 * resid, -0.45, 0.45)
            full = np.concatenate([q_goal + bias, [grip]])
            for _ in range(8):
                obs_state.append(env.data.qpos[env.qadr].copy())
                actions.append(full.copy())
                env.step_ctrl(full)
                imgs = env.render_all()
                for k in frames:
                    frames[k].append(imgs[k])
        q_now = q_goal.copy()
        if precise:
            from so101_ik import TIP_LEN as _TL
            _sm = env.data.site_xmat[env.ik.site].reshape(3, 3)
            _tip = env.data.site_xpos[env.ik.site] + _sm[:, 0] * _TL
            print(f"   precise-wp executed tip-err = {np.linalg.norm(_tip - target)*1000:.1f}mm "
                  f"(ik-err {err*1000:.1f}mm)")

    def approach_dir(target):
        """Tool approach axis (unit, points from pregrasp toward target)."""
        q_sol, _ = env.ik.solve(env.data, target)
        scratch = mujoco.MjData(env.model)
        scratch.qpos[:] = env.data.qpos
        scratch.qpos[env.qadr[:5]] = q_sol
        mujoco.mj_kinematics(env.model, scratch)
        return scratch.site_xmat[env.ik.site].reshape(3, 3)[:, 0].copy()

    # -- approach high, then over the piece --
    exec_waypoint(np.array([x0, y0, zh + 0.05]), GRIP_OPEN, 22)
    # re-target on the piece's LIVE position, then approach ALONG the tool axis so
    # a tilted gripper doesn't sweep sideways through the piece on descent
    ppos, _ = env.piece_pos(body)
    gx, gy = float(ppos[0]), float(ppos[1])
    gtarget = np.array([gx, gy, zg])
    pre_descent_piece = ppos.copy()
    adir = approach_dir(gtarget)
    pocket = 0.5 * geom["width"] + 0.002
    exec_waypoint(gtarget - adir * 0.06, grip_approach, 16, z_off=pocket)
    for lam in (0.66, 0.33):
        exec_waypoint(gtarget - adir * 0.06 * lam, grip_approach, 5, z_off=pocket)
    exec_waypoint(gtarget, grip_approach, 8, precise=True, z_off=pocket)
    # attach on the converged pose, then close the jaws around the held piece
    ppos2, _ = env.piece_pos(body)
    print(f"   piece drift during descent = {np.linalg.norm(ppos2[:2]-pre_descent_piece[:2])*1000:.1f}mm")
    env.attach(body)
    print(f"   attach: tip-to-piece offset = "
          f"{np.linalg.norm(env.weld['xy_offset'])*1000:.1f}mm")
    exec_waypoint(gtarget, grip_hold, 10)
    # -- lift (back along approach), transit, place --
    exec_waypoint(gtarget - adir * 0.07, grip_hold, 14)
    # transit altitude: carried piece hangs below the grip; clear the tallest
    # possible standing piece (~46mm) plus margin
    zt = env.board_top + 0.058 + 0.55 * geom["height"]
    exec_waypoint(np.array([gx, gy, zt]), grip_hold, 8)
    exec_waypoint(np.array([x1, y1, zt]), grip_hold, 24)
    # live grasp offset right before placement: land the PIECE on the square
    from so101_ik import TIP_LEN as _TL
    smat = env.data.site_xmat[env.ik.site].reshape(3, 3)
    tip_now = env.data.site_xpos[env.ik.site] + smat[:, 0] * _TL
    ppos_now, _ = env.piece_pos(body)
    off = (ppos_now - tip_now)[:2]
    ptarget = np.array([x1 - off[0], y1 - off[1], zg])
    adir2 = approach_dir(ptarget)
    exec_waypoint(ptarget - adir2 * 0.06, grip_hold, 14)
    for lam in (0.66, 0.33):
        exec_waypoint(ptarget - adir2 * 0.06 * lam, grip_hold, 5)
    exec_waypoint(ptarget, grip_hold, 8, precise=True)

    env.release()
    exec_waypoint(ptarget, grip_release, 8)
    exec_waypoint(ptarget - adir2 * 0.07, grip_release, 14)
    exec_waypoint(np.array([x1, y1, zh + 0.04]), GRIP_OPEN, 10)

    # verify: piece near target square center, upright, and nothing else moved
    pos, bid = env.piece_pos(body)
    dist = float(np.hypot(pos[0] - x1, pos[1] - y1))
    zaxis = env.data.xmat[bid].reshape(3, 3)[2, 2]
    disturbed = []
    for bname, p0 in start_pose.items():
        if bname == body:
            continue
        p_now, _ = env.piece_pos(bname)
        if np.linalg.norm(p_now[:2] - p0[:2]) > 0.010:
            disturbed.append(bname)
    success = ok and dist < 0.011 and zaxis > 0.85 and not disturbed
    if disturbed:
        print(f"   disturbed: {disturbed}")
    if not success:
        print(f"   debug: ok={ok} dist={dist*1000:.1f}mm zaxis={zaxis:.2f}")

    ep_dir = os.path.join(out_dir, f"episode_{ep_idx:04d}")
    os.makedirs(ep_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(ep_dir, "data.npz"),
        observation_state=np.array(obs_state, dtype=np.float32),
        action=np.array(actions, dtype=np.float32),
    )
    for k, fr in frames.items():
        imageio.mimsave(os.path.join(ep_dir, f"{k}.mp4"), fr, fps=CTRL_HZ)
    meta = {
        "task": "chess_move", "instruction": instruction, "fen": board.fen(),
        "move": mv.uci(), "success": bool(success), "final_dist_m": dist,
        "disturbed": disturbed,
        "fps": CTRL_HZ, "n_steps": len(actions),
        "joints": JOINTS, "grasp_mode": "kinematic_attach",
    }
    with open(os.path.join(ep_dir, "meta.json"), "w") as fjson:
        json.dump(meta, fjson, indent=2)
    env.renderer.close()
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--out", default="/home/kamil/chess_so101/datasets/chess")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    manifest = open(os.path.join(args.out, "manifest.jsonl"), "a")
    n_ok = 0
    for i in range(args.episodes):
        meta = run_episode(i, rng, args.out)
        if meta is None:
            print(f"ep {i}: no reachable move, skipped")
            continue
        manifest.write(json.dumps({"episode": i, **meta}) + "\n")
        manifest.flush()
        n_ok += meta["success"]
        print(f"ep {i}: {meta['instruction']} -> success={meta['success']} dist={meta['final_dist_m']*1000:.1f}mm")
    print(f"done: {n_ok}/{args.episodes} verified successes")


if __name__ == "__main__":
    main()
