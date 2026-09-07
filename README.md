# chess_so101

A MuJoCo framework for an SO-101 arm playing chess: sim-ready assets, one
assembled scene, and a small API to set up positions, command moves with a
scripted expert, drive the joints yourself, and record datasets.

```
chess_sim/                 the framework package
  assets/                  sim-ready files: so101/ (MJCF + STL), pieces/<name>/ (OBJ + texture),
                           scene/ (board texture, backdrop, exported chess_so101.xml)
  calib/                   executed_reach.json — measured per-square fingertip accuracy
  board.py                 board geometry (BoardSpec), square <-> world coordinates
  assets.py                asset registry, piece scaling
  scene.py                 MjSpec assembly (table, board, 32 pieces, arm, cameras) + XML export
  ik.py                    IK for the arm: mink QP tasks on an arm-only kinematic model
  gripper.py               measured pinch-pocket / jaw-gap calibration
  controller.py            scripted pick-and-place expert
  reach.py                 which squares/moves the expert can execute
  env.py                   ChessSimEnv — the public API
  recorder.py              EpisodeRecorder — LeRobot-style episode files
examples/                  play_random_moves.py, play_opening.py, collect_dataset.py, calibrate_reach.py
```

## Quick start

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate embodiedgen
export MUJOCO_GL=egl PYTHONPATH=~/chess_so101
python examples/play_random_moves.py --moves 5
```

```python
from chess_sim import ChessSimEnv, START_FEN

env = ChessSimEnv()                     # builds and compiles the scene once (~4 s)
env.reset(START_FEN)                    # any FEN board field; pieces absent from it park off-board
result = env.move("g1", "f3")           # scripted expert: MoveResult(success, placement_error, ...)
print(env.board.fen())                  # python-chess mirror of the physical board

obs = env.step(action)                  # or drive the 6 joints yourself (radians, 30 Hz)
frame = env.render("top")               # 'external' or 'top' camera
env.export_xml("scene.xml")             # standalone MuJoCo XML of the scene
```

Recording an episode:

```python
from chess_sim import EpisodeRecorder
rec = EpisodeRecorder(env, "datasets/chess")
rec.begin("move the white knight from g1 to f3", fen=env.board.fen(), move="g1f3")
result = env.move("g1", "f3", on_step=rec.on_step)
rec.end(result.success, placement_error=result.placement_error)
```

Each episode directory holds `data.npz` (`observation_state`, `action`), one mp4
per camera, and `meta.json`; `manifest.jsonl` indexes the dataset. Train only on
episodes with `"success": true`.

## What the scripted expert does

`env.move()` picks the piece with a tool-axis approach, closes the jaws onto
it, carries it world-upright above the other pieces, and sets it down on the
target square, then verifies: piece within 11 mm of the square center, upright,
and no other piece displaced by more than 10 mm. Everything is simulated
physics: the piece is held only by friction between the fingertip pads, and
the arm's position servos track IK joint targets under gravity and contact.
`env.executable_moves()` lists the quiet legal moves the arm can perform in
the current position with clearance for the jaws.

## Design notes

- **Fixed topology.** All 32 pieces always exist as free bodies; `reset(fen)`
  moves them onto squares or into off-board graveyard slots. One compiled model
  serves every episode, and the scene exports as a plain XML.
- **Mini board.** 2.8 cm squares, arm on a 14 cm riser 4 cm from the edge: the
  SO-101 reaches every square; the sampler further restricts moves to squares
  it can grasp within a 30° tool tilt and with measured tracking accuracy
  (`examples/calibrate_reach.py` regenerates that map).
- **Piece colliders** are profiled stacks of cylinders sampled from the mesh
  (flat base for stable settling, true radii above it). Generated convex hulls
  rest on a rounded nub and creep across the board.
- **Where a piece is held.** Straight prongs cannot pinch a narrow neck below a
  wider crown, so each piece is grasped at its widest segment above the base
  (a king by its crown, a pawn by its head) and the jaws close to that radius.
  The 2 cm pads must not reach down into a wider segment below the one they
  clamp, so a rook is held high enough that the pads stay above its base
  flange (it drops 7 mm at release), and open prongs are positioned to clear
  the widest radius anywhere in the band the pads span, on the way in and out.
- **The grasp is physical.** Pieces are held purely by friction between
  fingertip contact pads: the jaws close 2.5 mm past the piece surface and the
  piece rides on the resulting clamp (~1–2 N) through lift, transit and
  placement, with the live piece-to-pocket offset measured before setting
  down. What made it work (see `gripper.py`):
  the stock finger meshes collide as convex hulls that fill the pinch pocket, so
  thin pads replace them at the fingertips; MuJoCo's default soft contact is
  mass-normalized and clamps a 4 g piece with ~0.1 N, so the pads carry an
  explicit stiffness (`solref` in N/m), full 6-D friction and contact priority;
  the moving pad is mounted parallel to the fixed one at the nominal grasp
  angle; and the scene uses elliptic friction cones with `impratio` 10,
  because with MuJoCo's default pyramidal cones a held piece creeps through
  the pads by about a centimeter over a two-second carry.
- **IK is only a target generator.** `So101Ik` turns "pinch pocket at this
  square" into joint targets; the position actuators then track them under
  full dynamics, with a small integral bias per waypoint to cancel the servos'
  pose-dependent steady-state error. The solver is [mink](https://github.com/kevinzakka/mink):
  each solve is a few quadratic programs over an arm-only copy of the SO-101
  mounted as in the scene (`scene.build_arm`; the full scene's 32 free pieces
  would make each QP 25x slower), with a tool-point task, an axis-align task
  for the approach direction, a roll-only jaw-span task, a weak posture
  tie-breaker, and joint limits plus a per-iteration step bound as hard
  constraints. The step bound matters: the arm's zero pose is a singular
  vertical stack, and an unbounded Gauss-Newton step from there lands in a
  wrong basin on the near squares.
- **Jaw-span selection.** The wrist roll is chosen per move so the moving jaw
  opens toward the freest neighboring square (`ChessSimEnv.free_span_direction`,
  `So101Ik.solve(span=...)`), and the piece rides next to the fixed prong while
  the jaws are open so all the opening slack lands on the free side. In the
  solver the span objective drives the roll alone: the roll's 320° range puts
  every direction within ~20° of a reachable angle, and a span past the limit
  is realized as far as possible instead of bending the arm to serve it.
- **Crowding limit.** On a fully populated opening position the ~5 mm prongs
  have only ~2 mm of corridor between 28 mm squares, and the roll joint cannot
  reach a full 180°; expect occasional neighbor contact there. Sparse and
  moderately crowded positions execute reliably: 50/50 random moves over five
  seeds of `examples/play_random_moves.py` at 0.2–4.2 mm placement, and a
  5-move ladder mate 5/5. A larger square pitch trades reach for clearance.
- Captures, castling, promotion and en passant are not executed by the expert
  yet; the graveyard slots make captures a small follow-up.
