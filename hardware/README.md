# Building the rig

The physical counterpart of the simulated scene, dimension for dimension. Every
number here is generated from `chess_sim` by `hardware/generate.py` and stored
in `dimensions.json`, so the rig you build is the one the policy trained in.

**These are simulated dimensions.** The scene was tuned until the arm could work
a chess position reliably; nothing in this folder has been built or measured on a
physical bench yet, and the sim-to-real gap is unmeasured. Treat it as a
buildable specification, not a validated product.

```bash
python hardware/generate.py      # rebuilds dimensions.json, the STLs, parts.scad and the board art
```

Change a dimension in `chess_sim/config.py` (a different square size, a taller
table) and re-run it - the parts follow.

## Buy

| item | what it has to be |
| --- | --- |
| SO-101 follower arm | the standard build from [SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) (BOM and print files there); its meshes are vendored here under `chess_sim/assets/so101/` |
| overhead camera | ≥ 21.2° vertical field of view at 678 mm; the sim renders 24°, 640x480 |
| wrist camera | wide angle, ~75° field of view, small enough to sit beside the wrist-roll motor |
| table | flat, 430 mm to the top (or set your own `BoardConfig.table_top` and re-generate) |
| chess set | 28 mm squares, king no taller than 46 mm - a travel set, or print the board below |
| board substrate | 254 x 254 mm, 12 mm thick, for the printed artwork |
| fasteners | M3 for the arm plate, the mast and the camera mounts |

## Board and pieces

| | mm |
| --- | --- |
| square | 28.0 |
| playing field (8 squares) | 224.0 |
| border | 15.0 each side |
| board overall | 254.0 x 254.0 |
| board thickness | 12.0 |

`board_254mm.png` is the board artwork at 300 dpi - print at 100% scale (254 mm
across, borders included) and mount it on the substrate.

Piece dimensions, as the simulator models them at a 28 mm square:

| piece | per side | height | widest diameter | grasp diameter | grasp height |
| --- | --- | --- | --- | --- | --- |
| pawn | 8 | 27.2 | 13.1 | 8.3 | 22.4 |
| rook | 2 | 30.4 | 21.9 | 12.0 | 10.6 |
| knight | 2 | 33.6 | 18.2 | 17.6 | 18.5 |
| bishop | 2 | 36.8 | 13.4 | 12.1 | 20.2 |
| queen | 1 | 41.6 | 22.1 | 12.7 | 34.3 |
| king | 1 | 46.4 | 17.3 | 12.6 | 38.3 |

Grasp diameter is where the jaws close and grasp height is how far up the piece
that is - a real set only has to match those two within a couple of millimetres
for the trained grasp to transfer. The widest diameter matters for clearance:
nothing may exceed the 28 mm pitch.

## Workstation

Everything is measured from the centre of the board, looking from white's side
(the arm's side is -y):

| | mm |
| --- | --- |
| table top above the floor | 430 |
| board edge to arm base | 80 |
| arm base from board centre | (0, -207) |
| arm riser under the base | 60 tall |
| arm base footprint | 111 x 72 |
| discard tray, first slot | (-160, -100) |
| tray slots | 4, at 40 pitch, running +y |

The riser is what makes the near ranks reachable: mounted flat on the table the
servos cannot track ranks 1-3, and no game from the initial position is possible.

## Camera mast

A plate under the arm base carries a square tube beside the arm, camera on top,
looking straight down at the board.

| | mm |
| --- | --- |
| mast axis beside the arm axis (+x) | 160 |
| tube | 30 x 30, 600 tall above the plate |
| plate | 280 x 140 x 10 |
| lens above the table | 690 |
| lens above the board surface | 678 |
| what it must see | the whole 254 mm board: ≥ 21.2° vertical FOV |

The wrist camera sits at (-75, +45, +12) mm in the gripper's tool frame - beside
the wrist-roll motor, pointing at the fingertips - with a ~75° field of view.
Both cameras record at 640x480; those two views are the entire observation the
policy gets, so their placement matters more than their resolution.

## Print

`stl/` holds the four parts as solid blocks, sized from the scene:

| file | what it is |
| --- | --- |
| `arm_riser.stl` | 111 x 72 x 60 pedestal under the arm base |
| `mast_plate.stl` | 280 x 140 x 10 plate carrying arm and mast |
| `mast_tube.stl` | one half of the 600 mm mast, 30 mm square |
| `capture_tray.stl` | four 40 mm pockets for captured pieces |

`parts.scad` is the same set parametrically (hollow tube, real tray pockets, one
variable per dimension) for anyone who wants to adjust wall thickness, split the
mast differently, or fit their own camera mount. Both are generated - edit
`hardware/generate.py`, not the outputs.
