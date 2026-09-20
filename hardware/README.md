# hardware

Generated build files for the physical rig. The guide, with the drawings and
what to buy or print, is [docs/hardware.md](../docs/hardware.md).

| | |
| --- | --- |
| `dimensions.json` | the whole build sheet, as generated from `chess_sim` |
| `stl/` | four printable parts: riser, mast plate, mast tube, capture tray |
| `parts.scad` | the same parts parametrically, for adjusting them |
| `board_254mm.png` | the board artwork at 300 dpi, print at 100% |
| `generate.py` | rewrites all of the above, plus the figures in `docs/media/` |
| `diagrams.py` | the scale drawings, as SVG |

```bash
MUJOCO_GL=egl python hardware/generate.py
```

Nothing here is hand-written: change `chess_sim/config.py` and re-run.
