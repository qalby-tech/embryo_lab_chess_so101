#!/bin/bash
# Check the machine can actually train before starting a run that takes a day.
#
# Written after a day lost to a corrupt driver library: a job that had worked
# the day before began failing with impossible out-of-memory errors (22 MiB
# refused with 8.7 GiB free) and then hanging. The cause was in the kernel log
# the whole time. These three checks would have found it in a minute.
set -u
fail=0

echo "=== 1. GPU present ==="
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv,noheader || { echo "FAIL: no GPU"; exit 1; }

echo
echo "=== 2. kernel log clean of driver faults ==="
bad=$(dmesg 2>/dev/null | grep -ciE "is truncated|dxgkio_escape|Xid |gpu has fallen" || true)
if [ "${bad:-0}" -gt 0 ]; then
  echo "FAIL: $bad driver fault lines in dmesg:"
  dmesg 2>/dev/null | grep -iE "is truncated|dxgkio_escape|Xid |gpu has fallen" | tail -5
  echo "  -> a truncated NVIDIA library or failing GPU ioctls will surface as"
  echo "     out-of-memory errors and hangs. Reinstall the GPU driver (on WSL,"
  echo "     a clean reinstall of the Windows driver) before training."
  fail=1
else
  echo "ok: no truncated libraries, no GPU ioctl failures"
fi

echo
echo "=== 3. a single process can reserve what training needs ==="
PY=${PYTHON:-python3}
$PY - <<'PYEOF'
import sys
try:
    import torch
except ImportError:
    print("SKIP: torch not importable here"); sys.exit(0)
if not torch.cuda.is_available():
    print("FAIL: torch cannot see the GPU"); sys.exit(1)
free, total = torch.cuda.mem_get_info()
print(f"device reports {free/2**30:.1f} GiB free of {total/2**30:.1f} GiB")
blocks, gib = [], 0
try:
    while gib < 31:
        blocks.append(torch.empty(2**30, dtype=torch.uint8, device="cuda")); gib += 1
except RuntimeError:
    pass
print(f"largest cumulative allocation: {gib} GiB")
# training peaks near 30 GB at batch 8; 27 GiB was the symptom of a sick driver
print("ok" if gib >= 30 else "WARNING: under 30 GiB, batch 8 will not fit")
PYEOF

echo
[ $fail -eq 0 ] && echo "preflight passed" || echo "preflight FAILED - fix the above before training"
exit $fail
