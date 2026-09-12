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
# dxgkio_escape failures are NOT included: nvidia-smi emits them on every run
# under WSL and they rise only when nvidia-smi is called, never from actual CUDA
# work. Treating them as faults blocks a perfectly healthy machine.
bad=$(dmesg 2>/dev/null | grep -ciE "is truncated|Xid |gpu has fallen" || true)
if [ "${bad:-0}" -gt 0 ]; then
  echo "FAIL: $bad driver fault lines in dmesg:"
  dmesg 2>/dev/null | grep -iE "is truncated|Xid |gpu has fallen" | tail -5
  echo "  -> a truncated NVIDIA library surfaces as out-of-memory errors with"
  echo "     gigabytes free, and as hangs with the GPU idle holding memory."
  echo "     Reinstall the GPU driver (on WSL, a clean reinstall of the Windows"
  echo "     driver, then wsl --shutdown) before training."
  fail=1
else
  echo "ok: no truncated libraries, no Xid errors"
fi

echo
echo "=== 2b. WSL driver libraries match the installed driver ==="
if [ -d /usr/lib/wsl/lib ]; then
  # WSL copies these from the Windows driver at VM boot and never again, so a
  # driver update does nothing until wsl --shutdown. Comparing the library's
  # date with the boot time catches the case where someone updated the driver
  # and expected it to take effect.
  libdate=$(stat -c %Y /usr/lib/wsl/lib/libnvidia-gpucomp.so 2>/dev/null || echo 0)
  boot=$(( $(date +%s) - $(cut -d. -f1 /proc/uptime) ))
  if [ "$libdate" -gt 0 ] && [ "$libdate" -lt "$boot" ]; then
    echo "note: driver libraries predate this boot ($(date -d @$libdate '+%Y-%m-%d'))."
    echo "      that is normal, unless the driver was updated since - in which"
    echo "      case run 'wsl --shutdown' so WSL picks the new one up."
  else
    echo "ok: driver libraries refreshed at this boot"
  fi
else
  echo "not WSL, skipping"
fi

echo
echo "=== 3. a single process can reserve what training needs ==="
PY=${PYTHON:-python3}
# A missing interpreter or a missing torch must fail the preflight: once, the
# training virtualenv had been deleted and this check was silently skipped, so
# the script reported "passed" on a machine that could not train at all.
if ! command -v "$PY" >/dev/null 2>&1 && [ ! -x "$PY" ]; then
  echo "FAIL: interpreter $PY not found"; fail=1
else
$PY - <<'PYEOF'
import sys
try:
    import torch
except ImportError:
    print("FAIL: torch not importable with this interpreter"); sys.exit(1)
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
[ $? -eq 0 ] || fail=1
fi

echo
echo "=== 4. the training stack imports ==="
# Torch and the GPU can be perfectly healthy while the trainer dies on its first
# import: a rebuilt environment once lacked LeRobot's [dataset] extra, passed
# every check above, and failed at startup with "'datasets' is required".
if command -v "$PY" >/dev/null 2>&1 || [ -x "$PY" ]; then
  if $PY -c "import datasets, peft, accelerate, lerobot.datasets.lerobot_dataset; from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy" 2>/tmp/preflight_import.err; then
    echo "ok: lerobot dataset stack and MolmoAct2 policy import"
  else
    echo "FAIL: training imports broken:"; grep -E "Error" /tmp/preflight_import.err | tail -2
    echo "  -> install lerobot with extras: pip install -e 'lerobot[molmoact2,dataset,training]'"
    fail=1
  fi
fi

echo
[ $fail -eq 0 ] && echo "preflight passed" || echo "preflight FAILED - fix the above before training"
exit $fail
