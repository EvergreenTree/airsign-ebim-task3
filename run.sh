#!/usr/bin/env bash
set -euo pipefail

ROOT="${AIRSIGN_TASK3_ROOT:-/mnt/nas/evergreen/ebim-task3}"
VENV="${AIRSIGN_TASK3_VENV:-${ROOT}/.venv}"
WORKSPACE="${AIRSIGN_TASK3_WORKSPACE:-${ROOT}/workspace}"
PYTHON="${AIRSIGN_TASK3_PYTHON:-${VENV}/bin/python}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing managed runtime: ${PYTHON}" >&2
  echo "Run ${WORKSPACE}/scripts/bootstrap_remote.sh first." >&2
  exit 2
fi

export PYTHONPATH="${WORKSPACE}${PYTHONPATH:+:${PYTHONPATH}}"
export OMNI_KIT_ACCEPT_EULA="YES"
export PRIVACY_CONSENT="Y"
export OMNI_USER_CONFIG_PATH="${ROOT}/cache/ov/config"
export OMNI_CACHE_PATH="${ROOT}/cache/ov/cache"
export OMNI_DATA_PATH="${ROOT}/cache/ov/data"
export NVIDIA_SHADER_CACHE_PATH="${ROOT}/cache/nvidia-shaders"
export XDG_CACHE_HOME="${ROOT}/cache/xdg"
export XDG_RUNTIME_DIR="${ROOT}/cache/xdg-runtime"
export VK_DRIVER_FILES="${WORKSPACE}/config/nvidia_icd.json"
export VK_ICD_FILENAMES="${WORKSPACE}/config/nvidia_icd.json"
export __GLX_VENDOR_LIBRARY_NAME="nvidia"

# The DSW headless image omits libXrandr, which Isaac's WebRTC encoder plugin
# loads even without an X display.  Bootstrap extracts Ubuntu's libxrandr2
# package here without mutating host packages.
VENDORED_RUNTIME_LIBS="${ROOT}/runtime-libs/xrandr/usr/lib/x86_64-linux-gnu"
if [[ -d "${VENDORED_RUNTIME_LIBS}" ]]; then
  export LD_LIBRARY_PATH="${VENDORED_RUNTIME_LIBS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

mkdir -p "${XDG_RUNTIME_DIR}"
chmod 700 "${XDG_RUNTIME_DIR}"

# Exit 75 is an intentional dashboard reset request. Restarting the whole
# Isaac process restores authored rigid-body and bean state without mutating
# any object during an active episode. Other exits propagate unchanged.
RESET_MARKER="${AIRSIGN_TASK3_RUN_ROOT:-${ROOT}/runs}/.reset-requested"
MAX_CRASH_RESTARTS="${AIRSIGN_TASK3_MAX_CRASH_RESTARTS:-2}"
crash_restarts=0
while true; do
  set +e
  "${PYTHON}" -m airsign_task3.main "$@"
  status=$?
  set -e
  # Isaac Kit's fast shutdown may force status 0 before Python can return 75.
  # The runtime writes this exact marker before closing, preserving the same
  # full-process reset semantics without touching any episode object.
  if [[ -f "${RESET_MARKER}" ]]; then
    rm -f -- "${RESET_MARKER}"
    continue
  fi
  # Isaac Sim's renderer plugin has been observed to segfault mid-episode --
  # twice in 34 launches here, at unrelated points in the plan, and always
  # while several Isaac instances shared one GPU. The episode dies with no
  # summary written, which scores as nothing at all. A crash is not a policy
  # decision, so restart the episode a bounded number of times rather than
  # handing back a dead container. Ordinary exits, including a policy failure,
  # propagate unchanged on the first attempt.
  if [[ ${status} -gt 128 ]] && (( crash_restarts < MAX_CRASH_RESTARTS )); then
    crash_restarts=$(( crash_restarts + 1 ))
    echo "airsign: simulator died with status ${status};" \
         "restarting episode (${crash_restarts}/${MAX_CRASH_RESTARTS})" >&2
    continue
  fi
  if [[ ${status} -ne 75 ]]; then
    exit "${status}"
  fi
done
