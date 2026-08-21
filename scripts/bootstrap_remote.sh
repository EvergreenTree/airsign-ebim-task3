#!/usr/bin/env bash
set -euo pipefail

# The DSW-managed loopback proxy is useful for some services but stalls large
# GitHub release downloads after notebook restarts. These endpoints are
# reachable directly from DSW, so keep the reproducible bootstrap independent
# of that proxy process.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

ROOT="${AIRSIGN_TASK3_ROOT:-/mnt/nas/evergreen/ebim-task3}"
BENCHMARK_COMMIT="e36119cc43e949dc6269bfe5c1e7f613f9f24d0c"

# Package index. The default is PyPI so this bootstrap works anywhere; the
# development host overrides it with a local mirror because it is far faster
# from there. NVIDIA's index is always needed for the Isaac Sim wheels.
PYPI_INDEX="${AIRSIGN_TASK3_PYPI_INDEX:-https://pypi.org/simple}"
NVIDIA_INDEX="${AIRSIGN_TASK3_NVIDIA_INDEX:-https://pypi.nvidia.com}"

# Artifact mirroring is optional and off unless a destination is given.
OSS_ROOT="${AIRSIGN_TASK3_OSS_ROOT:-}"

mkdir -p "$ROOT"/{artifacts,cache,logs,runs,tools/bin,workspace}
if [[ -n "$OSS_ROOT" ]]; then
  mkdir -p "$OSS_ROOT"/{artifacts,logs,runs}
fi

# DSW's NVIDIA driver is injected as read-only files.  The minimal host image
# does not include the generic loader libraries that native Isaac/Kit needs.
# libGL itself is already injected, so tolerate its package's cross-device
# backup failure while configuring every successfully unpacked dependency.
if ! ldconfig -p 2>/dev/null | grep -q 'libGLX.so.0'; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    libgl1 libglx0 libglu1-mesa libxt6 libvulkan1 vulkan-tools || true
  DEBIAN_FRONTEND=noninteractive dpkg --configure -a
  ldconfig
fi

export PATH="$ROOT/tools/bin:$PATH"

export UV_INSTALL_DIR="$ROOT/tools/bin"
export UV_CACHE_DIR="$ROOT/cache/uv"
export UV_PYTHON_INSTALL_DIR="$ROOT/tools/python"
export PIP_CACHE_DIR="$ROOT/cache/pip"
export OMNI_USER_CONFIG_PATH="$ROOT/cache/ov/config"
export OMNI_CACHE_PATH="$ROOT/cache/ov/cache"
export OMNI_DATA_PATH="$ROOT/cache/ov/data"
export NVIDIA_SHADER_CACHE_PATH="$ROOT/cache/nvidia-shaders"
export XDG_CACHE_HOME="$ROOT/cache/xdg"

if [[ ! -x "$ROOT/tools/bin/uv" ]]; then
  curl --noproxy '*' -LsSf https://astral.sh/uv/install.sh | sh
fi

UV="$ROOT/tools/bin/uv"
"$UV" python install 3.11

if [[ ! -d "$ROOT/benchmark/.git" ]]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --filter=blob:none \
    --recurse-submodules --shallow-submodules --branch main \
    https://github.com/EBiM-Benchmark/benchmark.git "$ROOT/benchmark"
fi

git -C "$ROOT/benchmark" fetch origin "$BENCHMARK_COMMIT"
git -C "$ROOT/benchmark" checkout --detach "$BENCHMARK_COMMIT"
# Task 3 Isaac uses the Franka description submodule. The Newton physics
# submodule is unrelated to this runtime and is intentionally omitted from the
# critical path; the exact gitlink remains pinned in the checkout.
git -C "$ROOT/benchmark" submodule update --init --depth 1 \
  third_party/franka_description

if [[ ! -x "$ROOT/tools/bin/git-lfs" ]]; then
  curl --noproxy '*' -fL --retry 3 \
    -o /tmp/git-lfs-linux-amd64-v3.7.1.tar.gz \
    https://github.com/git-lfs/git-lfs/releases/download/v3.7.1/git-lfs-linux-amd64-v3.7.1.tar.gz
  mkdir -p /tmp/git-lfs-3.7.1
  tar -xzf /tmp/git-lfs-linux-amd64-v3.7.1.tar.gz \
    -C /tmp/git-lfs-3.7.1 --strip-components=1
  cp /tmp/git-lfs-3.7.1/git-lfs "$ROOT/tools/bin/git-lfs"
  chmod +x "$ROOT/tools/bin/git-lfs"
fi
git -C "$ROOT/benchmark" lfs install --local
if [[ "$(stat -c %s "$ROOT/benchmark/assets/robot_room.usd")" != "55384382" ]]; then
  AIRSIGN_TASK3_ROOT="$ROOT" \
  AIRSIGN_TASK3_BENCHMARK_ROOT="$ROOT/benchmark" \
    bash "$ROOT/workspace/scripts/fetch_room_lfs.sh"
fi

if [[ ! -f "$ROOT/benchmark/task1_isaacsim/assets/Robotiq_2f_85_with_d405_mobile_fr3_duo_v0_2.usd" ]]; then
  # OneDrive is the exception: its share-link redirect works through DSW's
  # managed HTTP proxy, unlike the large GitHub/NVIDIA downloads.
  if [[ -r /etc/profile.d/99-codex-proxy.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/99-codex-proxy.sh
  fi
  if ! bash "$ROOT/benchmark/task1_isaacsim/scripts/download_large_assets.sh"; then
    echo "Official OneDrive bundle unavailable; fetching identical USD from benchmark Robotiq_DEMO branch" >&2
    mkdir -p "$ROOT/benchmark/task1_isaacsim/assets"
    curl --noproxy '*' -fL --retry 3 \
      -o "$ROOT/benchmark/task1_isaacsim/assets/Robotiq_2f_85_with_d405_mobile_fr3_duo_v0_2.usd" \
      https://raw.githubusercontent.com/EBiM-Benchmark/benchmark/Robotiq_DEMO/DEMO/Robotiq_2f_85_with_d405_mobile_fr3_duo_v0_2.usd
  fi
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
fi

if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  "$UV" venv --python 3.11 "$ROOT/.venv"
fi

"$UV" pip install --python "$ROOT/.venv/bin/python" \
  --default-index "$PYPI_INDEX" \
  --index "$NVIDIA_INDEX" \
  "isaacsim[all,extscache]==5.1.0"

"$UV" pip install --python "$ROOT/.venv/bin/python" \
  --default-index "$PYPI_INDEX" \
  -r "$ROOT/workspace/requirements-runtime.txt"

"$ROOT/.venv/bin/python" - <<'PY'
import json
import platform
import sys
from pathlib import Path

root = Path("/mnt/nas/evergreen/ebim-task3")
manifest = {
    "python": sys.version,
    "platform": platform.platform(),
    "benchmark_commit": "e36119cc43e949dc6269bfe5c1e7f613f9f24d0c",
    "isaacsim": "5.1.0",
    "robot_usd_sha256": "aa1a833de48cc543c73957461dab82fe0979320b7c0b6a0a113d24b500075e5c",
}
(root / "artifacts" / "bootstrap.json").write_text(
    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(manifest, indent=2))
PY
