#!/usr/bin/env bash
# Rebuild the Task 3 benchmark root directly from the official
# EBiM-Benchmark/benchmark repository.
#
# The container can be built from this path instead of the vendored copy:
#   docker build --build-arg BENCHMARK_SOURCE=upstream -t airsign-ebim-task3 .
#
# Both paths produce byte-identical trees; vendor/benchmark/PROVENANCE.md
# carries the SHA-256 of every file.
#
# Two assets are not carried by a plain checkout of the pinned commit:
#   - assets/robot_room.usd is a Git LFS pointer on main
#   - the Robotiq robot USD lives on the Robotiq_DEMO branch
# Both are fetched here from those official locations and hash-verified.
set -euo pipefail

BENCHMARK_COMMIT="${BENCHMARK_COMMIT:-e36119cc43e949dc6269bfe5c1e7f613f9f24d0c}"
ROBOT_ASSET_REF="${ROBOT_ASSET_REF:-c2439d961b652b1eda6122bf530c58cb9559b219}"
REMOTE="${BENCHMARK_REMOTE:-https://github.com/EBiM-Benchmark/benchmark.git}"

ROOM_SHA256="bd04da2643bb515ebe311a6a17fd36bf9b32be95ad9e8893a68d44cf2dcc56d3"
ROOM_SIZE=55384382
ROBOT_SHA256="aa1a833de48cc543c73957461dab82fe0979320b7c0b6a0a113d24b500075e5c"

destination="${1:?usage: $0 /path/to/benchmark-root}"

verify() {
  local path="$1" expected="$2"
  local actual
  actual="$(sha256sum "${path}" | cut -d' ' -f1)"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "hash mismatch for ${path}: ${actual} != ${expected}" >&2
    exit 3
  fi
}

echo "cloning ${REMOTE} at ${BENCHMARK_COMMIT}"
git clone --filter=blob:none --no-checkout "${REMOTE}" "${destination}"
git -C "${destination}" checkout --detach "${BENCHMARK_COMMIT}"
printf '%s\n' "${BENCHMARK_COMMIT}" > "${destination}/BENCHMARK_COMMIT"

# The room USD is stored in Git LFS. Resolve the object through the public LFS
# batch API rather than requiring a git-lfs client in the build environment.
echo "resolving the robot_room.usd LFS object"
lfs_response="$(
  curl -sSfL \
    -H 'Accept: application/vnd.git-lfs+json' \
    -H 'Content-Type: application/vnd.git-lfs+json' \
    --data "{\"operation\":\"download\",\"transfers\":[\"basic\"],\"objects\":[{\"oid\":\"${ROOM_SHA256}\",\"size\":${ROOM_SIZE}}]}" \
    "https://github.com/EBiM-Benchmark/benchmark.git/info/lfs/objects/batch"
)"
room_url="$(
  printf '%s' "${lfs_response}" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["objects"][0]["actions"]["download"]["href"])'
)"
curl -sSfL --retry 5 --retry-delay 3 -o "${destination}/assets/robot_room.usd" "${room_url}"
verify "${destination}/assets/robot_room.usd" "${ROOM_SHA256}"

# The Task 3 launchers load the Robotiq robot from the Robotiq_DEMO branch.
echo "fetching the Robotiq robot USD from ${ROBOT_ASSET_REF}"
mkdir -p "${destination}/task1_isaacsim/assets"
curl -sSfL --retry 5 --retry-delay 3 \
  -o "${destination}/task1_isaacsim/assets/Robotiq_2f_85_with_d405_mobile_fr3_duo_v0_2.usd" \
  "https://raw.githubusercontent.com/EBiM-Benchmark/benchmark/${ROBOT_ASSET_REF}/DEMO/Robotiq_2f_85_with_d405_mobile_fr3_duo_v0_2.usd"
verify \
  "${destination}/task1_isaacsim/assets/Robotiq_2f_85_with_d405_mobile_fr3_duo_v0_2.usd" \
  "${ROBOT_SHA256}"

echo "benchmark root ready at ${destination}"
