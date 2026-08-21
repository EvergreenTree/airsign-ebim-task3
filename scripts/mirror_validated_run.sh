#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /mnt/nas/evergreen/ebim-task3/runs/seed-N-TIMESTAMP" >&2
  exit 2
fi

run_dir="$(realpath "$1")"
case "${run_dir}" in
  /mnt/nas/evergreen/ebim-task3/runs/seed-*) ;;
  *)
    echo "refusing to mirror unexpected run directory: ${run_dir}" >&2
    exit 2
    ;;
esac

for required in episode.jsonl summary.json evidence.mp4; do
  if [[ ! -s "${run_dir}/${required}" ]]; then
    echo "validated run is missing ${required}: ${run_dir}" >&2
    exit 3
  fi
done

oss_root="${AIRSIGN_TASK3_OSS_ROOT:-/mnt/oss/evergreen/ebim-task3}"
destination="${oss_root}/runs/$(basename "${run_dir}")"
mkdir -p "${destination}"
cp -p "${run_dir}/episode.jsonl" "${destination}/episode.jsonl"
cp -p "${run_dir}/summary.json" "${destination}/summary.json"
cp -p "${run_dir}/evidence.mp4" "${destination}/evidence.mp4"
(
  cd "${destination}"
  sha256sum episode.jsonl summary.json evidence.mp4 > SHA256SUMS
)
echo "${destination}"
