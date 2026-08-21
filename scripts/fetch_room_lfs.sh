#!/usr/bin/env bash
set -euo pipefail

ROOT="${AIRSIGN_TASK3_ROOT:-/mnt/nas/evergreen/ebim-task3}"
REPO="${AIRSIGN_TASK3_BENCHMARK_ROOT:-$ROOT/benchmark}"
PYTHON="${AIRSIGN_TASK3_PYTHON:-python3}"
OID="bd04da2643bb515ebe311a6a17fd36bf9b32be95ad9e8893a68d44cf2dcc56d3"
OBJECT_SIZE=55384382
CHUNK_SIZE=1048576
LAST_PART=$(((OBJECT_SIZE - 1) / CHUNK_SIZE))
PARALLELISM="${AIRSIGN_TASK3_DOWNLOAD_PARALLELISM:-4}"
TRANSFER_TIMEOUT="${AIRSIGN_TASK3_TRANSFER_TIMEOUT:-600}"

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

response="$({
  curl --noproxy '*' -sSfL \
    -H 'Accept: application/vnd.git-lfs+json' \
    -H 'Content-Type: application/vnd.git-lfs+json' \
    --data "{\"operation\":\"download\",\"transfers\":[\"basic\"],\"objects\":[{\"oid\":\"$OID\",\"size\":$OBJECT_SIZE}]}" \
    https://github.com/EBiM-Benchmark/benchmark.git/info/lfs/objects/batch
})"
download_url="$(printf '%s' "$response" | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["objects"][0]["actions"]["download"]["href"])')"

export download_url OBJECT_SIZE CHUNK_SIZE
export TRANSFER_TIMEOUT
seq 0 "$LAST_PART" | xargs -P "$PARALLELISM" -I PART bash -c '
  part_index=PART
  range_start=$((part_index * CHUNK_SIZE))
  range_end=$((range_start + CHUNK_SIZE - 1))
  if [ "$range_end" -ge "$OBJECT_SIZE" ]; then range_end=$((OBJECT_SIZE - 1)); fi
  expected=$((range_end - range_start + 1))
  output="/tmp/airsign-room.part.${part_index}"
  if [ -f "$output" ] && [ "$(stat -c %s "$output")" -eq "$expected" ]; then exit 0; fi
  for attempt in $(seq 1 8); do
    curl --noproxy "*" -sSfL --connect-timeout 30 --max-time "$TRANSFER_TIMEOUT" \
      --range "${range_start}-${range_end}" -o "$output" "$download_url" || true
    if [ -f "$output" ] && [ "$(stat -c %s "$output")" -eq "$expected" ]; then exit 0; fi
  done
  echo "failed LFS range $part_index" >&2
  exit 1
'

for part_index in $(seq 0 "$LAST_PART"); do
  range_start=$((part_index * CHUNK_SIZE))
  expected=$CHUNK_SIZE
  if [ "$part_index" -eq "$LAST_PART" ]; then expected=$((OBJECT_SIZE - range_start)); fi
  actual=$(stat -c %s "/tmp/airsign-room.part.${part_index}")
  if [ "$actual" -ne "$expected" ]; then
    echo "range $part_index size mismatch: $actual != $expected" >&2
    exit 1
  fi
done

cp /tmp/airsign-room.part.0 /tmp/robot_room.usd
for part_index in $(seq 1 "$LAST_PART"); do
  dd if="/tmp/airsign-room.part.${part_index}" of=/tmp/robot_room.usd \
    bs=$CHUNK_SIZE seek=$part_index conv=notrunc status=none
done

actual_oid=$(sha256sum /tmp/robot_room.usd | awk '{print $1}')
if [[ "$actual_oid" != "$OID" ]]; then
  echo "room USD checksum mismatch: $actual_oid != $OID" >&2
  exit 1
fi

object_dir="$REPO/.git/lfs/objects/${OID:0:2}/${OID:2:2}"
mkdir -p "$object_dir"
cp /tmp/robot_room.usd "$object_dir/$OID"
cp /tmp/robot_room.usd "$REPO/assets/robot_room.usd"

printf 'room USD verified: sha256=%s size=%s\n' "$actual_oid" "$(stat -c %s "$REPO/assets/robot_room.usd")"
