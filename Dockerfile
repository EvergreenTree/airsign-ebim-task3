FROM nvcr.io/nvidia/isaac-sim:5.1.0

USER root

ARG BENCHMARK_COMMIT=e36119cc43e949dc6269bfe5c1e7f613f9f24d0c
ARG ROBOT_ASSET_REF=c2439d961b652b1eda6122bf530c58cb9559b219

# Where the official EBiM benchmark tree comes from.
#   upstream  (default) - cloned from EBiM-Benchmark/benchmark at build time by
#                         scripts/fetch_benchmark.sh, so the image is built
#                         against the official repository.
#   vendored            - the copy checked into vendor/benchmark/. Needs no
#                         network, for an offline or air-gapped build.
# Both produce byte-identical trees; the SHA-256 checks below run either way and
# vendor/benchmark/PROVENANCE.md lists the hash of every file.
ARG BENCHMARK_SOURCE=upstream

COPY vendor/benchmark/ /opt/ebim-benchmark/
COPY scripts/fetch_benchmark.sh /usr/local/bin/fetch_benchmark.sh

RUN if [ "${BENCHMARK_SOURCE}" = "upstream" ]; then \
        rm -rf /opt/ebim-benchmark \
        && if ! command -v git >/dev/null || ! command -v curl >/dev/null; then \
               apt-get update \
               && apt-get install -y --no-install-recommends git curl ca-certificates \
               && rm -rf /var/lib/apt/lists/*; \
           fi \
        && BENCHMARK_COMMIT="${BENCHMARK_COMMIT}" ROBOT_ASSET_REF="${ROBOT_ASSET_REF}" \
           bash /usr/local/bin/fetch_benchmark.sh /opt/ebim-benchmark; \
    fi \
    && test "$(cat /opt/ebim-benchmark/BENCHMARK_COMMIT)" = "${BENCHMARK_COMMIT}" \
    && echo 'bd04da2643bb515ebe311a6a17fd36bf9b32be95ad9e8893a68d44cf2dcc56d3  /opt/ebim-benchmark/assets/robot_room.usd' \
       | sha256sum -c - \
    && echo 'aa1a833de48cc543c73957461dab82fe0979320b7c0b6a0a113d24b500075e5c  /opt/ebim-benchmark/task1_isaacsim/assets/Robotiq_2f_85_with_d405_mobile_fr3_duo_v0_2.usd' \
       | sha256sum -c -

WORKDIR /workspace
COPY airsign_task3/ /workspace/airsign_task3/
COPY config/ /workspace/config/
COPY run.sh /workspace/run.sh
RUN chmod +x /workspace/run.sh \
    && /isaac-sim/python.sh -c 'import cv2, fastapi, httptools, numpy, PIL, pydantic, uvicorn; import airsign_task3.main' \
    && /isaac-sim/python.sh -m compileall -q /workspace/airsign_task3

ENV AIRSIGN_TASK3_ROOT=/workspace/runtime \
    AIRSIGN_TASK3_WORKSPACE=/workspace \
    AIRSIGN_TASK3_PYTHON=/isaac-sim/python.sh \
    AIRSIGN_TASK3_BENCHMARK_ROOT=/opt/ebim-benchmark \
    AIRSIGN_TASK3_RUN_ROOT=/workspace/runs \
    OMNI_KIT_ACCEPT_EULA=YES \
    PRIVACY_CONSENT=Y

EXPOSE 18091
ENTRYPOINT ["/workspace/run.sh"]
CMD ["--seed", "0", "--headless", "--ui-port", "18091"]
