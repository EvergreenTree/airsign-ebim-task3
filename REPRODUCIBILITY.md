# Reproducibility manifest

- Benchmark: `EBiM-Benchmark/benchmark`
- Benchmark commit: `e36119cc43e949dc6269bfe5c1e7f613f9f24d0c`
- Isaac Sim: `5.1.0`
- Native Python: CPython `3.11`
- Robot: competition Mobile FR3 Duo with Robotiq 2F-85 grippers
- Renderer: RTX Ray-Traced Lighting, headless
- Default physics/render rates: 120 Hz / 20 Hz
- Default base/TCP speeds: 0.15 m/s / 0.15 m/s
- Head-zone TCP speed: 0.08 m/s
- Submission image base: `nvcr.io/nvidia/isaac-sim:5.1.0`
- Vendored benchmark revision marker: `vendor/benchmark/BENCHMARK_COMMIT`

Benchmark asset provenance. Two assets the policy needs are not carried by a
plain checkout of the pinned commit, so record where each actually comes from:

| Asset | Official source | SHA-256 |
|---|---|---|
| `assets/robot_room.usd` | Git LFS object of `main@e36119cc`; the tracked blob is a 133-byte pointer | `bd04da2643bb515ebe311a6a17fd36bf9b32be95ad9e8893a68d44cf2dcc56d3` |
| `task1_isaacsim/assets/Robotiq_2f_85_with_d405_mobile_fr3_duo_v0_2.usd` | branch `Robotiq_DEMO` (`c2439d96`), path `DEMO/…`; not tracked on `main` | `aa1a833de48cc543c73957461dab82fe0979320b7c0b6a0a113d24b500075e5c` |

Everything else under `vendor/benchmark/` — scene loader, Lula configuration,
head model and textures, `bowl2.usd`, LICENSE and NOTICE — is a plain checkout
of `e36119cc`. `vendor/benchmark/PROVENANCE.md` lists the SHA-256 of every
vendored file, and `scripts/fetch_benchmark.sh` rebuilds the identical tree
from the official repository.

Runtime Python package versions are pinned in `requirements-runtime.txt`.
The Docker build uses the matching packages already present in the pinned
Isaac Sim base image. The default `BENCHMARK_SOURCE=vendored` build needs no
network access after that base is pulled; `BENCHMARK_SOURCE=upstream` instead
clones `EBiM-Benchmark/benchmark` at the pinned commit and fetches the two
assets above from their official locations, verifying the same hashes.
`scripts/bootstrap_remote.sh` installs native dependencies and external
competition assets beneath `/mnt/nas/evergreen/ebim-task3`, leaving system
Python untouched. Run-specific seed, head placement, resolved prim paths,
scores, timings, safety events, and failure reason are saved in machine-readable
artifacts.

The minimum policy-submission gate is three consecutive clean-reset runs with
different seeds that terminate autonomously, score above zero, and have no
watchdog intervention or unexplained emergency stop. The target release gate is
Stage 4 completion in at least two runs and at least three Stage 4 object points
in every run. A perfect 16/16 is a stretch target, not a prerequisite for an
honest runnable submission. Any reported mean includes all three frozen
acceptance runs rather than only successful runs.
