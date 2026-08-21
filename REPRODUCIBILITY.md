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
- Room USD SHA-256: `bd04da2643bb515ebe311a6a17fd36bf9b32be95ad9e8893a68d44cf2dcc56d3`
- Robot USD SHA-256: `aa1a833de48cc543c73957461dab82fe0979320b7c0b6a0a113d24b500075e5c`

Runtime Python package versions are pinned in `requirements-runtime.txt`.
The Docker build uses the matching packages already present in the pinned
Isaac Sim base image and requires no network access after that base is pulled.
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
