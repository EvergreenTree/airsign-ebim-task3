# AirSign EBiM Task 3 policy

**Team AirSign** — Changqing Fu, Shangyu Yao, Sichen Su, Ziqi Ma, Changda Tian.


This repository contains AirSign's physically actuated policy for EBiM Task 3
(Assisted Living and Feeding), targeting Isaac Sim 5.1.0. The policy may read
simulator ground-truth poses and states, as allowed for Phase I. It never moves
task objects by editing their transforms or kinematic state after an episode
starts: the mobile base, FR3 arms, Robotiq grippers, and contact physics are the
only object-motion mechanisms.

No learned checkpoint is required. The submitted policy is a deterministic
ground-truth behavior tree with bounded physical skills, verification, retries,
and safety stops.

The runtime is pinned to EBiM benchmark commit
`e36119cc43e949dc6269bfe5c1e7f613f9f24d0c`.

The policy runs the official benchmark's own scene loader, Lula configuration
and assets. By default the image builds them from the official repository:
`scripts/fetch_benchmark.sh` clones `EBiM-Benchmark/benchmark` at the pinned
commit, resolves `assets/robot_room.usd` through the public Git LFS API, and
fetches the Robotiq robot USD from the `Robotiq_DEMO` branch. Neither of those
two assets is carried by a plain checkout of the pinned commit — the room USD
is a 133-byte LFS pointer there, and the robot USD is not tracked on `main` at
all.

The same tree is also checked into `vendor/benchmark/` as an unmodified copy,
for an offline or air-gapped build:

```bash
docker build --build-arg BENCHMARK_SOURCE=vendored -t airsign-ebim-task3 .
```

Both paths verify the same SHA-256 hashes during the build, and every
benchmark file the policy loads is byte-for-byte identical between them; the
upstream fetch additionally materialises other files from the same paths and
keeps its `.git` directory. The fetch takes about 90 seconds and produces
roughly 125 MB. `vendor/benchmark/PROVENANCE.md` records the official source
and hash of every vendored file.

## Run

The container is the supported entry point; see *Container usage* below. The
native path additionally requires `scripts/bootstrap_remote.sh`, which builds a
managed virtualenv and stages the benchmark assets under
`$AIRSIGN_TASK3_ROOT` (default `/mnt/nas/evergreen/ebim-task3`). With that in
place:

```bash
./run.sh --seed 0 --headless --ui-port 18091
```

After scene calibration, this command starts the autonomous episode without an
operator action. Add `--wait-for-start` only for an interactive dashboard
rehearsal that should remain idle until `POST /api/control/start`.

The dashboard binds to loopback only. To reach it from a workstation, forward
the port from whichever host runs the simulator:

```bash
ssh -N -L 127.0.0.1:18091:127.0.0.1:18091 <simulator-host>
```

Then open `http://127.0.0.1:18091`. The dashboard exposes synchronized camera,
score, safety, and lifecycle telemetry, plus Start/Pause/Resume/Reset and
timestamped realism feedback.

Container usage:

```bash
docker build -t airsign-ebim-task3 .
docker run --rm --gpus all --network host \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  airsign-ebim-task3 --seed 0 --headless --ui-port 18091
```

To keep the episode artifacts after the container exits, mount a host
directory over the run root:

```bash
docker run --rm --gpus all --network host \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -v "$PWD/runs:/workspace/runs" \
  airsign-ebim-task3 --seed 0 --headless --ui-port 18091
```

The container writes `episode.jsonl`, `summary.json`, and `evidence.mp4` into
a per-episode directory beneath that path, and exits on its own once the
episode terminates.

The dashboard's web stack (`fastapi`, `uvicorn`, `httptools`, `pydantic`) and
`opencv-python-headless` are not part of `isaacsim[all]`. The build prefers
whatever the base image already provides, so no version is forced onto Isaac's
own dependencies, and installs the pinned set from `requirements-runtime.txt`
only if an import is missing. It then re-checks every import and
byte-compiles the policy, so a build that could not satisfy them fails at
build time rather than at episode start.

The service endpoints are:

- `GET /api/state`
- `POST /api/control/{start,pause,resume,reset}`
- `POST /api/feedback`
- `GET /stream/{overview,head,left_wrist}`
- `GET /ws/telemetry`

Three cameras are served. `overview` is the room camera also burned into
`evidence.mp4`; `head` is a fixed head-safety view; `left_wrist` is a
supply-table grasp-detail view, labelled GRASP DETAIL in the dashboard, rather
than a camera mounted on the left wrist.

## Scoring and evidence

The live official estimate scores only plate, cup, bowl, and spoon in Stages 1
and 4. Stage 2 is four points only after a bean-bearing three-second feeding
hold and bean return. Stage 3 is continuous `4 × recovered/original mass`.
The upstream development scorer remains diagnostic and is never presented as
the competition estimate because it includes the tray and discretizes recovery.

Each run writes `episode.jsonl`, `summary.json`, camera/video evidence, and
safety telemetry under the configured run directory. `episode.jsonl` is the
authoritative record: it carries every primitive boundary, gripper contact
report, navigation decision and score update. `summary.json` holds the final
telemetry snapshot, and `scripts/summarize_runs.py` prints the stage table for
one or more run directories.

## Status and known limitations

Measured behaviour, not intent. Every claim here comes from a run whose
`episode.jsonl` is reproducible with the command above.

**What works.** Stage 1 places the bowl and the cup at their assigned dining
seats. Episodes terminate autonomously with no watchdog intervention and no
emergency stop, and every stage failure is contained: a failed object is
deferred and the policy continues to the next scope rather than ending the run.

**The spoon is not placed, and this costs Stage 2 entirely.** The grasp itself
succeeds -- both inner fingers are confirmed on the handle under 2.8-6 N.m of
squeeze -- but the lift then reports `lifted: false` with the spoon having
risen under 2 cm while the gripper driver climbs from 0.75 to 0.79. The thin
handle is being squeezed out from between the Robotiq pads as the jaws
continue to close. Because Stage 2 needs the spoon staged at the head-adjacent
seat, `check spoon staged for feeding` fails and the feeding stage is skipped.
This costs Stage 1's third point, all four Stage 2 points, and the Stage 4
spoon point. It is a genuine manipulation limit rather than a threshold that
can be retuned, and it is the largest single gap in this submission.

**Stage 3's bowl pickup is at the arm's reach limit.** The base completes its
0.28 m final approach, and the pregrasp then converges with z on target and xy
roughly 0.15 m short, stalling there through every retry. Where it fails, the
recovery ratio stays at zero and the stage is deferred.

**Stage 4 is geometry-dependent.** Cleanup handles each object independently,
so a failure costs one point rather than the stage. Objects at a reachable
seat are grasped, lifted and carried; some seat geometries leave the pregrasp
outside the arm's envelope in the same way Stage 3 does.

**The container has not been built.** Neither machine available for this work
has a container runtime, so the image itself is unbuilt and untested. What was
verified instead: the vendored benchmark slice loads the scene, robot, head and
beans in a real Isaac session; `scripts/fetch_benchmark.sh` was run end to end
and returns every benchmark file byte-for-byte; and `import airsign_task3.main`
succeeds outside a running Kit app. The runtime dependency step is written to
prefer the base image's packages and install the pinned set only if an import
is missing, but that fallback has not been exercised.

## Tests

```bash
python -m pytest -q
```

The pure tests cover assignment resolution, official scoring, continuous bean
recovery, state transitions, waypoint clearance, recovery retries/timeouts,
and a policy-source guard against simulator-side task-object mutation.

See [GROUND_TRUTH.md](GROUND_TRUTH.md) and
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) for track and environment details.
