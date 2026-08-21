# AirSign EBiM Task 3 policy

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

The minimal pinned benchmark runtime and the two large USD files required by
the policy are vendored under `vendor/benchmark/`. The image verifies the
benchmark revision marker and both USD SHA-256 hashes during the build. No
model, dataset, package install, Git checkout, Git LFS transfer, or
evaluator-time download is required after the Isaac base image is available.

## Run

On a GPU host with the native NAS installation:

```bash
./run.sh --seed 0 --headless --ui-port 18091
```

After scene calibration, this command starts the autonomous episode without an
operator action. Add `--wait-for-start` only for an interactive dashboard
rehearsal that should remain idle until `POST /api/control/start`.

Forward the SSH-only dashboard from the simulator host:

```bash
ssh -N -L 127.0.0.1:18091:127.0.0.1:18091 dsw-evergreen
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

The vendored benchmark slice contains the official scene loader, Task 3 Lula
configuration, room support asset, notices, and the room/robot USDs from the
pinned public benchmark revision. The image uses the matching Python packages
already shipped in the pinned Isaac Sim base image and performs an import and
bytecode-compilation smoke check during the build.

The service endpoints are:

- `GET /api/state`
- `POST /api/control/{start,pause,resume,reset}`
- `POST /api/feedback`
- `GET /stream/{overview,head,left_wrist,right_wrist}`
- `GET /ws/telemetry`

## Scoring and evidence

The live official estimate scores only plate, cup, bowl, and spoon in Stages 1
and 4. Stage 2 is four points only after a bean-bearing three-second feeding
hold and bean return. Stage 3 is continuous `4 × recovered/original mass`.
The upstream development scorer remains diagnostic and is never presented as
the competition estimate because it includes the tray and discretizes recovery.

Each run writes `episode.jsonl`, `summary.json`, camera/video evidence, and
safety telemetry under the configured run directory. Validated artifacts are
mirrored to OSS with checksums.

## Tests

```bash
python -m pytest -q
```

The pure tests cover assignment resolution, official scoring, continuous bean
recovery, state transitions, waypoint clearance, recovery retries/timeouts,
and a policy-source guard against simulator-side task-object mutation.

See [GROUND_TRUTH.md](GROUND_TRUTH.md) and
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) for track and environment details.
