# Task 3 Phase I submission plan

Deadline: **2026-08-23 19:59 Asia/Shanghai** (2026-08-23 11:59 UTC).
The submission declares the Phase I **simulator ground-truth** track.

## Rule decision: do not train a VLA for Phase I

The controlling documents ask for an autonomous runnable **policy**, not a
particular learned architecture. Rulebook 1.0 defines only task behavior and
scoring. The Phase I notice explicitly permits simulator ground-truth object
positions and states, requires a public repository with a Dockerfile and run
instructions for Option A, and treats model weights or datasets as optional
implementation details rather than submission requirements.

The Phase I deliverable should therefore be the most reliable autonomous
policy we can validate by Sunday: a deterministic ground-truth behavior tree
using physical robot actuation. Training a VLA now would add data, inference,
packaging, and validation failure modes without satisfying any missing rule.
Learned perception or VLA work belongs on the Phase II migration path after the
Phase I policy is frozen.

## Official problem contract

- **Stage 1 — Table setup, 4 points:** place plate, cup, bowl, and spoon at
  their randomly assigned dining targets. Bowl and spoon share the
  head-adjacent target.
- **Stage 2 — Feeding, 4 points:** scoop beans with the spoon, move it in front
  of the head, hold it there for at least three seconds with beans present,
  then return beans to the bowl.
- **Stage 3 — Bean recovery, 4 points:** transfer beans to recycling; score is
  continuous at `4 × recovered mass / original mass`.
- **Stage 4 — Cleanup, 4 points:** place each of the four official objects
  inside the sink for one point each. The tray is not a scored object.
- **Ranking:** highest fully completed stage, then total score, then completion
  time. Three evaluation runs are averaged. A container that fails to build or
  run scores zero.

## Policy objective

Optimize for **highest completed stage first**, not a perfect Stage 1. A plate
failure must not prevent feeding, recovery, or cleanup. Every object or skill
has bounded retries; exhausted recoverable work is deferred while the policy
continues. Emergency stop is reserved for explicit safety failures, lost robot
control, or an unrecoverable carried-object hazard.

The intended fallback path is:

- retain three Stage 1 object points if the plate cannot be seated;
- complete Stage 2 with bowl and spoon;
- preserve every Stage 3 recovery fraction rather than requiring 99% to
  continue;
- complete Stage 4 with independent transfers, including a final plate attempt
  from its actual location.

This can reach Stage 4 and approximately 14–15 points even without a reliable
Stage 1 plate placement, and can still obtain all four Stage 4 points if the
plate is recovered directly into the sink.

## Implementation sequence

1. **Lock the rule contract in tests.** Keep four official objects, random
   assignment targets, all-or-nothing feeding conditions, continuous recovery,
   per-object sink scoring, and score-derived highest-stage ranking covered.
2. **Finish score-seeking control flow.** Verify bounded retries defer only the
   current object or stage, later stages continue, and explicit unrecoverable
   actuator failures still emergency-stop.
3. **Validate Stage 2 first.** Use the table-supported bowl, then exercise spoon
   acquisition, minimum bean count, rescoop, a guarded 3.2-second hold,
   retraction, and bean return. Do not depend on the unreachable left-arm bowl
   brace observed at the head station.
4. **Validate Stage 3.** Use the internal bowl spread and direct transport to
   recycling. Measure the live TCP-to-bowl offset, rotate through reachable
   pour waypoints, and compensate TCP translation so the bowl center and lip
   remain above the recycling opening instead of orbiting around a fixed wrist
   point. Use bounded dither with live recovery-ratio and retained-bean checks,
   then upright the bowl and carry it directly to the sink. If the recovery
   target is missed, preserve the proportional partial score and continue.
5. **Validate Stage 4 independently.** Precheck objects already in the sink;
   transfer bowl, spoon, cup, and plate separately; verify each settled object;
   attempt the plate last.
6. **Continue plate work without blocking the run.** Use the exposed rim/tray
   edge and calibrated side pinch. Treat a failed Stage 1 plate seat as a
   deferred point, not an episode failure.
7. **Run full clean-reset seeds 0, 1, and 2.** Fix the earliest repeatable
   failure that reduces highest completed stage, then rerun from a clean reset.
8. **Rehearse the clean container.** Build from a clean source copy, launch the
   documented command, confirm no hidden NAS/source dependency, and retain the
   image digest.
9. **Freeze and submit.** Preserve code hash, benchmark commit, environment
   lock, three summaries, JSONL logs, videos, checksums, image digest, exact run
   command, declared ground-truth usage, limitations, and the submission issue.

## Acceptance gates

### Policy submission gate

- clean checkout builds and starts the container;
- three distinct clean-reset seeds complete autonomously without operator
  intervention;
- every run scores above zero and reaches later stages after any recoverable
  Stage 1 failure;
- no safety watchdog intervention or unexplained emergency stop;
- reported score is the honest arithmetic mean of all three acceptance runs;
- README commands match the frozen image and evaluator-visible paths.

Release targets, in priority order:

1. all three runs terminate autonomously and score above zero;
2. all three runs reach cleanup and earn at least three Stage 4 object points;
3. at least two runs fully complete Stage 4;
4. maximize the honest three-run mean, with 16/16 retained as a stretch goal.

Do not require 16/16 to submit. A reproducible policy with a strong later-stage
score is better aligned with the ranking rules than an unvalidated perfect-run
claim.

### Technical report fallback

Use Option B only if the policy cannot pass the clean-container gate or cannot
produce repeatable nonzero autonomous runs. Include the exact current code,
all attempted-run results, logs, telemetry, video evidence, known physical
failures, and a concrete real-robot migration plan. Submit only one option for
Task 3.

## Execution calendar

- **Thu Aug 20:** land score-seeking runner, independent cleanup, scorer fixes,
  and unit/contract tests; run one full seed 0 to expose the first Stage 2 issue.
- **Fri Aug 21:** stabilize Stage 2 and Stage 3; run focused development episodes
  and one end-to-end seed 0.
- **Sat Aug 22:** stabilize independent Stage 4 transfers; execute clean seeds
  0, 1, and 2; fix only repeatable score-limiting failures.
- **Sun Aug 23 morning:** clean container build and evaluator-command rehearsal;
  rerun any invalid acceptance seed.
- **Sun Aug 23 by 16:00 CST:** freeze source and artifacts, calculate the mean,
  choose Policy or Report, and draft the issue.
- **Sun Aug 23 by 18:00 CST:** file and read back the issue, leaving nearly two
  hours before the 19:59 CST deadline.

## Evidence retained per run

- seed, benchmark commit `e36119cc43e949dc6269bfe5c1e7f613f9f24d0c`,
  source hash, runtime lock, image digest, and command line;
- `episode.jsonl`, `summary.json`, evidence video, simulator log, and SHA-256
  manifest;
- per-stage score, highest completed stage from scorer evidence, total score,
  completion time, deferred scopes, and terminal reason;
- recovery ratio, feeding hold evidence, head-contact peak, and watchdog count;
- ground-truth declaration and no-object-mutation test result.
