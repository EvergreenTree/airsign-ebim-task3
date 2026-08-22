from __future__ import annotations

from pathlib import Path

def test_dashboard_does_not_install_uvloop_process_wide() -> None:
    source = (Path(__file__).parents[1] / "airsign_task3" / "dashboard.py").read_text()
    assert 'loop="asyncio"' in source


def test_dashboard_reset_requests_a_fresh_simulator_process() -> None:
    root = Path(__file__).parents[1]
    dashboard = (root / "airsign_task3" / "dashboard.py").read_text(
        encoding="utf-8"
    )
    native = (root / "airsign_task3" / "isaac_native.py").read_text(
        encoding="utf-8"
    )
    launcher = (root / "run.sh").read_text(encoding="utf-8")
    assert "store.request_reset()" in dashboard
    assert 'reset_marker = args.run_root / ".reset-requested"' in native
    assert "reset_marker.write_text" in native
    assert "return 75 if store.reset_requested else 0" in native
    assert 'RESET_MARKER="${AIRSIGN_TASK3_RUN_ROOT:-${ROOT}/runs}/.reset-requested"' in launcher
    assert 'if [[ -f "${RESET_MARKER}" ]]' in launcher
    assert 'if [[ ${status} -ne 75 ]]' in launcher
    assert "and not store.stop_requested" in native
    main_loop = native.index("simulation_app.is_running()")
    assert "world.reset()" not in native[main_loop:]


def test_runtime_records_completed_stage_from_live_score_evidence() -> None:
    source = (
        Path(__file__).parents[1] / "airsign_task3" / "isaac_native.py"
    ).read_text(encoding="utf-8")
    assert "highest_completed_stage=highest_completed_stage(breakdown)" in source
    # The reported completed stage must come from scored evidence, never from
    # the runner's internal cursor. The cursor may still be read for other
    # purposes, such as the exploratory --stop-after-stage harness.
    assert "highest_completed_stage=policy_runner.stage_index" not in source
    assert "highest_completed_stage=policy_runner" not in source
    assert 'if message == "complete":' in source
    assert "stopping = True" in source
    assert '"episode_failed"' in source
    failure_block = source.split("elif not success and policy_runner.failed:", 1)[1]
    assert "stopping = True" in failure_block


def test_native_runtime_can_isolate_one_physical_table_object_for_calibration() -> None:
    source = (
        Path(__file__).parents[1] / "airsign_task3" / "isaac_native.py"
    ).read_text(encoding="utf-8")
    assert '"--development-table-object"' in source
    assert '"development_plan_override"' in source
    assert '"--development-feeding"' in source
    assert '"development_feeding_plan_override"' in source
    assert '"--development-recovery"' in source
    assert '"development_recovery_plan_override"' in source
    assert 'for object_name in ("bowl", "cup", "spoon")' in source
    assert "Stage.TABLE_SETUP: table_plan[start:end]" in source


def test_native_runtime_autostarts_unless_wait_is_explicit() -> None:
    source = (
        Path(__file__).parents[1] / "airsign_task3" / "isaac_native.py"
    ).read_text(encoding="utf-8")
    assert 'parser.add_argument("--wait-for-start", action="store_true")' in source
    assert "if not args.preview_only and not args.wait_for_start:" in source
    assert 'store.apply_command("start")' in source


def test_validated_run_mirror_requires_complete_evidence_and_checksums() -> None:
    source = (
        Path(__file__).parents[1] / "scripts" / "mirror_validated_run.sh"
    ).read_text(encoding="utf-8")
    assert "episode.jsonl summary.json evidence.mp4" in source
    assert "/mnt/oss/evergreen/ebim-task3" in source
    assert "sha256sum episode.jsonl summary.json evidence.mp4" in source


def test_launcher_restarts_a_crashed_simulator() -> None:
    """A simulator segfault must not hand back a dead container.

    Isaac's renderer plugin segfaulted twice in 34 launches during development,
    at unrelated points in the plan. The episode dies with no summary, which
    scores as nothing.
    """
    source = (Path(__file__).parents[1] / "run.sh").read_text(encoding="utf-8")
    assert 'MAX_CRASH_RESTARTS="${AIRSIGN_TASK3_MAX_CRASH_RESTARTS:-2}"' in source
    assert "if [[ ${status} -gt 128 ]] && (( crash_restarts < MAX_CRASH_RESTARTS )); then" in source
    # An ordinary non-zero exit still propagates.
    assert "if [[ ${status} -ne 75 ]]; then" in source
