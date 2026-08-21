import pytest

from airsign_task3.scoring import ScoreEvidence, compute_score, highest_completed_stage


def test_official_score_ignores_tray_and_is_16_points() -> None:
    evidence = ScoreEvidence(
        table_objects_correct={"plate": True, "cup": True, "bowl": True, "spoon": True, "tray": False},
        feeding_beans_present=True,
        feeding_hold_seconds=3.2,
        feeding_beans_returned=True,
        original_bean_mass=100.0,
        recovered_bean_mass=100.0,
        sink_objects={"plate": True, "cup": True, "bowl": True, "spoon": True, "tray": False},
    )
    assert compute_score(evidence).total == 16.0


@pytest.mark.parametrize(("ratio", "points"), [(0.0, 0.0), (0.25, 1.0), (0.725, 2.9), (1.0, 4.0), (1.2, 4.0)])
def test_stage3_is_continuous_and_clamped(ratio: float, points: float) -> None:
    evidence = ScoreEvidence(original_bean_mass=100.0, recovered_bean_mass=100.0 * ratio)
    assert compute_score(evidence).stage3 == pytest.approx(points)


def test_feeding_requires_all_conditions() -> None:
    evidence = ScoreEvidence(feeding_beans_present=True, feeding_hold_seconds=2.99, feeding_beans_returned=True)
    assert compute_score(evidence).stage2 == 0.0


def test_highest_completed_stage_uses_score_evidence_not_plan_progress() -> None:
    evidence = ScoreEvidence(
        table_objects_correct={"plate": False, "cup": True, "bowl": True, "spoon": True},
        feeding_beans_present=True,
        feeding_hold_seconds=3.2,
        feeding_beans_returned=True,
        original_bean_mass=100.0,
        recovered_bean_mass=95.0,
        sink_objects={"plate": False, "cup": True, "bowl": True, "spoon": True},
    )
    assert highest_completed_stage(compute_score(evidence)) == 2


def test_highest_completed_stage_can_advance_despite_an_earlier_partial_stage() -> None:
    evidence = ScoreEvidence(
        table_objects_correct={"plate": False, "cup": True, "bowl": True, "spoon": True},
        sink_objects={"plate": True, "cup": True, "bowl": True, "spoon": True},
    )
    assert highest_completed_stage(compute_score(evidence)) == 4
