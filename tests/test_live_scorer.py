from airsign_task3.live_scorer import AssignmentTarget, OfficialLiveScorer
from airsign_task3.types import Bounds, OFFICIAL_OBJECTS, Pose


class FakeReader:
    def __init__(self) -> None:
        self.positions = {name: (float(index), 0.0, 1.0) for index, name in enumerate(OFFICIAL_OBJECTS)}
        self.positions.update({"head": (0.0, 1.0, 1.2), "recycling": (5.0, 0.0, 0.0), "sink": (7.0, 0.0, 0.0)})
        self.beans = [(5.0, 0.0, 0.2)] * 75 + [(0.0, 0.0, 1.0)] * 25

    def pose(self, name: str) -> Pose:
        return Pose(self.positions[name], (1.0, 0.0, 0.0, 0.0))

    def bounds(self, name: str) -> Bounds:
        if name == "recycling":
            return Bounds((4.5, -0.5, 0.0), (5.5, 0.5, 1.0))
        if name == "sink":
            return Bounds((6.5, -0.5, 0.0), (7.5, 0.5, 1.0))
        x, y, z = self.positions[name]
        return Bounds((x - 0.05, y - 0.05, z - 0.05), (x + 0.05, y + 0.05, z + 0.05))

    def bean_positions(self):
        return tuple(self.beans)


def test_live_scorer_uses_continuous_recovery_and_four_objects() -> None:
    reader = FakeReader()
    assignments = {name: AssignmentTarget(reader.positions[name][:2], 0.1) for name in OFFICIAL_OBJECTS}
    scorer = OfficialLiveScorer(reader, assignments)
    result = scorer.update()
    assert result.stage1 == 4.0
    assert result.stage3 == 3.0
    assert result.stage4 == 0.0


def test_recovery_excludes_beans_hovering_above_the_container() -> None:
    reader = FakeReader()
    reader.beans = [(5.0, 0.0, 1.2)] * 100
    assignments = {
        name: AssignmentTarget(reader.positions[name][:2], 0.1)
        for name in OFFICIAL_OBJECTS
    }
    assert OfficialLiveScorer(reader, assignments).update().stage3 == 0.0


def test_stage1_latches_each_successful_assignment_independently() -> None:
    reader = FakeReader()
    assignments = {name: AssignmentTarget(reader.positions[name][:2], 0.1) for name in OFFICIAL_OBJECTS}
    scorer = OfficialLiveScorer(reader, assignments)
    reader.positions["spoon"] = (20.0, 20.0, 1.0)
    assert scorer.update().stage1 == 3.0
    reader.positions["bowl"] = (20.0, 20.0, 1.0)
    reader.positions["spoon"] = assignments["spoon"].xy + (1.0,)
    assert scorer.update().stage1 == 4.0


def test_feeding_requires_the_held_beans_to_return_to_the_bowl() -> None:
    reader = FakeReader()
    assignments = {
        name: AssignmentTarget(reader.positions[name][:2], 0.1)
        for name in OFFICIAL_OBJECTS
    }
    scorer = OfficialLiveScorer(reader, assignments)
    scorer.record_feeding_hold(
        bean_indices={2, 7},
        dt=3.2,
        in_head_zone=True,
    )
    scorer.record_feeding_return({2})
    assert scorer.update().stage2 == 0.0

    scorer.record_feeding_return({2, 7})
    assert scorer.update().stage2 == 4.0


def test_feeding_hold_must_keep_the_same_verified_payload_continuously() -> None:
    reader = FakeReader()
    assignments = {
        name: AssignmentTarget(reader.positions[name][:2], 0.1)
        for name in OFFICIAL_OBJECTS
    }
    scorer = OfficialLiveScorer(reader, assignments)
    scorer.record_feeding_hold(bean_indices={2}, dt=1.6, in_head_zone=True)
    scorer.record_feeding_hold(bean_indices=set(), dt=0.1, in_head_zone=True)
    scorer.record_feeding_hold(bean_indices={2}, dt=1.6, in_head_zone=True)
    scorer.record_feeding_return({2})
    assert scorer.update().stage2 == 0.0

    scorer.record_feeding_hold(bean_indices={2}, dt=3.1, in_head_zone=True)
    scorer.record_feeding_return({2})
    assert scorer.update().stage2 == 4.0
