from pathlib import Path


BANNED_POLICY_TOKENS = (
    "set_world_pose",
    "set_local_pose",
    "set_translate",
    "set_rigid_body_kinematic",
    "set_kinematic_enabled",
    "teleport",
)


def test_policy_has_no_object_transform_or_kinematic_mutation() -> None:
    package = Path(__file__).parents[1] / "airsign_task3"
    policy_sources = [package / "policy.py", package / "planning.py", package / "ground_truth.py"]
    for source in policy_sources:
        text = source.read_text(encoding="utf-8").lower()
        for token in BANNED_POLICY_TOKENS:
            assert token not in text, f"{source.name} contains prohibited policy mutation token {token!r}"

