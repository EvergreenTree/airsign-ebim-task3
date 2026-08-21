from pathlib import Path

from airsign_task3.isaac_state import PRIM_ALIASES


def test_pinned_room_object_prim_names() -> None:
    assert PRIM_ALIASES["plate"] == "plate2_01"
    assert PRIM_ALIASES["bowl"] == "bowl2"
    assert PRIM_ALIASES["spoon"] == "spoon2_01"
    assert PRIM_ALIASES["cup"] == "cup"


def test_head_geometry_inventory_reports_collision_authoring() -> None:
    source = (
        Path(__file__).parents[1] / "airsign_task3" / "isaac_state.py"
    ).read_text(encoding="utf-8")
    assert '"collision_api"' in source
    assert '"rigid_body_api"' in source
