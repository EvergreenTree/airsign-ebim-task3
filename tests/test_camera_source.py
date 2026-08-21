from pathlib import Path


def test_transient_annotator_failure_does_not_abort_policy_runtime() -> None:
    source = (
        Path(__file__).parents[1] / "airsign_task3" / "isaac_native.py"
    ).read_text(encoding="utf-8")
    assert "def _read_annotator_frame" in source
    assert "except Exception:" in source
    assert "return _frame_to_bgr(data)" in source


def test_native_runtime_provides_a_dedicated_grasp_detail_camera() -> None:
    source = (
        Path(__file__).parents[1] / "airsign_task3" / "isaac_native.py"
    ).read_text(encoding="utf-8")
    assert 'camera_path = "/World/AirSignCameras/Supply"' in source
    assert "SetLookAt" in source
    assert 'store.set_frame("left_wrist", detail_frame)' in source
    assert "camera_capture_degraded = frame is None" in source


def test_native_runtime_provides_a_dedicated_head_safety_camera() -> None:
    source = (
        Path(__file__).parents[1] / "airsign_task3" / "isaac_native.py"
    ).read_text(encoding="utf-8")
    assert 'camera_path = "/World/AirSignCameras/HeadSafety"' in source
    assert "def _create_head_capture" in source
    assert "0.75 * inward" in source
    assert "1.75" in source
    assert 'store.set_frame("head", head_frame)' in source
    assert "head_annotator.detach(head_render_product)" in source


def test_dashboard_labels_the_dedicated_grasp_view() -> None:
    source = (
        Path(__file__).parents[1] / "airsign_task3" / "dashboard.py"
    ).read_text(encoding="utf-8")
    assert 'data-stream="/stream/left_wrist"' in source
    assert "GRASP DETAIL" in source


def test_runtime_records_score_and_safety_overlay_video() -> None:
    source = (
        Path(__file__).parents[1] / "airsign_task3" / "isaac_native.py"
    ).read_text(encoding="utf-8")
    assert "import cv2" in source
    assert 'video_path = run_dir / "evidence.mp4"' in source
    assert "cv2.VideoWriter(" in source
    assert "def record_overview" in source
    assert 'f"score {state[' in source
    assert 'f"head force {safety[' in source
    assert "video_writer.release()" in source


def test_native_webrtc_uses_official_streaming_experience() -> None:
    source = (
        Path(__file__).parents[1] / "airsign_task3" / "isaac_native.py"
    ).read_text(encoding="utf-8")
    assert '"--native-webrtc"' in source
    assert '"isaacsim.exp.full.streaming.kit"' in source
    assert "next(iter(isaacsim.__path__))" in source
    assert '"--/app/livestream/publicEndpointAddress="' in source
    assert '"--/app/livestream/port=' in source
    assert '"--/app/livestream/fixedHostPort=' in source
    assert '"--/app/livestream/publicEndpointPort=' in source

    launcher = (Path(__file__).parents[1] / "run.sh").read_text(encoding="utf-8")
    assert "runtime-libs/xrandr" in launcher
    assert "LD_LIBRARY_PATH" in launcher
