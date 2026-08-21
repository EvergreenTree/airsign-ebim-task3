from pathlib import Path

import pytest

from airsign_task3.benchmark import (
    BENCHMARK_COMMIT,
    benchmark_revision,
    validate_benchmark_revision,
)


def test_reads_vendored_benchmark_revision(tmp_path: Path) -> None:
    (tmp_path / "BENCHMARK_COMMIT").write_text(
        f"{BENCHMARK_COMMIT}\n", encoding="utf-8"
    )

    assert benchmark_revision(tmp_path) == BENCHMARK_COMMIT
    validate_benchmark_revision(tmp_path)


def test_reads_detached_git_revision(tmp_path: Path) -> None:
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text(f"{BENCHMARK_COMMIT}\n", encoding="utf-8")

    assert benchmark_revision(tmp_path) == BENCHMARK_COMMIT


def test_rejects_wrong_benchmark_revision(tmp_path: Path) -> None:
    (tmp_path / "BENCHMARK_COMMIT").write_text("wrong\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="benchmark revision mismatch"):
        validate_benchmark_revision(tmp_path)


def test_policy_source_hash_covers_the_package_and_is_stable() -> None:
    """A run artifact must identify the policy that produced it.

    Artifacts record the benchmark revision, which does not move when the
    policy changes, so without this a run cannot be attributed to a version.
    """
    from airsign_task3.benchmark import policy_source_hash

    first = policy_source_hash()
    assert first == policy_source_hash()
    assert len(first) == 64

    root = Path(__file__).parents[1] / "airsign_task3"
    scratch = root / "_hash_probe_tmp.py"
    scratch.write_text("# temporary\n", encoding="utf-8")
    try:
        assert policy_source_hash() != first
    finally:
        scratch.unlink()
    assert policy_source_hash() == first


def test_runtime_records_the_policy_source_hash() -> None:
    source = (
        Path(__file__).parents[1] / "airsign_task3" / "isaac_native.py"
    ).read_text(encoding="utf-8")
    assert "policy_source_sha256=policy_source_hash()" in source
