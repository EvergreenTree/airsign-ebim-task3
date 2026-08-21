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
