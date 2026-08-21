from pathlib import Path


BENCHMARK_COMMIT = "e36119cc43e949dc6269bfe5c1e7f613f9f24d0c"


def benchmark_revision(root: Path) -> str:
    marker = root / "BENCHMARK_COMMIT"
    if marker.is_file():
        return marker.read_text(encoding="utf-8").strip()
    head = root / ".git" / "HEAD"
    if not head.is_file():
        raise FileNotFoundError(f"benchmark revision marker not found: {root}")
    revision = head.read_text(encoding="utf-8").strip()
    if revision.startswith("ref: "):
        reference = root / ".git" / revision.removeprefix("ref: ")
        if not reference.is_file():
            raise FileNotFoundError(f"benchmark Git reference not found: {reference}")
        revision = reference.read_text(encoding="utf-8").strip()
    return revision


def validate_benchmark_revision(root: Path) -> None:
    revision = benchmark_revision(root)
    if revision != BENCHMARK_COMMIT:
        raise RuntimeError(
            f"benchmark revision mismatch: {revision} != {BENCHMARK_COMMIT}"
        )
