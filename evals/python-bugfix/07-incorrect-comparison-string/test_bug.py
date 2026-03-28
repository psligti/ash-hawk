from __future__ import annotations

from src.solution import is_status_ok


def test_status_ok_handles_new_string() -> None:
    status = "".join(["o", "k"])
    assert is_status_ok(status) is True
