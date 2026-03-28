from __future__ import annotations

from src.solution import Greeter


def test_greeter_instance_method() -> None:
    greeter = Greeter("Hello, ")
    assert greeter.greet("Ada") == "Hello, Ada!"
