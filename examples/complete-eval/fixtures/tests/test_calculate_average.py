import pytest


def test_calculate_average_basic():
    from collections.abc import Callable

    from calculate_average import calculate_average

    assert callable(calculate_average)
    result = calculate_average([1.0, 2.0, 3.0, 4.0, 5.0])
    assert result == 3.0


def test_calculate_average_single_value():
    from calculate_average import calculate_average

    result = calculate_average([42.0])
    assert result == 42.0


def test_calculate_average_empty_list():
    from calculate_average import calculate_average

    result = calculate_average([])
    assert result == 0.0


def test_calculate_average_none():
    from calculate_average import calculate_average

    result = calculate_average(None)
    assert result == 0.0


def test_calculate_average_negative_numbers():
    from calculate_average import calculate_average

    result = calculate_average([-1.0, -2.0, -3.0])
    assert result == -2.0


def test_calculate_average_mixed_numbers():
    from calculate_average import calculate_average

    result = calculate_average([-2.0, 0.0, 2.0])
    assert result == 0.0


def test_calculate_average_floats():
    from calculate_average import calculate_average

    result = calculate_average([1.5, 2.5, 3.5])
    assert result == 2.5
