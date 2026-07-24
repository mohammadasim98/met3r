from __future__ import annotations


def test_factorial_scorer_is_available_from_package_root() -> None:
    from met3r import score_factorial_known_camera_pair

    assert callable(score_factorial_known_camera_pair)
