from __future__ import annotations


def test_factorial_scorer_is_available_from_package_root() -> None:
    from met3r import score_factorial_known_camera_pair

    assert callable(score_factorial_known_camera_pair)


def test_geometry_support_api_is_available_from_package_root() -> None:
    from met3r import (
        DirectionalGeometrySupport,
        SymmetricGeometrySupport,
        score_symmetric_geometry_support,
    )

    assert DirectionalGeometrySupport.__name__ == "DirectionalGeometrySupport"
    assert SymmetricGeometrySupport.__name__ == "SymmetricGeometrySupport"
    assert callable(score_symmetric_geometry_support)
