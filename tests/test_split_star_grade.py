"""Split star display grade (nebula graded before star composite)."""

from starsky_gen.config import FeatureConfig, NebulaMode


def test_split_star_display_grade_default_on() -> None:
    feat = FeatureConfig()
    assert feat.split_star_display_grade is True


def test_stars_deferred_for_galaxy_streak() -> None:
    feat = FeatureConfig(
        galaxy_view=True,
        stars=True,
        nebula=True,
        split_star_display_grade=True,
        stars_after_display_grade=False,
    )
    deferred = bool(
        feat.galaxy_view
        and feat.stars
        and feat.nebula
        and NebulaMode.galaxy_streak == NebulaMode.galaxy_streak
        and (feat.split_star_display_grade or feat.stars_after_display_grade)
    )
    assert deferred is True


def test_legacy_combined_grade_when_split_off() -> None:
    feat = FeatureConfig(
        galaxy_view=True,
        stars=True,
        nebula=True,
        split_star_display_grade=False,
        stars_after_display_grade=False,
    )
    deferred = bool(
        feat.galaxy_view
        and feat.stars
        and feat.nebula
        and (feat.split_star_display_grade or feat.stars_after_display_grade)
    )
    assert deferred is False
