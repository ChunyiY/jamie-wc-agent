"""Tests for fixture loading helpers."""

import pandas as pd

from src.football_data import clean_fixture_field


def test_clean_fixture_field_handles_nan():
    assert clean_fixture_field(float("nan")) == ""
    assert clean_fixture_field(None) == ""
    assert clean_fixture_field("Round of 32") == "Round of 32"
