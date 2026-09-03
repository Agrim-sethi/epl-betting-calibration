"""Unit tests for the no-vig conversion."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_betting.odds import compute_no_vig_probs


def test_probabilities_sum_to_one():
    p = compute_no_vig_probs(2.0, 3.5, 4.0)
    assert pytest.approx(p["pH"] + p["pD"] + p["pA"], abs=1e-12) == 1.0


def test_shorter_odds_give_higher_probability():
    p = compute_no_vig_probs(1.5, 4.0, 7.0)
    assert p["pH"] > p["pD"] > p["pA"]


def test_fair_book_is_unchanged():
    # A book with no margin: 1/3 each at odds of 3.0
    p = compute_no_vig_probs(3.0, 3.0, 3.0)
    for k in ("pH", "pD", "pA"):
        assert pytest.approx(p[k], abs=1e-12) == 1 / 3


def test_margin_is_removed():
    # Overround book still yields probabilities summing to exactly 1
    p = compute_no_vig_probs(1.8, 3.4, 4.2)
    assert pytest.approx(sum(p.values()), abs=1e-12) == 1.0


@pytest.mark.parametrize("bad", [(0, 3.0, 4.0), (-2.0, 3.0, 4.0), (2.0, float("nan"), 4.0)])
def test_invalid_odds_raise(bad):
    with pytest.raises(ValueError):
        compute_no_vig_probs(*bad)
