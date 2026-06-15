"""Tests for the ExpertUsageModel statistical expert-usage model."""

from __future__ import annotations

import random

import pytest

from serve_sim.experts import ExpertUsageModel
from serve_sim.model import toy_moe_model


def make_usage(E=16, k=2, mean=16.0, var=4.0):
    return ExpertUsageModel(E, k, persistence_mean=mean, persistence_variance=var)


# --- construction / validation --------------------------------------------------


def test_from_model():
    model = toy_moe_model(num_experts=8, num_experts_per_token=2)
    usage = ExpertUsageModel.from_model(model)
    assert usage.num_experts == 8
    assert usage.num_experts_per_token == 2
    assert usage.persistence_mean == model.expert_persistence_mean


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_experts": 0},
        {"num_experts_per_token": 0},
        {"num_experts_per_token": 99},
        {"persistence_mean": 0},
        {"persistence_variance": -1},
    ],
)
def test_validation(kwargs):
    base = {"num_experts": 8, "num_experts_per_token": 2}
    base.update(kwargs)
    with pytest.raises(ValueError):
        ExpertUsageModel(**base)


# --- expected_distinct properties -----------------------------------------------


def test_zero_tokens_is_zero():
    assert make_usage().expected_distinct(0, consecutive=True) == 0.0


def test_single_token_consecutive_equals_topk():
    # one token => k picks => roughly k distinct (not exactly, due to collisions)
    usage = make_usage(E=1000, k=2)
    # with many experts, collisions are negligible -> ~2 distinct
    assert usage.expected_distinct(1, consecutive=True) == pytest.approx(2.0, abs=0.01)


def test_distinct_never_exceeds_num_experts():
    usage = make_usage(E=8, k=2)
    huge = usage.expected_distinct(100000, consecutive=False)
    assert huge <= 8.0
    assert huge == pytest.approx(8.0, rel=1e-6)


def test_distinct_monotonic_in_tokens():
    usage = make_usage(E=64, k=4)
    prev = 0.0
    for n in [1, 2, 4, 8, 16, 64, 256]:
        d = usage.expected_distinct(n, consecutive=False)
        assert d >= prev - 1e-9
        prev = d


def test_consecutive_touches_fewer_than_independent():
    # persistence makes consecutive tokens share experts -> fewer distinct
    usage = make_usage(E=64, k=2, mean=16.0)
    n = 32
    consec = usage.expected_distinct(n, consecutive=True)
    indep = usage.expected_distinct(n, consecutive=False)
    assert consec < indep


def test_higher_persistence_reduces_distinct_for_consecutive():
    n = 64
    low = make_usage(E=128, k=2, mean=4.0).expected_distinct(n, consecutive=True)
    high = make_usage(E=128, k=2, mean=64.0).expected_distinct(n, consecutive=True)
    assert high < low


def test_closed_form_matches_formula():
    E, k, mean, n = 32, 3, 16.0, 20
    usage = make_usage(E=E, k=k, mean=mean)
    picks = k * (1 + (n - 1) / mean)
    expected = E * (1 - (1 - 1 / E) ** picks)
    assert usage.expected_distinct(n, consecutive=True) == pytest.approx(expected)


# --- sampler --------------------------------------------------------------------


def test_sample_distinct_bounds():
    usage = make_usage(E=8, k=2)
    rng = random.Random(0)
    for _ in range(20):
        d = usage.sample_distinct(10, consecutive=False, rng=rng)
        assert 0 <= d <= 8


def test_sample_distinct_is_seed_reproducible():
    usage = make_usage(E=32, k=2)
    a = usage.sample_distinct(50, consecutive=True, rng=random.Random(123))
    b = usage.sample_distinct(50, consecutive=True, rng=random.Random(123))
    assert a == b


def test_sample_mean_approximates_expectation():
    # The independent (decode) regime is where the occupancy expectation is
    # designed to match the renewal simulation closely.
    usage = make_usage(E=64, k=2, mean=16.0, var=4.0)
    n = 24
    rng = random.Random(7)
    trials = 400
    avg = sum(
        usage.sample_distinct(n, consecutive=False, rng=rng) for _ in range(trials)
    ) / trials
    expected = usage.expected_distinct(n, consecutive=False)
    assert avg == pytest.approx(expected, rel=0.1)


def test_sample_consecutive_is_in_ballpark_of_expectation():
    # For consecutive tokens the closed form is an approximation of the renewal
    # simulation; require only same order of magnitude / loose agreement.
    usage = make_usage(E=64, k=2, mean=16.0, var=4.0)
    n = 64
    rng = random.Random(11)
    trials = 400
    avg = sum(
        usage.sample_distinct(n, consecutive=True, rng=rng) for _ in range(trials)
    ) / trials
    expected = usage.expected_distinct(n, consecutive=True)
    assert avg == pytest.approx(expected, rel=0.3)


def test_sample_independent_step_picks_distinct_per_sequence():
    # each independent sequence picks exactly k distinct experts
    usage = make_usage(E=100, k=3)
    rng = random.Random(1)
    # one sequence -> exactly 3 distinct
    assert usage.sample_distinct(1, consecutive=False, rng=rng) == 3
