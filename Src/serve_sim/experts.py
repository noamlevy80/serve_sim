"""Expert-usage statistical model for MoE layers.

Per the PRD, per-token expert routing is modelled statistically: there are
``num_experts_per_token`` active routed-expert "slots"; each slot keeps the same
expert live for ``expert_persistence`` consecutive tokens (a random variable with
a configured mean and variance) before switching to a different random expert.
Expert selection is assumed identical across all layers (a worst case for weight
movement).

What the rest of the simulator needs from this model is, for a group of tokens
processed together, **how many distinct routed experts** are touched -- this sets
the expert weight-movement bandwidth. We expose:

- :meth:`expected_distinct` -- a closed-form expectation used by the roofline
  (deterministic and reproducible).
- :meth:`sample_distinct` -- a seedable Monte-Carlo estimate that simulates the
  slot renewal process (uses the variance), for future stochastic runs.

The expectation uses an occupancy model: a group of ``n`` tokens generates
``picks`` expert selections spread (approximately uniformly) over ``E`` experts,
and the expected number of distinct experts is ``E * (1 - (1 - 1/E) ** picks)``.

For a single sequence's *consecutive* tokens (prefill), persistence reduces the
selection count to ``k_E * (1 + (n - 1) / mean)``. For a decode step across ``n``
*independent* sequences, each contributes a fresh full set, giving ``k_E * n``.
"""

from __future__ import annotations

import random


class ExpertUsageModel:
    """Statistical model of distinct routed-expert usage over token groups."""

    def __init__(
        self,
        num_experts: int,
        num_experts_per_token: int,
        persistence_mean: float = 16.0,
        persistence_variance: float = 4.0,
    ) -> None:
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if not 1 <= num_experts_per_token <= num_experts:
            raise ValueError("num_experts_per_token must be in [1, num_experts]")
        if persistence_mean <= 0:
            raise ValueError("persistence_mean must be positive")
        if persistence_variance < 0:
            raise ValueError("persistence_variance must be non-negative")
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.persistence_mean = persistence_mean
        self.persistence_variance = persistence_variance

    @classmethod
    def from_model(cls, model) -> "ExpertUsageModel":
        """Build from a MoE :class:`~serve_sim.model.Model`."""

        return cls(
            num_experts=model.num_experts,
            num_experts_per_token=model.num_experts_per_token,
            persistence_mean=model.expert_persistence_mean,
            persistence_variance=model.expert_persistence_variance,
        )

    def _picks(self, num_tokens: int, consecutive: bool) -> float:
        """Expected number of routed-expert selections over the group."""

        if num_tokens <= 0:
            return 0.0
        k = self.num_experts_per_token
        if consecutive:
            runs_per_slot = 1.0 + (num_tokens - 1) / self.persistence_mean
            return k * runs_per_slot
        return float(k * num_tokens)

    def expected_distinct(self, num_tokens: int, consecutive: bool) -> float:
        """Expected number of distinct routed experts touched by the group.

        Args:
            num_tokens: Tokens processed together (chunk size or batch size).
            consecutive: ``True`` when the tokens are consecutive positions of a
                single sequence (prefill); ``False`` for independent sequences
                in one decode step.
        """

        picks = self._picks(num_tokens, consecutive)
        if picks <= 0:
            return 0.0
        e = self.num_experts
        return e * (1.0 - (1.0 - 1.0 / e) ** picks)

    def _sample_persistence(self, rng: random.Random) -> int:
        """Draw one persistence run length (>= 1) honouring mean and variance."""

        std = self.persistence_variance ** 0.5
        value = rng.gauss(self.persistence_mean, std)
        return max(1, round(value))

    def sample_distinct(
        self,
        num_tokens: int,
        consecutive: bool,
        rng: random.Random | None = None,
    ) -> int:
        """Monte-Carlo count of distinct routed experts for one group draw.

        Simulates the per-slot renewal process; uses the configured variance.
        """

        if num_tokens <= 0:
            return 0
        rng = rng or random.Random()
        e = self.num_experts
        k = self.num_experts_per_token
        touched: set[int] = set()
        if consecutive:
            for _ in range(k):
                position = 0
                while position < num_tokens:
                    touched.add(rng.randrange(e))
                    position += self._sample_persistence(rng)
        else:
            for _ in range(num_tokens):
                # each independent sequence picks k distinct experts this step
                touched.update(rng.sample(range(e), k))
        return len(touched)
