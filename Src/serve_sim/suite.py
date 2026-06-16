"""Test suites: a list of workloads, each mapped to a model.

A *suite* is what the simulator runs: a sequence of :class:`SuiteEntry` pairs, each
binding one multi-turn :class:`~serve_sim.workload.Workload` to the name of the
model that serves it.

A *randomized* suite (the only kind implemented so far) draws a number of random
workloads from the source dataset and assigns each a model chosen uniformly at
random from a configured list. Dataset access goes through a
:class:`~serve_sim.dataset.WorkloadLoader`, which is injected so the selection
logic can be exercised offline with an in-memory fetcher. Randomness comes from a
caller-supplied :class:`random.Random`, so a fixed seed yields a reproducible
suite.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from .dataset import WorkloadLoader
from .workload import Workload


@dataclass(frozen=True)
class SuiteEntry:
    """One unit of work: a workload bound to the model that serves it.

    Attributes:
        workload: The multi-turn conversation to simulate.
        model: Name/stem of the model serving this workload (resolved to a
            concrete model by the simulation layer, e.g. ``load_model_config``).
    """

    workload: Workload
    model: str


@dataclass(frozen=True)
class Suite:
    """An ordered collection of :class:`SuiteEntry` pairs."""

    entries: tuple[SuiteEntry, ...]

    def __post_init__(self) -> None:
        if not self.entries:
            raise ValueError("a suite must contain at least one entry")

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[SuiteEntry]:
        return iter(self.entries)

    def __getitem__(self, index: int) -> SuiteEntry:
        return self.entries[index]

    @property
    def models(self) -> set[str]:
        """The distinct model names referenced by the suite."""

        return {entry.model for entry in self.entries}


@dataclass(frozen=True)
class RandomizedSuiteConfig:
    """Configuration for a randomized suite.

    Attributes:
        num_workloads: How many workloads to draw from the dataset.
        models: Model names to choose from (one is picked at random per workload).
    """

    num_workloads: int
    models: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.num_workloads < 1:
            raise ValueError("num_workloads must be >= 1")
        if not self.models:
            raise ValueError("models must list at least one model")

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "RandomizedSuiteConfig":
        return cls(
            num_workloads=int(config["num_workloads"]),
            models=tuple(config["models"]),
        )


def build_randomized_suite(
    config: RandomizedSuiteConfig,
    loader: WorkloadLoader,
    rng: random.Random | int | None = None,
) -> Suite:
    """Draw ``num_workloads`` random workloads and bind each to a random model.

    Each workload is the full session containing a uniformly random dataset row;
    sampling is with replacement (the same session may be drawn more than once).
    Each draw is paired with a model chosen uniformly from ``config.models``.

    Args:
        config: The randomized-suite parameters.
        loader: Dataset loader (injected, so this is testable offline).
        rng: A :class:`random.Random`, an int seed, or ``None`` for system entropy.
    """

    if not isinstance(rng, random.Random):
        rng = random.Random(rng)

    total = loader.num_rows()
    if total < 1:
        raise ValueError("dataset is empty; cannot build a randomized suite")

    entries: list[SuiteEntry] = []
    for _ in range(config.num_workloads):
        offset = rng.randrange(total)
        workload = loader.load_session_at(offset)
        model = rng.choice(config.models)
        entries.append(SuiteEntry(workload=workload, model=model))

    return Suite(tuple(entries))


def build_suite_from_config(
    config: Mapping[str, Any],
    loader: WorkloadLoader,
    rng: random.Random | int | None = None,
) -> Suite:
    """Build a suite from a parsed config, dispatching on its ``type``.

    Only ``"randomized"`` is implemented; ``"directed"`` is reserved (TBD).
    """

    suite_type = config.get("type", "randomized")
    if suite_type == "randomized":
        return build_randomized_suite(
            RandomizedSuiteConfig.from_config(config), loader, rng
        )
    if suite_type == "directed":
        raise NotImplementedError("directed suites are not implemented yet")
    raise ValueError(f"unknown suite type: {suite_type!r}")


def load_suite(
    path: str | Path,
    loader: WorkloadLoader,
    rng: random.Random | int | None = None,
) -> Suite:
    """Load a suite from a JSON config file (see ``Suites/*.json``)."""

    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    return build_suite_from_config(config, loader, rng)
