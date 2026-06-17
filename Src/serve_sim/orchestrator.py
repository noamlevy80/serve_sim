"""Orchestrator v0: a strictly event-driven serving loop.

This is the first end-to-end driver that ties the pieces together: requests
*arrive* over time, are collected into a concurrency window, dispatched as
batches onto a fixed slice of the system's compute devices, co-run on the shared
timeline by the :class:`~serve_sim.arbiter.IncrementalArbiter` (which rescales
events when batches contend for the same resource), and retired on completion.

v0 deliberately uses the *trivial fixed policy* the plan calls for:

- **No PDD:** a request's prefill and decode for its turn run as one job (the
  existing work-shard -> event pipeline), not split across prefill/decode pools.
- **Fixed parallelism:** the engine is the first ``pipeline_parallel x
  expert_parallel`` compute devices of the system; the roofline parallelism
  search comes in a later stage.
- **Target concurrency only:** the one admission knob caps how many sequences
  are in flight; beyond that, ready sequences wait for a completion.
- **Concurrency-window batching:** ready sequences of the same model are grouped
  and dispatched when the window fills (max batch size) or its duration elapses.

Single-turn requests, one-shot batched decode, single-tier devices and no KV /
weight reuse are v0 simplifications; multi-turn conversations, iteration-level
batching, expert streaming and prefix reuse are layered on in later stages. The
loop itself is strictly event driven: it advances to the next arrival, window
deadline or in-flight completion -- never a fixed step.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .arbiter import IncrementalArbiter
from .events import EventGenerator
from .shards import WorkShardGenerator
from .system import System
from .tokenizer import Tokenizer
from .tracker import SequenceWork
from .workload import Workload


@dataclass(frozen=True)
class StrategyConfig:
    """The orchestration knobs (v0 subset).

    Attributes:
        max_batch_size: Sequences per dispatched batch (window fill threshold).
        max_window_duration: Seconds a window stays open before dispatching
            whatever it has collected (``0`` dispatches as soon as work is ready).
        target_concurrency: Max sequences in flight, or ``None`` for unbounded.
        pipeline_parallel: Fixed pipeline-parallel degree of the engine.
        expert_parallel: Fixed expert-parallel degree of the engine.
        prefill_chunk_size: Optional prefill chunking applied to every batch.
        event_random_factor_range: Per-event time is multiplied by
            ``1 + U(-range, range)`` to model system randomness; ``0`` disables it.
        random_seed: Seed for the run's randomness (event-time perturbation);
            ``None`` draws a non-deterministic seed.
    """

    max_batch_size: int = 1
    max_window_duration: float = 0.0
    target_concurrency: int | None = None
    pipeline_parallel: int = 1
    expert_parallel: int = 1
    prefill_chunk_size: int | None = None
    event_random_factor_range: float = 0.0
    random_seed: int | None = None

    def __post_init__(self) -> None:
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        if self.max_window_duration < 0:
            raise ValueError("max_window_duration must be non-negative")
        if self.target_concurrency is not None and self.target_concurrency < 1:
            raise ValueError("target_concurrency must be >= 1 when set")
        if self.pipeline_parallel < 1 or self.expert_parallel < 1:
            raise ValueError("parallelism degrees must be >= 1")
        if self.prefill_chunk_size is not None and self.prefill_chunk_size < 1:
            raise ValueError("prefill_chunk_size must be >= 1")
        if not 0.0 <= self.event_random_factor_range < 1.0:
            raise ValueError("event_random_factor_range must be in [0, 1)")


@dataclass(frozen=True)
class Request:
    """One serving request: a single sequence to prefill and decode.

    Attributes:
        request_id: Caller-assigned identifier (unique within a run).
        model: The model object serving the request (flat ``Model`` or
            ``LayeredModel``). Requests sharing a model *instance* may batch.
        prompt_tokens: Total prompt length (including any cached prefix).
        output_tokens: Tokens to generate.
        arrival_time: When the request enters the system.
        cached_tokens: Prompt tokens already in cache (prefill skips these).
    """

    request_id: int
    model: object
    prompt_tokens: int
    output_tokens: int
    arrival_time: float = 0.0
    cached_tokens: int = 0

    def __post_init__(self) -> None:
        if self.prompt_tokens < 0 or self.output_tokens < 0 or self.cached_tokens < 0:
            raise ValueError("token counts must be non-negative")
        if self.cached_tokens > self.prompt_tokens:
            raise ValueError("cached_tokens cannot exceed prompt_tokens")
        if self.arrival_time < 0:
            raise ValueError("arrival_time must be non-negative")

    @property
    def work(self) -> SequenceWork:
        """The per-sequence work description for the shard generator."""

        return SequenceWork(
            cached_tokens=self.cached_tokens,
            prefill_tokens=self.prompt_tokens - self.cached_tokens,
            decode_tokens=self.output_tokens,
        )

    @classmethod
    def from_workload(
        cls,
        request_id: int,
        workload: Workload,
        model: object,
        tokenizer: Tokenizer,
        *,
        arrival_time: float = 0.0,
        turn_index: int = 0,
    ) -> "Request":
        """Build a single-sequence request from one turn of a workload."""

        from .tracker import SequenceTracker

        tracker = SequenceTracker.from_turn(workload, turn_index, tokenizer)
        return cls(
            request_id=request_id,
            model=model,
            prompt_tokens=tracker.prompt_tokens,
            output_tokens=tracker.output_tokens,
            arrival_time=arrival_time,
            cached_tokens=tracker.cached_tokens,
        )


@dataclass
class RequestRecord:
    """Per-request timing produced by a run."""

    request_id: int
    arrival_time: float
    dispatch_time: float
    completion_time: float
    prompt_tokens: int
    output_tokens: int
    batch_index: int

    @property
    def queue_delay(self) -> float:
        """Seconds spent waiting between arrival and dispatch."""

        return self.dispatch_time - self.arrival_time

    @property
    def latency(self) -> float:
        """End-to-end seconds from arrival to completion."""

        return self.completion_time - self.arrival_time


@dataclass
class RunResult:
    """Minimal run state: per-request records and the overall makespan.

    The rich run report (TTFT/TPOT, utilization, occupancy) is a later
    *outputs* stage; v0 exposes only the raw per-request timings.
    """

    records: list[RequestRecord] = field(default_factory=list)
    num_batches: int = 0

    @property
    def makespan(self) -> float:
        """Wall-clock end of the last request (0 if none)."""

        return max((r.completion_time for r in self.records), default=0.0)

    def record_for(self, request_id: int) -> RequestRecord:
        for record in self.records:
            if record.request_id == request_id:
                return record
        raise KeyError(f"no record for request {request_id}")


class Simulator:
    """Runs a set of requests through a system under a fixed strategy."""

    def __init__(self, system: System, strategy: StrategyConfig | None = None) -> None:
        self.system = system
        self.strategy = strategy or StrategyConfig()
        degree = self.strategy.pipeline_parallel * self.strategy.expert_parallel
        devices = system.compute_devices
        if len(devices) < degree:
            raise ValueError(
                f"system has {len(devices)} compute devices but the engine needs "
                f"{degree} (pipeline_parallel x expert_parallel)"
            )
        self._engine_devices = devices[:degree]
        self._rng = random.Random(self.strategy.random_seed)

    def run(self, requests: list[Request]) -> RunResult:
        """Event-driven serving loop over ``requests``; returns per-request timing."""

        strategy = self.strategy
        arrivals = sorted(requests, key=lambda r: r.arrival_time)
        arrival_pos = 0
        ready: list[Request] = []
        window_open: float | None = None

        arbiter = IncrementalArbiter()
        in_flight = 0
        batch_index = 0
        # job_index -> (requests, dispatch_time, batch_index)
        jobs: dict[int, tuple[list[Request], float, int]] = {}
        reported: set[int] = set()
        result = RunResult()

        def next_arrival_time() -> float | None:
            if arrival_pos < len(arrivals):
                return arrivals[arrival_pos].arrival_time
            return None

        while True:
            # Candidate next decision times.
            candidates: list[float] = []
            na = next_arrival_time()
            if na is not None:
                candidates.append(na)
            if window_open is not None:
                deadline = window_open + strategy.max_window_duration
                # Only a *future* deadline is an event to advance to; once the
                # window has elapsed, further dispatch is gated by concurrency /
                # completions, not by the (past) deadline.
                if deadline > arbiter.time:
                    candidates.append(deadline)
            ne = arbiter.next_event_time()
            if ne is not None:
                candidates.append(ne)

            if not candidates:
                # Nothing pending anywhere. If ready work remains it must be
                # dispatchable now (window already accounted for above).
                if not ready:
                    break
                now = arbiter.time
            else:
                now = min(candidates)
                arbiter.advance_to(now)

            # 1) Retire any jobs that have finished by ``now``.
            for job_index, (reqs, dispatch_time, b_index) in jobs.items():
                if job_index in reported or not arbiter.job_is_done(job_index):
                    continue
                reported.add(job_index)
                end = arbiter.job_end_time(job_index)
                in_flight -= len(reqs)
                for req in reqs:
                    result.records.append(
                        RequestRecord(
                            request_id=req.request_id,
                            arrival_time=req.arrival_time,
                            dispatch_time=dispatch_time,
                            completion_time=end,
                            prompt_tokens=req.prompt_tokens,
                            output_tokens=req.output_tokens,
                            batch_index=b_index,
                        )
                    )

            # 2) Admit arrivals that have occurred by ``now``.
            while arrival_pos < len(arrivals) and arrivals[arrival_pos].arrival_time <= now:
                ready.append(arrivals[arrival_pos])
                arrival_pos += 1
                if window_open is None:
                    window_open = now

            # 3) Dispatch decisions.
            window_elapsed = (
                window_open is not None
                and now >= window_open + strategy.max_window_duration
            )
            progressed = False
            while ready:
                fill = len(ready) >= strategy.max_batch_size
                if not (fill or window_elapsed):
                    break
                batch = self._take_batch(ready, in_flight)
                if batch is None:
                    break  # concurrency full; wait for a completion
                in_flight += len(batch)
                job_index = self._dispatch(arbiter, batch)
                jobs[job_index] = (batch, now, batch_index)
                batch_index += 1
                progressed = True

            if not ready:
                window_open = None

            # Termination: everything arrived, nothing queued, nothing in flight.
            if (
                arrival_pos >= len(arrivals)
                and not ready
                and arbiter.is_idle()
            ):
                break

            # Guard against a stall: no future event and we made no progress.
            if not candidates and not progressed:
                break

        result.num_batches = batch_index
        return result

    # --- helpers ------------------------------------------------------------

    def _take_batch(self, ready: list[Request], in_flight: int) -> list[Request] | None:
        """Pull up to ``max_batch_size`` same-model requests, honoring concurrency.

        Returns ``None`` if concurrency is exhausted and work is already in
        flight (so a completion must free a slot first). When nothing is in
        flight a batch is always returned, to guarantee progress.
        """

        strategy = self.strategy
        if strategy.target_concurrency is None:
            slots = strategy.max_batch_size
        else:
            slots = strategy.target_concurrency - in_flight
            if slots <= 0:
                if in_flight > 0:
                    return None
                slots = strategy.max_batch_size  # force progress
        limit = min(strategy.max_batch_size, slots)

        model = ready[0].model
        batch: list[Request] = []
        rest: list[Request] = []
        for req in ready:
            if len(batch) < limit and req.model is model:
                batch.append(req)
            else:
                rest.append(req)
        ready[:] = rest
        return batch

    def _dispatch(self, arbiter: IncrementalArbiter, batch: list[Request]) -> int:
        """Build a batch's events and admit them to the arbiter at the current time."""

        model = batch[0].model
        work = [req.work for req in batch]
        shards = WorkShardGenerator(model).generate(
            work, prefill_chunk_size=self.strategy.prefill_chunk_size
        )
        generator = EventGenerator(
            model,
            self._engine_devices,
            pipeline_parallel=self.strategy.pipeline_parallel,
            expert_parallel=self.strategy.expert_parallel,
            event_random_factor_range=self.strategy.event_random_factor_range,
            rng=self._rng,
        )
        return arbiter.admit(generator, shards)
