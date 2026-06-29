"""Multi-turn conversation simulation: stitch a workload's turns into one timeline.

A :class:`~serve_sim.workload.Workload` is a multi-turn agentic conversation.
Each turn is one LLM iteration (a prefill of the newly appended messages followed
by a decode of the completion) and the turns are separated by *tool-call waits*:
the client-side gap during which the sequence waits for a tool/function call to
return before sending the next request.

This module drives the per-turn pipeline already provided by the simulator --
:class:`~serve_sim.tracker.SequenceTracker` (tokenize a turn into cached / prefill
/ decode token counts), :class:`~serve_sim.shards.WorkShardGenerator` (turn that
work into shards) and :class:`~serve_sim.events.EventGenerator` (time the shards
into compute events) -- and lays the per-turn schedules out end-to-end on one
global clock, inserting a tool-call wait event in front of every turn that has a
``pre_gap``.

Per the PRD we do not model the computation of a tool call, only the wait for it
to complete: a tool-call event has no FLOPs or bytes, just a duration. Its
duration is the dataset-provided ``pre_gap`` divided by ``tool_calling_speedup``
(a global system parameter; ``1.0`` leaves the trace untouched, ``2.0`` makes
tools return twice as fast).
"""

from __future__ import annotations

from dataclasses import replace

from .events import ComputeEvent, EventGenerator, EventSchedule
from .shards import WorkShardGenerator
from .tokenizer import Tokenizer
from .tracker import SequenceTracker
from .workload import Workload

#: Phase tag and sentinel device index for a tool-call wait event.
TOOL_CALL_PHASE = "tool_call"
_NO_DEVICE = -1


def _build_tool_call_event(turn_index: int, duration: float, start: float) -> ComputeEvent:
    """A client-side wait for a tool call to return (no FLOPs/bytes)."""

    return ComputeEvent(
        group_index=turn_index,
        phase=TOOL_CALL_PHASE,
        device_index=_NO_DEVICE,
        flops=0.0,
        bytes_read=0.0,
        compute_time=0.0,
        bandwidth_time=0.0,
        duration=duration,
        start=start,
        end=start + duration,
    )


def run_conversation(
    workload: Workload,
    model,
    compute_devices,
    tokenizer: Tokenizer,
    *,
    pipeline_parallel: int = 1,
    expert_parallel: int = 1,
    prefill_chunk_size: int | None = None,
    tool_calling_speedup: float = 1.0,
    scale_up_bandwidth_bytes_per_s: float | None = None,
    scale_up_latency_s: float = 0.0,
) -> EventSchedule:
    """Simulate a whole multi-turn conversation on a single timeline.

    Each turn is tokenized (its predecessor's messages form the KV-cache prefix),
    turned into work shards and timed by an :class:`EventGenerator`; the turn's
    events are then shifted onto the global clock. A tool-call wait event of
    ``turn.pre_gap / tool_calling_speedup`` seconds precedes every turn whose
    ``pre_gap`` is positive (the first turn has none).

    Args:
        workload: The multi-turn conversation to simulate.
        model: The model serving the conversation.
        compute_devices: Devices the turns run on (re-used every turn).
        tokenizer: Counts tokens per message to size each turn.
        pipeline_parallel: Pipeline-parallel degree for each turn.
        expert_parallel: Expert-parallel degree for each turn.
        prefill_chunk_size: Optional prefill chunking for each turn.
        tool_calling_speedup: Global divisor applied to every ``pre_gap``.
        scale_up_bandwidth_bytes_per_s: Scale-up network bandwidth used to time
            parallelism communication collectives; ``None`` disables comm.
        scale_up_latency_s: Per-collective scale-up network latency.

    Returns:
        One :class:`EventSchedule` whose events span the whole conversation,
        including ``phase == "tool_call"`` waits between turns.
    """

    if tool_calling_speedup <= 0:
        raise ValueError("tool_calling_speedup must be positive")

    schedule = EventSchedule()
    clock = 0.0

    for turn in workload:
        wait = turn.pre_gap / tool_calling_speedup
        if wait > 0:
            event = _build_tool_call_event(turn.index, wait, clock)
            schedule.events.append(event)
            clock = event.end

        tracker = SequenceTracker.from_turn(workload, turn.index, tokenizer)
        shards = WorkShardGenerator(model).generate(
            [tracker.to_work()], prefill_chunk_size
        )
        generator = EventGenerator(
            model,
            compute_devices,
            pipeline_parallel=pipeline_parallel,
            expert_parallel=expert_parallel,
            scale_up_bandwidth_bytes_per_s=scale_up_bandwidth_bytes_per_s,
            scale_up_latency_s=scale_up_latency_s,
        )
        turn_schedule = generator.run(shards)
        for event in turn_schedule.events:
            schedule.events.append(
                replace(event, start=event.start + clock, end=event.end + clock)
            )
        clock += turn_schedule.makespan

    return schedule
