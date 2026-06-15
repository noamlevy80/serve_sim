"""serve_sim: datacenter-scale LLM inference workload simulator.

Stage 1 exposes the workload model and a loader that downloads multi-turn
agentic sessions from the source dataset. Stage 2 adds the roofline simulation
path: model sizing, devices, trackers, work-shard generation and event
generation.
"""

from .workload import Message, ToolCall, Turn, Workload, build_workload_from_rows
from .dataset import HttpRowFetcher, RowFetcher, WorkloadLoader
from .model import Model, toy_model
from .hardware import ComputeDevice, MemoryDevice, dtype_compute_scale
from .tokenizer import Tokenizer, TiktokenTokenizer, WhitespaceTokenizer
from .tracker import BatchTracker, SequenceTracker, SequenceWork
from .shards import WorkShard, WorkShardGenerator
from .events import ComputeEvent, EventGenerator, EventSchedule

__all__ = [
    # workloads
    "Message",
    "ToolCall",
    "Turn",
    "Workload",
    "build_workload_from_rows",
    "HttpRowFetcher",
    "RowFetcher",
    "WorkloadLoader",
    # model + hardware
    "Model",
    "toy_model",
    "ComputeDevice",
    "MemoryDevice",
    "dtype_compute_scale",
    # tokenization + trackers
    "Tokenizer",
    "TiktokenTokenizer",
    "WhitespaceTokenizer",
    "BatchTracker",
    "SequenceTracker",
    "SequenceWork",
    # shards + events
    "WorkShard",
    "WorkShardGenerator",
    "ComputeEvent",
    "EventGenerator",
    "EventSchedule",
]
