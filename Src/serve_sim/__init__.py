"""serve_sim: datacenter-scale LLM inference workload simulator.

Stage 1 exposes the workload model and a loader that downloads multi-turn
agentic sessions from the source dataset. Stage 2 adds the roofline simulation
path: model sizing, devices, trackers, work-shard generation and event
generation.
"""

from .workload import Message, ToolCall, Turn, Workload, build_workload_from_rows
from .dataset import HttpRowFetcher, RowFetcher, WorkloadLoader
from .model import Model, toy_model, toy_moe_model
from .blocks import (
    Attention,
    DenseFFN,
    Layer,
    LayeredModel,
    MambaBlock,
    MoEFFN,
)
from .model_config import load_model_config, model_from_config
from .hardware import ComputeDevice, MemoryDevice, dtype_compute_scale
from .device_config import (
    compute_device_from_config,
    load_compute_device,
    load_memory_device,
    memory_device_from_config,
)
from .tokenizer import Tokenizer, TiktokenTokenizer, WhitespaceTokenizer
from .tracker import BatchTracker, SequenceTracker, SequenceWork
from .kv_cache import KVCacheTracker
from .experts import ExpertUsageModel
from .shards import WorkShard, WorkShardGenerator
from .tiering import (
    ExpertResidencyCache,
    GroupActivation,
    build_activation_trace,
    derive_expert_cache_capacity,
)
from .events import ComputeEvent, EventGenerator, EventSchedule
from .weights import ModelWeightsTracker, WeightShard
from .transfer import (
    INTRA_PACKAGE,
    TransferLink,
    make_transfer_event,
    transfer_duration,
)
from .arbiter import ArbiterResult, IncrementalArbiter, ResourceArbiter
from .orchestrator import (
    Request,
    RequestRecord,
    RunResult,
    Simulator,
    StrategyConfig,
)
from .conversation import TOOL_CALL_PHASE, run_conversation
from .system import (
    Network,
    Node,
    System,
    load_system,
    system_from_config,
)
from .suite import (
    RandomizedSuiteConfig,
    Suite,
    SuiteEntry,
    build_randomized_suite,
    build_suite_from_config,
    load_suite,
)

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
    "toy_moe_model",
    "Attention",
    "DenseFFN",
    "Layer",
    "LayeredModel",
    "MambaBlock",
    "MoEFFN",
    "load_model_config",
    "model_from_config",
    "ComputeDevice",
    "MemoryDevice",
    "dtype_compute_scale",
    "compute_device_from_config",
    "load_compute_device",
    "load_memory_device",
    "memory_device_from_config",
    # tokenization + trackers
    "Tokenizer",
    "TiktokenTokenizer",
    "WhitespaceTokenizer",
    "BatchTracker",
    "SequenceTracker",
    "SequenceWork",
    "KVCacheTracker",
    # experts + shards + events
    "ExpertUsageModel",
    "WorkShard",
    "WorkShardGenerator",
    "ExpertResidencyCache",
    "GroupActivation",
    "build_activation_trace",
    "derive_expert_cache_capacity",
    "ComputeEvent",
    "EventGenerator",
    "EventSchedule",
    "ModelWeightsTracker",
    "WeightShard",
    "INTRA_PACKAGE",
    "TransferLink",
    "make_transfer_event",
    "transfer_duration",
    "ArbiterResult",
    "ResourceArbiter",
    "IncrementalArbiter",
    "Request",
    "RequestRecord",
    "RunResult",
    "Simulator",
    "StrategyConfig",
    "TOOL_CALL_PHASE",
    "run_conversation",
    "Network",
    "Node",
    "System",
    "load_system",
    "system_from_config",
    "RandomizedSuiteConfig",
    "Suite",
    "SuiteEntry",
    "build_randomized_suite",
    "build_suite_from_config",
    "load_suite",
]
