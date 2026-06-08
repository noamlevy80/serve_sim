# Product Requirements Document (PRD)

## 1. Purpose
This project simulates LLM serving workloads at datacenter scale by replaying trace-derived sessions over a configurable system model. Its primary goal is to quantify bottlenecks caused by memory hierarchy design and data movement, especially for KV cache, MoE expert weights, and intermediate tensors.

The simulator is intended for architecture exploration, not cycle-accurate hardware modeling.

## 2. Problem Statement
Real serving systems exhibit performance behavior that depends on compute capability, memory tiering, network topology, orchestration policy, and workload shape. Existing trace analysis alone does not answer "what if" questions for alternate hardware and placement strategies.

This project must provide a repeatable environment to:
- Ingest real traces.
- Model event-level execution and contention.
- Compare architecture and scheduling options with standardized metrics.

## 3. Goals
- Simulate end-to-end prefill and decode behavior for many concurrent conversations.
- Capture effects of data placement and movement across memory tiers.
- Model MoE-specific behavior, including expert residency and weight movement.
- Evaluate orchestration strategies including prefix reuse and disaggregation.
- Output serving metrics and detailed event logs for offline analysis and visualization.

## 4. Non-Goals
- Cycle-level microarchitecture simulation.
- Exact kernel-level modeling for specific vendor stacks.
- Training workloads.
- Full tool execution modeling; only trace-observed tool latency is represented.
- Non-autoregressive model families in the initial version.

## 5. Scope and Assumptions
- Models are autoregressive and MoE in v1.
- Time is modeled as event boundaries, not hardware ticks.
- Trace input may come from LMCache Agentic Traces and other sources after normalization.
- The simulator must remain deterministic when randomness is disabled and random seed is fixed.

Reference dataset:
https://huggingface.co/datasets/sammshen/lmcache-agentic-traces

## 6. Personas and Primary Use Cases
### Personas
- Serving architect evaluating memory and network designs.
- Performance engineer validating bottleneck hypotheses.
- Researcher comparing orchestration policies under realistic traces.

### Use Cases
- Compare baseline GPU-only serving against tiered memory serving.
- Quantify impact of prefill-decode disaggregation on latency and throughput.
- Evaluate prefix reuse effectiveness with multi-turn conversational traces.
- Estimate network pressure from remote KV/expert fetches under high concurrency.

## 7. High-Level System Design
Each simulation element is represented by Python classes and instantiated from JSON configuration.

Core entities:
- Model
- Workload
- Test suite
- Compute device
- Memory device
- Node
- System
- Orchestrator
- Event engine

## 8. Data Contracts and Configuration
### 8.1 Model Definition
A model describes the computational and memory behavior needed to estimate event durations and resource usage for prefill and decode.

Required parameters:
- model_id: unique model name.
- architecture_type: fixed to "autoregressive_moe" in v1.
- num_layers: transformer layer count.
- hidden_size: model hidden dimension.
- num_attention_heads: total attention heads.
- num_kv_heads: KV heads for grouped/multi-query attention.
- vocab_size: tokenizer vocabulary size.
- max_context_tokens: maximum supported context length.
- precision: default compute/storage precision assumptions.
- prefill_flops_per_token: effective FLOPs/token for prefill.
- decode_flops_per_token: effective FLOPs/token for decode.
- kv_bytes_per_token: KV cache bytes per token per sequence.
- activation_bytes_per_token: transient activation footprint for prefill/decode.
- moe_num_experts: total experts.
- moe_top_k: experts selected per token.
- moe_expert_param_bytes: bytes of parameters per expert.
- moe_router_overhead_flops_per_token: routing overhead.
- expert_residency_model: statistical parameters for expert locality.
- tensor_parallel_supported: boolean.
- pipeline_parallel_supported: boolean.

Optional parameters:
- calibration_profile: empirical scaling corrections by device family.
- quantization_profiles: alternate precision-specific coefficients.

### 8.2 Workload Definition
A workload represents a single conversation/session trace.

Required parameters:
- workload_id: unique identifier.
- source: dataset/source metadata.
- model_id: model used by this workload.
- turns: ordered list of turns.
- arrival_time_ms: external arrival timestamp or synthetic start.

Each turn includes:
- turn_index
- input_tokens
- output_tokens
- has_tool_call
- tool_latency_ms (0 when absent)
- inter_turn_gap_ms
- trace_metadata (opaque source fields)

Standard normalized trace format (v1):
- Flat JSON schema with explicit token counts and timing fields.
- No source-specific parsing logic in simulation core.
- Adapters convert raw datasets into normalized format.

### 8.3 Test Suite Definition
A test suite is a collection of workloads plus run policy.

Required parameters:
- suite_id
- workloads: list of workload references or embedded workloads.
- default_model_id
- concurrency_policy
- sampling_policy
- random_seed

Optional parameters:
- workload_weights for weighted replay.
- suite_duration_limit_ms.
- warmup_workloads_count.

### 8.4 Compute Device Definition
A compute device models execution capability and local memory.

Required parameters:
- device_id
- device_type (GPU/CPU/TPU/other)
- peak_flops_by_precision
- sustained_flops_efficiency
- local_memory_capacity_bytes
- local_memory_bandwidth_bytes_per_s
- local_memory_latency_ms
- interconnect_ports
- max_concurrent_compute_events
- max_active_sequences
- supported_precisions

Two-tier memory support on compute device:
- tier0: on-device native memory (for example HBM).
- tier1: attached/remote memory mapping with added latency/bandwidth limits.

### 8.5 Memory Device Definition
Standalone memory devices can serve as capacity or bandwidth tiers for compute devices.

Required parameters:
- memory_device_id
- memory_type (DRAM/CXL/NVMe-backed cache/other)
- capacity_bytes
- read_bandwidth_bytes_per_s
- write_bandwidth_bytes_per_s
- access_latency_ms
- max_concurrent_transfers
- placement_scope (node-local or remote)

Optional parameters:
- bandwidth_sharing_policy
- eviction_policy for managed caches.

### 8.6 Node Definition
A node groups management and inference devices.

Required parameters:
- node_id
- management_device profile
- compute_devices list
- memory_devices list
- intra_node_topology
- intra_node_link_bandwidth_bytes_per_s
- intra_node_link_latency_ms

### 8.7 System Definition
The system contains one or more nodes and inter-node fabric.

Required parameters:
- system_id
- nodes
- inter_node_topology
- scale_up_link_bandwidth_bytes_per_s
- scale_up_link_latency_ms
- network_bisection_bandwidth_bytes_per_s
- routing_policy

The system configuration references concrete implemented component types and named instances. The repository must include sample JSONs for representative setups.

## 9. Simulation Behavior
### 9.1 Event-Based Engine
Simulation is event-based with dependency-aware scheduling.

Event examples:
- Compute forward step (prefill/decode micro-batch).
- KV movement between memory tiers.
- Expert weight fetch or migration.
- Tensor transfer between devices.
- Tool wait event (latency-only).

Event properties:
- event_id
- event_type
- requested_start_time_ms
- actual_start_time_ms
- end_time_ms
- dependencies
- resource_claims
- payload_size_bytes (for transfer events)
- workload/session attribution metadata

The engine computes event end times from model, device, and contention state. Ongoing events consume resources for their active interval and affect overlap behavior.

### 9.2 Randomness Model
Randomness applies multiplicative noise to expected event duration:

duration_ms = expected_duration_ms * (1 + randomness_scale * randn())

Requirements:
- randomness_scale is configurable globally and optionally by event type.
- Random seed must fully control reproducibility.
- Randomness can be disabled.
- Negative or near-zero durations are clamped to a minimum duration floor.

### 9.3 Resource Utilization Model
At minimum, track utilization for:
- Compute throughput capacity.
- Memory capacity occupancy.
- Memory bandwidth.
- Intra-node network bandwidth.
- Inter-node network bandwidth.

Resource contention rules must define whether bandwidth is shared fairly, priority-based, or weighted.

## 10. Orchestration Requirements
### 10.1 Concurrency Construction
Because traces are mostly single-session timelines, the simulator synthesizes multi-session concurrency using suite policy.

Baseline policy (v1):
- Eager grouping of every N conversations in suite order.
- Configurable batch window and max in-flight sessions.

### 10.2 Prefix Reuse and KV Rehydration
The orchestrator must:
- Detect reusable prompt prefixes across active sessions.
- Reuse compatible prefill results when valid.
- Retrieve prior-turn KV for multi-turn sessions when available.
- Account for misses, re-prefill cost, and movement overhead.

### 10.3 Data Movement Decisions
The orchestrator chooses when and where to place, prefetch, evict, and move KV, expert weights, and tensors. Policies must be pluggable so multiple heuristics can be compared.

### 10.4 Prefill-Decode Disaggregation
Optional mode in which prefill and decode may run on different devices.

v1 assignment policy:
- Eager assignment to device with highest expected throughput at dispatch time.

Must account for:
- Handoff latency and movement overhead.
- Queueing effects from assignment choices.

### 10.5 Attention-FFN Disaggregation
Not implemented in v1, but architecture must support later separation of attention and FFN execution onto different devices through extensible event and placement abstractions.

## 11. Runtime Parameters
Required runtime controls:
- random_seed
- randomness_scale
- simulation_start_time_ms
- simulation_end_condition (all workloads done or time limit)
- max_in_flight_workloads
- orchestration_policy_id
- placement_policy_id
- prefetch_policy_id
- enable_prefix_reuse
- enable_kv_rehydration
- enable_prefill_decode_disaggregation
- metrics_sampling_interval_ms
- event_log_level
- output_directory

Optional runtime controls:
- warmup_duration_ms
- cooldown_duration_ms
- device_failure_injection profile (future-facing)

## 12. Outputs
### 12.1 Standard Serving Metrics
The simulator must output AIperf-style metrics where applicable:
- TTFT
- TTFOT
- Output TPS per user
- Prefill TPS per user

Additional required metrics:
- Time to task completion
- End-to-end latency percentiles (P50/P90/P99)
- Throughput (sessions/s and tokens/s)
- Compute utilization (aggregate and per device)
- Memory capacity utilization (aggregate and per device)
- Memory bandwidth utilization
- Network utilization (intra-node and inter-node)
- Prefix reuse hit rate
- KV rehydration hit rate
- Data moved by type (KV/weights/tensors)
- Queueing delay by stage (prefill/decode/transfer)

### 12.2 Event Log
The simulator must emit a complete event log for offline analysis.

Event log requirements:
- Structured format (JSONL or Parquet).
- Stable schema versioning.
- One record per event with timestamps, resource attribution, and causality references.
- Optional compressed output.

### 12.3 Run Manifest
Each run must include a manifest containing:
- Input config digests.
- Runtime parameters.
- Simulator version.
- Random seed.
- Summary metrics pointers.

## 13. Visualization Requirements
Web-based visualization is required for interactive result analysis.

Minimum features:
1. Time-series graphs of compute, memory capacity, memory bandwidth, and network utilization.
2. Per-device drill-down views for hotspots and idle time.
3. Latency and throughput distributions (histogram and percentile curves).
4. Event timeline (Gantt-style) by workload and resource.
5. Data movement breakdown by type (KV/weights/tensors) and path.
6. Policy comparison view between two or more simulation runs.
7. Filters by workload subset, model, node, device, and event type.

## 14. Validation and Acceptance Criteria
### 14.1 Correctness Criteria
- Deterministic replay with fixed seed and randomness disabled.
- Conservation checks for data movement accounting.
- No resource over-allocation beyond configured capacities unless explicitly modeled.

### 14.2 Performance Criteria
- Must support at least 10k session traces per run in offline mode.
- Runtime should scale sub-quadratically with number of events for typical workloads.

### 14.3 Product Acceptance
The v1 PRD is considered satisfied when:
- All required entities and parameters are supported in config.
- Baseline orchestration and disaggregation modes run end-to-end.
- Required metrics and logs are generated with schema documentation.
- Visualization supports required minimum features.

## 15. Risks and Open Questions
- How to calibrate FLOPs/token and bandwidth efficiency for realism across hardware families.
- How sensitive conclusions are to trace quality and adapter normalization.
- How to model MoE expert locality shifts under changing concurrency.
- Whether fairness or tail-latency optimization should drive default orchestration policy.
- Which event log format should be default for scale: JSONL simplicity vs Parquet efficiency.

## 16. Roadmap
### v1
- Event engine, core entities, normalized trace ingestion, baseline orchestration, required metrics/logging, basic visualization.

### v1.1
- Additional orchestration heuristics, richer policy comparison, improved calibration workflow.

### v2 (future)
- Attention-FFN disaggregation.
- Failure modeling and resilience scenarios.
- Multi-tenant SLA-aware scheduling policies.
