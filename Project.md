# Overview
The purpose of this project is to ingest inference session traces and simulate workloads at the datacenter scale, extracting useful statistics to allow architectural exploration and discovery of bottlenecks.

# Elements
## Models
A model is an abstract representation of an LLM, which contains all the information required to allow to calculate the inference rate given the compute device description, and the type of parallelism chosen.
The model contains information about that allows calculating the required memory, bandwidth and compute to run a forward pass, which means the sizing of the tensors of each layer so that the compute and bandwidth load could be calculated.


### Supported Layers
- FFN - dense or MoE
- Attention: MHA, GQA, MLA, DSA
- Linear attention: Mamba V2

### Expert usage:
Per token expert usage will be modelled when generating work shards. It will be based on a statistical model where each expert remains live for expert_persistance consecutive tokens before being switched to a different expert, where expert_persistance is a random poisson variable and defined globally per model (default mean - 16, variance - 4)
For simplicity we will assume expert selection is the same across all the layers of a model. That will be a worse case in terms of weight movement bandwidth peak.

When a model is split across two memory tiers, the first tier holds the KV cache, the non-expert weights, and a working set of routed experts; the rest of the experts live in the second tier. Residency follows an LRU policy keyed on expert index (one index covers that expert across all MoE layers, since selection is shared). A group that touches an expert not currently resident incurs a transfer event moving it up; persistence keeps recently used experts resident so they are reused across decode steps. The first-tier expert capacity is derived from its memory minus the peak KV cache and the always-resident non-expert weights.

### Model loading and expert streaming (orchestration)
A real serving stack stages a model through host RAM before it can serve: the full model is read once from the shared input NVM into a **home node's** RAM (the chosen node whose `node_memory` can hold the whole model), then the resident weights are copied from that RAM onto each serving device. The simulator models this as two `weight_transfer` stages — NVM → home RAM (charged once per model for the life of the run) and home RAM → device — so the first batch of a freshly placed model waits for both. This staging applies to every model, dense or MoE.

For MoE models, only the **non-expert** weights are staged onto the device; the routed experts are streamed on demand as an **orchestration decision**. Each batch's expert activations are traced ahead of time; an LRU residency cache (sized to the batch's peak working set per expert-parallel rank) decides which experts are already on the device and which must be fetched. A miss emits an `expert_transfer` event (moving the absent experts up from the home RAM) and an `expert_load` decision; an LRU eviction to make room emits an `expert_eviction` decision. Because only the working set is pinned, the device reserves far less memory than full-expert residency would require — which is what lets very large MoE models fit at all.

When **no single node** can home the whole model (it is larger than any node's RAM), the model stays in the NVM: the device loads only its resident non-expert weights straight from the NVM, and routed experts stream from that same NVM rather than from a node's RAM. Dense models that cannot be homed fall back to the legacy single stage (NVM → each device of the whole resident footprint). A node that homes one or more models must hold all of their full weights at once for the life of the run; oversubscribing a node's RAM aborts the run with the same capacity check used for device memory.

### Model parameters

These are the minimal parameters required to size every tensor in a forward pass so that compute (FLOPs), parameter/KV memory, and bandwidth can be derived. Parameters are grouped by scope; the `Applies to` column indicates when a parameter is relevant.

#### Global

| Parameter | Symbol | Description | Applies to |
|---|---|---|---|
| `num_layers` | $L$ | Number of transformer/block layers. | All |
| `hidden_size` | $d_\text{model}$ | Residual stream / token embedding width. | All |
| `vocab_size` | $V$ | Vocabulary size; sizes the embedding and LM head matrices. | All |
| `tie_word_embeddings` | — | Whether the LM head shares weights with the input embedding (affects parameter memory). | All |
| `param_dtype_bytes` | $b_w$ | Bytes per stored weight element (e.g. 2 for fp16/bf16, 1 for fp8). Sizes parameter memory and weight-load bandwidth. | All |
| `kv_dtype_bytes` | $b_{kv}$ | Bytes per KV cache element. Sizes KV memory and KV transfer bandwidth. | Attention |
| `layer_pattern` | — | Per-layer assignment of block type (e.g. first $k$ dense then MoE, or interleaved attention/Mamba). Lets per-layer params vary across the stack. | All |

#### Attention block

| Parameter | Symbol | Description | Applies to |
|---|---|---|---|
| `attention_type` | — | Base attention mechanism: `MHA`, `GQA`, or `MLA`. | Attention |
| `num_query_heads` | $h_q$ | Number of query heads. | MHA, GQA, MLA |
| `num_kv_heads` | $h_{kv}$ | Number of key/value heads (= $h_q$ for MHA, < $h_q$ for GQA). Sizes KV cache. | MHA, GQA |
| `head_dim` | $d_h$ | Per-head dimension for Q/K/V. | MHA, GQA |
| `q_lora_rank` | $r_q$ | Rank of the down-projected query latent. | MLA |
| `kv_lora_rank` | $r_{kv}$ | Rank of the compressed KV latent (this is what is cached). | MLA |
| `qk_rope_head_dim` | $d_h^\text{rope}$ | Per-head dimension carrying rotary position info. | MLA |
| `qk_nope_head_dim` | $d_h^\text{nope}$ | Per-head dimension without rotary embedding. | MLA |
| `v_head_dim` | $d_h^v$ | Per-head value dimension. | MLA |
| `sliding_window` | $W$ | If set, each query attends only to the most recent $W$ tokens (local / sliding-window attention) instead of the full context. Caps KV cache size and attention compute for that layer; `null`/absent means full attention. Overlays any base `attention_type`. | Attention |
| `sparse_attention` | — | Whether a lightweight indexer selects a sparse subset of past tokens to attend to (DeepSeek Sparse Attention / DSA). Overlays any base `attention_type`. | Attention |
| `sparse_topk` | $k_\text{attn}$ | Number of past tokens the indexer selects per query (sparse attention budget). | DSA |
| `index_n_heads` | $h_\text{idx}$ | Number of heads in the indexer that scores tokens for selection. | DSA |
| `index_head_dim` | $d_\text{idx}$ | Per-head dimension of the indexer. | DSA |

#### FFN block

| Parameter | Symbol | Description | Applies to |
|---|---|---|---|
| `ffn_type` | — | `dense` or `MoE`. | FFN |
| `intermediate_size` | $d_\text{ff}$ | FFN hidden dimension (per routed expert for MoE). | FFN |
| `gated` | — | Whether the FFN is gated (e.g. SwiGLU → 3 weight matrices vs. 2 for an ungated activation like ReLU²). Affects FLOPs and weight memory. | FFN |
| `num_experts` | $E$ | Total number of routed experts. | MoE |
| `num_experts_per_token` | $k_E$ | Active experts per token (router top-k); drives MoE compute. | MoE |
| `num_shared_experts` | $E_s$ | Always-active shared experts. | MoE |
| `shared_expert_intermediate_size` | $d_\text{ff}^s$ | FFN hidden dimension of the shared expert(s) when it differs from the routed experts. | MoE |
| `moe_latent_size` | $d_\text{lat}$ | Latent width that tokens are down-projected to for routing and expert computation (LatentMoE). When set, adds hidden→latent and latent→hidden projections and experts operate at $d_\text{lat}$ instead of $d_\text{model}$. | MoE |

#### Mamba V2 (linear attention) block

| Parameter | Symbol | Description | Applies to |
|---|---|---|---|
| `mamba_d_state` | $N$ | SSM state dimension. | Mamba V2 |
| `mamba_d_conv` | $k_\text{conv}$ | Causal convolution kernel width. | Mamba V2 |
| `mamba_expand` | $e$ | Expansion factor for the inner dimension ($d_\text{inner} = e \cdot d_\text{model}$). | Mamba V2 |
| `mamba_num_heads` | $h_m$ | Number of SSM heads. | Mamba V2 |
| `mamba_head_dim` | $d_h^m$ | Per-head dimension of the SSM. | Mamba V2 |
| `mamba_n_groups` | $g$ | Number of B/C projection groups; sizes the input projection and causal conv ($d_\text{inner} + 2 g N$ channels). | Mamba V2 |

#### Multi-Token Prediction (MTP)

Optional speculative-decoding head appended after the main stack. Needed to model the extra per-step compute and the draft tokens it produces.

| Parameter | Symbol | Description | Applies to |
|---|---|---|---|
| `num_mtp_layers` | $L_\text{mtp}$ | Number of MTP modules (predicted future tokens per step). | MTP |
| `mtp_layer_pattern` | — | Block type(s) composing each MTP module (reuses the attention/FFN/Mamba block definitions above). | MTP |

## Batch Tracker
The batch tracker is responsible for tracking a batch of workloads and instantiating all the sequence trackers for each sequence.

## Sequence Tracker
The sequence tracker takes a conversation and tokenizes it. Conversations are tokenized using a standard tokenizer (just to get indices from the conversation string, not to actually use them as tokens) - only one tokenizer is ever used in the simulator, there is no need to model different tokenizers of different models.
All KV Cache Trackers are bounded to the sequence tracker so it provides an abstraction, and KV cache trackers only need to track the index within a sequence.
Two seperate sequence trackers can find the common prefix between sequences, and create the links between the KV cache trackers accordingly.
The sequence tracker can also report on wait for tool events which derive from tool calling.

## KV Cache Tracker
The KV cache tracker element tracks KV cache placement (we will have a seperate tracker per model, per convesation, per layer).
When a conversation is initiated, all the tokens (input and output) are tracked via a KV cache tracker.
KV cache trackers are instantiated under a sequence tracker.
The KV cache tracker tracks whether a KV token was generated or not, and on which devices it is currently found (each token might be found on more than one device).
To allow for sharing KV cache across different sequences, KV cache trackers may track KV tokens to different KV cache trackers instead of to a physical device, in case a common prefix is identified for an already prefilled sequence.

## Model Weights Tracker
The model weights tracker, like the KV cache tracker, divides the model weights into all it's different layers, and for each layers the different expert FFNs, attention or MAMBA matrices, etc. Each of these is a weight shard.
As in the KV cache tracker, each weight shard may be stored on one or more devices.
Unlike the KV cache tracker, weight shards cannot have a state of "not generated" yet. They typically start off on an NVM memory device.
A batch can only run on a device set whose memory holds the relevant weight shards; serving several models concurrently means their weights are resident on different device sets at once, multiplying the weight memory footprint.

## Compute Devices
A compute device is a GPU or other inference oriented device.

The compute device is defined by the native compute performance, and native memory (e.g. HBM) volume and bandwidth. These parameters will be used to calculate the inference events duration and resource usage.
A compute event is an ask for the compute device to do a forward pass; the actual amount of work depends on the model, the parallelism and concurrency.
The output will of the event will be the expected duration of the event.
The expected duration will be affected by the amount of compute and bandwidth needed, vs. what is available, where availability of resources could be influenced by prior events not ending yet.

A compute device is tightly coupled to at least one memory device, which represents its first tier memory.
Optionally, it may be tied to a second memory device which represents its second tier memory.
A memory device may be connected to the compute device directly (representing memory within the same package), or via the inter-node CXL, or via the scale-up network.

A compute device has the following parameter:
- Nominal FP16 FLOPs

## Memory Devices
A memory device represents volatile or non volatile memory in the system. The memory device contains parts of the model weights and KV cache.
The memory device class tracks the information "stored" (no actual information is stored) on it and may return available space. 
Memory device does not "know" what parts of the model and what parts of the KV cache are stored on it. That is up to the respective trackers.
The memory device, depending on the type, has an intrinsic bandwidth which is the absolute limit to how fast data can be read from it (i.e., the aboslute maximum limit to the bandwidth allocated to compute or data transfer events related to it)

A memory device has the following parameter:
- Capacity
- Nominal unconstrained bandwidth

## Work shard generator
A work shard is an atomic piece of work executable by a single compute device.
Work shards are precursors to compute or data transfer events.
Example of work shards:
- Prefilling a chunk of C tokens in a batch of B (in a layer )
- Decoding a batch of B for decode (in a layer)
- Launching a new kernel

The work shard contains the total FLOPs (and type of FLOPs) and the total bytes read needed to perform the action, in addition to KV cache dependencies as pointers to the sequence, and model weights dependecies.

The work shard generator is instantiated in connection to a batch tracker. Per sequence tracker, it receives the following:
- Index of the last existing cache token (or None if no existing cache)
- Index of the last prefill token
- Index of the last decode token
And outputs all the relevant shards.

## System Level Definitions
The system is defined by the following parameters:
- Scale up network bandwidth (we will model point to point, we will not model network congestion at this stage)
- Scale up network latency
- In-node (CXL) bandwidth
- In-node (CXL) latency

### System Architecture
The simulated system architecture is defined by a JSON file which defines all the devices in the system.
We assume that all devices are connected to each via the scale up network (and we only model point to point connnections, ignoring for now network topology and bissection bandwidth)

Each device must be an instance of the devices defined in the configuration files.

One memory device (typically NVM) will be defined as the input device, where the weights for all models are stored at system init.
The JSON will have specific designation of this input device.

Data transfers can happen between any memory device and any other memory device via the scale up network (as we assume but do not model that each memory device is connected to a physical device that manages it)

Each device may optionally be linked to a node memory device, representing the memory device managed by the node's CPU. The communication to this memory is via CXL to devices inside the current node, or scale up network to devices outside the node.

## Workloads
A workload is a multi turn conversation with or without tool calling
https://huggingface.co/datasets/sammshen/lmcache-agentic-traces

## Test Suite
A test suite is a list of workloads that the system should simulate, each workload is mapped to a model.

### Randomized suite
A randomized test suite selects random workloads from the dataset and matches them to a random model from the list of models.
Each workload is also assigned an arrival time: the time at which its first turn is submitted to the system. Arrivals are generated by sampling inter-arrival gaps from a random model with a configurable mean and variance, then accumulating them (the first workload arrives at time 0). A zero variance yields deterministic, evenly spaced arrivals.
The JSON for configuring a randomized test suite contains:
1. Number of workloads in the test suite
2. List of models to choose from
3. Inter-arrival mean (arrival_interval_sec) and variance (arrival_variance_sec2)

### Directed suite
TBD - not implemented for now

## Event Generator
The event generator is the heart of the simulation.
It is tasked with generating simulation events to produce a single output batch, i.e. to convert one batch of work shards to events.
The main scope of it is to map the work shards to actual work and calculate the duration of the work.

The event generator gets a list of compute devices and memory devices needed to execute the sequence, as well as the actual batch tracker with its associated sequence and KV trackers. The event generator decides upon initiatization on the division of work (in case more then one compute device is chosen). It supports pipeline and expert parellilism and accepts it as a parameter of the simulation.
Expert parellelism is performed eagerly, with experts hosted on devices based on the best available knowledge at the time, and evicted by LRU.

The devices form a `pipeline_parallel x expert_parallel x tensor_parallel` grid. Pipeline parallelism partitions the layers across stages; for a single batch the stages of a group run sequentially (no pipeline overlap), so latency is the sum of the stage times. Expert parallelism partitions the routed experts across the `expert_parallel` devices of a stage (expert `e` is owned by rank `e mod expert_parallel`) and splits that stage's compute evenly across those ranks (balanced routing), so the ranks run concurrently and a balanced stage is `expert_parallel` times faster. Each rank keeps its own LRU residency of the experts it owns: when every device's first tier can hold the whole model there is no expert movement, and when the second tier is a single shared device (a system NVM) its bandwidth bounds the aggregate movement of all ranks. Tensor parallelism then shards *every* tensor and the KV cache across the `tensor_parallel` devices sitting under each `(stage, expert-rank)` cell, dividing both that cell's compute and its per-device memory footprint by `tensor_parallel`; so a stage's work spreads evenly over all `expert_parallel x tensor_parallel` of its devices and runs that many times faster. (Tensor parallelism is not combined with two-tier expert streaming.)

The number of devices must be devisable by the product of the parallelism options (for simplicity). An engine therefore occupies `pipeline_parallel x expert_parallel x tensor_parallel` devices.

The event generator can and should consolidate adjacent work shards if they run on the same device, so as to minimize thrashing of the simulation with events.

### Events generated:

#### Data transfer
Memory to memory data transfers generate data transfer events
Data transfer events cost latency + transfer time, where transfer time is the volume / bandwidth
Bandwidth is the minimum across both ends of the transfer and the link connecting them; latency is the maximum across the two ends and the link.
The link is the scale-up network when the two memory devices are in different nodes, and the in-node (CXL) link when they share a node (intra-package first-tier accesses use the memory's own bandwidth and incur no link latency).

#### Compute
Compute events take the maximum of compute-bound time and bandwidth-bound time
Bandwidth is the bandwidth of the 1st tier memory (using 2nd tier memory requires a transfer event to the 1st tier)
Compute is defined by the nominal capabilities of the device. If using a different "data type", the compute is scaled so that 8 bits is 2x faster than nominal, 4 bits is 4x faster, and FP32 is 2x slower.

#### Kernel launch
When a new kernel must be launched (as depicted by the approriate work shard), the device must wait the kernel_launch_latency.
This event models the wait.

#### Tool call
We do not model the event of computing a tool call, but we do model waiting for the tool call to complete.
Tool call events represents time that the sequence must wait before receiving the tool response.
Tool call times are provided by the dataset, but they may be globally scaled by the tool_calling_speedup system parameter.

### Event Rescaling
An event generator shall support rescaling of events in case the associated resource is loaded.
For example: a memory device that started with supporting a single compute event, which in a time step concurrent to the event needed to support another compute event, will need to be rescaled with half the bandwidth, but prorated for the time already passed.
Generally speaking, we assume compute and bandwidth is divided equally when devices support more than one workload.

### Event time randomization
In order to model system randomness, the total event time is scaled by a random scaling factor, such that event_time = calculated_time*(1+factor), where factor is distrubted uniformly between (-event_random_factor_range:event_random_factor_range)

# Simluation Flow
Simulation is performed by running a test suite through a system.
The simulator class is responsible for making the technical connections between all the different parts of the simulator.
The class uses abstraction and methods from the other classes to perform its work.
This includes:
1. Keeping track of inflight sequences. Instantiating and deleting event generators
2. Prefix comparisons between every incoming sequence and the KV of every sequence that has not yet been evicted (not only the previous turn of the same conversation, and not only sequences currently in flight). Keeping track of simulated KV cache, system-wide, to allow actual prefix reuse across conversations. See "Global KV cache" below.
3. Keeping track of device usage and making sure event generators are correctly aware of one another in order to rescale events.
4. Executing the orchestrator decision with regards to placement of batches onto hardware, and movement of batches and KV cache between devices.

Multiple batches may be in flight at the same time. Each batch is placed on a *device set* (its engine slot); the simulator tracks which set each in-flight batch occupies. Batches on disjoint device sets run independently, while batches whose sets overlap contend and are rescaled by the arbiter. Because different models cannot be batched together, distinct models always run as separate in-flight batches (on separate device sets, or time-sharing the same one).

The simulation is strictly event driven. The simulator maintains a time-ordered event queue and advances to the next event; there is no fixed time step. The order in which orchestration acts between events derives logically from the events themselves (an arrival, an event completion, or a window timeout is itself an event that may trigger an orchestration decision).

Because event generators come and go and the set of co-running events on a resource changes over time, the rescaling arbiter is not run once over a fixed set of jobs. Instead the simulator keeps the active events and re-solves the equal-share rescaling incrementally: whenever an event starts or finishes (or a generator is added or removed), the rates of all in-flight events sharing the affected resources are recomputed and prorated for the time already elapsed. The existing fluid (processor-sharing) arbiter logic is reused, but driven incrementally from the event queue rather than as a single batch solve.

## Batching
Incoming sequences that are ready to run are collected into a concurrency window before dispatch. Only sequences mapped to the same model can share a batch. A batch is dispatched when either the batch fills up (reaches the configured maximum batch size) or the window time elapses, whichever comes first. The window is therefore defined by two knobs: a maximum batch size and a maximum window duration. These knobs are per batch, not a datacenter-wide limit; several batches (e.g. one per model) may be in flight concurrently.

## Orchestration
Orchestration will generally be eager, mapping sequences to the devices which it expects will best execute them according to the information it has at the point in time where the decision is made.

For each batch the orchestrator selects a *device set* (engine slot), not just a parallelism degree. A batch may only be placed where its model's weights are already resident, or can be made resident by a weight-load transfer first; serving several models concurrently therefore replicates their non-expert weights across device sets, which bounds how many engines fit. Different models, since they cannot share a batch, occupy distinct device sets or time-share one.

The orchestration eagerly decides on prefill-decode dissagregation if enabled.
The orchestration eagerly decides on parallelism using a high-level roofline estimate: for the candidate device sets and parallelism degrees available at the time of issue, it estimates the batch's compute-bound and bandwidth-bound time (as the event generator would) and picks the option with the best expected outcome, subject to the device and memory-capacity constraints it knows about.

**Memory is a hard constraint.** A single batch is rejected up front if its per-device footprint does not fit the serving device's memory. Beyond that, the simulator also enforces the *aggregate* constraint: the reserved footprint (weights + KV) of all jobs concurrently resident on a device may never exceed the memory available to it (its first tier, plus any second tier it can spill into). If a run would ever oversubscribe a device, the simulation aborts with an informative out-of-memory error naming the device, its peak reserved footprint, its capacity and the oversubscription factor, rather than reporting a physically impossible occupancy. The fix is to reduce concurrency (`max_concurrency`/`max_batch_size`), increase the parallelism degree (or enable `auto_parallelism`) to shard the footprint across more devices, give the device a second memory tier, or use a smaller model.

When prefill-decode disaggregation is enabled (`allow_pdd`), the engine slots are split into two disjoint pools — a prefill pool and a decode pool — with the partition point set by `prefill_engine_fraction` (a fixed fraction for now; dynamic repartitioning is left as future work, since fixed partitions tend to be predictably suboptimal). A request is prefilled on the prefill pool; on completion its KV cache is moved to the decode pool as one modeled transfer of the prompt's KV bytes over the link between the two engines' memories, and decode begins only after that transfer completes. Prefill and decode batch independently, while max concurrency counts a sequence as in flight from prefill dispatch through decode completion.

Concretely, an engine slot is a *fixed device budget* (the parallelism degree); the search only chooses how to wire it into a `pipeline x expert` arrangement. Since a single batch has no pipeline overlap, the speed estimate favours expert parallelism (`time ~ max(compute, bandwidth) / expert_parallel`), while pipeline parallelism is what relieves memory: pipeline stages shard the layers, whereas expert parallelism shards only the routed experts and replicates the attention/dense/shared/LM-head weights and the KV cache across ranks. The search therefore takes the most expert-parallel arrangement that still fits each device's memory, reaching for more pipeline stages only when a batch would otherwise not fit. This search is opt-in (`auto_parallelism`); by default the configured pipeline/expert degrees are used as-is.

Tensor parallelism (`tensor_parallel`) is a third, always-fixed degree layered on top: the engine occupies `pipeline_parallel x expert_parallel x tensor_parallel` devices, and tensor parallelism shards every weight tensor and the KV cache across its ranks while splitting their compute, so it both speeds a batch up by `tensor_parallel` and divides the per-device footprint by the same factor (relieving dense-weight, expert and KV pressure alike — unlike expert parallelism, which only shards routed experts). It is applied verbatim whether or not `auto_parallelism` is set; the search re-factors only the `pipeline x expert` budget while `tensor_parallel` rides along through the footprint and time estimate.

### Global KV cache
The orchestrator keeps a single, system-wide record of every sequence's KV cache that has not yet been evicted, so prefixes can be reused across conversations (not merely between consecutive turns of one conversation) and the KV may live anywhere in the memory hierarchy (not only on the device that last served the sequence). This is enabled by `global_kv_cache` (default true); when off, the only reuse is the previous-turn-of-the-same-conversation heuristic.

**Where KV lives.** A compute device's first-tier memory (e.g. HBM) holds a sequence's KV only while that device is actively computing the sequence. Once a turn completes, its KV is moved off the device into a *floating* memory — any node's CPU-managed node memory — where it persists for later reuse. The system NVM (the weight input device) is never used to hold KV. The floating pool therefore spans every node memory in the system; an entry may be placed on, or migrated to, a node memory on a *different* node than the one that served it.

**Admission (prefix comparison).** When a sequence is about to be dispatched, the orchestrator compares it against every non-evicted entry of the same model and takes the longest message-aligned common prefix. That prefix length becomes the sequence's cached-token count (its prefill skips those tokens), and the matched entry is recorded as the reuse source (its conversation, turn and the floating memory holding it). The cached prefix is fetched from that floating memory into the serving device's first-tier memory as a modeled transfer over the appropriate link (in-node CXL, or the scale-up network across nodes); the prefill waits on that fetch, and the arbiter contends its bytes against all other in-flight work, so a slow KV link shows up as added time-to-first-token. This produces a `kv_reuse` decision (the logical reuse) and a `kv_transfer` decision (the physical fetch).

**Completion (offload/migration).** When a turn completes, its full context KV (prompt + generated tokens) is moved from the serving device into a floating memory as a modeled, arbiter-accounted transfer (a `kv_transfer` decision), freeing the device's first-tier memory for the next batch. The orchestrator places the entry on any floating memory that has room, migrating across nodes if necessary to avoid evicting a still-useful entry.

**Capacity and eviction.** Stored KV competes for floating-memory capacity with everything else resident there. As long as some floating memory has room the entry is kept; only when the whole floating pool is full does the orchestrator evict, choosing victims by least-recently-used at the granularity of a whole stored sequence (sequence-level resolution, for simplicity). Each eviction is a `kv_eviction` decision. A reuse refreshes an entry's recency, so hot prefixes survive while cold ones are reclaimed first.

### Model weight residency
A batch can only run on an engine slot (device set) whose first-tier memory already holds that model's weights. The orchestrator tracks, per slot, which model's weights are currently resident, and follows a simple residency/eviction policy:

- **Affinity.** When dispatching a batch, the orchestrator prefers a free slot that already hosts the batch's model, so back-to-back batches of the same model reuse the resident weights and pay no reload.
- **Load on (re)placement.** When a batch lands on a slot that does not currently host its model (an empty slot, or one last used by a different model), the model's full weight footprint is streamed from the system input NVM into each of the slot's devices as a modeled, arbiter-accounted transfer over the appropriate link; compute waits for that load to finish. This is a `weight_load` decision. The footprint is sized by the parallelism planner for the chosen pipeline × expert arrangement, and its bytes contend the input-NVM bandwidth against all other in-flight loads, so weight streaming can itself become the bottleneck.
- **Eviction (displacement).** A slot holds exactly one model's weights at a time, so placing a different model on a slot displaces the previous resident. That displacement is a `weight_eviction` decision naming the evicted model and the slot it left; the evicted model simply pays a fresh `weight_load` the next time it is dispatched there. Eviction is therefore implicit, single-model-per-slot, and driven by which slot the placement chooses — there is no capacity-aware multi-model weight cache per slot.
- This behaviour is gated on `model_weight_loading`; when it is off, weights are assumed pre-resident everywhere and no weight loads or evictions are charged or recorded.


### Strategy
The orchestration strategy is configured by a few knobs:
- Max batch size: the fundamental inference knob -- the largest batch a single engine slot forms and runs at once. This sets how wide each individual batch is.
- Max concurrency: the high-level orchestration requirement -- the cap on the *total* sequences in flight datacenter-wide across all in-flight batches and engine slots. When several engine slots are available, setting max batch size below max concurrency lets the remaining budget spill into additional concurrent batches on other slots; setting them equal funnels all in-flight work into a single batch on a single slot.
- Max window duration: the batching window (see above), applied per batch.
- Allow PDD: whether prefill-decode disaggregation may be used.
- Prefill engine fraction: when PDD is enabled, the share of engine slots devoted to prefill (the rest serve decode).
- Max parallelism degrees: caps on pipeline and expert parallelism the roofline search may choose from.
- Global KV cache: whether the system-wide, cross-conversation KV cache (with LRU eviction and KV migration across floating memories) is active.
- Model weight loading: whether model weights are streamed from the input NVM onto an engine slot on (re)placement (with per-slot single-model residency and displacement), or assumed pre-resident.

## List of simulation parameters (to appear in config.JSON)
- max_batch_size (default 8)
    the fundamental inference knob: the largest batch a single engine slot forms and runs at once (the window fill threshold)
- max_concurrency (default null)
    high-level orchestration cap on the total sequences kept in flight datacenter-wide across all batches/slots; null means unbounded (admit up to max_batch_size per batch). Lower max_batch_size than max_concurrency to spread concurrent batches across multiple engine slots.
- concurrency_window_sec (default 1)
    the max time the simulator waits incoming sequences before issuing a batch
- allow_pdd (default true)
    If true, prefill and decode run on disjoint engine pools with a KV-cache transfer between them; if false a single pool serves both phases.
- prefill_engine_fraction (default 0.5)
    Fraction of engine slots assigned to the prefill pool when PDD is enabled (the rest serve decode); clamped so each pool keeps at least one slot.
- auto_parallelism (default false)
    If true, the orchestrator treats the configured pipeline x expert degrees as a fixed engine size and, per batch, searches their factorizations for the fastest memory-feasible arrangement; if false the configured degrees are used as-is.
- max_parallelism (default 32)
    The maximum parallelism rank the orchestrator should explore
- prefill_chunk_size (default null)
    If set, prefill is split into chunks of this many tokens; null means prefill the whole prompt in one shard
- global_kv_cache (default true)
    If true, the orchestrator keeps a system-wide record of non-evicted KV, reuses the longest message-aligned prefix across conversations, offloads completed KV to floating (node) memories with arbiter-accounted transfers, and evicts by LRU at sequence granularity when the floating pool is full. If false, only the previous-turn-of-the-same-conversation reuse applies.

- expert_persistance_mean (default 16)    
- expert_persistance_var (default 4)
    The mean and variance of the number of tokens an expert persists (before it is replaced by another expert)
- event_random_factor_range (default 0.05)
    The maximum deviation factor of calculated event time due to randomization
- tool_calling_speedup (default 1)
    Global multiplier applied to every tool-call wait time from the dataset (e.g. 2 makes tool calls return twice as fast)
- random_seed (default null)
    Seed for all simulation randomness (expert persistence, suite arrivals, event-time randomization); null draws a non-deterministic seed. A fixed seed makes a run fully reproducible.
- arrival_interval_sec (default 0)
    Mean of the randomized inter-arrival gap between successive workloads (the first arrives at time 0); 0 admits the whole suite at once.
- arrival_variance_sec2 (default 0)
    Variance (in seconds squared) of the inter-arrival gap. Gaps are drawn from a Gamma distribution with the configured mean and variance, so 0 gives deterministic, evenly spaced arrivals and a variance equal to arrival_interval_sec squared gives a Poisson arrival process (rate 1/mean). Draws are governed by random_seed.
- max_turns_per_workload (default null)
    Cap on how many conversation turns of each workload are issued as requests; null issues every turn.
- model_weight_loading (default true for config-driven runs)
    If true, a model's weights are streamed from the input NVM onto an engine slot the first time it is placed there (and reloaded if the slot is later repurposed to another model), as a modeled transfer that contends input-NVM bandwidth; if false, weights are assumed pre-resident and no load is charged.
- report_time_buckets (default 64)
    Number of time buckets in the per-device timeline output.

The config also names the inputs to load: `system` (system-architecture JSON), `models_dir` (folder of model JSONs, keyed by suite model name), `tokenizer` (`tiktoken` or `whitespace`), `suite` (an inline suite config or a path to one), optional `dataset` (the workload source), `run_id`, and `output_root`.

The optional `dataset` block selects the workload source: `dataset`/`config`/`split` (Hugging Face datasets-server coordinates) and `cache_dir` (a local cache directory, resolved relative to the config; default `Dataset`). If a populated cache exists there the run reads from it (offline and reproducible); otherwise it falls back to the live API. Set `dataset.require_cache` to fail fast instead of going to the network.

Note: `kernel_launch_latency` is a per-device property defined in the device/system architecture JSON, not a global config parameter; likewise the scale-up / CXL bandwidth and latency are system-architecture parameters.


# Outputs and Visualization

## Running
The simulator is driven from a single config JSON: `python run_sim.py Configs/example.json` (or `python -m serve_sim Configs/example.json`). Paths inside the config are resolved relative to the config file. The runner loads the system and models, builds the suite (drawing workloads through the dataset loader), issues one request per conversation turn, runs the simulation, and writes all outputs under `<output_root>/<run_id>/`. `--output-root`, `--run-id` and `--tokenizer` override the config.

As the run progresses it reports — refreshed in place — how many of the suite's sequences have completed, the elapsed simulation time and the elapsed wall-clock time. Pass `--quiet` to suppress these updates.

## Dataset cache
To avoid hitting (and being rate-limited by) the live Hugging Face datasets-server on every run, the source dataset is cached locally under `./Dataset/`. Populate it once with `python cache_dataset.py` (use `--max-rows N` for a smaller, faster cache; `--overwrite` to refresh). The cache is a deterministic prefix of the split (`rows.jsonl` + `meta.json`), so running the same command on another machine reproduces the same cache; the data itself is gitignored. A run automatically prefers the cache when present and falls back to the live API otherwise.

## Raw data and textual analysis
The simulator produces a run report aggregated over the test suite.
- Count of requests, batches ran, memory used and total DMA transfers
- Per-request latency, time-to-first-token (TTFT), and time-per-output-token (TPOT). TTFT is the time the request's first decode step completes (after prefill and any KV transfer) relative to arrival; TPOT is the remaining decode time divided by the remaining output tokens.
- Throughput (requests and tokens per second) and overall makespan.
- Per-device utilization (compute and bandwidth) and memory occupancy over time. Memory occupancy is the reserved per-device footprint (weights + KV) of the jobs active at each instant, as sized by the parallelism planner -- a reservation estimate, not a byte-accurate residency trace. This occupancy can never exceed a device's memory: a run that would oversubscribe any device aborts with an out-of-memory error (see "Memory is a hard constraint" above) instead of producing a report.
- Per-device execution-state breakdown, a finer view than busy/idle. At each instant a compute device is attributed to exactly one of: **compute-bound** (running a forward pass limited by FLOPs), **bandwidth-bound** (running a forward pass limited by memory bandwidth), **waiting for KV** (stalled fetching KV cache), **waiting for weights** (stalled staging non-expert model weights), **waiting for experts** (stalled streaming routed MoE experts), **kernel-launch** (launch-latency overhead), or **idle** (no work assigned). The states partition the run, so their fractions sum to one. They are reported both as run-level aggregates (in `device_summary.csv`) and bucketed over time (in `device_timeline.csv`, for later visualization). When events overlap on a device (e.g. a transfer prefetching while a forward pass runs) the higher-priority state -- compute over waiting -- is charged for that interval. (A "waiting for tensors" state covering inter-stage activations / tensor-parallel collectives is not yet modelled as events and so is not reported.)
- Raw list of all simulation events in CSV format. Rescaled events appear seperately before and after rescaling.
- A high level list of all orchestration decisions: model-weight load, model-weight eviction, prefill, KV reuse, KV transfer, decode, KV eviction. Each decision is described with the mapped to device(s), the model, the sequence identifier (unique workload ID and turn number), the second sequence ID and device(s) (in case of KV reuse or KV transfer), and the source memory (the input NVM for a weight load). A weight eviction names the displaced model and the slot it leaves. Alongside the time the decision was made, each row also records the execution window in which it was carried out -- a start time and a completed time taken from the rescaled events that realise it -- so the log doubles as a timeline of when each decision actually ran. Bookkeeping acts that have no corresponding compute or transfer (such as evictions) report the same value for both.

These are written as `run_report.json`/`run_report.txt`, `requests.csv`, `orchestration_decisions.csv`, `device_summary.csv`, `device_timeline.csv`, `events_before_rescaling.csv` and `events_after_rescaling.csv`, alongside an echo of the input `config.json`.

## Visualization tool
A web based visualization tool shall show the results of the run.
It will be implemented via FLASK and have a seperate entry point then the simulator runtime.

The tools will have a tabbed webpage appearance. Each tab shall be vertically scrollable to show all the content contained within.

The visualtion tool will display numbers in concise easy to read notation, using no more than 3 integer digits + no more than one decimal and a letter to denote the magnitude for example:
2.4P, 10T, 1.2G, 128M, 10.1K, 12m, 75.4u, 1.2n

The following tabs shall exist:

### Summary Tab
This will display an AI-Perf style summary of the run with detailed tables showing aggregate values for the run, workload, system, and simulated performance.

### Timeline Tab
The timeline shall show a set of graphs all with the same horizontal axis of time.

The graphs will display in a matrix of 1,2,3, or 4 columns (a selector at the top of the tab chooses the matrix configuration) and as many lines as needed, at the order specified in the list below (filling row and then column), however it shall be possible to rearrange graphs, by dragging and dropping (the graph shall pin to a location in the display matrix and shift elements to other pinned locations)

Graphs that display discrete object shall display as bars, choose different colors for different states, abbreviations on the graph and detailed text on mouse hover.
When a state does not exist (example - transfer source when no transfer is happening) - the graph shall be empty in that range.
When a state represents the same object on different graphs (a device ID, a sequence ID, etc.) it shall be the same color.

Graphs that represent values shall have absolute values on the left, and relative values on the right. The horizontal line of max value shall be displayed, but not used to autoscale the graph (so for example if the whole graph is in 0-0.01 max range, the line will not be visible)

The tab will also allow scaling of the horizontal (time) axis by a couple of sliders at the top of the tab. This will affect all the graphs in the tab the same way. The default display is the full simulation span.

#### Graph selection section
On the left there will be a panel that shows a hierarchical list (arranged as below, the lowest level being the individual devices and workloads) of the graphs displayed and allows by clicking to remove or enable a graph.
The list is compacted, and each part is expandable by clicking on a "+" next to it or contractable by clicking on a "-".
Clicking on the actual name of the graph hides it or displays it.
Clicking on the name of the hierarchy displays or hides all the graphs under it (overriding anything done inside the hierarchy)
The section is seperately vertically scrollable and (if needed) also horizontally (in case the names don't fit)
The section takes 20% of the screen width.

The default for all graphs is displayed.

#### List of timeline graphs
1. For each compute Device:
1.1 Compute (with both absolute and relative to max)
1.2 Bandwidth used of the 1st tier device (with both absolute and relative to max)
1.3 Capacity of the 1st tier device (with both absolute and relative to max) - broken down by content - KV and weights
1.4 Reason idle (the most dominant of the reasons in the reporting section above)
1.5 Transfer source (the device from which an incoming transfer is happening)
1.6 Transfer object (KV of which sequence, weights)
1.7 Current task batch size (0 if device is idle)
1.8 Effective device output token throughput (total tokens per second of current task) - if device is a rank in a parallelism group, this is the effective throughput of the part of the token generation the device is responsible for. When not decoding this is 0
1.9 Effective device input token throughput, similar to 1.8 but for prefill; when not prefilling, this is 0.
2. For each independent memory device:
2.1 Bandwidth used (with both absolute and relative to max)
2.2 Capacity used (with both absolute and relative to max) - broken down by content - KV and weights
2.3 Transfer source (the device from which an incoming transfer is happening)
2.4 Transfer object (KV of which sequence, weights)
3. For each workload:
3.1 Device computing workload
3.2 Current turn
3.3 State (of current turn sequence): Not arrived, In queue, KV Fetch, Prefill, Decode, Done

### Running the tool
Each run writes a self-contained `viz.json` into its output directory (alongside
the CSVs). The visualization tool is launched separately from the simulator:

    python run_viz.py                      # serves the latest run under Outputs/
    python run_viz.py Outputs/<run-dir>    # serves a specific run
    python run_viz.py Outputs/<run-dir> --port 8000

All derivation (summary tables and the declarative timeline-graph descriptors)
is done in Python (`serve_sim.viz.graphs.build_view_model`); the browser only
renders that view model, so the entire visualization is described, and testable,
textually.





 