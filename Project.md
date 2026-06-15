# Overview
The purpose of this project is to ingest inference session traces and simulate workloads at the datacenter scale, extracting useful statistics to allow architectural exploration and discovery of bottlenecks.

# Elements
## Models
A model is an abstract representation of an LLM, which contains all the information required to allow to calculate the inference rate given the compute device description, and the type of parallelism chosen.
The model contains information about that allows calculating the required memory, bandwidth and compute to run a forward pass, which means the sizing of the tensors of each layer so that the compute and bandwidth load could be calculated.

Models will support:
- FFN - dense or MoE
- Attention: MHA, GQA, MLA, DSA
- Linear attention: Mamba V2

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

The work shard contains the total FLOPs (and type of FLOPs) and the total bytes read needed to perform the action, in addition to KV cache dependencies as pointers to the sequence, and model weights dependecies.

The work shard generator is instantiated in connection to a batch tracker. Per sequence tracker, it receives the following:
- Index of the last existing cache token (or None if no existing cache)
- Index of the last prefill token
- Index of the last decode token
And outputs all the relevant shards.

## Node
A node contains a management device (CPU), and a number of inference devices, which could be the identical or different from each other.

The node is characterized by compute devices (with linked memory devices), optionally free-floating memory devices.

The node accepts the following parameter:
- CXL Bandwidth (we will model point to point bandwidth)

## System Level Definitions
The system is defined by the following parameter:
- Scale up network bandwidth (we will model point to point, we will not model network congestion at this stage)
- Scale up network latency

## Workloads
A workload is a multi turn conversation with or without tool calling
https://huggingface.co/datasets/sammshen/lmcache-agentic-traces

## Test Suite
A test suite is a list of workloads that the system should simulate, each workload is mapped to a model.

## Event Generator
The event generator is the heart of the simulation.
It is tasked with generating simulation events to produce a single output batch, i.e. to convert one batch of work shards to events.
The main scope of it is to map the work shards to actual work and calculate the duration of the work.

The event generator gets a list of compute devices and memory devices needed to execute the sequence, as well as the actual batch tracker with its associated sequence and KV trackers. The event generator decides upon initiatization on the division of work (in case more then one compute device is chosen). It supports pipeline and expert parellilism and accepts it as a parameter of the simulation.

The number of devices must be devisable by the product of the parallelism options (for simplicity).

The event generator can and should consolidate adjacent work shards if they run on the same device, so as to minimize thrashing of the simulation with events.

Events generated:

### Data transfer
Memory to memory data transfers generate data transfer events
Data transfer events cost latency + transfer time, where transfer time is the volume / bandwidth
Bandwidth is the minimum across both ends of the transfer, and latency is the maximum 

### Compute
Compute events take the maximum of compute-bound time and bandwidth-bound time
Bandwidth is the bandwidth of the 1st tier memory (using 2nd tier memory requires a transfer event to the 1st tier)
Compute is defined by the nominal capabilities of the device. If using a different "data type", the compute is scaled so that 8 bits is 2x faster than nominal, 4 bits is 4x faster, and FP32 is 2x slower.

### Tool call
We do not model the event of computing a tool call, but we do model waiting for the tool call to complete.
Tool call events represents time that the sequence must wait before receiving the tool response.
Tool call times are provided by the dataset, but they may be globally scaled by the tool_calling_speedup system parameter.

# Simluation Flow
1. Preprocessing:
Workloads -> Batch + Model -> Work Shards
(for all workloads and all turns - each turn is a different set of work shards)

2. Running:
- Choose sequences and map to devices, initiate event generators (Orchestration)
- Generate events to execute the chosen sequences (meaning, find the end time of each event)
- Repeat

3. Post processing:
Analyze and create logs and reports

## Orchestration
Orchestration will generally be eager, mapping sequences to the devices which it expects will best execute them according to the information it has at the point in time where the decision is made.

The orchestration eagerly decides on prefill-decode dissagregation if enabled.

Orchestration parameters:
- Target concurrency
- Allow PDD






 