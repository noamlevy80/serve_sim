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

## Workloads
A workload is a multi turn conversation with or without tool calling

https://huggingface.co/datasets/sammshen/lmcache-agentic-traces

In the simulator, we will 'run' several workloads in parallel to simulate a high concurrency system

## Test Suite
A test suite is a list of workloads that the system should simulate.

## Compute Devices
A compute device is a GPU or other inference oriented device.

The compute device is defined by the native compute performance, and native memory (e.g. HBM) volume and bandwidth. These parameters will be used to calculate the inference events duration and resource usage.
A compute event is an ask for the compute device to do a forward pass; the actual amount of work depends on the model, the parallelism and concurrency.
The output will of the event will be the expected duration of the event.
The expected duration will be affected by the amount of compute and bandwidth needed, vs. what is available, where availability of resources could be influenced by prior events not ending yet.



## Memory Devices
The system will support standalone memory devices, which are not necessarily part of an inference device, and are connected to the other devices in the node and in other nodes via the node interconnect or the scale up network. Memory devices are characterized by volume and bandwidth.
A device may use a memory device in lieu of its internal storage, and pay the extra bandwidth and latency

## Node
A node contains a management device (CPU), and a number of inference devices, which could be the identical or different from each other.
The node is charachetarized by it's internal latency and bandwidth between components (e.g. CXL)

## System
The full system contains a number of nodes, each could be different
The nodes are connected via a scale up network.
The system is characterized by the latency and bandwidth of this scale up network.

# Behavior
## Event based simulation
The simulation will model events. An event can be: calculating one time step of a batch on a device, transferring a chunk of KV cache between memory tiers

## Orchestration
As the workloads are traces of a single concurrency, the system must orchestrate concurrency to make the simulation interesting. We will take a greedy approach, where each N conversations are grouped at once, according to the order they appeared in the test suite.

The system should identify common prefixes and attempt to reuse them as much as possible to avoid unneccesary prefill computations.
In a multi turn conversation, the system should try to fetch the existing KV cache conversation from memory and prefill the previous turns.
The heart of the simulation will be to capture how efficient this process can be.

## Randomness




 