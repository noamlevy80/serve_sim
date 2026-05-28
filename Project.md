# Overview
The purpose of this project is to ingest inference session traces and simulate workloads at the datacenter scale, extracting useful statistics to allow architectural exploration and discovery of bottlenecks.

# Elements
## Models
A model is an abstract representation of an LLM, which contains all the information required to allow to calculate the inference rate given the compute device description, and the type of parallelism chosen.
The model contains information about that allows calculating the required memory, bandwidth and compute to run prefill and decode.
We assume that all supported models are auto regressive (ignore linear models for now), and that all are MoE.

In order to model expert weights movement, we will use a statistical model of expert usage per output token. Each expert will have a relative affinity score, and when modelling inference each expert will or will not be selected by a random draw, having a relative probability matching its score.

A model will have a default normal distribution of scores between experts, so that %20 of the experts are within 1 standard deviation.

## Workloads
A workload is a multi turn conversation with or without tool calling

https://huggingface.co/datasets/sammshen/lmcache-agentic-traces

In the simulator, we will 'run' several workloads in parallel to simulate a high concurrency system

## Test Suite
A test suite is a list of workloads that the system should simulate.

## Compute Devices
A compute device is a GPU, CPU, TPU, etc.
The compute device is defined by the native compute performance, and native memory (e.g. HBM), that includes the volume and bandwidth. These parameters will be used to calculate the inference events duration and resource usage.
A compute event is an ask for the compute device to do a forward pass; the actual amount of works depends on the model, the parallelism and concurrency.
The output will of the event will be the expected 

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




 