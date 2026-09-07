# Runtime Architecture and Resource Reuse

This document explains the implementation boundaries behind LightRFT's `torchrun`/SPMD runtime, Strategy abstraction, model roles, colocated execution, memory lifecycle, and Actor-to-engine weight synchronization. It describes the current repository; low-level vLLM and SGLang memory behavior may vary by installed version.

## 1. Component responsibilities

```text
training entry point
        │
        ▼
PPO Trainer ── training loop, evaluation, logging, checkpoints
        │
        ├── FastExperienceMaker
        │       ├── rollout engine: response generation
        │       ├── Actor: behavior log-probabilities
        │       ├── Reference Model: reference log-probabilities
        │       ├── Critic: values, when required
        │       └── RewardComputationEngine: reward computation and aggregation
        │
        └── Strategy
                ├── DeepSpeed or FSDP v2 training operations
                └── vLLM or SGLang runtime integration
```

| Component | Responsibility | Explicit non-responsibility |
| --- | --- | --- |
| `Strategy` | Distributed setup, backend wrapping, collectives, optimization primitives, checkpoints, engine lifecycle, and weight broadcast | Does not define task correctness or reward semantics |
| `PPOTrainer` / `SPMDPPOTrainerVL` | Rollout/update loop, evaluation, logging, and checkpoints | Does not implement ZeRO/FSDP collectives |
| `FastExperienceMaker` | Runs model roles and constructs `Experience` objects | Does not implement parameter sharding |
| `RewardComputationEngine` | Calls local/remote reward models or custom reward functions and aggregates scores | Does not update policy parameters |
| vLLM/SGLang engine | High-throughput generation with a rollout copy of Actor weights | Is not the differentiable training Actor |

The Strategy layer supports execution of training, rollout, and evaluation phases, but it does not control evaluation semantics. Evaluation is orchestrated by the Trainer; answer verification and reward definitions belong to reward functions, reward models, and `RewardComputationEngine`.

## 2. Native `torchrun` and SPMD

A typical job starts as follows:

```bash
torchrun --nproc_per_node 8 examples/gsm8k_geo3k/train_colocate.py ...
```

`torchrun` creates Python processes and supplies `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, and `MASTER_PORT`. Normally each process binds to one GPU and executes the same program over different data. FSDP and ZeRO-3 configurations may also give each rank a different parameter shard.

The FSDP path initializes `torch.distributed` directly. The DeepSpeed path calls `deepspeed.init_distributed`, which still establishes PyTorch process groups. vLLM uses `external_launcher`; SGLang reuses the external distributed environment. Ray is therefore not required for global process scheduling.

SPMD means Single Program, Multiple Data: all ranks follow compatible control flow while operating on their local data and parameter state. With eight ranks and `engine_tp_size=2`, LightRFT creates groups equivalent to:

```text
rollout TP groups: [0, 1] [2, 3] [4, 5] [6, 7]
orthogonal DP groups: [0, 2, 4, 6] [1, 3, 5, 7]
```

Within each rollout TP group, `gather_and_generate` gathers local inputs, performs TP generation, and slices the replicated output so each rank receives its original share. Collective calls must occur in the same order on every participating rank; divergent control flow can cause a hang.

## 3. Strategy selection and boundaries

`get_strategy(args)` returns `FSDPV2Strategy` when `--fsdp` is set and `DeepspeedStrategy` otherwise. The shared interface covers:

- distributed setup, data samplers, and process groups;
- model and optimizer preparation;
- backward, optimizer step, clipping, and scheduler interaction;
- checkpoint and model persistence;
- rollout-engine creation, generation, sleep, and wake-up;
- Actor-to-engine weight synchronization;
- explicit FSDP model and optimizer CPU/GPU movement.

This interface isolates backend mechanics. It does not imply that every method has identical memory semantics in DeepSpeed and FSDP, or that every backend/LoRA/engine combination is implemented.

## 4. Actor, rollout policy, and Reference Model

### Actor representations

The same logical policy has two runtime representations:

- the training Actor, a Hugging Face model wrapped by DeepSpeed or FSDP for forward/backward and optimization;
- the rollout policy, a vLLM or SGLang representation used only for generation.

They are different Python objects and do not share parameters automatically. After optimization, LightRFT explicitly sends updated Actor parameters to the rollout engine.

### Reference Model

The Reference Model is usually called `initial_model` in source. When KL regularization is enabled, it starts from the same pretrained/SFT checkpoint as the Actor and remains frozen:

```text
initial checkpoint
    ├── Actor: trainable
    └── Reference Model: frozen KL anchor
```

The Reference Model supplies `log π_ref(a|s)` while the Actor supplies `log π_θ(a|s)`. Their difference constrains policy drift through a reward penalty or an explicit KL loss. Current examples omit the Reference Model when `init_kl_coef == 0`.

The PPO old policy is different. LightRFT stores update-time behavior log-probabilities in each `Experience`; later epochs compare the current Actor against those stored values:

```text
ratio = exp(log π_θ(a|s) - log π_old(a|s))
```

The stored old probability is short-lived PPO state; the frozen Reference Model is a longer-lived regularization anchor.

## 5. Colocated execution and memory lifecycle

Colocation means that the training Actor, local Reference/Critic/Reward models, and rollout engine use the same GPU/rank set in different phases. It is logical co-location and time sharing, not simultaneous execution of all models and not a guarantee that every weight leaves GPU memory.

| Operation | Target | Current meaning |
| --- | --- | --- |
| engine `sleep` / `wake_up` | vLLM/SGLang | Release or restore engine-managed memory; the exact released set is backend-dependent |
| model `offload_model` / `reload_model` | FSDP v2 PyTorch model | Move parameters and buffers between CPU and the current CUDA device |
| optimizer offload/load | FSDP v2 optimizer | Move optimizer state at phase boundaries when `--adam_offload` is enabled |

“Reload” here means CPU-to-GPU movement, not reading a checkpoint from disk.

### Engine-specific sleep behavior

LightRFT calls parameterless `sleep()` and `wake_up()` on both engines:

- for vLLM, LightRFT does not set a sleep level; the installed vLLM version's default determines which memory categories are released;
- for SGLang, the LightRFT wrapper defaults to `release_weights=False`, releasing KV-cache and CUDA-Graph memory while retaining rollout weights.

It is therefore inaccurate to claim that both engines always offload model weights during sleep.

### Model-role execution order

The fast experience path batches work by role:

1. Actor log-probabilities;
2. Reference log-probabilities, with explicit FSDP reload/offload when present;
3. Critic values, optionally managed by FSDP `CPUOffloadPolicy`;
4. reward computation.

For multiple local PyTorch reward models, the current implementation reloads all models first, then computes them sequentially and offloads each model after use. This shortens post-compute residency but peak memory can still contain multiple reloaded reward models.

Explicit `reload_model`/`offload_model` is implemented by `FSDPV2Strategy`. DeepSpeed primarily relies on ZeRO and its configured offload behavior; do not assume the FSDP manual-movement path applies to every DeepSpeed/local-RM combination.

The replay buffer can keep experience tensors on CPU. FSDP can also offload Adam state between training phases. These mechanisms trade GPU memory for host memory and transfer latency.

## 6. Weight synchronization

`Strategy.update_engine_weights(actor)` performs the following work:

1. wakes the engine when necessary;
2. creates or reuses `BroadcastManager`;
3. reconstructs current Actor parameters from the training backend;
4. maps parameter names and sends tensors to the selected engine;
5. synchronizes ranks and clears unused CUDA cache.

DeepSpeed ZeRO-3 uses `deepspeed.zero.GatheredParameters`; FSDP v2 reconstructs `DTensor` values with `full_tensor()`. LoRA weights are merged according to the supported backend/engine path. Source code contains explicit `NotImplementedError` checks for unsupported vLLM + LoRA synchronization combinations.

Wake-up and weight synchronization are distinct: `wake_up()` restores engine availability, whereas `BroadcastManager.broadcast_to_engine()` installs the latest Actor weights.

## 7. End-to-end iteration

```text
initialization
  ├─ torchrun creates ranks and Strategy creates process groups
  ├─ DeepSpeed/FSDP wraps training models
  ├─ rollout engine is created
  ├─ Actor weights are synchronized once
  └─ engine sleeps when enabled

iteration
  1. wake rollout engine
  2. generate with vLLM/SGLang
  3. sleep engine when enabled
  4. compute Actor behavior log-probabilities
  5. compute Reference probabilities when enabled
  6. compute Critic values when required
  7. compute and aggregate rewards
  8. construct Experience and optionally store it on CPU
  9. update Actor and optional Critic
 10. broadcast updated Actor weights to the engine
```

Evaluation uses the same generation and reward path and constructs Experiences for metrics, but does not train on them or update model parameters.

## 8. Operational checks

- A hang usually indicates that ranks entered collectives in different orders or that `WORLD_SIZE % engine_tp_size != 0`.
- Stale rollout behavior usually indicates a missing `update_engine_weights(actor)` call; wake-up alone is insufficient.
- Lower-than-expected memory release requires distinguishing weights, KV cache, CUDA Graphs, optimizer state, and replay-buffer tensors.
- `torch.cuda.empty_cache()` releases only unused allocator cache, not live tensors.
- Colocation is not an unconditional throughput guarantee; results depend on model size, sequence length, interconnect, host memory, and the rollout/training time ratio.

## 9. Source map

- `lightrft/strategy/strategy.py`: Strategy factory
- `lightrft/strategy/strategy_base.py`: distributed setup and engine lifecycle
- `lightrft/strategy/fsdp/fsdpv2.py`: FSDP model/optimizer movement
- `lightrft/strategy/deepspeed/deepspeed.py`: DeepSpeed backend
- `lightrft/strategy/utils/distributed_util.py`: TP/DP groups and input gathering
- `lightrft/strategy/utils/broadcast_utils.py`: Actor-to-engine weight broadcast
- `lightrft/strategy/sglang_utils/sglang_engine.py`: SGLang memory release/resume
- `lightrft/strategy/vllm_utils/`: vLLM external-launcher integration
- `lightrft/trainer/fast_exp_maker.py`: model-role ordering and reward computation
- `lightrft/trainer/spmd_ppo_trainer.py`: optimization and engine update loop

See also the [Strategy guide](strategy.rst) and [Strategy design](strategy_design_philosophy.md).
