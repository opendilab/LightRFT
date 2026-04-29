<div align="center">

# ORM RL Demo

Complete ORM trajectory-scoring RL training demo based on Geo3K.

</div>

## Overview

This demo shows the full pipeline of using an ORM to score trajectories for RL training:
- dataset: Geo3K
- actor: Qwen2.5-VL 7B model
- reward: one general outcome reward model combined with rule-based accuracy reward and format reward, all contributing to the GRPO loss
- training engine: FSDP, inference engine: SGLang

The actor generates Geo3K trajectories, the general ORM scores them, and the scores are combined with a rule-based accuracy reward (`accuracy_reward`) and a format reward (`format_reward`) to compute the final GRPO loss. To avoid rewriting the Geo3K dataset files, the demo overrides the dataset label to `geo3k_general` at runtime so the original dataset path can be reused while routing through the general ORM reward mix.

Environment requirements stay aligned with the repository-level [README.md](../../README.md). Refer to the main project document.

## Project Structure

```text
orm_rl_demo/
├── train_colocate.py
├── reward_models.py
├── reward_models_utils.py
├── test_reward_models.py
└── run_general_fsdp_qwenvl.sh
```

## Quick Start

Set the data and model paths, then run the entry script:

```bash
export DATA_PATH=/path/to/geo3k
export PRETRAIN_PATH=/path/to/Qwen2.5-VL-7B-Instruct
export REWARD_PRETRAIN_PATHS='{"general":"/path/to/general-reward-model"}'
bash examples/orm_rl_demo/run_general_fsdp_qwenvl.sh
```

Set the dataset and model paths via the environment variables above before running.

## Results

### Experiment Setup

This demo has been validated with one real 2-GPU full training run (W&B: [ORM-RL-Demo-QwenVL-7B-Geo3K](https://wandb.ai/hansbug/ORM-RL-Demo-QwenVL-7B-Geo3K/runs/zrekazyw)):

| Item | Value |
| --- | --- |
| Actor | Qwen2.5-VL-7B-Instruct |
| General RM | Qwen2.5-VL-7B general reward model |
| Dataset | Geo3K |
| Training engine | FSDP |
| Inference engine | SGLang (`rm_use_engine=True`) |
| Reward mixing | `format_reward × 0.1 + general_model_reward × 0.2 + accuracy_reward × 0.7` |
| Batch sizes | `train_batch_size=128`, `rollout_batch_size=128` |
| Sampling | `n_samples_per_prompt=8`, `num_episodes=20` |
| Sequence length | `prompt_max_len=1024`, `generate_max_len=2048` |
| Optimizer / KL | `actor_learning_rate=1e-6`, `init_kl_coef=0.001`, `lr_warmup_ratio=0.03` |

Three reward components — rule-based accuracy reward (`accuracy_reward`), ORM scoring (`general_model_reward`, coefficient 0.2), and format reward (`format_reward`) — are combined with the weights above and together compute the final GRPO loss. The `general_model_reward` values shown (e.g. `0.2`) reflect the ORM output (range 0.0 / 0.5 / 1.0) multiplied by the 0.2 coefficient, not the raw model score.

### Curve Results

The run completed successfully (`train/global_step=320`, 16 eval passes):
- `eval/reward_mean` improved from `0.4636` to `0.5679`
- Best `eval/reward_mean=0.5686` at `train_step=260`
- Final `eval/accuracy_reward_mean=0.5166`, `eval/format_reward_mean=0.9956`, `eval/general_model_reward_mean=0.1067`

![](assets/exp_20260417/summary_card.png)

![](assets/exp_20260417/reward_dashboard.png)

![](assets/exp_20260417/optimization_dashboard.png)

### Case Study

Between step 80 and step 320, two question stems appear in both saved trajectories. The following shows the same two questions compared across early and late training.

#### Question A: Parallelogram Area

![](assets/exp_20260417/question_a_step80.png)

![](assets/exp_20260417/question_a_step320.png)

- Step 80 rewards: `total=0.3`, `format=1.0`, `accuracy=0.0`, `general_model=0.2`, `rule=0.1`
- Step 320 rewards: `total=1.0`, `format=1.0`, `accuracy=1.0`, `general_model=0.2`, `rule=0.8`
- The actor already produced a close answer at step 80 so the ORM scored it near 1.0 (contributing 0.2 after the 0.2 coefficient); by step 320 the output moved from `38.97` to the rule-matching `39.0`, flipping `accuracy_reward` from `0.0` to `1.0`.

#### Question B: Tangent Geometry `y`

![](assets/exp_20260417/question_b_step80.png)

![](assets/exp_20260417/question_b_step320.png)

- Step 80 rewards: `total=0.1`, `format=1.0`, `accuracy=0.0`, `general_model=0.0`, `rule=0.1`
- Step 320 rewards: `total=1.0`, `format=1.0`, `accuracy=1.0`, `general_model=0.2`, `rule=0.8`
- At step 80 only format was preserved while both accuracy and ORM failed to reward the answer; by step 320 both became positive contributions.

## License

This project is licensed under the Apache 2.0 License. See [LICENSE](../../LICENSE) for details.
