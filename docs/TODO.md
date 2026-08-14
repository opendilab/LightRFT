# Deferred engineering work

This document records implementation issues identified during the documentation and code audit of the `dev-readme-blog` branch. They are intentionally excluded from the documentation-focused change set and should be addressed in dedicated branches with targeted tests.

Status labels:

- **P0**: a user-selectable path fails or silently does not perform the requested operation;
- **P1**: correctness, packaging, or maintainability risk that should be scheduled soon;
- **P2**: cleanup or optimization with a narrower operational impact.

## P0: incomplete algorithm options

### `reinforce++` is accepted by the CLI but rejected by the calculator factory

- **Evidence**: the `--advantage_estimator` choices in `examples/gsm8k_geo3k/train_colocate.py` include `reinforce++`; `get_advantage_calculator()` in `lightrft/trainer/advantage_calculator.py` has no corresponding mapping.
- **Impact**: selecting the advertised value raises `ValueError` during experience-maker initialization.
- **Resolution**: either implement and test a calculator under an unambiguous algorithm specification, or remove the CLI choice and every related claim.
- **Acceptance**: a parser-to-training unit test covers every accepted estimator value.

### GSPO flags do not change the instantiated policy loss

- **Evidence**: `SPMDPPOTrainer` builds `policy_loss_kwargs` and stores `self.use_gspo`, but the dictionary is not used to construct or replace `actor_loss_fn`. The concrete PPO trainers instantiate `PolicyLoss` without GSPO parameters.
- **Impact**: `--use_gspo`, `--normalize_advantages`, and `--use_sequence_rewards` can be accepted without producing the documented loss behavior.
- **Resolution**: complete the loss/advantage path and define its aggregation semantics, or remove the inactive arguments.
- **Acceptance**: a numerical unit test demonstrates that enabling GSPO changes the loss on a fixed batch; CLI defaults must also be disableable.

### Non-default `gamma` is consumed only for the first experience

- **Evidence**: `FastExperienceMaker.compute_advantages_and_returns()` calls `generate_kwargs.pop("gamma", 1.0)` inside the loop over experiences.
- **Impact**: when more than one experience is processed, only the first receives the configured value; subsequent items use `1.0`.
- **Resolution**: read `gamma` once before the loop without mutating shared generation arguments.
- **Acceptance**: a multi-experience unit test verifies identical configured discounting for every item.

## P1: runtime and evaluation gaps

### Local PyTorch reward-model offload assumes an FSDP-only Strategy interface

- **Evidence**: `_compute_local_rewards()` in `lightrft/trainer/fast_exp_maker.py` unconditionally calls `strategy.reload_model()` and `strategy.offload_model()` for `torch.nn.Module` reward models. These methods are implemented by `FSDPV2Strategy`, but not by `StrategyBase` or `DeepspeedStrategy`.
- **Impact**: a DeepSpeed configuration that reaches the local PyTorch reward path can fail with `AttributeError`.
- **Resolution**: define an explicit lifecycle contract for both backends, or guard and document supported combinations.
- **Acceptance**: integration tests cover one local reward model under both supported training strategies.

### The dedicated text PPO trainer's evaluation branch is a no-op

- **Evidence**: `lightrft/trainer/ppo_trainer.py` reaches the `eval_steps` branch but the evaluation call is commented out and the branch executes `pass`; `ppo_trainer_vl.py` contains an implemented evaluation path. The primary GSM8K/Geo3K entry currently instantiates `SPMDPPOTrainerVL` even with `--text_only`, so that entry does not exercise this branch.
- **Impact**: direct users of `PPOTrainer` or `SPMDPPOTrainer` can configure an evaluation interval without receiving evaluation metrics.
- **Resolution**: implement text evaluation or reject the unsupported configuration explicitly.
- **Acceptance**: a small text run emits deterministic `eval/*` metrics at the configured interval.

### `--image` in the chat smoke test does not reach the interactive loop

- **Evidence**: `examples/chat/test_chat.py` builds `initial_images` and `initial_image_list`, then calls `interactive_mode(chatbot)`; that function initializes its own empty `current_images` list.
- **Impact**: the script reports an initial image as loaded, but the first prompt is generated without it.
- **Resolution**: pass one validated initial list to `interactive_mode` and remove the duplicated branches.
- **Acceptance**: a mocked test verifies that the first chat call receives the image exactly once.

### Boolean CLI options with `store_true` and `default=True` cannot be disabled

- **Evidence**: `--rm_use_engine` in `lightrft/utils/cli_args.py`, and the GSPO-related `--normalize_advantages` and `--use_sequence_rewards` options in the main training entry, use this combination.
- **Impact**: users cannot select the false state from the command line; examples conditionally adding `--rm_use_engine` do not actually control its value.
- **Resolution**: use `argparse.BooleanOptionalAction` or paired positive/negative flags and document the default.
- **Acceptance**: parser tests cover both states for every Boolean option.

### `--load_in_4bit` is not applied to Actor model loading

- **Evidence**: the training entry points pass `load_in_4bit` to `ActorLanguage`, `ActorVL`, and `ActorAL`; their constructors accept it only through `**kwargs` and do not forward it to `from_pretrained()`.
- **Impact**: enabling the CLI option does not quantize the policy or Reference Actor, although some Critic paths may forward the same keyword to Transformers.
- **Resolution**: define supported quantization and distributed-backend combinations, pass an explicit quantization configuration through every intended model path, or remove the option.
- **Acceptance**: model-loading tests inspect the effective `from_pretrained()` arguments for text, vision-language, and audio-language actors, followed by one supported distributed smoke test.

## P1: packaging and reproducibility

### PEP 517 wheel omits all LightRFT subpackages

- **Evidence**: `setup.py` uses `find_packages()`, while `[tool.setuptools]` in `pyproject.toml` declares only `packages = ["lightrft"]`. A PEP 517 wheel built during this audit contained only `lightrft/__init__.py` plus distribution metadata; `lightrft.datasets`, `lightrft.models`, `lightrft.strategy`, `lightrft.trainer`, and `lightrft.utils` were absent.
- **Impact**: the published wheel cannot provide the documented training, model, Strategy, or utility modules.
- **Resolution**: use setuptools package discovery in one authoritative configuration and remove duplicated metadata.
- **Acceptance**: CI builds the wheel in isolation, inspects its file list, installs it into a clean environment, and imports representative subpackages.

### Docker and source metadata specify different PyTorch requirements

- **Evidence**: `pyproject.toml` and `requirements.txt` require `torch>=2.9.1`; the Dockerfile explicitly installs `torch==2.9.0`.
- **Impact**: the container does not satisfy the source package's declared dependency constraint, and a later resolver step may upgrade or reject it.
- **Resolution**: select one tested PyTorch version and align the Dockerfile, package metadata, README, and compatibility matrix.
- **Acceptance**: the built image passes `pip check` and a minimal distributed import/smoke test.

### Formatter versions are inconsistent

- **Evidence**: the `dev` extra pins `yapf==0.29.0`; `requirements-dev.txt`, used by the style workflow, installs `yapf>=0.40.0`.
- **Impact**: local formatting and CI formatting can produce different diffs as new YAPF releases appear.
- **Resolution**: pin one formatter version in a single development dependency definition.
- **Acceptance**: `make format` is idempotent locally and in CI from a clean checkout.

### The `docs` package extra does not reproduce the documented Sphinx environment

- **Evidence**: `[project.optional-dependencies].docs` installs Sphinx, `sphinx-rtd-theme`, and `sphinx-autobuild`, while `docs/source/conf.py` requires MyST, `pytorch_sphinx_theme`, and `sphinx-copybutton`; `requirements-doc.txt` is a separate, working dependency list.
- **Impact**: `pip install -e ".[docs]"` does not supply the extensions and theme needed by `make docs`.
- **Resolution**: define the documentation dependencies once and reference the same set from local instructions and CI.
- **Acceptance**: a clean environment can run `pip install -e ".[docs]"` followed by `make docs` with warnings treated as errors.

## P1: code-quality baseline

### The repository does not pass its configured Flake8 check

- **Evidence**: running the Flake8 options from `Makefile` against all tracked Python files reports 699 findings in the audited snapshot. Most are formatting or whitespace findings, but the result also contains eight `F821` undefined-name errors, one bare `except`, and two unused local variables. Representative undefined names occur in `examples/gsm8k_geo3k/train_colocate.py`, `examples/orm_rl_demo/reward_models.py`, and `examples/orm_rl_demo/train_colocate.py`.
- **Impact**: `make fcheck` cannot act as a merge gate, and actionable correctness signals are obscured by the formatting backlog.
- **Resolution**: first fix the undefined names and other semantic findings with focused tests; then apply formatter and whitespace cleanup in a dedicated mechanical commit. Pin the formatter version before the bulk rewrite.
- **Acceptance**: `make fcheck` succeeds from a clean checkout and runs as a required CI job. Files modified by the documentation audit already pass the same Flake8 options and YAPF check.

## P2: resource use and public interfaces

### Multiple local reward models are all reloaded before sequential evaluation

- **Evidence**: `_compute_local_rewards()` first reloads every PyTorch reward model, then evaluates and offloads them one at a time.
- **Impact**: peak GPU memory can include all local models even though their computation is sequential.
- **Resolution**: reload, evaluate, and offload each model in one scoped sequence, subject to distributed-state safety.
- **Acceptance**: output rewards remain equivalent and a memory regression test shows that peak allocated memory no longer scales with the sum of model sizes.

### Public exports and modality annotations require cleanup

- **Evidence**: `lightrft/models/__init__.py` imports `ActorLanguage`, `ActorAL`, and reward-model classes but omits them from `__all__`; `ActorModality.AUDIO_LANGUAGE` is still annotated as a future extension although `ActorAL` is implemented; `PolicyLoss.use_dapo` is stored but not read by `forward()`.
- **Impact**: wildcard-import behavior and code comments do not match the implemented surface, while an inactive loss option can be mistaken for released behavior.
- **Resolution**: define the intended public API, update annotations, and remove or implement inactive fields.
- **Acceptance**: export tests and API documentation are generated from the same explicit public list.

## Cross-cutting test work

- Generate a parser manifest from each entry point and validate documentation examples against accepted arguments.
- Add a documentation link/path check and run Sphinx with warnings treated as errors.
- Add CPU-only unit tests for advantage calculators, reward aggregation, and argument semantics; keep multi-GPU tests as a separate marked suite.
- Build release artifacts in CI and smoke-test the installed wheel rather than importing from the checkout.
