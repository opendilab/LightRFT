from abc import ABC
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from PIL import Image
import torch
import numpy as np
import torch.nn as nn
from functools import reduce

from lightrft.models.actor_vl import ActorVL

from lightrft.models.utils import compute_approx_kl, masked_mean
from lightrft.utils import init_logger, remote_rm_fn

logger = init_logger(__name__)


def to(tensor: Union[torch.Tensor, list[torch.Tensor]], device):
    """
    Move tensor(s) to the specified device.

    :param tensor: Tensor or list of tensors to move.
    :type tensor: Union[torch.Tensor, list[torch.Tensor]]
    :param device: Target device.
    :type device: torch.device or str
    :return: Tensor(s) on the target device.
    :rtype: Union[torch.Tensor, list[torch.Tensor]]
    """
    if isinstance(tensor, list):
        return [to(t, device) for t in tensor]
    return tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor


def pin_memory(tensor: Union[torch.Tensor, list[torch.Tensor]]):
    """
    Pin tensor(s) in memory for faster GPU transfer.

    :param tensor: Tensor or list of tensors to pin.
    :type tensor: Union[torch.Tensor, list[torch.Tensor]]
    :return: Pinned tensor(s).
    :rtype: Union[torch.Tensor, list[torch.Tensor]]
    """
    if isinstance(tensor, list):
        return [pin_memory(t) for t in tensor]
    return tensor.pin_memory() if isinstance(tensor, torch.Tensor) else tensor


@dataclass
class ExperienceVL:
    """
    Experience is a batch of data for Vision-Language models.

    These data should have the same sequence length and number of actions.
    Left padding for sequences is applied.

    Tensor shapes:
        - sequences: (B, S) where B is batch size, S is sequence length
        - pixel_values: (B * h, w) - image pixels processed by HF processor
        - image_grid_thws: (B, 3) - image grid thw
        - raw_images: Optional[List[Image.Image]] - raw images before processing
        - pixel_values_videos: (B * f, c * h * w) - video pixels processed by HF processor
        - video_grid_thws: (B, 3) - video grid thw
        - action_log_probs: (B, A) where A is number of actions
        - base_action_log_probs: (B, A)
        - values: (B, A)
        - returns: (B, A)
        - advantages: (B, A)
        - attention_mask: (B, S)
        - action_mask: (B, A)
        - kl: (B, A)
        - action_entropy: (B, A) - Entropy values for high-entropy token filtering

    :param sequences: Token sequences including both prompt and response.
    :type sequences: torch.Tensor
    :param pixel_values: Image pixel values processed by HF processor, defaults to None.
    :type pixel_values: Optional[torch.Tensor]
    :param image_grid_thws: Image grid thw, defaults to None.
    :type image_grid_thws: Optional[torch.Tensor]
    :param raw_images: Raw image data list, defaults to None.
    :type raw_images: Optional[List[Image.Image]]
    :param pixel_values_videos: Video pixel values processed by HF processor, defaults to None.
    :type pixel_values_videos: Optional[torch.Tensor]
    :param video_grid_thws: Video grid thw, defaults to None.
    :type video_grid_thws: Optional[torch.Tensor]
    :param action_log_probs: Log probabilities of actions from the current policy, defaults to None.
    :type action_log_probs: torch.Tensor
    :param base_action_log_probs: Log probabilities from the reference policy, defaults to None.
    :type base_action_log_probs: torch.Tensor
    :param values: Value estimates from the critic, defaults to None.
    :type values: torch.Tensor
    :param returns: Discounted returns for each action, defaults to None.
    :type returns: Optional[torch.Tensor]
    :param advantages: Advantage estimates for each action, defaults to None.
    :type advantages: Optional[torch.Tensor]
    :param attention_mask: Mask indicating valid tokens in sequences, defaults to None.
    :type attention_mask: Optional[torch.LongTensor]
    :param action_mask: Mask indicating action (response) tokens, defaults to None.
    :type action_mask: Optional[torch.BoolTensor]
    :param info: Dictionary containing additional information, defaults to None.
    :type info: Optional[dict]
    :param kl: KL divergence between current and reference policy, defaults to None.
    :type kl: Optional[torch.Tensor]
    :param action_entropy: Entropy values for each action token, used for high-entropy token
        filtering. When provided, enables training only on high-entropy tokens (forking tokens
        that determine reasoning directions), improving training efficiency. Shape: (B, A).
        See: https://arxiv.org/abs/2506.01939
    :type action_entropy: Optional[torch.Tensor]
    """

    sequences: torch.Tensor
    # Image processing related
    pixel_values: Optional[torch.Tensor] = None
    image_grid_thws: Optional[torch.Tensor] = None
    raw_images: Optional[List[Image.Image]] = None

    # Video processing related
    pixel_values_videos: Optional[torch.Tensor] = None
    video_grid_thws: Optional[torch.Tensor] = None

    action_log_probs: torch.Tensor = None
    base_action_log_probs: torch.Tensor = None
    values: torch.Tensor = None
    returns: Optional[torch.Tensor] = None
    advantages: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.LongTensor] = None
    action_mask: Optional[torch.BoolTensor] = None
    info: Optional[dict] = None
    kl: Optional[torch.Tensor] = None
    action_entropy: Optional[torch.Tensor] = None  # Entropy for high-entropy token filtering

    @torch.no_grad()
    def to_device(self, device: torch.device):
        """
        Move all tensors in the experience to the specified device.

        :param device: Target device.
        :type device: torch.device
        :return: Self with tensors moved to device.
        :rtype: ExperienceVL
        """
        self.sequences = to(self.sequences, device)
        self.action_log_probs = to(self.action_log_probs, device)
        self.base_action_log_probs = to(self.base_action_log_probs, device)
        self.returns = to(self.returns, device)
        self.advantages = to(self.advantages, device)
        if self.pixel_values is not None:
            self.pixel_values = to(self.pixel_values, device)
        if self.image_grid_thws is not None:
            self.image_grid_thws = to(self.image_grid_thws, device)
        if self.pixel_values_videos is not None:
            self.pixel_values_videos = to(self.pixel_values_videos, device)
        if self.video_grid_thws is not None:
            self.video_grid_thws = to(self.video_grid_thws, device)
        self.values = to(self.values, device)
        self.attention_mask = to(self.attention_mask, device)
        self.action_mask = to(self.action_mask, device)
        self.kl = to(self.kl, device)
        if self.action_entropy is not None:
            self.action_entropy = to(self.action_entropy, device)
        self.info = {key: to(value, device) for key, value in self.info.items()}
        return self

    def pin_memory(self):
        """
        Pin all tensors in memory for faster GPU transfer.

        :return: Self with pinned tensors.
        :rtype: ExperienceVL
        """
        self.sequences = pin_memory(self.sequences)
        self.action_log_probs = pin_memory(self.action_log_probs)
        self.base_action_log_probs = pin_memory(self.base_action_log_probs)
        self.returns = pin_memory(self.returns)
        self.advantages = pin_memory(self.advantages)
        if self.pixel_values is not None:
            self.pixel_values = pin_memory(self.pixel_values)
        if self.image_grid_thws is not None:
            self.image_grid_thws = pin_memory(self.image_grid_thws)
        if self.pixel_values_videos is not None:
            self.pixel_values_videos = pin_memory(self.pixel_values_videos)
        if self.video_grid_thws is not None:
            self.video_grid_thws = pin_memory(self.video_grid_thws)
        self.values = pin_memory(self.values)
        self.attention_mask = pin_memory(self.attention_mask)
        self.action_mask = pin_memory(self.action_mask)
        self.kl = pin_memory(self.kl)
        if self.action_entropy is not None:
            self.action_entropy = pin_memory(self.action_entropy)
        self.info = {key: pin_memory(value) for key, value in self.info.items()}
        return self


@dataclass
class SamplesVL:
    """
    Samples is a batch of data for Vision-Language models.

    There can be 2 formats to store the samples, batched or packed.
    The batched format means padding is applied to the sequences, while the packed format
    will concatenate the prompt and response without padding.

    Tensor shapes (batched / packed):
        - sequences: (B, S) or (1, total_length) - tokens of both prompt and response
        - attention_mask: (B, S) or (1, total_length) - attention mask for sequences
        - action_mask: (B, A) or None - response mask showing which part is the response
        - pixel_values: Optional[torch.Tensor] - image pixels processed by HF processor
        - image_grid_thws: Optional[torch.Tensor] - image grid thw
        - raw_images: Optional[List[Image.Image]] - raw image data list
        - pixel_values_videos: Optional[torch.Tensor] - video pixels processed by HF processor
        - video_grid_thws: Optional[torch.Tensor] - video grid thw
        - num_actions: int or (B,) - number of actions (tokens) in the response
        - packed_seq_lens: None or (B,) - length of each sample in packed format
        - response_length: (B,) - number of tokens in the response
        - total_length: (B,) - total number of tokens in sequences
        - prompts: list[str] - the prompts used to generate responses
        - references: Optional[List[str]] - reference texts
        - labels: Optional[List[str]] - ground truth labels
        - output_texts: list[str] - generated output texts
        - image_num: Optional[List[int]] - image numbers
        - video_num: Optional[List[int]] - video numbers

    :param sequences: Token sequences including both prompt and response.
    :type sequences: torch.Tensor
    :param attention_mask: Attention mask for sequences, defaults to None.
    :type attention_mask: Optional[torch.LongTensor]
    :param action_mask: Mask indicating action (response) tokens, defaults to None.
    :type action_mask: Optional[torch.BoolTensor]
    :param pixel_values: Image pixels processed by HF processor, defaults to None.
    :type pixel_values: Optional[torch.Tensor]
    :param image_grid_thws: Image grid thw, defaults to None.
    :type image_grid_thws: Optional[torch.Tensor]
    :param raw_images: Raw image data list, defaults to None.
    :type raw_images: Optional[List[Image.Image]]
    :param num_actions: Number of actions per sample, defaults to None.
    :type num_actions: Union[int, torch.Tensor]
    :param packed_seq_lens: Sequence lengths for packed format, defaults to None.
    :type packed_seq_lens: Optional[torch.Tensor]
    :param response_length: Length of each response, defaults to None.
    :type response_length: torch.Tensor
    :param total_length: Total length of each sequence, defaults to None.
    :type total_length: torch.Tensor
    :param references: Reference texts, defaults to None.
    :type references: Optional[List[str]]
    :param labels: Ground truth labels, defaults to None.
    :type labels: Optional[List[str]]
    :param prompts: List of prompt strings, defaults to None.
    :type prompts: list[str]
    :param output_texts: Generated output texts, defaults to None.
    :type output_texts: list[str]
    :param image_num: Image numbers, defaults to None.
    :type image_num: Optional[List[int]]
    :param video_num: Video numbers, defaults to None.
    :type video_num: Optional[List[int]]
    """

    sequences: torch.Tensor
    attention_mask: Optional[torch.LongTensor] = None
    action_mask: Optional[torch.BoolTensor] = None

    pixel_values: Optional[torch.Tensor] = None
    image_grid_thws: Optional[torch.Tensor] = None
    raw_images: Optional[List[Image.Image]] = None
    image_num: Optional[List[int]] = None

    pixel_values_videos: Optional[torch.Tensor] = None
    video_grid_thws: Optional[torch.Tensor] = None
    video_num: Optional[List[int]] = None

    num_actions: Union[int, torch.Tensor] = None
    packed_seq_lens: Optional[torch.Tensor] = None
    response_length: torch.Tensor = None
    total_length: torch.Tensor = None

    references: Optional[List[str]] = None
    labels: Optional[List[str]] = None
    prompts: list[str] = None

    output_texts: list[str] = None


class NaiveExperienceMakerVL(ABC):
    """
    A naive experience maker for Vision-Language reinforcement learning.

    This class is responsible for generating experiences (sequences of prompts, actions, rewards, etc.)
    which are then used to train the actor and critic models for Vision-Language tasks.

    :param actor: The Vision-Language policy model to be trained.
    :type actor: ActorVL
    :param critic: The value model to be trained.
    :type critic: nn.Module
    :param reward_model: The reward model used to score generated responses.
    :type reward_model: nn.Module
    :param initial_model: The reference model for KL divergence calculation.
    :type initial_model: ActorVL
    :param tokenizer: The tokenizer for encoding and decoding text.
    :type tokenizer: Tokenizer
    :param processor: The processor for handling multi-modal inputs.
    :type processor: Processor
    :param prompt_max_len: The maximum length of input prompts after tokenization.
    :type prompt_max_len: int
    :param kl_controller: The controller for managing the KL penalty coefficient.
    :type kl_controller: KLController
    :param strategy: The training strategy containing configurations, defaults to None.
    :type strategy: Strategy, optional
    :param remote_rm_url: A list of URLs for remote reward models, defaults to None.
    :type remote_rm_url: list[str], optional
    :param reward_fn: A custom reward function, defaults to None.
    :type reward_fn: Callable, optional
    """
    def __init__(
        self,
        actor: ActorVL,
        critic: nn.Module,
        reward_model: nn.Module,
        initial_model: ActorVL,
        tokenizer,
        processor,
        prompt_max_len: int,
        kl_controller,
        strategy=None,
        remote_rm_url: list[str] = None,
        reward_fn=None,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.remote_rm_url = remote_rm_url
        self.initial_model = initial_model
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_max_len = prompt_max_len
        self.kl_ctl = kl_controller
        self.strategy = strategy
        self.reward_fn = reward_fn
        self.perf_stats = None
        self.advantage_estimator = strategy.args.advantage_estimator

        # Custom reward function for reinforced fine-tuning
        self.custom_reward_func = None
        remote_rm_url = [remote_rm_url] if isinstance(remote_rm_url, str) else remote_rm_url
        if remote_rm_url and remote_rm_url[0].endswith(".py"):
            print(f"Loading custom `reward_func(queries, prompts, references)` from {remote_rm_url[0]}")
            import importlib.util

            spec = importlib.util.spec_from_file_location("reward_func", remote_rm_url[0])
            reward_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(reward_module)
            self.custom_reward_func = reward_module.reward_func

    def tokenize_fn(self, texts, max_length, padding=True, device=None):
        """
        Tokenize input texts.

        :param texts: List of text strings to tokenize.
        :type texts: List[str]
        :param max_length: Maximum sequence length.
        :type max_length: int
        :param padding: Whether to apply padding, defaults to True.
        :type padding: bool
        :param device: Target device for tensors, defaults to None.
        :type device: torch.device or str, optional
        :return: Tokenized batch (as dict if padding=True, otherwise as list).
        :rtype: dict or list
        """
        if not padding:
            # When padding is False, return tokenized texts as list
            return self.tokenizer(
                texts,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
            )
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}

    def processor_fn(self, texts, images, max_length, padding=True, device=None):
        """
        Process multi-modal inputs (text and images).

        :param texts: List of text strings to process.
        :type texts: List[str]
        :param images: List of images to process.
        :type images: List[Image.Image]
        :param max_length: Maximum sequence length.
        :type max_length: int
        :param padding: Whether to apply padding, defaults to True.
        :type padding: bool
        :param device: Target device for tensors, defaults to None.
        :type device: torch.device or str, optional
        :return: Processed batch (as dict if padding=True, otherwise as list).
        :rtype: dict or list
        """
        if images[0]:
            if not padding:
                # When padding is False, return tokenized texts as list
                return self.processor(
                    text=texts,
                    images=images,
                    add_special_tokens=False,
                    max_length=max_length,
                    truncation=True,
                )
            batch = self.processor(
                text=texts,
                images=images,
                return_tensors="pt",
                add_special_tokens=False,
                max_length=max_length,
                padding=True,
                truncation=True,
            )
            return {k: v.to(device) for k, v in batch.items()}
        else:
            return self.tokenize_fn(texts, max_length, padding, device)

    @torch.no_grad()
    def make_experience_list(
        self, all_prompts: Union[str, List[str]], all_images, all_references, all_labels, **generate_kwargs
    ) -> List[ExperienceVL]:
        """
        Make a list of experiences with the micro_rollout_batch_size.

        This method will first calculate the response sequences and rewards for the given prompts.
        Then, if we need certain processing for the rewards or filtering, we process the rollout as a whole.
        After that, we calculate the advantages and returns for each experience.

        :param all_prompts: Prompts to generate responses for.
        :type all_prompts: Union[str, List[str]]
        :param all_images: Images corresponding to prompts.
        :type all_images: List
        :param all_references: Reference texts for evaluation.
        :type all_references: List[str]
        :param all_labels: Ground truth labels.
        :type all_labels: List[str]
        :param generate_kwargs: Additional generation parameters (gamma, lambd, etc.).
        :type generate_kwargs: dict
        :return: List of ExperienceVL objects.
        :rtype: List[ExperienceVL]
        """
        raise NotImplementedError

    @torch.no_grad()
    def generate_samples(self, all_prompts: List[str], all_images, all_references, all_labels,
                         **generate_kwargs) -> List[SamplesVL]:
        """
        Generate samples and return in batches.

        :param all_prompts: List of prompt strings.
        :type all_prompts: List[str]
        :param all_images: List of images corresponding to prompts.
        :type all_images: List
        :param all_references: List of reference texts.
        :type all_references: List[str]
        :param all_labels: List of ground truth labels.
        :type all_labels: List[str]
        :param generate_kwargs: Additional generation parameters.
        :type generate_kwargs: dict
        :return: List of SamplesVL objects.
        :rtype: List[SamplesVL]
        """
        assert not getattr(self, "packing_samples", False)
        args = self.strategy.args
        self.actor.eval()
        # Sample multiple responses per prompt
        all_prompts = sum([[prompt] * args.n_samples_per_prompt for prompt in all_prompts], [])
        all_images = sum([[image] * args.n_samples_per_prompt for image in all_images], [])
        all_references = sum([[reference] * args.n_samples_per_prompt for reference in all_references], [])
        all_labels = sum([[label] * args.n_samples_per_prompt for label in all_labels], [])
        samples_list = []
        for i in range(0, len(all_prompts), args.micro_rollout_batch_size):
            prompts = all_prompts[i:i + args.micro_rollout_batch_size]
            images = all_images[i:i + args.micro_rollout_batch_size]
            references = all_references[i:i + args.micro_rollout_batch_size]
            labels = all_labels[i:i + args.micro_rollout_batch_size]
            inputs = self.processor_fn(prompts, images, self.prompt_max_len, device="cuda")
            sequences, attention_mask, action_mask = self.actor.generate(**inputs, **generate_kwargs)
            samples = SamplesVL(
                sequences=sequences,
                attention_mask=attention_mask,
                action_mask=action_mask,
                pixel_values=inputs["pixel_values"],
                image_grid_thws=inputs["image_grid_thw"],
                raw_images=images,
                num_actions=action_mask.size(1),
                packed_seq_lens=None,
                response_length=action_mask.float().sum(dim=-1),
                total_length=attention_mask.float().sum(dim=-1),
                references=references,
                labels=labels,
                prompts=prompts,
            )
            samples_list.append(samples)
        return samples_list

    @torch.no_grad()
    def make_experience(self, samples: SamplesVL) -> ExperienceVL:
        """
        Turn samples into experience by calculating log probs, values, rewards, and KL divergence.

        :param samples: Samples object containing sequences and metadata.
        :type samples: SamplesVL
        :return: ExperienceVL object with all computed values.
        :rtype: ExperienceVL
        """
        self.actor.eval()
        if self.initial_model is not None:
            self.initial_model.eval()
        if self.reward_model is not None:
            self.reward_model.eval()
        if self.critic is not None:
            self.critic.eval()

        # Extract values from samples
        sequences = samples.sequences
        attention_mask = samples.attention_mask
        action_mask = samples.action_mask
        pixel_values = samples.pixel_values
        image_grid_thws = samples.image_grid_thws
        raw_images = samples.raw_images
        num_actions = samples.num_actions

        # Log probabilities from current policy
        action_log_probs = self.actor(sequences, num_actions, attention_mask, pixel_values, image_grid_thws)

        # Log probabilities from initial/reference policy
        if self.initial_model is not None:
            base_action_log_probs = self.initial_model(
                sequences, num_actions, attention_mask, pixel_values, image_grid_thws
            )
        else:
            base_action_log_probs = None

        # Values from critic
        if self.critic is not None:
            value = self.critic(sequences, num_actions, attention_mask, pixel_values, image_grid_thws)
        else:
            value = None

        # Rewards
        if self.remote_rm_url is not None:
            # Remote reward model
            queries = self.tokenizer.batch_decode(sequences.cpu(), skip_special_tokens=False)
            references = samples.references if hasattr(samples, "references") else None
            if self.custom_reward_func:
                r = self.custom_reward_func(queries, samples.prompts, references).to(device=action_log_probs.device)
            else:
                r = remote_rm_fn(
                    api_url=self.remote_rm_url,
                    queries=queries,
                    prompts=samples.prompts,
                    references=references,
                    raw_images=raw_images
                ).to(device=action_log_probs.device)
        else:
            # Local reward model
            r = self.reward_model(sequences, attention_mask, pixel_values, image_grid_thws)

        if (self.initial_model is not None) and (not self.strategy.args.use_kl_loss):
            kl = compute_approx_kl(
                action_log_probs,
                base_action_log_probs,
                action_mask=action_mask,
                kl_estimator=self.strategy.args.kl_estimator,
            )
        else:
            kl = torch.zeros_like(action_log_probs, dtype=action_log_probs.dtype, device=action_log_probs.device)

        info = {
            "kl": masked_mean(kl, action_mask, dim=-1),
            "reward": r,
            "response_length": samples.response_length,
            "total_length": samples.total_length,
            "num_actions": num_actions,
        }
        # Reset model state
        self.actor.train()
        if self.critic is not None:
            self.critic.train()

        return ExperienceVL(
            sequences,
            pixel_values,
            image_grid_thws,
            raw_images,
            action_log_probs,
            base_action_log_probs,
            value,
            None,
            None,
            attention_mask,
            action_mask,
            info,
            kl,
        )

    @torch.no_grad()
    def process_experiences(self, experiences: List[ExperienceVL]) -> Tuple[List[ExperienceVL], List[torch.Tensor]]:
        """
        Process experiences for reward shaping and filtering.

        This can be used to filter out some experiences or do some processing on the rewards.

        :param experiences: List of ExperienceVL objects.
        :type experiences: List[ExperienceVL]
        :return: Tuple of (processed experiences, processed rewards).
        :rtype: Tuple[List[ExperienceVL], List[torch.Tensor]]
        """
        args = self.strategy.args
        # Reward shaping for RLOO and REINFORCE baseline
        if args.advantage_estimator == "rloo":
            rewards = torch.cat([experience.info["reward"] for experience in experiences])
            rewards = rewards.reshape(-1, args.n_samples_per_prompt).to(device="cuda")
            baseline = (rewards.sum(-1, keepdim=True) - rewards) / (args.n_samples_per_prompt - 1)
            rewards = rewards - baseline
            rewards = rewards.flatten().to(device="cpu").chunk(len(experiences))
            return experiences, rewards
        elif args.advantage_estimator == "reinforce_baseline":
            # REINFORCE++ baseline removed the / std and K3 KL loss in GRPO.
            # `/ std` is not needed in RL variance reduction theory, and `K3 KL` has a larger variance
            # than `K1 KL` under a categorical distribution.
            rewards = torch.cat([experience.info["reward"] for experience in experiences])
            rewards = rewards.reshape(-1, args.n_samples_per_prompt).to(device="cuda")
            rewards = rewards - rewards.mean(-1, keepdim=True)
            rewards = rewards.reshape(-1).to(device="cpu").chunk(len(experiences))
            return experiences, rewards
        elif args.advantage_estimator in ["group_norm", "grpo"]:
            rewards = torch.cat([experience.info["reward"] for experience in experiences])
            rewards = rewards.reshape(-1, args.n_samples_per_prompt).to(device="cuda")
            rewards = (rewards - rewards.mean(-1, keepdim=True)) / (rewards.std(-1, keepdim=True) + 1e-9)
            rewards = rewards.reshape(-1).to(device="cpu").chunk(len(experiences))
            return experiences, rewards

        elif args.advantage_estimator == "cpgd":
            return experiences, [experience.info["reward"] for experience in experiences]

        else:
            raise ValueError(f"Unhandled advantage_estimator: {args.advantage_estimator}")

        # Default rewards
        return experiences, [experience.info["reward"] for experience in experiences]

    @torch.no_grad()
    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages and returns from rewards and values using GAE.

        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages formula:
            Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                  - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns formula:
            Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                       + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        :param values: Tensor of shape (batch_size, response_size).
        :type values: torch.Tensor
        :param rewards: Tensor of shape (batch_size, response_size).
        :type rewards: torch.Tensor
        :param action_mask: Tensor of shape (batch_size, response_size).
        :type action_mask: torch.Tensor
        :param gamma: Discount factor.
        :type gamma: float
        :param lambd: GAE lambda parameter.
        :type lambd: float
        :return: Tuple of (advantages, returns), both of shape (batch_size, response_size).
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        if isinstance(values, list):
            # Packing samples
            # TODO: This is slow...
            advantages = []
            returns = []
            for v, r in zip(values, rewards):
                adv, ret = self.get_advantages_and_returns(v.unsqueeze(0), r.unsqueeze(0), action_mask, gamma, lambd)
                advantages.append(adv.squeeze(0))
                returns.append(ret.squeeze(0))
            return advantages, returns

        lastgaelam = 0
        advantages_reversed = []
        response_length = rewards.size(1)

        # Mask invalid responses
        if action_mask is not None:
            values = action_mask * values
            rewards = action_mask * rewards

        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return advantages.detach(), returns

    @torch.no_grad()
    def get_cumulative_returns(
        self,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute cumulative returns from rewards using REINFORCE.

        REINFORCE uses cumulative returns without GAE (Generalized Advantage Estimation).

        :param rewards: Tensor of shape (batch_size, response_size).
        :type rewards: torch.Tensor
        :param action_mask: Binary mask tensor of shape (batch_size, response_size).
        :type action_mask: torch.Tensor
        :param gamma: Discount factor.
        :type gamma: float
        :return: Returns tensor of shape (batch_size, response_size).
        :rtype: torch.Tensor
        """

        if isinstance(rewards, list):
            # Packing samples
            # TODO: This is slow...
            returns = []
            for r in rewards:
                ret = self.get_cumulative_returns(r.unsqueeze(0), action_mask, gamma)
                returns.append(ret.squeeze(0))
            return returns

        response_length = rewards.size(1)
        returns = torch.zeros_like(rewards)
        cumulative_return = torch.zeros(rewards.size(0), device=rewards.device)

        # Mask invalid responses if action_mask is provided
        if action_mask is not None:
            rewards = action_mask * rewards

        # Calculate returns by accumulating discounted rewards
        for t in reversed(range(response_length)):
            cumulative_return = rewards[:, t] + gamma * cumulative_return
            returns[:, t] = cumulative_return

        return returns


def cumulative_product(data: Union[List[int], int, np.ndarray, torch.Tensor]) -> int:
    """
    Compute the cumulative product of a one-dimensional list, a tensor, or a single integer.

    :param data: Input can be an integer, a list of integers, or a tensor (NumPy/torch).
    :type data: Union[List[int], int, np.ndarray, torch.Tensor]
    :return: The cumulative product of the input.
    :rtype: int
    """
    if isinstance(data, int):
        # If data is a single integer, return it directly
        return data

    if isinstance(data, list):
        # Compute the product of all elements in the list
        return reduce(lambda x, y: x * y, data, 1)

    if isinstance(data, np.ndarray):
        # For NumPy arrays, use np.prod
        return int(np.prod(data))

    if isinstance(data, torch.Tensor):
        # For PyTorch tensors, use tensor.prod()
        return int(torch.prod(data).item())

    raise ValueError("Input must be an int, list of integers, or a NumPy/PyTorch tensor.")
