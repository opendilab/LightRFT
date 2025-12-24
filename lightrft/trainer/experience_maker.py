import time
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Callable, Dict

import ray
import torch
import torch.nn as nn
from tqdm import tqdm

from lightrft.models import ActorLanguage

from lightrft.models.utils import compute_approx_kl, compute_reward, masked_mean, unpacking_samples
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
    return tensor.to(device)


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
    return tensor.pin_memory()


def clip_filter_like_weight_func(rewards, clip_filter_like_weight_clip_eps=3.0, lamda=1.0):
    """
    Compute clip-filter-like weights for rewards.

    This function applies a weighting scheme similar to the clip-filter method used in
    early RLHF implementations, where samples with zero variance are given special weights.

    :param rewards: Reward tensor of shape [batch_size, n_samples].
    :type rewards: torch.Tensor
    :param clip_filter_like_weight_clip_eps: Maximum clipping value for weights, defaults to 3.0.
    :type clip_filter_like_weight_clip_eps: float
    :param lamda: Weight value for samples with zero variance, defaults to 1.0.
    :type lamda: float
    :return: Weight tensor of the same shape as rewards.
    :rtype: torch.Tensor
    """
    online_filter_mask = (rewards.std(-1, keepdim=True) == 0.0)
    if online_filter_mask.sum() == rewards.size(0):
        return torch.ones_like(rewards, device=rewards.device)
    weights = torch.ones_like(rewards,
                              device=rewards.device) * (rewards.size(0) / (rewards.size(0) - online_filter_mask.sum())
                                                        ).clamp(max=clip_filter_like_weight_clip_eps)
    weights[online_filter_mask.repeat(1, rewards.size(-1))] = lamda

    return weights


@dataclass
class Experience:
    """
    Experience is a batch of data containing sequences and associated RL training information.

    These data should have the same sequence length and number of actions.
    Left padding for sequences is applied.

    Tensor shapes:
        - sequences: (B, S) where B is batch size, S is sequence length
        - action_log_probs: (B, A) where A is number of actions
        - values: (B, A)
        - returns: (B, A)
        - advantages: (B, A)
        - attention_mask: (B, S)
        - action_mask: (B, A)
        - kl: (B, A)

    :param sequences: Token sequences including both prompt and response.
    :type sequences: torch.Tensor
    :param action_log_probs: Log probabilities of actions from the current policy.
    :type action_log_probs: torch.Tensor
    :param base_action_log_probs: Log probabilities from the reference (initial) policy.
    :type base_action_log_probs: torch.Tensor
    :param values: Value estimates from the critic.
    :type values: torch.Tensor
    :param returns: Discounted returns for each action.
    :type returns: Optional[torch.Tensor]
    :param advantages: Advantage estimates for each action.
    :type advantages: Optional[torch.Tensor]
    :param attention_mask: Mask indicating valid tokens in sequences.
    :type attention_mask: Optional[torch.LongTensor]
    :param action_mask: Mask indicating action (response) tokens.
    :type action_mask: Optional[torch.BoolTensor]
    :param info: Dictionary containing additional information (rewards, lengths, etc.).
    :type info: Optional[dict]
    :param kl: KL divergence between current and reference policy.
    :type kl: Optional[torch.Tensor]
    """

    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    base_action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: Optional[torch.Tensor]
    advantages: Optional[torch.Tensor]
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    info: Optional[dict]
    kl: Optional[torch.Tensor] = None

    @torch.no_grad()
    def to_device(self, device: torch.device) -> None:
        """
        Move all tensors in the experience to the specified device.

        :param device: Target device.
        :type device: torch.device
        """
        self.sequences = to(self.sequences, device)
        self.action_log_probs = to(self.action_log_probs, device)
        self.returns = to(self.returns, device)
        self.advantages = to(self.advantages, device)
        if self.values is not None:
            self.values = to(self.values, device)
        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.to(device)
        if self.action_mask is not None:
            self.action_mask = self.action_mask.to(device)

    def pin_memory(self):
        """
        Pin all tensors in memory for faster GPU transfer.

        :return: Self with pinned tensors.
        :rtype: Experience
        """
        self.sequences = pin_memory(self.sequences)
        self.action_log_probs = pin_memory(self.action_log_probs)
        self.returns = pin_memory(self.returns)
        self.advantages = pin_memory(self.advantages)
        if self.values is not None:
            self.values = pin_memory(self.values)
        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.pin_memory()
        if self.action_mask is not None:
            self.action_mask = self.action_mask.pin_memory()
        return self


@dataclass
class Samples:
    """
    Samples is a batch of data that can be in batched or packed format.

    The batched format applies padding to sequences, while the packed format
    concatenates prompt and response without padding.

    Tensor shapes (batched / packed):
        - sequences: (B, S) or (1, total_length) - tokens of both prompt and response
        - attention_mask: (B, S) or (1, total_length) - attention mask for sequences
        - action_mask: (B, A) or None - response mask showing which part is the response
        - num_actions: int or (B,) - number of actions (tokens) in the response
        - packed_seq_lens: None or (B,) - length of each sample in packed format
        - response_length: (B,) - number of tokens in the response
        - total_length: (B,) - total number of tokens in sequences
        - prompts: list[str] - the prompts used to generate responses
        - labels: list[str] - ground truth labels (if available)

    :param sequences: Token sequences including both prompt and response.
    :type sequences: torch.Tensor
    :param attention_mask: Attention mask for sequences.
    :type attention_mask: Optional[torch.LongTensor]
    :param action_mask: Mask indicating action (response) tokens.
    :type action_mask: Optional[torch.BoolTensor]
    :param num_actions: Number of actions per sample.
    :type num_actions: Union[int, torch.Tensor]
    :param packed_seq_lens: Sequence lengths for packed format.
    :type packed_seq_lens: Optional[torch.Tensor]
    :param response_length: Length of each response.
    :type response_length: torch.Tensor
    :param total_length: Total length of each sequence.
    :type total_length: torch.Tensor
    :param prompts: List of prompt strings.
    :type prompts: list[str]
    :param labels: List of label strings.
    :type labels: list[str]
    :param pad_len: Padding length applied.
    :type pad_len: Optional[int]
    """

    sequences: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    num_actions: Union[int, torch.Tensor]
    packed_seq_lens: Optional[torch.Tensor]
    response_length: torch.Tensor
    total_length: torch.Tensor
    prompts: list[str]
    labels: list[str]
    pad_len: Optional[int]


class NaiveExperienceMaker(ABC):
    """
    A naive experience maker for reinforcement learning.

    This class is responsible for generating experiences (sequences of prompts, actions, rewards, etc.)
    which are then used to train the actor and critic models. It orchestrates the interaction between
    the actor, critic, reward model, and the initial reference model to produce the data needed for
    a single step of PPO (or a similar RL algorithm).

    :param actor: The policy model to be trained.
    :type actor: ActorLanguage
    :param critic: The value model to be trained.
    :type critic: nn.Module
    :param reward_model: The reward model used to score generated responses.
    :type reward_model: nn.Module
    :param initial_model: The reference model for KL divergence calculation (typically a frozen copy of the SFT model).
    :type initial_model: ActorLanguage
    :param tokenizer: The tokenizer for encoding and decoding text.
    :type tokenizer: Tokenizer
    :param prompt_max_len: The maximum length of input prompts after tokenization.
    :type prompt_max_len: int
    :param kl_controller: The controller for managing the KL penalty coefficient.
    :type kl_controller: KLController
    :param strategy: The training strategy containing configurations and distributed training logic, defaults to None.
    :type strategy: Strategy, optional
    :param remote_rm_url: A list of URLs for remote reward models, defaults to None.
    :type remote_rm_url: List[str], optional
    :param reward_fn: A custom reward function, defaults to None.
    :type reward_fn: Callable, optional
    :param reward_fn_label_map: A map for reward function labels, defaults to None.
    :type reward_fn_label_map: Dict, optional
    :param reward_recipe: A dictionary defining how to combine different reward sources, defaults to None.
    :type reward_recipe: Dict, optional
    """
    def __init__(
        self,
        actor: 'ActorLanguage',
        critic: nn.Module,
        reward_model: nn.Module,
        initial_model: 'ActorLanguage',
        tokenizer,
        prompt_max_len: int,
        kl_controller,
        strategy,
        remote_rm_url: Optional[List[str]] = None,
        reward_fn: Optional[Callable] = None,
        reward_fn_label_map: Optional[Dict] = None,
        reward_recipe: Optional[Dict] = None,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.remote_rm_url = remote_rm_url
        self.initial_model = initial_model
        self.tokenizer = tokenizer
        self.prompt_max_len = prompt_max_len
        self.kl_ctl = kl_controller
        self.strategy = strategy
        self.reward_fn = reward_fn
        self.reward_fn_label_map = reward_fn_label_map
        self.reward_recipe = reward_recipe
        self.perf_stats = None
        self.advantage_estimator = strategy.args.advantage_estimator

        # Custom reward function for reinforced fine-tuning
        self.custom_reward_func = None
        if remote_rm_url and remote_rm_url[0].endswith(".py"):
            print(f"Loading custom `reward_func(queries, prompts)` from {remote_rm_url[0]}")
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

    @torch.no_grad()
    def make_experience_list(self, all_prompts: Union[str, List[str]], **generate_kwargs) -> List[Experience]:
        """
        Make a list of experiences with the micro_rollout_batch_size.

        This method first calculates the response sequences and rewards for the given prompts.
        Then, if we need certain processing for the rewards or filtering, we process the rollout as a whole.
        After that, we calculate the advantages and returns for each experience.

        :param all_prompts: Prompts to generate responses for.
        :type all_prompts: Union[str, List[str]]
        :param generate_kwargs: Additional generation parameters (gamma, lambd, etc.).
        :type generate_kwargs: dict
        :return: List of Experience objects.
        :rtype: List[Experience]
        """
        args = self.strategy.args
        experiences = []
        for samples in tqdm(
            self.generate_samples(all_prompts, **generate_kwargs),
            desc="make_experience",
            disable=not self.strategy.is_rank_0(),
        ):
            experiences.append(self.make_experience(samples))

        experiences, rewards = self.process_experiences(experiences)

        # Calculate return and advantages
        for experience, reward in zip(experiences, rewards):
            num_actions = experience.info["num_actions"]
            reward = compute_reward(
                reward,
                self.kl_ctl.value,
                experience.kl,
                action_mask=experience.action_mask,
                num_actions=num_actions,
                reward_clip_range=args.reward_clip_range,
            )

            if self.advantage_estimator == "gae":
                experience.advantages, experience.returns = self.get_advantages_and_returns(
                    experience.values,
                    reward,
                    experience.action_mask,
                    generate_kwargs["gamma"],
                    generate_kwargs["lambd"],
                )
            elif self.advantage_estimator in ["reinforce", "rloo", "reinforce_baseline", "group_norm"]:
                experience.returns = self.get_cumulative_returns(
                    reward,
                    experience.action_mask,
                    generate_kwargs["gamma"],
                )
                experience.advantages = deepcopy(experience.returns)
            else:
                raise Exception(f"Unknown advantage_estimator {self.advantage_estimator}")

            # Calculate the return info
            if not getattr(self, "packing_samples", False):
                return_sums = reward.sum(dim=-1)
            else:
                return_sums = torch.tensor([each_reward.sum() for each_reward in reward],
                                           device=torch.cuda.current_device())
            experience.info["return"] = return_sums
            # Remove unnecessary info
            experience.kl = None
            del experience.info["num_actions"]
        return experiences

    @torch.no_grad()
    def generate_samples(self, all_prompts: List[str], **generate_kwargs) -> List[Samples]:
        """
        Generate samples and return in batches.

        :param all_prompts: List of prompt strings.
        :type all_prompts: List[str]
        :param generate_kwargs: Additional generation parameters.
        :type generate_kwargs: dict
        :return: List of Samples objects.
        :rtype: List[Samples]
        """
        assert not getattr(self, "packing_samples", False)
        args = self.strategy.args
        self.actor.eval()
        # Sample multiple responses per prompt
        all_prompts = sum([[prompt] * args.n_samples_per_prompt for prompt in all_prompts], [])
        samples_list = []
        for i in range(0, len(all_prompts), args.micro_rollout_batch_size):
            prompts = all_prompts[i:i + args.micro_rollout_batch_size]
            inputs = self.tokenize_fn(prompts, self.prompt_max_len, device="cuda")
            sequences, attention_mask, action_mask = self.actor.generate(**inputs, **generate_kwargs)
            samples = Samples(
                sequences=sequences,
                attention_mask=attention_mask,
                action_mask=action_mask,
                num_actions=action_mask.size(1),
                packed_seq_lens=None,
                response_length=action_mask.float().sum(dim=-1),
                total_length=attention_mask.float().sum(dim=-1),
                prompts=prompts,
            )
            samples_list.append(samples)
        return samples_list

    @torch.no_grad()
    def make_experience(self, samples: Samples) -> Experience:
        """
        Turn samples into experience by calculating log probs, values, rewards, and KL divergence.

        :param samples: Samples object containing sequences and metadata.
        :type samples: Samples
        :return: Experience object with all computed values.
        :rtype: Experience
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
        num_actions = samples.num_actions

        # Log probabilities from current policy
        action_log_probs = self.actor(sequences, num_actions, attention_mask)

        # Log probabilities from initial/reference policy
        if self.initial_model is not None:
            base_action_log_probs = self.initial_model(sequences, num_actions, attention_mask)
        else:
            base_action_log_probs = None

        # Values from critic
        if self.critic is not None:
            value = self.critic(sequences, num_actions, attention_mask)
        else:
            value = None

        # Rewards
        if self.remote_rm_url is not None:
            # Remote reward model
            queries = self.tokenizer.batch_decode(sequences.cpu(), skip_special_tokens=False)
            if self.custom_reward_func:
                r = self.custom_reward_func(queries, samples.prompts).to(device=action_log_probs.device)
            else:
                r = remote_rm_fn(self.remote_rm_url, queries=queries,
                                 prompts=samples.prompts).to(device=action_log_probs.device)
        else:
            # Local reward model
            r = self.reward_model(sequences, attention_mask)

        if self.initial_model is not None:
            kl = compute_approx_kl(
                action_log_probs,
                base_action_log_probs,
                action_mask=action_mask,
                use_kl_estimator_k3=self.strategy.args.use_kl_estimator_k3,
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

        return Experience(
            sequences,
            action_log_probs,
            value,
            None,
            None,
            attention_mask,
            action_mask,
            info,
            kl,
        )

    @torch.no_grad()
    def process_experiences(self, experiences: List[Experience]) -> Tuple[List[Experience], List[torch.Tensor]]:
        """
        Process experiences for reward shaping and filtering.

        This can be used to filter out some experiences or do some processing on the rewards.

        :param experiences: List of Experience objects.
        :type experiences: List[Experience]
        :return: Tuple of (processed experiences, processed rewards).
        :rtype: Tuple[List[Experience], List[torch.Tensor]]
        """
        args = self.strategy.args

        # Reward shaping for RLOO
        if args.advantage_estimator == "rloo":
            rewards = torch.cat([experience.info["reward"] for experience in experiences])
            rewards = rewards.reshape(-1, args.n_samples_per_prompt)
            baseline = (rewards.sum(-1, keepdim=True) - rewards) / (args.n_samples_per_prompt - 1)
            rewards = rewards - baseline
            rewards = rewards.flatten().chunk(len(experiences))
            return experiences, rewards
        # Reward shaping for GRPO
        if args.advantage_estimator in ["grpo", "group_norm"]:
            rewards = torch.cat([experience.info["reward"] for experience in experiences])
            rewards = rewards.reshape(-1, args.n_samples_per_prompt)

            baseline = rewards.mean(-1, keepdim=True)
            rewards = (rewards - baseline) / (rewards.std(1, keepdim=True) + 1e-8)

            rewards = rewards.flatten().chunk(len(experiences))
            return experiences, rewards
        elif args.advantage_estimator == "reinforce_baseline":
            # REINFORCE++ baseline removed the / std and K3 KL loss in GRPO
            # `/ std` is not needed in RL variance reduction theory, and `K3 KL` has a larger variance
            # than `K1 KL` under a categorical distribution
            rewards = torch.cat([experience.info["reward"] for experience in experiences])
            rewards = rewards.reshape(-1, args.n_samples_per_prompt).to(device="cuda")
            rewards = rewards - rewards.mean(-1, keepdim=True)
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
