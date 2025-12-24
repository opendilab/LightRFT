from typing import List
from abc import ABC

import random
import torch

from .experience_maker import Experience
from .replay_buffer_utils import (
    BufferItem, split_experience_batch, make_experience_batch, remove_padding_in_sequences
)


class NaiveReplayBuffer(ABC):
    """
    Naive replay buffer class. It stores experience samples.

    :param sample_batch_size: Batch size when sampling.
    :type sample_batch_size: int
    :param limit: Limit of number of experience samples. A number <= 0 means unlimited, defaults to 0.
    :type limit: int
    :param cpu_offload: Whether to offload experience to CPU when sampling, defaults to True.
    :type cpu_offload: bool
    :param packing_samples: Whether to use packed samples format, defaults to False.
    :type packing_samples: bool
    """
    def __init__(
        self, sample_batch_size: int, limit: int = 0, cpu_offload: bool = True, packing_samples: bool = False
    ) -> None:
        super().__init__()
        self.sample_batch_size = sample_batch_size
        # Limit <= 0 means unlimited
        self.limit = limit
        self.cpu_offload = cpu_offload
        self.packing_samples = packing_samples
        self.target_device = torch.device(f"cuda:{torch.cuda.current_device()}")
        self.items: List[BufferItem] = []

    @torch.no_grad()
    def append(self, experience: Experience) -> None:
        """
        Append experience to the replay buffer.

        :param experience: Experience batch to append.
        :type experience: Experience
        """
        if self.cpu_offload:
            experience.to_device(torch.device("cpu"))
        items = split_experience_batch(experience)
        # The packed samples come with no padding
        if not self.packing_samples:
            items = remove_padding_in_sequences(items)
        self.items.extend(items)
        if self.limit > 0:
            samples_to_remove = len(self.items) - self.limit
            if samples_to_remove > 0:
                self.items = self.items[samples_to_remove:]

    def clear(self) -> None:
        """
        Clear all items from the replay buffer.
        """
        self.items.clear()

    @torch.no_grad()
    def sample(self) -> Experience:
        """
        Sample a batch of experiences from the replay buffer.

        :return: Batch of sampled experiences.
        :rtype: Experience
        """
        items = random.sample(self.items, self.sample_batch_size)
        experience = make_experience_batch(items, self.packing_samples)
        if self.cpu_offload:
            experience.to_device(self.target_device)
        return experience

    def __len__(self) -> int:
        """
        Get the number of items in the replay buffer.

        :return: Number of items.
        :rtype: int
        """
        return len(self.items)

    def __getitem__(self, idx: int) -> BufferItem:
        """
        Get an item from the replay buffer by index.

        :param idx: Index of the item.
        :type idx: int
        :return: Buffer item at the specified index.
        :rtype: BufferItem
        """
        return self.items[idx]

    def collate_fn(self, batch) -> Experience:
        """
        Collate function for DataLoader.

        :param batch: Batch of buffer items.
        :type batch: List[BufferItem]
        :return: Batched experience.
        :rtype: Experience
        """
        experience = make_experience_batch(batch, self.packing_samples)
        return experience

    def normalize(self, attribute: str, strategy) -> None:
        """
        Normalize a specified attribute across all items in the buffer.

        This method computes the mean and standard deviation of the specified attribute
        across all items and normalizes them. Currently only supports "advantages".

        :param attribute: Name of the attribute to normalize (currently only "advantages" is supported).
        :type attribute: str
        :param strategy: Distributed training strategy for all_reduce operations.
        :type strategy: Strategy
        """
        assert attribute == "advantages"
        items = []
        action_masks = []
        for item in self:
            items.append(getattr(item, attribute))
            action_masks.append(item.action_mask)

        items_vector = torch.cat(items).float().flatten()

        if action_masks[0] is None:
            # Packing samples has no action mask
            action_masks_vector = 1
            num_actions = items_vector.numel()
        else:
            action_masks_vector = torch.cat(action_masks).flatten()
            num_actions = action_masks_vector.sum()

        # For distributed data parallel: compute mean
        sum_and_count = torch.tensor([items_vector.sum(), num_actions], device=items_vector.device)
        all_sum, all_count = strategy.all_reduce(sum_and_count, "sum")
        mean = all_sum / all_count

        # Compute standard deviation
        std = ((items_vector - mean).pow(2) * action_masks_vector).sum()
        all_std = strategy.all_reduce(std, "sum")
        rstd = (all_std / all_count).clamp(min=1e-8).rsqrt()

        for i, item in enumerate(self):
            setattr(item, attribute, (items[i] - mean) * rstd)
