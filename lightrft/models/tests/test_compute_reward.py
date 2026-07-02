import unittest

import torch

from lightrft.models.utils import compute_reward


class TestComputeReward(unittest.TestCase):
    def test_per_step_mixed_rows_fall_back_to_scalar_reward(self):
        r = torch.tensor([1.5, 2.5], dtype=torch.float32)
        kl = torch.zeros(2, 5, dtype=torch.float32)
        action_mask = torch.tensor(
            [
                [1, 1, 1, 1, 0],
                [1, 1, 1, 0, 0],
            ],
            dtype=torch.long,
        )
        step_rewards = torch.tensor(
            [
                [0.2, 0.8],
                [0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        step_token_indices = torch.tensor(
            [
                [1, 3],
                [-1, -1],
            ],
            dtype=torch.long,
        )

        reward = compute_reward(
            r,
            kl_coef=0.0,
            kl=kl,
            action_mask=action_mask,
            step_rewards=step_rewards,
            step_token_indices=step_token_indices,
        )

        expected = torch.tensor(
            [
                [0.0, 0.2, 0.0, 0.8, 0.0],
                [0.0, 0.0, 2.5, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        self.assertTrue(torch.equal(reward, expected))


if __name__ == "__main__":
    unittest.main()
