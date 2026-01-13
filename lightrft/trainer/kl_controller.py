import numpy as np


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """
    def __init__(self, init_kl_coef, target, horizon):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current, n_steps):
        """
        Update KL coefficient using adaptive algorithm.

        :param current: Current KL divergence value.
        :type current: float
        :param n_steps: Number of training steps taken.
        :type n_steps: int
        """
        target = self.target
        proportional_error = np.clip(current / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""
    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current, n_steps):
        """
        Update KL controller state (no-op for fixed KL).

        :param current: Current KL divergence value (unused).
        :type current: float
        :param n_steps: Number of training steps (unused).
        :type n_steps: int
        """
        pass
