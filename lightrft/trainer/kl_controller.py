import numpy as np


class AdaptiveKLController:
    """
    Adaptive KL controller for PPO training.

    Implements the adaptive KL penalty coefficient adjustment described in:
    "Fine-Tuning Language Models from Human Preferences"
    (https://arxiv.org/pdf/1909.08593.pdf)

    This controller dynamically adjusts the KL penalty coefficient based on how
    the current KL divergence compares to a target value, helping maintain stable
    training while preventing the policy from deviating too far from the reference.
    """
    def __init__(self, init_kl_coef: float, target: float, horizon: int) -> None:
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current: float, n_steps: int) -> None:
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
    """
    Fixed KL controller that maintains a constant KL penalty coefficient.

    Unlike the adaptive controller, this keeps the KL coefficient fixed throughout
    training, providing consistent regularization strength.
    """
    def __init__(self, kl_coef: float) -> None:
        self.value = kl_coef

    def update(self, current: float, n_steps: int) -> None:
        """
        Update KL controller state (no-op for fixed KL).

        :param current: Current KL divergence value (unused).
        :type current: float
        :param n_steps: Number of training steps (unused).
        :type n_steps: int
        """
        pass
