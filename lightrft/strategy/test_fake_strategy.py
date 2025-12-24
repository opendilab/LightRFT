"""
Unit tests for FakeStrategy in LightRFT.

This module contains comprehensive tests for the FakeStrategy class,
ensuring it behaves correctly as a drop-in replacement for real distributed
strategies in testing environments.
"""

import os
import tempfile
import unittest
from unittest.mock import MagicMock

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from lightrft.strategy.config import StrategyConfig

from .fake_strategy import FakeStrategy, get_fake_strategy


class SimpleModel(nn.Module):
    """Simple model for testing purposes."""
    def __init__(self, input_size=10, hidden_size=20, output_size=1):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class TestFakeStrategy(unittest.TestCase):
    """Test cases for FakeStrategy."""
    def setUp(self):
        """Set up test fixtures."""
        self.strategy = FakeStrategy(seed=42, max_norm=1.0, micro_train_batch_size=2, train_batch_size=8, args=None)

        self.model = SimpleModel()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)

        # Create dummy data
        self.x_data = torch.randn(16, 10)
        self.y_data = torch.randn(16, 1)
        self.dataset = TensorDataset(self.x_data, self.y_data)
        self.dataloader = DataLoader(self.dataset, batch_size=2)

    def test_initialization(self):
        """Test that FakeStrategy initializes correctly."""
        self.assertEqual(self.strategy.seed, 42)
        self.assertEqual(self.strategy.max_norm, 1.0)
        self.assertEqual(self.strategy.micro_train_batch_size, 2)
        self.assertEqual(self.strategy.train_batch_size, 8)
        self.assertEqual(self.strategy.world_size, 1)

    def test_setup_distributed(self):
        """Test that setup_distributed works without errors."""
        # Should not raise any exceptions
        self.strategy.setup_distributed()
        self.assertEqual(self.strategy.world_size, 1)

    def test_create_optimizer(self):
        """Test optimizer creation."""
        optimizer = self.strategy.create_optimizer(self.model, lr=0.001, weight_decay=0.01)
        self.assertIsInstance(optimizer, torch.optim.Optimizer)

        # Test with actor model
        actor_model = SimpleModel()
        setattr(actor_model, "is_actor", True)
        setattr(actor_model, "model", self.model)
        optimizer = self.strategy.create_optimizer(actor_model, lr=0.001)
        self.assertIsInstance(optimizer, torch.optim.Optimizer)

    def test_prepare(self):
        """Test model preparation."""
        # Test with single model
        prepared = self.strategy.prepare(self.model)
        self.assertEqual(prepared, self.model)

        # Test with model-optimizer pair
        prepared = self.strategy.prepare((self.model, self.optimizer, None))
        self.assertEqual(prepared, (self.model, self.optimizer, None))

        # Test with multiple inputs
        prepared = self.strategy.prepare(self.model, (self.model, self.optimizer, None))
        self.assertEqual(len(prepared), 2)
        self.assertEqual(prepared[0], self.model)
        self.assertEqual(prepared[1], (self.model, self.optimizer, None))

    def test_backward(self):
        """Test backward pass."""
        x = torch.randn(2, 10, requires_grad=True)
        y = self.model(x)
        loss = y.sum()

        # Clear gradients
        self.optimizer.zero_grad()

        # Perform backward pass
        self.strategy.backward(loss, self.model, self.optimizer)

        # Check that gradients are computed
        for param in self.model.parameters():
            self.assertIsNotNone(param.grad)

    def test_optimizer_step(self):
        """Test optimizer step with gradient clipping."""
        # Create a simple training step
        x = torch.randn(2, 10)
        y = self.model(x)
        loss = y.sum()

        # Perform backward pass
        loss.backward()

        # Record initial parameters
        initial_params = [param.clone() for param in self.model.parameters()]

        # Take optimizer step
        self.strategy.optimizer_step(self.optimizer, self.model, name="test")

        # Check that parameters changed
        for initial, current in zip(initial_params, self.model.parameters()):
            self.assertFalse(torch.equal(initial, current))

    def test_save_and_load_ckpt(self):
        """Test checkpoint saving and loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save checkpoint
            client_state = {"step": 100, "loss": 0.5}
            self.strategy.save_ckpt(self.model, temp_dir, tag="test_checkpoint", client_state=client_state)

            # Check that file was created
            checkpoint_path = os.path.join(temp_dir, "test_checkpoint.pt")
            self.assertTrue(os.path.exists(checkpoint_path))

            # Create a new model and load checkpoint
            new_model = SimpleModel()

            # Record initial parameters
            initial_params = [param.clone() for param in new_model.parameters()]

            # Load checkpoint
            load_path, loaded_state = self.strategy.load_ckpt(new_model, temp_dir, tag="test_checkpoint")

            # Check that parameters changed after loading
            for initial, current in zip(initial_params, new_model.parameters()):
                self.assertFalse(torch.equal(initial, current))

            # Check client state
            self.assertEqual(loaded_state["step"], 100)
            self.assertEqual(loaded_state["loss"], 0.5)

    def test_all_reduce(self):
        """Test all_reduce operation."""
        # Test with tensor
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = self.strategy.all_reduce(tensor, op="mean")
        self.assertTrue(torch.equal(tensor, result))

        # Test with dict
        data_dict = {"loss": torch.tensor(1.5), "accuracy": torch.tensor(0.8)}
        result = self.strategy.all_reduce(data_dict, op="sum")
        self.assertTrue(torch.equal(data_dict["loss"], result["loss"]))
        self.assertTrue(torch.equal(data_dict["accuracy"], result["accuracy"]))

    def test_all_gather(self):
        """Test all_gather operation."""
        # Test with tensor
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = self.strategy.all_gather(tensor)
        self.assertEqual(len(result), 1)
        self.assertTrue(torch.equal(tensor, result[0]))

        # Test with dict
        data_dict = {"loss": torch.tensor(1.5), "accuracy": torch.tensor(0.8)}
        result = self.strategy.all_gather(data_dict)
        self.assertEqual(len(result["loss"]), 1)
        self.assertEqual(len(result["accuracy"]), 1)
        self.assertTrue(torch.equal(data_dict["loss"], result["loss"][0]))
        self.assertTrue(torch.equal(data_dict["accuracy"], result["accuracy"][0]))

    def test_rank_methods(self):
        """Test rank-related methods."""
        self.assertTrue(self.strategy.is_rank_0())
        self.assertEqual(self.strategy.get_rank(), 0)

    def test_inference_engine_methods(self):
        """Test inference engine related methods."""
        # These should not raise exceptions
        self.strategy.setup_inference_engine(None)
        self.strategy.maybe_sleep_inference_engine()
        self.strategy.wakeup_inference_engine()

        # Test generation methods
        result = self.strategy.engine_generate_local(None)
        self.assertEqual(result, [])

        result = self.strategy.gather_and_generate(None)
        self.assertEqual(result, [])

    def test_update_engine_weights(self):
        """Test engine weight update."""
        # Should not raise exceptions
        self.strategy.update_engine_weights(self.model)

    def test_context_managers(self):
        """Test context managers."""
        with self.strategy.init_model_context():
            # Should not raise exceptions
            pass

    def test_optimizer_offloading(self):
        """Test optimizer offloading methods."""
        original_optimizer = self.optimizer

        # Test offloading
        offloaded = self.strategy.maybe_offload_optimizer(original_optimizer)
        self.assertIs(offloaded, original_optimizer)

        # Test loading
        loaded = self.strategy.maybe_load_optimizer(original_optimizer)
        self.assertIs(loaded, original_optimizer)

    def test_get_fake_strategy(self):
        """Test get_fake_strategy convenience function."""
        # Test with None args
        strategy = get_fake_strategy()
        self.assertIsInstance(strategy, FakeStrategy)

        # Test with mock args
        mock_args = MagicMock()
        mock_args.seed = 123
        mock_args.max_norm = 2.0
        mock_args.micro_train_batch_size = 4
        mock_args.train_batch_size = 16

        strategy = get_fake_strategy(mock_args)
        self.assertIsInstance(strategy, FakeStrategy)
        self.assertEqual(strategy.seed, 123)
        self.assertEqual(strategy.max_norm, 2.0)
        self.assertEqual(strategy.micro_train_batch_size, 4)
        self.assertEqual(strategy.train_batch_size, 16)


class TestFakeStrategyWithConfig(unittest.TestCase):
    """Test FakeStrategy with StrategyConfig integration."""
    def test_strategy_config_integration(self):
        """Test that FakeStrategy works with StrategyConfig."""
        config = StrategyConfig(seed=123, max_norm=2.0, micro_train_batch_size=4, train_batch_size=16, bf16=False)

        strategy = FakeStrategy(
            seed=config.seed,
            max_norm=config.max_norm,
            micro_train_batch_size=config.micro_train_batch_size,
            train_batch_size=config.train_batch_size,
            args=None
        )

        self.assertEqual(strategy.seed, 123)
        self.assertEqual(strategy.max_norm, 2.0)
        self.assertEqual(strategy.micro_train_batch_size, 4)
        self.assertEqual(strategy.train_batch_size, 16)

    def test_config_from_args_in_strategy(self):
        """Test that StrategyConfig.from_args is used internally."""

        # Create mock args with various attributes
        class MockArgs:
            def __init__(self):
                self.seed = 999
                self.max_norm = 3.0
                self.micro_train_batch_size = 8
                self.train_batch_size = 32
                self.bf16 = True
                self.fsdp = False
                self.adam_offload = True
                self.zpg = 2
                self.grad_accum_dtype = "fp32"
                self.overlap_comm = True
                self.engine_type = "vllm"
                self.engine_tp_size = 4
                self.enable_engine_sleep = True
                self.local_rank = 0
                self.sp_size = 2
                self.actor_learning_rate = 1e-4
                self.critic_learning_rate = 1e-4
                self.adam_betas = (0.9, 0.999)
                self.l2 = 0.01
                self.lr_warmup_ratio = 0.1
                self.critic_pretrain = True
                self.remote_rm_url = None
                self.pretrain_data = None
                self.fused_linear_logprob = False
                self.mixed_mm_data = False
                self.use_mp_opt = True
                self.plot_every = 100
                self.use_tensorboard = True
                self.fsdp_cpu_offload = False

        mock_args = MockArgs()
        strategy = FakeStrategy(args=mock_args)

        # Verify that config was created and used
        self.assertIsNotNone(strategy.config)
        self.assertEqual(strategy.config.seed, 999)
        self.assertEqual(strategy.config.max_norm, 3.0)
        self.assertEqual(strategy.config.adam_offload, True)
        self.assertEqual(strategy.config.zpg, 2)


if __name__ == "__main__":
    unittest.main()
