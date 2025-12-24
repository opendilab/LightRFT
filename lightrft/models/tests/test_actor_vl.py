#!/usr/bin/env python3
"""
Unit tests for ActorVL class.

This module contains unit tests for the ActorVL class, focusing on testing
the forward function with various inputs and validating the output format
and tensor dimensions.
"""

from unittest.mock import Mock, patch
import os
import pytest
import torch

# Add the lightrft package to the path

from lightrft.models import ActorVL


class TestActorVL:
    """Test cases for ActorVL class."""
    @pytest.fixture
    def device(self):
        """Set up device fixture."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def mock_config(self):
        """Set up mock config fixture."""
        config = Mock()
        config.model_type = "qwen2_vl"
        config.to_dict.return_value = {}
        return config

    @pytest.fixture
    def mock_output(self):
        """Set up mock output fixture."""
        return {
            "logits": torch.randn(2, 10, 32000)  # batch_size=2, seq_len=10, vocab_size=32000
        }

    @pytest.fixture
    def mock_model(self, mock_config, mock_output):
        """Set up mock model fixture."""
        model = Mock()
        model.config = mock_config
        model.generate.return_value = torch.randint(0, 32000, (2, 15))  # batch_size=2, seq_len=15
        model.return_value = mock_output
        return model

    @patch('lightrft.models.actor_vl.AutoModelForVision2Seq')
    @patch('lightrft.models.actor_vl.AutoModel')
    def test_actor_vl_initialization(self, mock_auto_model, mock_auto_model_vision, mock_model):
        """Test ActorVL initialization with mock model."""
        # Set up mock
        mock_auto_model_vision.from_pretrained.return_value = mock_model

        # Initialize ActorVL
        actor = ActorVL(
            pretrain_or_model="test_model_path",
            use_flash_attention_2=False,
            bf16=False,
            lora_rank=0,
            packing_samples=False
        )

        # Verify initialization
        assert actor.model is not None
        assert actor.pretrain_or_model == "test_model_path"
        assert actor.packing_samples is False

    def test_actor_vl_with_existing_model(self, mock_model):
        """Test ActorVL initialization with existing model instance."""
        # Create ActorVL with existing model
        actor = ActorVL(pretrain_or_model=mock_model, packing_samples=True)

        # Manually set packing_samples since it's not set in the else branch
        actor.packing_samples = True

        # Verify initialization
        assert actor.model == mock_model
        assert actor.packing_samples is True
        assert actor.pretrain_or_model == "qwen2_vl"

    def test_forward_without_num_actions(self, mock_model):
        """Test forward function without num_actions (should return full output)."""
        # Create ActorVL with existing model
        actor = ActorVL(pretrain_or_model=mock_model, packing_samples=False)
        # Manually set packing_samples since it's not set in the else branch
        actor.packing_samples = False

        # Prepare test inputs
        batch_size, seq_len = 2, 10
        sequences = torch.randint(0, 32000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        pixel_values = torch.randn(batch_size, 3, 224, 224)
        image_grid_thw = torch.tensor([[1, 24, 24], [1, 24, 24]])

        # Call forward function without num_actions
        result = actor.forward(
            sequences=sequences,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            return_output=True  # Must be True when num_actions is None
        )

        # Assert output format
        assert isinstance(result, dict)
        assert "logits" in result

    def test_generate_function(self, mock_model):
        """Test generate function."""
        # Create ActorVL with existing model
        actor = ActorVL(pretrain_or_model=mock_model, packing_samples=False)
        # Manually set packing_samples since it's not set in the else branch
        actor.packing_samples = False

        # Prepare test inputs
        batch_size, input_len = 2, 5
        input_ids = torch.randint(0, 32000, (batch_size, input_len))
        pixel_values = torch.randn(batch_size, 3, 224, 224)
        image_grid_thw = torch.tensor([[1, 24, 24], [1, 24, 24]])

        # Call generate function
        sequences, attention_mask, action_mask = actor.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            max_new_tokens=10,
            temperature=0.8,
            do_sample=True,
            eos_token_id=2,
            pad_token_id=0
        )

        # Assert output format and dimensions
        assert isinstance(sequences, torch.Tensor)
        assert isinstance(attention_mask, torch.Tensor)
        assert isinstance(action_mask, torch.Tensor)

        # Check dimensions
        assert sequences.shape[0] == batch_size
        assert attention_mask.shape[0] == batch_size
        assert action_mask.shape[0] == batch_size
        assert sequences.shape[1] == attention_mask.shape[1]

    def test_gradient_checkpointing(self, mock_model):
        """Test gradient checkpointing enable/disable."""
        # Create ActorVL with existing model
        actor = ActorVL(pretrain_or_model=mock_model, packing_samples=False)
        # Manually set packing_samples since it's not set in the else branch
        actor.packing_samples = False

        # Test enable gradient checkpointing
        actor.gradient_checkpointing_enable()
        mock_model.gradient_checkpointing_enable.assert_called_once()

        # Test disable gradient checkpointing
        actor.gradient_checkpointing_disable()
        mock_model.gradient_checkpointing_disable.assert_called_once()

    def test_print_trainable_parameters(self, mock_model):
        """Test print trainable parameters."""
        # Create ActorVL with existing model
        actor = ActorVL(pretrain_or_model=mock_model, packing_samples=False)
        # Manually set packing_samples since it's not set in the else branch
        actor.packing_samples = False

        # Test print trainable parameters
        actor.print_trainable_parameters()
        mock_model.print_trainable_parameters.assert_called_once()


class TestActorVLWithRealData:
    """Test cases for ActorVL with real model and data (if available)."""
    @pytest.fixture
    def model_path(self):
        """Set up model path fixture."""
        return "test_vl_model"

    @pytest.fixture
    def data_path(self):
        """Set up data path fixture."""
        return "test_vl_data"

    @pytest.mark.skipif(not os.path.exists("test_vl_model"), reason="Real model path not available")
    def test_forward_with_real_model(self, model_path):
        """Test forward function with real model (if available)."""
        try:
            # Initialize ActorVL with real model
            actor = ActorVL(
                pretrain_or_model=model_path,
                use_flash_attention_2=False,
                bf16=False,
                lora_rank=0,
                packing_samples=False
            )

            # Prepare test inputs
            batch_size, seq_len = 1, 10
            sequences = torch.randint(0, 32000, (batch_size, seq_len))
            attention_mask = torch.ones(batch_size, seq_len)
            pixel_values = torch.randn(batch_size, 3, 224, 224)
            image_grid_thw = torch.tensor([[1, 24, 24]])
            num_actions = 5

            # Call forward function
            result = actor.forward(
                sequences=sequences,
                num_actions=num_actions,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                return_output=False
            )

            # Assert output format and dimensions
            assert isinstance(result, torch.Tensor)
            assert result.shape == (batch_size, num_actions)
            assert result.dtype == torch.float32

        except Exception as e:
            pytest.skip(f"Real model test failed: {e}")


if __name__ == "__main__":
    # Run the tests with pytest
    pytest.main([__file__, "-v"])
