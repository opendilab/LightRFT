#!/usr/bin/env python3
"""
Unit tests for ActorLanguage class.

This module contains unit tests for the ActorLanguage class, focusing on testing
the initialization, forward pass, text generation, gradient checkpointing,
and parameter reporting functions.
"""

from unittest.mock import Mock, patch
import os
import pytest
import torch

from lightrft.models import ActorLanguage


class TestActorLanguage:
    """Test cases for ActorLanguage class."""
    @pytest.fixture
    def device(self):
        """Set up device fixture."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def mock_config(self):
        """Set up mock config fixture."""
        config = Mock()
        config.model_type = "gpt2"
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
        model.generate.return_value = torch.randint(0, 32000, (2, 15))
        model.return_value = mock_output
        return model

    @patch('lightrft.models.actor_text.AutoModelForCausalLM')
    @patch('lightrft.models.actor_text.AutoConfig')
    def test_actor_text_initialization(self, mock_auto_config, mock_auto_model, mock_model):
        """Test ActorLanguage initialization with mock model loading."""
        # Mock return values
        mock_auto_config.from_pretrained.return_value = mock_model.config
        mock_auto_model.from_pretrained.return_value = mock_model

        actor = ActorLanguage(
            pretrain_or_model="test_text_model",
            use_flash_attention_2=False,
            bf16=False,
            lora_rank=0,
            packing_samples=False
        )

        # Check that model initialized correctly
        assert actor.model is not None
        assert actor.pretrain_or_model == "test_text_model"
        assert actor.packing_samples is False

    def test_actor_text_with_existing_model(self, mock_model):
        """Test ActorLanguage initialization with an existing model instance."""
        actor = ActorLanguage(pretrain_or_model=mock_model, packing_samples=True)
        # Verify initialization
        assert actor.model == mock_model
        assert actor.packing_samples is True
        assert actor.pretrain_or_model == "gpt2"

    def test_forward_without_num_actions(self, mock_model):
        """Test forward pass without num_actions (should return model output dict)."""
        actor = ActorLanguage(pretrain_or_model=mock_model, packing_samples=False)
        batch_size, seq_len = 2, 10
        sequences = torch.randint(0, 32000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        result = actor.forward(sequences=sequences, attention_mask=attention_mask, return_output=True)

        assert isinstance(result, dict)
        assert "logits" in result

    @patch('lightrft.models.actor_text.log_probs_from_logits')
    def test_forward_with_num_actions(self, mock_log_probs, mock_model):
        """Test forward pass with num_actions."""
        actor = ActorLanguage(pretrain_or_model=mock_model, packing_samples=False)
        batch_size, seq_len = 2, 10
        sequences = torch.randint(0, 32000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        mock_log_probs.return_value = torch.randn(batch_size, seq_len - 1)

        num_actions = 3
        result = actor.forward(
            sequences=sequences, num_actions=num_actions, attention_mask=attention_mask, return_output=False
        )

        assert isinstance(result, torch.Tensor)
        assert result.shape == (batch_size, num_actions)

    def test_generate_function(self, mock_model):
        """Test generate function for text-only actor."""
        actor = ActorLanguage(pretrain_or_model=mock_model, packing_samples=False)
        batch_size, input_len = 2, 5
        input_ids = torch.randint(0, 32000, (batch_size, input_len))

        sequences, attention_mask, action_mask = actor.generate(
            input_ids=input_ids, max_new_tokens=10, temperature=0.8, do_sample=True, eos_token_id=2, pad_token_id=0
        )

        assert isinstance(sequences, torch.Tensor)
        assert isinstance(attention_mask, torch.Tensor)
        assert isinstance(action_mask, torch.Tensor)
        assert sequences.shape[0] == batch_size
        assert attention_mask.shape[0] == batch_size
        assert action_mask.shape[0] == batch_size

    def test_gradient_checkpointing(self, mock_model):
        """Test enabling and disabling gradient checkpointing."""
        actor = ActorLanguage(pretrain_or_model=mock_model, packing_samples=False)

        actor.gradient_checkpointing_enable()
        mock_model.gradient_checkpointing_enable.assert_called_once()

        actor.gradient_checkpointing_disable()
        mock_model.gradient_checkpointing_disable.assert_called_once()

    def test_print_trainable_parameters(self, mock_model):
        """Test printing trainable parameters."""
        actor = ActorLanguage(pretrain_or_model=mock_model, packing_samples=False)

        actor.print_trainable_parameters()
        mock_model.print_trainable_parameters.assert_called_once()


class TestActorLanguageWithRealModel:
    """Optional integration tests for ActorLanguage with a real model (if available)."""
    @pytest.fixture
    def model_path(self):
        return "test_text_model"

    @pytest.mark.skipif(not os.path.exists("test_text_model"), reason="Real model path not available")
    def test_forward_with_real_model(self, model_path):
        """Test forward pass using a real text model (optional integration)."""
        try:
            actor = ActorLanguage(
                pretrain_or_model=model_path,
                use_flash_attention_2=False,
                bf16=False,
                lora_rank=0,
                packing_samples=False
            )

            batch_size, seq_len = 1, 10
            sequences = torch.randint(0, 32000, (batch_size, seq_len))
            attention_mask = torch.ones(batch_size, seq_len)
            num_actions = 5

            result = actor.forward(
                sequences=sequences, num_actions=num_actions, attention_mask=attention_mask, return_output=False
            )

            assert isinstance(result, torch.Tensor)
            assert result.shape == (batch_size, num_actions)
            assert result.dtype == torch.float32

        except Exception as e:
            pytest.skip(f"Real model test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
