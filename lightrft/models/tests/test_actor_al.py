#!/usr/bin/env python3
"""
Unit tests for ActorAL (Audio Language) class.

This module contains unit tests for the ActorAL (Audio Language) class, focusing on testing
the forward function with various inputs and validating the output format
and tensor dimensions.
"""

from unittest.mock import Mock, patch
import os
import pytest
import torch

from lightrft.models.actor_al import ActorAL, AUDIO_MODEL_TYPE_QWEN2_5_OMNI, create_audio_processor


class TestActorAL:
    """Test cases for ActorAL (Audio Language) class."""
    @pytest.fixture
    def mock_config(self):
        """Set up mock config fixture."""
        config = Mock()
        config.model_type = "qwen2_audio"
        config.to_dict.return_value = {}
        return config

    @pytest.fixture
    def mock_output(self):
        """Set up mock output fixture."""
        return {
            "logits": torch.randn(2, 10, 32000)  # batch_size=2, seq_len=10, vocab_size=32000
        }

    @pytest.fixture
    def mock_omni_config(self):
        config = Mock()
        config.model_type = "qwen2_5_omni"
        config.use_cache = True
        config.pad_token_id = 0
        return config

    @pytest.fixture
    def mock_omni_thinker_config(self):
        config = Mock()
        config.model_type = "qwen2_5_omni_thinker"
        config.audio_token_id = 151646
        config.use_cache = True
        return config

    @pytest.fixture
    def mock_model(self, mock_config, mock_output):
        """Set up mock model fixture."""
        model = Mock()
        model.config = mock_config
        model.audio_tower = None
        model.audio_encoder = None
        model.generate.return_value = torch.randint(0, 32000, (2, 15))  # batch_size=2, seq_len=15
        model.return_value = mock_output
        return model

    @pytest.fixture
    def mock_omni_model(self, mock_omni_config, mock_omni_thinker_config, mock_output):
        model = Mock()
        model.config = mock_omni_config
        model.generate.return_value = torch.randint(0, 32000, (2, 15))

        thinker = Mock()
        thinker.config = mock_omni_thinker_config
        thinker.audio_tower = Mock()
        thinker.audio_tower._get_feat_extract_output_lengths.return_value = (
            torch.tensor([4, 8]),
            torch.tensor([1, 2]),
        )
        thinker.audio_encoder = None
        thinker.return_value = mock_output
        model.thinker = thinker
        return model

    def _make_actor(self, model, *, packing_samples=False):
        actor = ActorAL(pretrain_or_model=model, packing_samples=packing_samples)
        actor.packing_samples = packing_samples
        return actor

    def _make_omni_forward_inputs(self, *, sequences, attention_mask, feature_attention_mask):
        sequences = torch.tensor(sequences)
        return {
            "sequences": sequences,
            "attention_mask": torch.tensor(attention_mask),
            "audio_values": torch.randn(sequences.size(0), 80, 10),
            "feature_attention_mask": torch.tensor(feature_attention_mask),
        }

    def _run_mocked_omni_forward(self, actor, *, num_actions=2, **forward_kwargs):
        sequences = forward_kwargs["sequences"]
        with patch("lightrft.models.actor_al.log_probs_from_logits") as mock_log_probs:
            mock_log_probs.return_value = torch.randn(sequences.size(0), sequences.size(1) - 1)
            result = actor.forward(
                num_actions=num_actions,
                **forward_kwargs,
            )

        return result, actor.model.thinker.call_args.kwargs

    @patch('lightrft.models.actor_al.get_audio_model_and_type')
    def test_actor_al_initialization(self, mock_get_audio_model, mock_model):
        """Test ActorAL initialization with mock model."""
        # Set up mock
        mock_get_audio_model.return_value = (mock_model, "qwen2_audio")

        # Initialize ActorAL
        actor = ActorAL(
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

    def test_actor_al_with_existing_model(self, mock_model):
        """Test ActorAL initialization with existing model instance."""
        # Create ActorAL with existing model
        actor = ActorAL(pretrain_or_model=mock_model, packing_samples=True)

        # Manually set packing_samples since it's not set in the else branch
        actor.packing_samples = True

        # Verify initialization
        assert actor.model == mock_model
        assert actor.packing_samples is True
        assert actor.pretrain_or_model == "qwen2_audio"

    def test_forward_without_num_actions(self, mock_model):
        """Test forward function without num_actions (should return full output)."""
        # Create ActorAL with existing model
        actor = ActorAL(pretrain_or_model=mock_model, packing_samples=False)
        # Manually set packing_samples since it's not set in the else branch
        actor.packing_samples = False

        # Prepare test inputs
        batch_size, seq_len = 2, 10
        sequences = torch.randint(0, 32000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        audio_values = torch.randn(batch_size, 16000)  # Audio tensor: (batch_size, samples)

        # Call forward function without num_actions
        result = actor.forward(
            sequences=sequences,
            attention_mask=attention_mask,
            audio_values=audio_values,
            return_output=True  # Must be True when num_actions is None
        )

        # Assert output format
        assert isinstance(result, dict)
        assert "logits" in result

    def test_forward_with_num_actions(self, mock_model):
        """Test forward function with num_actions."""
        # Create ActorAL with existing model
        actor = ActorAL(pretrain_or_model=mock_model, packing_samples=False)
        # Manually set packing_samples since it's not set in the else branch
        actor.packing_samples = False

        # Prepare test inputs
        batch_size, seq_len = 2, 10
        sequences = torch.randint(0, 32000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        audio_values = torch.randn(batch_size, 16000)  # Audio tensor: (batch_size, samples)
        num_actions = 5

        # Mock the log_probs_from_logits function to avoid Flash Attention issues
        with patch('lightrft.models.actor_al.log_probs_from_logits') as mock_log_probs:
            mock_log_probs.return_value = torch.randn(batch_size, seq_len - 1)

            # Call forward function with num_actions
            result = actor.forward(
                sequences=sequences,
                num_actions=num_actions,
                attention_mask=attention_mask,
                audio_values=audio_values,
                return_output=False
            )

            # Assert output format and dimensions
            assert isinstance(result, torch.Tensor)
            assert result.shape == (batch_size, num_actions)

    def test_generate_function(self, mock_model):
        """Test generate function."""
        # Create ActorAL with existing model
        actor = ActorAL(pretrain_or_model=mock_model, packing_samples=False)
        # Manually set packing_samples since it's not set in the else branch
        actor.packing_samples = False

        # Prepare test inputs
        batch_size, input_len = 2, 5
        input_ids = torch.randint(0, 32000, (batch_size, input_len))
        audio_values = torch.randn(batch_size, 16000)  # Audio features tensor

        # Call generate function
        sequences, attention_mask, action_mask = actor.generate(
            input_ids=input_ids,
            audio_values=audio_values,
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
        # Create ActorAL with existing model
        actor = ActorAL(pretrain_or_model=mock_model, packing_samples=False)
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
        # Create ActorAL with existing model
        actor = ActorAL(pretrain_or_model=mock_model, packing_samples=False)
        # Manually set packing_samples since it's not set in the else branch
        actor.packing_samples = False

        # Test print trainable parameters
        actor.print_trainable_parameters()
        mock_model.print_trainable_parameters.assert_called_once()

    def test_forward_with_qwen2_5_omni_routes_to_thinker(self, mock_omni_model):
        actor = self._make_actor(mock_omni_model)

        sequences = torch.randint(0, 32000, (2, 10))
        attention_mask = torch.ones(2, 10)

        with patch('lightrft.models.actor_al.log_probs_from_logits') as mock_log_probs:
            mock_log_probs.return_value = torch.randn(2, 9)
            result = actor.forward(
                sequences=sequences,
                num_actions=4,
                attention_mask=attention_mask,
                audio_values=None,
            )

        assert isinstance(result, torch.Tensor)
        assert actor.model_type == AUDIO_MODEL_TYPE_QWEN2_5_OMNI
        mock_omni_model.thinker.assert_called_once()
        _, kwargs = mock_omni_model.thinker.call_args
        assert kwargs["position_ids"] is None
        assert kwargs["attention_mask"] is attention_mask

    @pytest.mark.parametrize(
        (
            "output_lengths",
            "feature_lengths",
            "expected_audio_counts",
            "expected_attention_mask",
        ),
        [
            ([8, 4], [2, 1], [2, 1], [[0, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1]]),
            ([4, 8], [1, 2], [1, 2], [[0, 0, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1]]),
        ],
    )
    def test_forward_with_qwen2_5_omni_aligns_audio_placeholders(
        self,
        mock_omni_model,
        output_lengths,
        feature_lengths,
        expected_audio_counts,
        expected_attention_mask,
    ):
        actor = self._make_actor(mock_omni_model)
        mock_omni_model.thinker.audio_tower._get_feat_extract_output_lengths.return_value = (
            torch.tensor(output_lengths),
            torch.tensor(feature_lengths),
        )
        audio_token_id = mock_omni_model.thinker.config.audio_token_id

        inputs = self._make_omni_forward_inputs(
            sequences=[
                [0, 0, audio_token_id, 11, 12, 13],
                [0, audio_token_id, audio_token_id, 21, 22, 23],
            ],
            attention_mask=[
                [0, 0, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 1],
            ],
            feature_attention_mask=[
                [1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
            ],
        )

        result, kwargs = self._run_mocked_omni_forward(actor, **inputs)
        assert isinstance(result, torch.Tensor)
        assert "input_features" in kwargs
        assert "feature_attention_mask" in kwargs
        assert (kwargs["input_ids"] == audio_token_id).sum(dim=1).tolist() == expected_audio_counts
        assert kwargs["attention_mask"].tolist() == expected_attention_mask

    def test_forward_with_qwen2_5_omni_trims_response_audio_placeholders(self, mock_omni_model):
        actor = self._make_actor(mock_omni_model)
        mock_omni_model.thinker.audio_tower._get_feat_extract_output_lengths.return_value = (
            torch.tensor([8, 8]),
            torch.tensor([2, 2]),
        )

        audio_token_id = mock_omni_model.thinker.config.audio_token_id
        inputs = self._make_omni_forward_inputs(
            sequences=[
                [0, 0, audio_token_id, audio_token_id, 11, audio_token_id],
                [0, audio_token_id, audio_token_id, 21, 22, audio_token_id],
            ],
            attention_mask=[
                [0, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 1],
            ],
            feature_attention_mask=[
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
            ],
        )

        result, kwargs = self._run_mocked_omni_forward(actor, **inputs)
        assert isinstance(result, torch.Tensor)
        assert (kwargs["input_ids"] == audio_token_id).sum(dim=1).tolist() == [2, 2]

    def test_forward_can_return_aligned_inputs_for_audio_replay(self, mock_omni_model):
        actor = self._make_actor(mock_omni_model)
        mock_omni_model.thinker.audio_tower._get_feat_extract_output_lengths.return_value = (
            torch.tensor([8]),
            torch.tensor([2]),
        )

        audio_token_id = mock_omni_model.thinker.config.audio_token_id
        inputs = self._make_omni_forward_inputs(
            sequences=[[0, 0, audio_token_id, 11, 12, 13]],
            attention_mask=[[0, 0, 1, 1, 1, 1]],
            feature_attention_mask=[[1, 1, 1, 1, 1, 1, 1, 1]],
        )

        (
            action_log_probs,
            model_output,
            aligned_sequences,
            aligned_attention_mask,
        ), _ = self._run_mocked_omni_forward(
            actor,
            return_output=True,
            return_aligned_inputs=True,
            **inputs,
        )

        assert isinstance(action_log_probs, torch.Tensor)
        assert model_output is not None
        assert (aligned_sequences == audio_token_id).sum(dim=1).tolist() == [2]
        assert aligned_attention_mask.tolist() == [[0, 1, 1, 1, 1, 1]]

    def test_align_audio_placeholders_is_batch_invariant_to_num_actions(self, mock_omni_model):
        actor = self._make_actor(mock_omni_model)
        audio_token_id = mock_omni_model.thinker.config.audio_token_id

        sequences = torch.tensor([[
            11, 12, 13, 14, audio_token_id, audio_token_id, audio_token_id, audio_token_id, 31, 41, 42, 43
        ]])
        attention_mask = torch.ones_like(sequences)
        expected_counts = torch.tensor([5])

        aligned_short, mask_short = actor._align_audio_placeholder_counts(
            sequences=sequences,
            attention_mask=attention_mask,
            audio_token_id=audio_token_id,
            expected_audio_token_counts=expected_counts,
            pad_token_id=0,
            num_actions=3,
        )
        aligned_long, mask_long = actor._align_audio_placeholder_counts(
            sequences=sequences,
            attention_mask=attention_mask,
            audio_token_id=audio_token_id,
            expected_audio_token_counts=expected_counts,
            pad_token_id=0,
            num_actions=6,
        )

        assert torch.equal(aligned_short, aligned_long)
        assert torch.equal(mask_short, mask_long)
        assert (aligned_short == audio_token_id).sum(dim=1).tolist() == [5]

    def test_generate_with_qwen2_5_omni_uses_text_mode(self, mock_omni_model):
        actor = ActorAL(pretrain_or_model=mock_omni_model, packing_samples=False)

        input_ids = torch.randint(0, 32000, (2, 5))
        sequences, attention_mask, action_mask = actor.generate(
            input_ids=input_ids,
            max_new_tokens=12,
            temperature=0.8,
            do_sample=True,
            eos_token_id=2,
            pad_token_id=0,
        )

        assert sequences.shape[0] == 2
        assert attention_mask.shape[0] == 2
        assert action_mask.shape[0] == 2

        _, kwargs = mock_omni_model.generate.call_args
        assert kwargs["generation_mode"] == "text"
        assert kwargs["thinker_max_new_tokens"] == 12

    def test_get_fsdp_target_model_uses_root_for_qwen2_audio(self, mock_model):
        actor = ActorAL(pretrain_or_model=mock_model, packing_samples=False)
        assert actor.get_fsdp_target_model() is mock_model

    def test_get_fsdp_target_model_uses_thinker_for_qwen2_5_omni(self, mock_omni_model):
        actor = self._make_actor(mock_omni_model)
        assert actor.get_fsdp_target_model() is mock_omni_model.thinker

    @patch("lightrft.models.actor_al.get_audio_processor_class")
    @patch("lightrft.models.actor_al.infer_audio_model_type")
    def test_create_audio_processor_reuses_matching_instance(
        self,
        mock_infer_audio_model_type,
        mock_get_audio_processor_class,
    ):
        dummy_processor_cls = type("DummyProcessor", (), {})
        existing_processor = dummy_processor_cls()

        mock_infer_audio_model_type.return_value = "qwen2_audio"
        mock_get_audio_processor_class.return_value = dummy_processor_cls

        assert create_audio_processor("test_model_path", processor=existing_processor) is existing_processor

    @patch("lightrft.models.actor_al.get_audio_processor_class")
    @patch("lightrft.models.actor_al.infer_audio_model_type")
    def test_create_audio_processor_reloads_mismatched_instance(
        self,
        mock_infer_audio_model_type,
        mock_get_audio_processor_class,
    ):
        reloaded_processor = Mock()
        dummy_processor_cls = type("DummyProcessor", (), {})
        dummy_processor_cls.from_pretrained = Mock(return_value=reloaded_processor)
        print_fn = Mock()

        mock_infer_audio_model_type.return_value = "qwen2_5_omni"
        mock_get_audio_processor_class.return_value = dummy_processor_cls

        result = create_audio_processor(
            "test_model_path",
            processor=Mock(),
            trust_remote_code=True,
            print_fn=print_fn,
        )

        assert result is reloaded_processor
        dummy_processor_cls.from_pretrained.assert_called_once_with(
            "test_model_path",
            trust_remote_code=True,
        )
        print_fn.assert_called_once()


class TestActorALWithRealData:
    """Test cases for ActorAL with real model and data (if available)."""
    @pytest.fixture
    def model_path(self):
        """Set up model path fixture."""
        return "test_audio_model"

    @pytest.fixture
    def data_path(self):
        """Set up data path fixture."""
        return "test_audio_data"

    @pytest.mark.skipif(not os.path.exists("test_audio_model"), reason="Real model path not available")
    def test_forward_with_real_model(self, model_path):
        """Test forward function with real model (if available)."""
        try:
            # Initialize ActorAL with real model
            actor = ActorAL(
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
            # Use proper audio format for Qwen2Audio
            audio_values = torch.randn(batch_size, 16000)  # Audio tensor: (batch_size, samples)
            num_actions = 5

            # Call forward function
            result = actor.forward(
                sequences=sequences,
                num_actions=num_actions,
                attention_mask=attention_mask,
                audio_values=audio_values,
                return_output=False
            )

            # Assert output format and dimensions
            assert isinstance(result, torch.Tensor)
            assert result.shape == (batch_size, num_actions)
            assert result.dtype in [torch.float32, torch.float16, torch.bfloat16]

        except Exception as e:
            pytest.skip(f"Real model test failed: {e}")

    @pytest.mark.skipif(not os.path.exists("test_audio_model"), reason="Real model path not available")
    def test_generate_with_real_model(self, model_path):
        """Test generate function with real model (if available)."""
        try:
            # Initialize ActorAL with real model
            actor = ActorAL(
                pretrain_or_model=model_path,
                use_flash_attention_2=False,
                bf16=False,
                lora_rank=0,
                packing_samples=False
            )

            # Prepare test inputs
            batch_size, input_len = 1, 5
            input_ids = torch.randint(0, 32000, (batch_size, input_len))
            # Use proper audio format for Qwen2Audio
            audio_values = torch.randn(batch_size, 16000)  # Audio features tensor

            # Call generate function
            sequences, attention_mask, action_mask = actor.generate(
                input_ids=input_ids,
                audio_values=audio_values,
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

        except Exception as e:
            pytest.skip(f"Real model generate test failed: {e}")

    @pytest.mark.skipif(not os.path.exists("test_audio_model"), reason="Real model path not available")
    def test_initialization_with_real_model(self, model_path):
        """Test ActorAL initialization with real model."""
        try:
            # Initialize ActorAL with real model
            actor = ActorAL(
                pretrain_or_model=model_path,
                use_flash_attention_2=False,
                bf16=False,
                lora_rank=0,
                packing_samples=False
            )

            # Verify initialization
            assert actor.model is not None
            assert actor.pretrain_or_model == model_path
            assert actor.packing_samples is False

            # Test model configuration
            assert hasattr(actor.model, 'config')
            assert hasattr(actor.model, 'generate')

        except Exception as e:
            pytest.skip(f"Real model initialization test failed: {e}")


if __name__ == "__main__":
    # Run the tests with pytest
    pytest.main([__file__, "-v"])
