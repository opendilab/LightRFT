import unittest
import argparse
import torch

from lightrft.utils import get_strategy
from lightrft.models.actor_vl import ActorVL


class TestActorVL(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # prepare args
        parser = argparse.ArgumentParser()
        parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
        parser.add_argument("--zero_stage", type=int, default=3, help="DeepSpeed ZeRO stage")
        parser.add_argument("--rollout_batch_size", type=int, default=32)
        parser.add_argument("--prompt_max_len", type=int, default=4096, help="Max tokens for each prompt")
        parser.add_argument(
            "--disable_logprobs_flashattn",
            action="store_true",
            default=False,
            help="Disable flash attn implementation in log_probs calculation"
        )
        parser.add_argument(
            "--fused_linear_logprob",
            action="store_true",
            default=True,
            help="Use FusedLinearChunkedLogProb to calculate log_probs to avoid pytorch to cache the logits"
        )
        parser.add_argument(
            "--chunk_size",
            type=int,
            default=8192,
            help="The chunk size of calculating log_prob when using fused_linear_logprob method"
        )
        parser.add_argument("--fsdp", action="store_true", default=True, help="use fsdp strategy")
        args = parser.parse_args()

        strategy = get_strategy(args)

        model_path = ("/fs-computility/ai-shen/shared/shidongxing/actor_qwenvl7b/")
        actor = ActorVL(
            model_path,
            use_flash_attention_2=True,
            bf16=True,
            disable_logprobs_flashattn=args.disable_logprobs_flashattn,
            fused_linear_logprob=args.fused_linear_logprob,
        )

        strategy.setup_distributed()
        strategy.print(actor)

        # gradient_checkpointing
        actor.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        cls.max_test_steps = 2
        cls.args = args
        cls.actor = actor

    def test_fused_linear_logprob_accuracy(self):
        """
        Test fused linear logprob accuracy in ActorVL.
        """
        pixel_values = None
        image_grid_thw = None
        inputs_extra_kwargs = {
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }
        device = torch.cuda.current_device()
        self.actor = self.actor.to(device)
        baseline_log_probs = []
        fused_linear_log_probs = []
        for i in range(self.max_test_steps):
            float_tensor = torch.rand(8, 1000).to(device)
            scaled_tensor = float_tensor * 152064
            sequences = scaled_tensor.long()
            num_actions = 2000
            rand_mask = torch.rand(8, 1000).to(device)
            attn_mask = rand_mask < 1
            orig_sequences = sequences
            with torch.no_grad():
                baseline_log_prob = self.actor(
                    sequences,
                    num_actions,
                    attn_mask,
                    **inputs_extra_kwargs,
                )
                baseline_log_probs.append(baseline_log_prob)

            self.assertTrue(torch.equal(sequences, orig_sequences))

            with torch.no_grad():
                fused_linear_log_prob = self.actor(
                    sequences,
                    num_actions,
                    attn_mask,
                    fwd_fused_linear_logprob=True,
                    chunk_size=self.args.chunk_size,
                    **inputs_extra_kwargs,
                )
                fused_linear_log_probs.append(fused_linear_log_prob)

        for i in range(self.max_test_steps):
            self.assertEqual(baseline_log_probs[i].shape, fused_linear_log_probs[i].shape)
            self.assertTrue(torch.allclose(baseline_log_probs[i], fused_linear_log_probs[i], atol=1e-7))


if __name__ == "__main__":
    unittest.main(verbosity=2)
