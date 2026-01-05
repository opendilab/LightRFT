"""
Trainer for generative reward models (vision-language capable).

This module contains a trainer that optimizes a generative reward model using
next-token prediction loss (``GPTLMLoss``). It integrates with the Strategy
abstraction for distributed training, gradient accumulation, checkpointing, and
logging via Weights & Biases or TensorBoard.
"""

import os
import json
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import Optimizer
from typing import Dict, Any

from lightrft.models import GPTLMLoss
from lightrft.datasets.utils import extract_answer
from lightrft.utils import DistributedSampler, all_gather_and_flatten, all_reduce_dict


class GRMTrainerVL:
    """
    Trainer for generative reward models.

    :param model: The model to be trained. Expected to return logits for
        next-token prediction when called with token ids and optional image or
        video features.
    :type model: torch.nn.Module
    :param strategy: The training strategy to apply, handling distributed
        setup, gradient accumulation, logging and checkpointing.
    :type strategy: Strategy
    :param optim: Optimizer to use during training.
    :type optim: torch.optim.Optimizer
    :param train_dataloader: Dataloader for the training dataset.
    :type train_dataloader: torch.utils.data.DataLoader
    :param scheduler: Learning rate scheduler for dynamic adjustments during
        training.
    :type scheduler: torch.optim.lr_scheduler._LRScheduler
    :param tokenizer: Tokenizer for input data (used when padding/processing
        sequences as needed by some helpers).
    :type tokenizer: Callable
    :param eval_dataloader: Dataloader for the evaluation dataset.
    :type eval_dataloader: Optional[torch.utils.data.DataLoader]
    :param max_epochs: Maximum number of training epochs.
    :type max_epochs: int
    :param loss: The loss function selector.
    :type loss: str
    """
    def __init__(
        self,
        model: nn.Module,
        strategy,
        optim: Optimizer,
        train_dataloader,
        scheduler,
        tokenizer,
        eval_dataloader=None,
        max_epochs: int = 2,
        loss: str = "GPTLMLoss",
    ) -> None:
        self.strategy = strategy
        self.epochs = max_epochs
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.scheduler = scheduler
        self.optimizer = optim
        self.tokenizer = tokenizer
        self.args = strategy.args

        if loss == 'GPTLMLoss':
            self.loss_fn = GPTLMLoss()
            self.strategy.print("GPT Language Model Loss")
        else:
            raise ValueError(f"invalid loss type: {loss}")

        # wandb/tensorboard setting
        self._wandb = None
        self._tensorboard = None
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=strategy.args.use_wandb)
            wandb.init(
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                config=strategy.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/global_step")
            wandb.define_metric("eval/*", step_metric="eval/global_step", step_sync=True)

        # Initialize TensorBoard writer if wandb is not available
        if self.strategy.args.use_tensorboard and self._wandb is None and self.strategy.is_rank_0():
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(self.strategy.args.use_tensorboard, exist_ok=True)
            log_dir = os.path.join(self.strategy.args.use_tensorboard, strategy.args.wandb_run_name)
            self._tensorboard = SummaryWriter(log_dir=log_dir)

    def fit(self, args, consumed_samples=0, num_update_steps_per_epoch=None) -> None:
        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = num_update_steps_per_epoch  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        # Restore step and start_epoch
        step = consumed_samples // args.train_batch_size * self.strategy.accumulated_gradient + 1
        start_epoch = consumed_samples // args.train_batch_size // num_update_steps_per_epoch
        consumed_samples = consumed_samples % (num_update_steps_per_epoch * args.train_batch_size)

        epoch_bar = tqdm(range(start_epoch, self.epochs), desc="Train epoch", disable=not self.strategy.is_rank_0())

        loss_mean = 0.0
        for epoch in range(start_epoch, self.epochs):
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(
                    epoch, consumed_samples=0 if epoch > start_epoch else consumed_samples
                )

            step_bar = tqdm(
                range(self.train_dataloader.__len__()),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            self.model.train()
            for data in self.train_dataloader:
                ids, mask, pixel_values, image_grid_thws, pixel_values_videos, video_grid_thws, labels, extras = data
                device = torch.cuda.current_device()
                ids = ids.squeeze(1).to(device)
                mask = mask.squeeze(1).to(device)
                labels = labels.squeeze(1).to(device)

                if pixel_values is not None:
                    pixel_values = pixel_values.to(device)
                    image_grid_thws = image_grid_thws.to(device)

                if pixel_values_videos is not None:
                    pixel_values_videos = pixel_values_videos.to(device)
                    video_grid_thws = video_grid_thws.to(device)

                logits = self.model(
                    ids,
                    attention_mask=mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thws,
                    pixel_values_videos=pixel_values_videos,
                    video_grid_thw=video_grid_thws,
                    return_outputs=False,
                )
                gpt_loss = self.loss_fn(logits, labels)

                total_loss = gpt_loss
                self.strategy.backward(total_loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                loss_mean = loss_mean * 0.9 + 0.1 * total_loss.item()

                logs_dict = {
                    "loss": total_loss.item(),
                    "loss_mean": loss_mean,
                    "lr": self.scheduler.get_last_lr()[0],
                }

                # step bar
                for k in logs_dict.keys():
                    logs_dict[k] = self.strategy.all_reduce(logs_dict[k])

                logs_dict = all_reduce_dict(logs_dict, op="mean")
                step_bar.set_postfix(logs_dict)
                step_bar.update()

                # logs/checkpoints/evaluation
                if step % self.strategy.accumulated_gradient == 0:
                    global_step = step // self.strategy.accumulated_gradient
                    client_states = {"consumed_samples": global_step * args.train_batch_size}
                    self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict, client_states)

                step += 1
            epoch_bar.update()

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        if self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()

    # logs/checkpoints/evaluate
    def save_logs_and_checkpoints(
        self,
        args,
        global_step: int,
        step_bar,
        logs_dict: Dict[str, float] = {},
        client_states: Dict[str, Any] = {}
    ) -> None:
        """
        Log metrics and optionally run evaluation and checkpointing.

        :param args: Training arguments providing logging/eval/save intervals
            and checkpoint configurations.
        :type args: Any
        :param global_step: Current global optimization step.
        :type global_step: int
        :param step_bar: Progress bar for step-level updates.
        :type step_bar: tqdm.tqdm
        :param logs_dict: Dictionary of metrics to log.
        :type logs_dict: Dict[str, float]
        :param client_states: Extra state to persist with checkpoints (e.g.,
            consumed samples).
        :type client_states: Dict[str, Any]

        :returns: None
        :rtype: NoneType
        """
        if global_step % args.logging_steps == 0:
            # wandb
            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {"train/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}
                self._wandb.log(logs)
            # TensorBoard
            elif self._tensorboard is not None and self.strategy.is_rank_0():
                for k, v in logs_dict.items():
                    self._tensorboard.add_scalar(f"train/{k}", v, global_step)

        # eval
        if global_step % args.eval_steps == 0:
            # do eval when len(dataloader) > 0, avoid zero division in eval.
            if self.eval_dataloader and len(self.eval_dataloader) > 0:
                # Pass args first to match evaluate signature (args, dataloader, steps)
                self.evaluate(args, self.eval_dataloader, global_step)
        # save ckpt
        # TODO: save best model on dev, use loss/perplexity on whole dev dataset as metric
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            self.strategy.save_ckpt(
                self.model, args.ckpt_path, tag, args.max_ckpt_num, args.max_ckpt_mem, client_states
            )

    def evaluate(self, args, eval_dataloader, steps: int = 0) -> None:
        """
        Evaluate the model on the provided dataloader by generating text responses and saving them to a
        JSON file. This method handles distributed gathering of generated text, extracted assistant
        responses, and extra metadata across all processes, with the rank 0 process writing the final
        results to disk. Evaluation results and metrics are also logged to Weights & Biases or
        TensorBoard if configured.

        :param args: Training arguments containing generation configurations.
        :type args: Any
        :param eval_dataloader: Dataloader for evaluation samples.
        :type eval_dataloader: torch.utils.data.DataLoader
        :param steps: Global step id for logging and naming the output file.
        :type steps: int

        :returns: None
        :rtype: NoneType
        """
        step_bar = tqdm(
            range(len(eval_dataloader)),
            desc="Eval stage of steps %d" % steps,
            disable=not self.strategy.is_rank_0(),
        )
        self.model.eval()

        # Create JSON file path (only on rank 0)
        if self.strategy.is_rank_0():
            output_file = f"eval_result_{steps}.json"
            output_file = os.path.join(self.strategy.args.save_path, "eval", output_file)
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

        all_eval_records = []

        with torch.no_grad():
            for data in eval_dataloader:
                ids, mask, pixel_values, image_grid_thws, pixel_values_videos, video_grid_thws, labels, extras = data

                device = torch.cuda.current_device()
                ids = ids.squeeze(1).to(device)
                mask = mask.squeeze(1).to(device)

                if pixel_values is not None:
                    pixel_values = pixel_values.to(device)
                    image_grid_thws = image_grid_thws.to(device)

                if pixel_values_videos is not None:
                    pixel_values_videos = pixel_values_videos.to(device)
                    video_grid_thws = video_grid_thws.to(device)

                # Generation
                # Unwrap the model if it is wrapped in a DistributedDataParallel or similar wrapper
                unwrapped_model = self.model.module if hasattr(self.model, "module") else self.model
                generated_ids = unwrapped_model.generate(
                    input_ids=ids,
                    attention_mask=mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thws,
                    pixel_values_videos=pixel_values_videos,
                    video_grid_thw=video_grid_thws,
                    max_new_tokens=args.generate_max_len,
                    synced_gpus=True,  # Use synced_gpus=True for Zero-3 compatibility
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

                generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

                responses_text = []
                predicted_answers = []
                for gen_text in generated_text:
                    # Extract only the assistant's response part
                    # Qwen/Llama-3 templates use <|im_start|>assistant\n or assistant\n
                    if "<|im_start|>assistant" in gen_text:
                        response = gen_text.split("<|im_start|>assistant")[-1]
                    elif "assistant\n" in gen_text:
                        response = gen_text.split("assistant\n")[-1]
                    else:
                        response = gen_text
                    responses_text.append(response)

                    # Extract predicted answer from the response
                    predicted_answers.append(extract_answer(response))

                # Construct records locally and gather them across all ranks
                local_records = []
                for gen_text, resp_text, pred_ans, extra in zip(
                    generated_text, responses_text, predicted_answers, extras
                ):
                    local_records.append({
                        "info": extra,
                        "generated_text": gen_text,
                        "response_text": resp_text,
                        "predicted_answer": pred_ans,
                        "gt_answer": extract_answer(extra["response"])
                    })

                gathered_records = all_gather_and_flatten(local_records)
                if self.strategy.is_rank_0():
                    all_eval_records.extend(gathered_records)

                step_bar.update()

        if self.strategy.is_rank_0():
            # Write JSON file
            with open(output_file, 'w') as f:
                json.dump(all_eval_records, f, indent=4, ensure_ascii=False)

            # Calculate accuracy
            correct = 0
            total = 0
            for r in all_eval_records:
                if r["gt_answer"] == r["predicted_answer"]:
                    correct += 1
                elif r["predicted_answer"] is None:
                    self.strategy.print(f"Could not extract answer from generated text: {r['generated_text']}")
                total += 1
            accuracy = correct / total if total > 0 else 0
            self.strategy.print(f"Step {steps} Evaluation Accuracy: {accuracy:.4f} ({correct}/{total})")

            # wandb/tensorboard logging
            if self._wandb is not None:
                columns = ["info", "generated_text", "response_text", "predicted_answer", "gt_answer"]
                # Log a subset of samples
                data = [[
                    str(r["info"]), r["generated_text"], r["response_text"], r["predicted_answer"], r["gt_answer"]
                ] for r in all_eval_records[:10]]
                self._wandb.log({
                    "eval/samples": self._wandb.Table(columns=columns, data=data),
                    "eval/accuracy": accuracy,
                    "eval/global_step": steps
                })

            if self._tensorboard is not None:
                self._tensorboard.add_scalar("eval/accuracy", accuracy, steps)
                for i, r in enumerate(all_eval_records[:5]):
                    text = (
                        f"Info: {r['info']}\n\nGenerated: {r['generated_text']}\n\n"
                        f"Response: {r['response_text']}\n\n"
                        f"Predicted Answer: {r['predicted_answer']}\n\n"
                        f"GT Answer: {r['gt_answer']}"
                    )
                    self._tensorboard.add_text(f"eval/sample_{i}", text, steps)

            self.strategy.print(f"Evaluation generations written to {output_file}")

        self.model.train()  # reset model state
