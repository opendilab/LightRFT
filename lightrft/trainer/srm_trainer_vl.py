"""
Trainer utilities for scalar reward models (vision-language capable).

This module provides a trainer that supports pairwise preference training with
different loss functions, including Bradley-Terry (BT) loss and Human Preference
Score (HPS) Loss. It integrates with the project's Strategy abstraction for distributed
training, gradient accumulation, and checkpointing, while optionally logging to
Weights & Biases or TensorBoard.
"""

import os
import json
from tqdm import tqdm
from typing import Tuple, Dict

import torch
import torch.nn as nn
from torch.optim import Optimizer
import torch.distributed as dist

from lightrft.models import LogExpLoss, LogSigmoidLoss, HPSLoss, pad_to_length
from lightrft.utils import DistributedSampler


class SRMTrainerVL:
    """
    Trainer for scalar vision-language reward models.

    :param model: The model to be trained; expected to return a dict of head
        scores for each head type when called with token ids and optional image
        or video features.
    :type model: torch.nn.Module
    :param strategy: Training strategy that manages distributed operations,
        gradient accumulation, checkpointing, logging helpers, and args.
    :type strategy: Strategy
    :param optim: Optimizer used for parameter updates.
    :type optim: torch.optim.Optimizer
    :param train_dataloader: Dataloader providing pairwise (A/B/Equal) batches.
    :type train_dataloader: torch.utils.data.DataLoader
    :param scheduler: Learning rate scheduler.
    :type scheduler: torch.optim.lr_scheduler._LRScheduler
    :param tokenizer: Tokenizer used to obtain pad token id for sequence
        padding when concatenating inputs.
    :type tokenizer: Callable
    :param eval_dataloader: Optional dataloader used for evaluation.
    :type eval_dataloader: Optional[torch.utils.data.DataLoader]
    :param max_epochs: Maximum number of training epochs.
    :type max_epochs: int
    :param loss: Loss function to use. Choices are 'sigmoid' (PairWiseLoss),
        'logexp' (LogExpLoss), and 'hps' (HPSLoss).
    :type loss: str
    :param margin: Margin value for BT loss (only used if ``loss`` is BT).
    :type margin: float
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
        loss: str = "sigmoid",
        margin: float = 0.1,
    ) -> None:
        self.strategy = strategy
        self.epochs = max_epochs
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.scheduler = scheduler
        self.optimizer = optim
        self.tokenizer = tokenizer
        self.margin = margin
        self.args = strategy.args

        if loss == "sigmoid":
            self.loss = "sigmoid"
            self.loss_fn = LogSigmoidLoss()
            self.strategy.print("LogSigmoid Loss")
        elif loss == 'logexp':
            self.loss = 'logexp'
            self.loss_fn = LogExpLoss()
            self.strategy.print("LogExp Loss")
        elif loss == 'hps':
            self.loss = 'hps'
            self.loss_fn = HPSLoss()
            self.strategy.print("HPS Loss")
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
        """
        Train the model for ``max_epochs`` using the provided dataloaders.

        :param args: Training arguments (typically ``strategy.args``) including
            logging/eval/save intervals and batch sizing.
        :type args: Any
        :param consumed_samples: Number of samples already consumed (for
            resuming training), defaults to ``0``.
        :type consumed_samples: int
        :param num_update_steps_per_epoch: Number of optimizer steps per epoch.
            If ``None``, it's inferred externally and passed in by caller.
        :type num_update_steps_per_epoch: Optional[int]

        :returns: None
        :rtype: NoneType

        Notes:
            - Supports HPS Scale training via ``scale_for_train`` flag in args.
            - Logs to Weights & Biases or TensorBoard when configured.
        """
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

        head_types = self.model.head_types
        loss_mean = {}
        acc_mean = {}
        acc = {}
        total_loss_mean = 0.0
        for head_type in head_types:
            loss_mean[head_type] = 0
            acc_mean[head_type] = 0
            acc[head_type] = 0
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
            scale_for_train = args.scale_for_train

            for data in self.train_dataloader:
                (
                    input0_ids, input0_mask, input1_ids, input1_mask, input0_img_pixels, input0_img_grid_thws,
                    input1_img_pixels, input1_img_grid_thws, input0_video_pixels, input0_video_grid_thws,
                    input1_video_pixels, input1_video_grid_thws, extras
                ) = data

                device = torch.cuda.current_device()

                input0_ids = input0_ids.squeeze(1).to(device)
                input0_mask = input0_mask.squeeze(1).to(device)
                input1_ids = input1_ids.squeeze(1).to(device)
                input1_mask = input1_mask.squeeze(1).to(device)

                if input0_img_pixels is not None:
                    input0_img_pixels = input0_img_pixels.to(device)
                    input0_img_grid_thws = input0_img_grid_thws.to(device)
                    input1_img_pixels = input1_img_pixels.to(device)
                    input1_img_grid_thws = input1_img_grid_thws.to(device)

                if input0_video_pixels is not None:
                    input0_video_pixels = input0_video_pixels.to(device)
                    input0_video_grid_thws = input0_video_grid_thws.to(device)
                    input1_video_pixels = input1_video_pixels.to(device)
                    input1_video_grid_thws = input1_video_grid_thws.to(device)

                scores0, scores1 = self.concatenated_forward(
                    self.model, input0_ids, input0_mask, input1_ids, input1_mask, input0_img_pixels,
                    input0_img_grid_thws, input1_img_pixels, input1_img_grid_thws, input0_video_pixels,
                    input0_video_grid_thws, input1_video_pixels, input1_video_grid_thws
                )

                labels = {}
                for head_type in head_types:
                    labels[head_type] = [e[head_type] if head_type in e else 'C' for e in extras]

                chosens = {}
                rejects = {}
                equals = {}
                for head_type in head_types:
                    chosens[head_type] = []
                    rejects[head_type] = []
                    equals[head_type] = []

                for i in range(len(extras)):
                    for head_type in head_types:
                        label = labels[head_type][i]

                        if label == 'A':
                            chosens[head_type].append(scores0[head_type][i])
                            rejects[head_type].append(scores1[head_type][i])
                        elif label == 'B':
                            chosens[head_type].append(scores1[head_type][i])
                            rejects[head_type].append(scores0[head_type][i])
                        else:
                            equals[head_type].append([scores0[head_type][i], scores1[head_type][i]])

                for head_type in head_types:
                    if len(chosens[head_type]) > 0:
                        chosens[head_type] = torch.stack(chosens[head_type])
                        rejects[head_type] = torch.stack(rejects[head_type])

                equals_loss = {}
                for head_type in head_types:
                    if len(equals[head_type]) > 0:
                        equals[head_type] = torch.stack([torch.stack(t) for t in equals[head_type]])
                        equal_loss = torch.abs(equals[head_type][:, 0] - equals[head_type][:, 1]).mean()
                        equals_loss[head_type] = equal_loss
                    else:
                        equals_loss[head_type] = torch.tensor(0.0, device=device)

                head_loss = {}
                for head_type in head_types:
                    if len(chosens[head_type]) == 0:
                        continue
                    # Compute per head loss based on the selected loss function
                    head_loss[head_type] = self.loss_fn(chosens[head_type], rejects[head_type], self.margin)

                total_loss = sum(head_loss.values()) + 0.01 * sum(equals_loss.values())
                self.strategy.backward(total_loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                for head_type in head_types:
                    if len(chosens[head_type]) == 0:
                        continue
                    acc[head_type] = (chosens[head_type] > rejects[head_type]).float().mean().item()
                    acc_mean[head_type] = acc_mean[head_type] * 0.9 + 0.1 * acc[head_type]
                    loss_mean[head_type] = loss_mean[head_type] * 0.9 + 0.1 * head_loss[head_type].item()

                total_loss_mean = total_loss_mean * 0.9 + 0.1 * total_loss.item()
                logs_dict = {
                    "loss": total_loss.item(),
                    "loss_mean": total_loss_mean,
                    "lr": self.scheduler.get_last_lr()[0],
                }
                for head_type in head_types:
                    if len(chosens[head_type]) > 0:
                        logs_dict[f"{head_type}_loss"] = head_loss[head_type].item()
                        logs_dict[f"{head_type}_acc"] = acc[head_type]
                        logs_dict[f"{head_type}_acc_mean"] = acc_mean[head_type]
                        logs_dict[f"{head_type}_loss_mean"] = loss_mean[head_type]
                        logs_dict[f"{head_type}_chosen_reward"] = round(
                            chosens[head_type].mean().item() * 0.07, 4
                        ) if scale_for_train else chosens[head_type].mean().item()
                        logs_dict[f"{head_type}_reject_reward"] = round(
                            rejects[head_type].mean().item() * 0.07, 4
                        ) if scale_for_train else rejects[head_type].mean().item()
                    else:
                        logs_dict[f"{head_type}_loss"] = 0.0
                        logs_dict[f"{head_type}_acc"] = 0.0
                        logs_dict[f"{head_type}_acc_mean"] = 0.0
                        logs_dict[f"{head_type}_loss_mean"] = 0.0
                        logs_dict[f"{head_type}_chosen_reward"] = 0.0
                        logs_dict[f"{head_type}_reject_reward"] = 0.0

                # step bar
                for k in logs_dict.keys():
                    if k.startswith('preference'):
                        logs_dict[k] = self.strategy.all_reduce(logs_dict[k], op='max')
                    else:
                        logs_dict[k] = self.strategy.all_reduce(logs_dict[k])
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

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        if self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()

    # logs/checkpoints/evaluate
    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, client_states={}) -> None:
        """
        Log metrics and optionally run evaluation and checkpointing.

        :param args: Training arguments providing logging/eval/save intervals
            and checkpoint configurations.
        :type args: Any
        :param global_step: Current global optimization step.
        :type global_step: int
        :param step_bar: Progress bar for step-level updates.
        :type step_bar: tqdm.tqdm
        :param logs_dict: Dictionary of metrics to log (will be reduced across
            ranks via ``strategy.all_reduce`` before display/logging).
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

    def evaluate(self, args, eval_dataloader, steps=0) -> None:
        """
        Evaluate the model on the provided dataloader and write a JSONL of
        scores to the save path indicated by ``strategy.args.save_path``.

        :param args: present for API compatibility with callers.
        :type args: Any
        :param eval_dataloader: Dataloader for evaluation samples.
        :type eval_dataloader: torch.utils.data.DataLoader
        :param steps: Global step id for naming the output file.
        :type steps: int

        :returns: None
        :rtype: NoneType

        The output file name format is ``eval_scores_{steps}.jsonl``.
        """
        step_bar = tqdm(
            range(len(eval_dataloader)),
            desc="Eval stage of steps %d" % steps,
            disable=not self.strategy.is_rank_0(),
        )
        self.model.eval()

        # Create JSONL file and write header (only on rank 0)
        if self.strategy.is_rank_0():
            output_file = f"eval_scores_{steps}.jsonl"
            output_file = os.path.join(self.strategy.args.save_path, output_file)
            with open(output_file, 'w') as f:
                f.write("")  # Just create/clear the file

        with torch.no_grad():
            results = []

            for data in eval_dataloader:
                (
                    input0_ids, input0_mask, input1_ids, input1_mask, input0_img_pixels, input0_img_grid_thws,
                    input1_img_pixels, input1_img_grid_thws, input0_video_pixels, input0_video_grid_thws,
                    input1_video_pixels, input1_video_grid_thws, extras
                ) = data
                device = torch.cuda.current_device()

                scores0, scores1 = self.concatenated_forward(
                    self.model, input0_ids, input0_mask, input1_ids, input1_mask, input0_img_pixels,
                    input0_img_grid_thws, input1_img_pixels, input1_img_grid_thws, input0_video_pixels,
                    input0_video_grid_thws, input1_video_pixels, input1_video_grid_thws
                )

                # Gather scores from all GPUs for each head_type
                gathered_scores0 = {}
                gathered_scores1 = {}
                for head_type in scores0.keys():
                    if head_type in scores0:
                        # Create tensor list for all_gather
                        tensor_list0 = [torch.zeros_like(scores0[head_type]) for _ in range(dist.get_world_size())]
                        tensor_list1 = [torch.zeros_like(scores1[head_type]) for _ in range(dist.get_world_size())]
                        # Use all_gather instead of all_gather_object for tensors
                        dist.all_gather(tensor_list0, scores0[head_type])
                        dist.all_gather(tensor_list1, scores1[head_type])
                        # Concatenate all tensors along batch dimension
                        gathered_scores0[head_type] = torch.cat(tensor_list0, dim=0)
                        gathered_scores1[head_type] = torch.cat(tensor_list1, dim=0)

                # Gather extras into
                gathered_extras = [None] * dist.get_world_size()
                dist.all_gather_object(gathered_extras, extras)
                # Flatten the list of lists
                all_extras = []
                for gpu_extras in gathered_extras:
                    all_extras.extend(gpu_extras)

                # write scores to JSONL file immediately (only on rank 0)
                if self.strategy.is_rank_0():
                    with open(output_file, 'a') as f:
                        for i, extras in enumerate(all_extras):
                            # build per-sample scores dict from gathered_scores
                            input0_scores = {
                                head_type: gathered_scores0[head_type][i].item()
                                for head_type in gathered_scores0
                            }
                            input1_scores = {
                                head_type: gathered_scores1[head_type][i].item()
                                for head_type in gathered_scores1
                            }
                            # build per-sample results dict
                            results = {
                                "info": extras,
                                "scores0": input0_scores,
                                "scores1": input1_scores,
                            }
                            f.write(json.dumps(results) + "\n")

                step_bar.update()

            # Final message about file completion
            if self.strategy.is_rank_0():
                self.strategy.print(f"Evaluation scores written to {output_file}")
        self.model.train()  # reset model state

    def concatenated_forward(
        self,
        model,
        input0_ids,
        input0_mask,
        input1_ids,
        input1_mask,
        input0_img_pixels,
        input0_img_grid_thws,
        input1_img_pixels,
        input1_img_grid_thws,
        input0_video_pixels,
        input0_video_grid_thws,
        input1_video_pixels,
        input1_video_grid_thws,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Run the model once on concatenated chosen/rejected inputs.

        Concatenation reduces the number of forward passes.

        :param model: Model to evaluate.
        :type model: nn.Module
        :param input0_ids: Token ids for chosen sequences.
        :type input0_ids: torch.Tensor
        :param input0_mask: Attention mask for chosen sequences.
        :type input0_mask: torch.Tensor
        :param input1_ids: Token ids for rejected sequences.
        :type input1_ids: torch.Tensor
        :param input1_mask: Attention mask for rejected sequences.
        :type input1_mask: torch.Tensor
        :param input0_img_pixels: Optional image features for chosen samples.
        :type input0_img_pixels: Optional[torch.Tensor]
        :param input0_img_grid_thws: Optional image grid meta for chosen.
        :type input0_img_grid_thws: Optional[torch.Tensor]
        :param input1_img_pixels: Optional image features for rejected samples.
        :type input1_img_pixels: Optional[torch.Tensor]
        :param input1_img_grid_thws: Optional image grid meta for rejected.
        :type input1_img_grid_thws: Optional[torch.Tensor]
        :param input0_video_pixels: Optional video features for chosen samples.
        :type input0_video_pixels: Optional[torch.Tensor]
        :param input0_video_grid_thws: Optional video grid meta for chosen.
        :type input0_video_grid_thws: Optional[torch.Tensor]
        :param input1_video_pixels: Optional video features for rejected samples.
        :type input1_video_pixels: Optional[torch.Tensor]
        :param input1_video_grid_thws: Optional video grid meta for rejected.
        :type input1_video_grid_thws: Optional[torch.Tensor]

        :returns: Tuple of dicts ``(scores0, scores1)`` separating the outputs
            for chosen and rejected samples.
        :rtype: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
        """
        input_ids, att_masks = self.concatenated_inputs(input0_ids, input0_mask, input1_ids, input1_mask)

        pixel_values = None
        image_grid_thws = None
        pixel_values_videos = None
        video_grid_thws = None
        with torch.no_grad():
            if input0_img_pixels is not None:
                pixel_values = torch.cat((input0_img_pixels, input1_img_pixels), dim=0)
                image_grid_thws = torch.cat((input0_img_grid_thws, input1_img_grid_thws), dim=0)

            if input0_video_pixels is not None:
                pixel_values_videos = torch.cat((input0_video_pixels, input1_video_pixels), dim=0)
                video_grid_thws = torch.cat((input0_video_grid_thws, input1_video_grid_thws), dim=0)

        scores = model(
            input_ids,
            attention_mask=att_masks,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thws,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thws
        )
        scores0 = {head_type: score[:input0_ids.shape[0]] for head_type, score in scores.items()}
        scores1 = {head_type: score[input0_ids.shape[0]:] for head_type, score in scores.items()}
        return scores0, scores1

    def concatenated_inputs(self, input0_ids, input0_mask, input1_ids,
                            input1_mask) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Concatenate two inputs into a single batch.

        :param input0_ids: Token ids for the first sequences, shape ``(N, Lc)``.
        :type input0_ids: torch.Tensor
        :param input0_mask: Attention mask for chosen sequences, shape ``(N, Lc)``.
        :type input0_mask: torch.Tensor
        :param input1_ids: Token ids for the second sequences, shape ``(N, Lr)``.
        :type input1_ids: torch.Tensor
        :param input1_mask: Attention mask for the second sequences, shape ``(N, Lr)``.
        :type input1_mask: torch.Tensor

        :returns: Tuple ``(input_ids, att_masks)`` where inputs are padded to
            a common max length across input0 and input1, then concatenated
            along the batch dimension to shape ``(2N, Lmax)``.
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        max_length = max(input0_ids.shape[1], input1_ids.shape[1])
        inputs_ids = torch.cat(
            (
                pad_to_length(input0_ids, max_length, self.tokenizer.pad_token_id),
                pad_to_length(input1_ids, max_length, self.tokenizer.pad_token_id),
            ),
            dim=0,
        )
        max_length = max(input0_mask.shape[1], input1_mask.shape[1])
        att_masks = torch.cat((pad_to_length(input0_mask, max_length, 0), pad_to_length(input1_mask, max_length, 0)),
                              dim=0)
        return inputs_ids, att_masks
