import os
import numpy as np
import json
import torch
from tqdm import tqdm
from loguru import logger
from collections import defaultdict
from typing import List, Dict, Any
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoTokenizer

from lightrft.models import ScalarRewardModelAL, concatenated_forward
from lightrft.datasets import RankDatasetAL


T2A_TASK_INSTRUCTION="""You will act as an expert audio evaluator for text-to-audio generation.
Given a text prompt and a generated audio clip, your task is to assess the overall quality of the audio in relation to the prompt.
Your evaluation should focus on the following key aspects:
• Preference: Which audio would a human listener find more satisfying or acoustically pleasing overall (considering audio fidelity, clarity, and musicality/naturalness).
• Alignment: How well the audio content matches the given text prompt in semantics, sound events, mood, and acoustic attributes (e.g., genre, tempo, instruments).
Your task is provided in the following, please give your judgement based on above criteria.
The prompt used for generation is as follows: {prompt}.
"""


def plot_histogram(per_head_scores: Dict[str, List[float]], save_dir: str) -> None:
    """
    Plot histograms using the default Matplotlib style with minimal aesthetic improvements.
    
    :param per_head_scores: Dictionary mapping head names to lists of float scores.
    :param save_dir: Directory where the plots will be saved.
    """
    if not per_head_scores:
        return

    for head, all_vals in per_head_scores.items():
        if len(all_vals) == 0:
            continue
        
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        min_v, max_v = min(all_vals), max(all_vals)
        
        # Avoid zero-width bins
        if max_v - min_v < 1e-6:
            min_v -= 1e-6
            max_v += 1e-6
            
        bins = np.linspace(min_v, max_v, 15)
        
        ax.hist(
            all_vals, 
            bins=bins, 
            color='#1f77b4',
            edgecolor='black',
            linewidth=0.8,
            alpha=0.7,
            density=False
        )
        
        ax.set_xlabel('Raw Score', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'Score Distribution - {head.capitalize()} Head', fontsize=14, pad=10)
        
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        # Disable scientific notation for axes
        y_formatter = ticker.ScalarFormatter(useOffset=False)
        y_formatter.set_scientific(False)
        ax.yaxis.set_major_formatter(y_formatter)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        
        # Add Statistics Box
        mu = np.mean(all_vals)
        sigma = np.std(all_vals)
        N = len(all_vals)
        
        stats_text = '\n'.join((
            r'$\mu=%.3f$' % (mu, ),
            r'$\sigma=%.3f$' % (sigma, ),
            r'$N=%d$' % (N, )
        ))
        
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', horizontalalignment='right', bbox=props)

        plt.tight_layout()
        
        # Save
        safe_head = head.replace('/', '_')
        fig_path = os.path.join(save_dir, f"hist_{safe_head}.png")
        plt.savefig(fig_path, bbox_inches='tight')
        plt.close()
        print(f"Histogram saved to {fig_path}")


class BaseScalarEvaluator(ABC):
    """
    Base class for Scalar Reward Model Evaluators.
    Responsible for interpreting raw scores and comparing them against ground truth.
    """
    
    def __init__(self):
        self.correct = 0
        self.total = 0
        self.results = []
        self.per_head_scores = defaultdict(list)
        # Store per-head reward gaps for GT-chosen vs GT-rejected: score_chosen - score_reject
        self.per_head_reward_gaps = defaultdict(list)

    @abstractmethod
    def evaluate_batch(self, scores0_batch: Dict[str, torch.Tensor], scores1_batch: Dict[str, torch.Tensor], extras: Dict[str, Any]) -> None:
        """
        Process a batch of raw scores.
        
        :param scores0_batch: Model outputs for the first candidate (Map: head_name -> tensor).
        :param scores1_batch: Model outputs for the second candidate.
        :param extras: Metadata from the dataset (contains Ground Truth).
        """
        pass

    def get_accuracy(self) -> float:
        """
        Compute evaluation accuracy.

        :return: Correct ratio in [0, 1].
        """
        return self.correct / self.total if self.total > 0 else 0.0

    def get_results(self) -> List[Dict]:
        """
        Return detailed per-sample evaluation records.

        :return: List of result dicts collected during evaluation.
        """
        return self.results
    
    def get_collected_scores(self) -> Dict[str, List[float]]:
        """
        Return collected per-head raw scores for visualization.

        :return: Dict mapping head name to list of float scores.
        """
        return self.per_head_scores
    
    def add_reward_gap(self, head: str, gap: float) -> None:
        """
        Collect a reward gap for a head (score_chosen - score_reject).

        :param head: Reward head name.
        :param gap: Gap value to record.
        """
        self.per_head_reward_gaps[head].append(gap)

    def get_reward_gaps(self) -> Dict[str, List[float]]:
        """
        Return raw collected reward gaps per head.

        :return: Dict mapping head name to list of gaps (float).
        """
        return self.per_head_reward_gaps

    def get_mean_reward_gap(self) -> Dict[str, float]:
        """
        Compute mean(score_chosen - score_reject) for each head.

        :return: Dict mapping head name to mean gap.
        """
        means = {}
        for head, gaps in self.per_head_reward_gaps.items():
            if len(gaps) == 0:
                continue
            means[head] = float(np.mean(gaps))
        return means


class OmniRewardBenchEvaluator(BaseScalarEvaluator):
    """
    Evaluator specifically for OmniRewardBench.
    """
    
    def evaluate_batch(self, scores0_batch: Dict[str, torch.Tensor], scores1_batch: Dict[str, torch.Tensor], extras: Dict[str, Any]) -> None:
        # Determine batch size from the metadata
        batch_size = len(extras)
        head_names = list(scores0_batch.keys())
        
        for i in range(batch_size):
            sample_scores0_map = {}
            sample_scores1_map = {}
            
            addup_scores0 = 0.0
            addup_scores1 = 0.0
            
            # Aggregate scores across all heads
            for head in head_names:
                # Extract scalar value
                val0 = float(scores0_batch[head][i].item())
                val1 = float(scores1_batch[head][i].item())
                
                sample_scores0_map[head] = val0
                sample_scores1_map[head] = val1
                
                # Collect data for global histogram (mixing Score 0 and Score 1)
                self.per_head_scores[head].append(val0)
                self.per_head_scores[head].append(val1)
                
                # Sum scores
                addup_scores0 += val0
                addup_scores1 += val1

            # Retrieve Ground Truth
            gt_choice = extras[i]['preference']
            # Collect reward gaps per head (chosen - reject) using ground-truth preference
            if gt_choice in ("A", "B"):
                sign = 1.0 if gt_choice == "A" else -1.0
                for head in head_names:
                    gap = sign * (sample_scores0_map[head] - sample_scores1_map[head])
                    self.add_reward_gap(head, gap)
            
            # Evaluation Logic
            is_correct = False
            if addup_scores0 > addup_scores1 and gt_choice == "A":
                self.correct += 1
                is_correct = True
            elif addup_scores1 > addup_scores0 and gt_choice == "B":
                self.correct += 1
                is_correct = True
            elif addup_scores1 == addup_scores0 and gt_choice == "C":
                self.correct += 1
                is_correct = True
            
            self.total += 1

            # Record detailed result
            self.results.append({
                "id": extras[i]['id'],
                "preference": gt_choice,
                "criteria": extras[i]['criteria'],
                "criteria_preference": extras[i]['criteria_preference'],
                "scores0": sample_scores0_map,
                "scores1": sample_scores1_map,
                "addup_scores0": addup_scores0,
                "addup_scores1": addup_scores1,
                "is_correct": is_correct
            })


@torch.no_grad()
def test_scalar_rm(
    model_path: str,
    data_path: List[str],
    evaluator: BaseScalarEvaluator,
    config: dict = None,
    batch_size: int = 1,
    save_dir: str = "./test_results"
):
    logger.info(f"Loading model from: {model_path}")
    model = ScalarRewardModelAL(
        model_path,
        use_flash_attention_2=True,
        bf16=True,
        lora_rank=0,
        lora_alpha=0,
        target_modules=None,
        lora_dropout=0,
        ds_config=None,
        device_map="cuda", 
        pooling_method=config["pooling_method"],
        scale_for_train=config["scale_for_train"],
        probing_layer=config["probing_layer"],
        head_types=config["head_types"],
    )
    state_dict = torch.load(os.path.join(model_path, 'pytorch_model.bin'), weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    logger.info(f"Model loaded successfully from {model_path}.")
    device = torch.cuda.current_device()
    model.to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

    dataset = RankDatasetAL(
        data_path,
        processor=processor,
        tokenizer=tokenizer,
        max_length=4096,
        config=config
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=dataset.collate_fn
    )

    logger.info(f"Starting evaluation on {len(dataset)} samples...")
    for idx, batch in enumerate(tqdm(data_loader)):
        (
            input0_ids, input0_mask,
            input1_ids, input1_mask,
            input0_input_features, input0_feature_attention_mask,
            input1_input_features, input1_feature_attention_mask,
            extras
        ) = batch

        input0_ids = input0_ids.squeeze(1).to(device)
        input0_mask = input0_mask.squeeze(1).to(device)
        input1_ids = input1_ids.squeeze(1).to(device)
        input1_mask = input1_mask.squeeze(1).to(device)

        input0_input_features = input0_input_features.to(device)
        input0_feature_attention_mask = input0_feature_attention_mask.to(device)
        input1_input_features = input1_input_features.to(device)
        input1_feature_attention_mask = input1_feature_attention_mask.to(device)

        
        scores0 = model(
            input0_ids,
            attention_mask=input0_mask,
            input_features=input0_input_features,
            feature_attention_mask=input0_feature_attention_mask,
        )

        scores1 = model(
            input1_ids,
            attention_mask=input1_mask,
            input_features=input1_input_features,
            feature_attention_mask=input1_feature_attention_mask,
        )

        evaluator.evaluate_batch(scores0, scores1, extras)

    accuracy = evaluator.get_accuracy()
    logger.info(f"Evaluation completed. Accuracy: {accuracy*100:.2f}% ({evaluator.correct}/{evaluator.total})")
    # Compute and log Mean Reward Gap per head
    mean_gaps = evaluator.get_mean_reward_gap()
    if mean_gaps:
        gap_str = ", ".join([f"{k}: {v:.6f}" for k, v in mean_gaps.items()])
        logger.info(f"Mean Reward Gap per head (score_chosen - score_reject): {gap_str}")
    else:
        logger.info("No reward gaps collected (possibly all ties).")

    if save_dir:
        final_save_dir = os.path.join(save_dir, os.path.basename(model_path), config["name"])
        os.makedirs(final_save_dir, exist_ok=True)
        
        # Save Summary Text
        with open(os.path.join(final_save_dir, "evaluation_info.txt"), "w") as f:
            f.write(f"Dataset paths: {data_path}\n")
            f.write(f"Model path: {model_path}\n")
            f.write(f"Accuracy: {accuracy*100:.2f}% ({evaluator.correct}/{evaluator.total})\n")
            if mean_gaps:
                f.write("Mean Reward Gap per head (score_chosen - score_reject):\n")
                for head, val in mean_gaps.items():
                    f.write(f"  - {head}: {val:.6f}\n")

        # Save Detailed Results JSON
        with open(os.path.join(final_save_dir, "evaluation_results.json"), "w") as f:
            json.dump(evaluator.get_results(), f, ensure_ascii=False, indent=4)
        
        # Save Reward Gap stats JSON (means and counts)
        reward_gap_stats = {
            "mean_reward_gap": mean_gaps,
            "count": {h: len(v) for h, v in evaluator.get_reward_gaps().items()}
        }
        with open(os.path.join(final_save_dir, "mean_reward_gap.json"), "w") as f:
            json.dump(reward_gap_stats, f, ensure_ascii=False, indent=4)
        
        # Generate Plots
        plot_histogram(evaluator.get_collected_scores(), final_save_dir)
        
        logger.info(f"Results and artifacts saved to {final_save_dir}")


if __name__ == "__main__":
    model_path = "path/to/your/srm_al/checkpoint"
    
    benchmark_configs = [
        {
            "name": "OmniRewardBenchT2A",
            "evaluator": OmniRewardBenchEvaluator(),
            "data_path": ["omnirewardbench-t2a:/path/to/OmniRewardBench-T2A/test.parquet"],
            "task_instruction": T2A_TASK_INSTRUCTION,
            "pooling_method": "attn",
            "scale_for_train": True,
            "probing_layer": -1,
            "head_types": ["preference"],
        },
    ]

    for config in benchmark_configs:
        print(f">>> Running {config['name']} Evaluation")
        test_scalar_rm(
            model_path, 
            data_path=config["data_path"], 
            evaluator=config["evaluator"], 
            config=config, 
            batch_size=8,
        )
