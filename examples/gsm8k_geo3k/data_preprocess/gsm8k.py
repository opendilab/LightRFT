"""
Preprocess the GSM8K dataset to parquet format for LightRFT training.

This script converts the openai/gsm8k dataset into a format compatible with LightRFT,
using rule-based reward model configuration for math problem solving.

Usage:
    python examples/gsm8k_geo3k/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k
"""

import argparse
import os
import re
from typing import Any, Callable, Dict, List, Optional

import datasets


def extract_answer(solution_str: str) -> str:
    """
    Extract the numerical answer from GSM8K solution string.

    GSM8K solutions end with "#### ANSWER" where ANSWER is the final numerical result.

    :param solution_str: Solution string containing reasoning and final answer
    :type solution_str: str
    :return: Extracted numerical answer (with commas removed)
    :rtype: str

    Example:
        Input: "Step 1: ... Step 2: ... #### 42"
        Output: "42"
    """
    solution_match = re.search(r"####\s*(\-?[0-9\.\,]+)", solution_str)
    if solution_match is None:
        # If no match found, return empty string or raise error
        print(f"Warning: Could not extract answer from: {solution_str[:100]}...")
        return ""

    final_answer = solution_match.group(1)
    # Remove commas from numbers (e.g., "1,000" -> "1000")
    final_answer = final_answer.replace(",", "")
    return final_answer


def preprocess_gsm8k(local_dataset_path: Optional[str], local_save_dir: str) -> None:
    """
    Main preprocessing function for GSM8K dataset.

    :param local_dataset_path: Optional local path to raw dataset
    :type local_dataset_path: Optional[str]
    :param local_save_dir: Directory to save preprocessed dataset
    :type local_save_dir: str
    """
    data_source = "openai/gsm8k"

    # Load dataset from HuggingFace or local path
    if local_dataset_path is not None:
        dataset = datasets.load_dataset(local_dataset_path, "main")
    else:
        dataset = datasets.load_dataset(data_source, "main")

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    # Instruction following prompt template
    # Encourage step-by-step reasoning with final answer after ####
    instruction_following = (
        r"You FIRST think about the reasoning process step by step as an internal monologue "
        r"and then provide the final answer. "
        r"The reasoning process MUST BE enclosed within <think> </think> tags. "
        r"The final answer MUST BE put in \boxed{} after the reasoning."
    )

    def make_map_fn(split: str) -> Callable[[Dict[str, Any], int], Dict[str, Any]]:
        """
        Create a mapping function for dataset preprocessing.

        :param split: Dataset split name ("train" or "test")
        :type split: str
        :return: A function that processes each example in the dataset
        :rtype: Callable[[Dict[str, Any], int], Dict[str, Any]]
        """
        def process_fn(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
            """
            Process a single example from the dataset.

            :param example: Dataset example containing question and answer
            :type example: Dict[str, Any]
            :param idx: Index of the example in the dataset
            :type idx: int
            :return: Processed data in LightRFT format
            :rtype: Dict[str, Any]
            """
            question_raw = example.pop("question")
            answer_raw = example.pop("answer")

            # Extract numerical answer
            ground_truth = extract_answer(answer_raw)

            # Create data format compatible with LightRFT
            # Use similar structure as geo3k but without images
            data = {
                "data_source": data_source,
                "prompt": [  # Use list format like SVKG/Geo3K dataset
                    {
                        "role": "system",
                        "content": instruction_following  # System instruction
                    },
                    {
                        "role": "user",
                        "content": question_raw  # Only the question
                    }
                ],
                "ability": "math",
                "reward_model": {
                    "ground_truth": ground_truth  # For compatibility with reward extraction
                },
                "extra_info": {
                    "label": "gsm8k_rule",  # This label will be used in RECIPE
                    "reference": ground_truth,  # Ground truth for rule-based reward
                    "split": split,
                    "index": idx,
                    "answer": ground_truth,
                    "question": question_raw,
                    "full_solution": answer_raw,  # Keep original solution for reference
                },
            }
            return data

        return process_fn

    # Apply preprocessing to both splits
    print(f"Preprocessing {data_source} dataset...")
    print(f"  Train set: {len(train_dataset)} examples")
    print(f"  Test set: {len(test_dataset)} examples")

    train_dataset = train_dataset.map(
        function=make_map_fn("train"),
        with_indices=True,
        num_proc=8
    )
    test_dataset = test_dataset.map(
        function=make_map_fn("test"),
        with_indices=True,
        num_proc=8
    )

    # Ensure save directory exists
    local_save_dir = os.path.expanduser(local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    # Save to parquet format
    train_path = os.path.join(local_save_dir, "train.parquet")
    test_path = os.path.join(local_save_dir, "test.parquet")

    train_dataset.to_parquet(train_path)
    test_dataset.to_parquet(test_path)

    print(f"\n✓ Successfully preprocessed GSM8K dataset:")
    print(f"  Train set: {len(train_dataset)} examples saved to {train_path}")
    print(f"  Test set: {len(test_dataset)} examples saved to {test_path}")
    print(f"\nDataset format:")
    print(f"  - prompt: Question with instruction following template")
    print(f"  - label: 'gsm8k_rule' (for recipe-based reward)")
    print(f"  - reference: Ground truth answer (for rule-based evaluation)")
    print(f"  - NO images (text-only dataset)")
    print(f"\nExample data:")
    if len(train_dataset) > 0:
        example = train_dataset[0]
        print(f"  Question: {example['extra_info']['question'][:100]}...")
        print(f"  Answer: {example['extra_info']['answer']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess GSM8K dataset for LightRFT training"
    )
    parser.add_argument(
        "--local_dataset_path",
        default=None,
        type=str,
        help="The local path to the raw dataset, if it exists."
    )
    parser.add_argument(
        "--local_save_dir",
        default="~/data/gsm8k",
        type=str,
        help="The save directory for the preprocessed dataset."
    )

    args = parser.parse_args()
    # Expand args to pass individual arguments
    preprocess_gsm8k(**vars(args))
