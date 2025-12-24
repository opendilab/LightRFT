"""
Preprocess the Geometry3k dataset to parquet format for LightRFT training.

This script converts the hiyouga/geometry3k dataset into a format compatible with LightRFT,
including multi-modal support (images) and rule-based reward model configuration.

Usage:
    python examples/gsm8k_geo3k/data_preprocess/geo3k.py --local_save_dir ~/data/geo3k
"""

import argparse
import os
from typing import Any, Callable, Dict, List, Optional

import datasets


def preprocess_geo3k(local_dataset_path: Optional[str], local_save_dir: str) -> None:
    """
    Main preprocessing function for geo3k dataset.

    :param local_dataset_path: Optional local path to raw dataset
    :type local_dataset_path: Optional[str]
    :param local_save_dir: Directory to save preprocessed dataset
    :type local_save_dir: str
    """
    data_source = "hiyouga/geometry3k"

    # Load dataset from HuggingFace or local path
    if local_dataset_path is not None:
        dataset = datasets.load_dataset(local_dataset_path)
    else:
        dataset = datasets.load_dataset(data_source)

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    # Instruction following prompt template
    # This matches the format used in verl and encourages step-by-step reasoning
    instruction_following = (
        r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
        r"The reasoning process MUST BE enclosed within <think> </think> tags. "
        r"The final answer MUST BE put in \boxed{}."
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

            :param example: Dataset example containing problem, answer, and images
            :type example: Dict[str, Any]
            :param idx: Index of the example in the dataset
            :type idx: int
            :return: Processed data in LightRFT format
            :rtype: Dict[str, Any]
            """
            problem = example.pop("problem")
            answer = example.pop("answer")
            images = example.pop("images")

            # Create data format compatible with LightRFT
            # Use SVKG-style format with chat structure in prompt
            data = {
                "data_source": data_source,
                "prompt": [  # Use list format like SVKG dataset
                    {
                        "role": "system",
                        "content": instruction_following  # System instruction
                    },
                    {
                        "role": "user",
                        "content": problem  # Only the question
                    }
                ],
                "images": images,
                "ability": "math",
                "reward_model": {
                    "ground_truth": answer  # For compatibility with reward extraction
                },
                "extra_info": {
                    "label": "geo3k_rule",  # This label will be used in RECIPE
                    "reference": answer,     # Ground truth for rule-based reward
                    "split": split,
                    "index": idx,
                    "answer": answer,
                    "question": problem,
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

    print(f"\n✓ Successfully preprocessed geo3k dataset:")
    print(f"  Train set: {len(train_dataset)} examples saved to {train_path}")
    print(f"  Test set: {len(test_dataset)} examples saved to {test_path}")
    print(f"\nDataset format:")
    print(f"  - prompt: Question with instruction following template")
    print(f"  - images: List of PIL images")
    print(f"  - label: 'geo3k_rule' (for recipe-based reward)")
    print(f"  - reference: Ground truth answer (for rule-based evaluation)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess Geometry3k dataset for LightRFT training"
    )
    parser.add_argument(
        "--local_dataset_path",
        default=None,
        type=str,
        help="The local path to the raw dataset, if it exists."
    )
    parser.add_argument(
        "--local_save_dir",
        default="~/data/geo3k",
        type=str,
        help="The save directory for the preprocessed dataset."
    )

    args = parser.parse_args()
    # Expand args to pass individual arguments
    preprocess_geo3k(**vars(args))
