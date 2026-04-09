"""
Preprocess the AVQA dataset to parquet format for LightRFT R1-AQA training.

This script converts R1-AQA's line-JSON (JSONL) training data into the
LightRFT-compatible parquet format. The AVQA JSONL is produced by the R1-AQA
project from the original AVQA training set.

You can download the AVQA dataset and the corresponding JSONL files from the
HuggingFace repository: https://huggingface.co/datasets/Joysw909/AVQA.

Expected JSONL fields (per line):
    - id (int)
    - video_name (str, optional)
    - video_id (int, optional)
    - question_text (str)
    - multi_choice (list[str])
    - answer (int) — index into multi_choice
    - question_relation (str, optional)
    - question_type (str, optional)
    - dataset_name (str) — e.g. "AVQA"
    - audio_path (str) — path to the .wav file

Output parquet fields (LightRFT compatible):
    - prompt: list[dict]  — chat-format messages with audio content type
    - reference: str       — correct answer text
    - label: str           — "avqa_rule" (routes to RECIPE in reward)
    - extra_info: dict     — metadata for traceability

Usage:
    # Example: after downloading from https://huggingface.co/datasets/Joysw909/AVQA
    huggingface-cli download --repo-type dataset --resume-download Joysw909/AVQA --local-dir path/to/AVQA
    cd path/to/AVQA
    mkdir -p all_audios
    # Copy all files from each VGG directory to all_audios
    cp VGG10000/* VGG20000/* VGG30000/* VGG40000/* all_audios/ 2>/dev/null || true

    python examples/r1_aqa/data_preprocess/avqa.py \\
        --input_jsonl path/to/AVQA/train_r1aqa_line.json \\
        --audio_dir path/to/AVQA/all_audios \\
        --local_save_dir ./avqa_lightrft
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import datasets


# ---------------------------------------------------------------------------
# Prompt template — faithfully ported from R1-AQA src/dataset/dataset.py
# ---------------------------------------------------------------------------

def build_prompt_and_solution(
    obj: Dict[str, Any],
    audio_dir: Optional[str] = None,
    enable_think: bool = False,
) -> Dict[str, Any]:
    """
    Convert one AVQA JSONL record into LightRFT prompt/solution fields.

    This replicates R1-AQA's ``_handle_avqa`` logic:
    - Replaces "video" with "audio" in the question text.
    - Formats multi-choice options.
    - Builds chat-format messages with audio content type.

    :param obj: One JSONL record.
    :param audio_dir: Optional base directory to prepend to ``audio_path``.
    :param enable_think: If True, include "think" instructions in the prompt
        (R1-AQA has this as a commented-out option).
    :return: dict with ``prompt``, ``solution``, and metadata fields.
    """
    question_text = obj["question_text"].replace("video", "audio")
    multi_choice = obj["multi_choice"]
    answer_idx = int(obj["answer"])
    audio_path = obj["audio_path"]

    # Resolve audio path
    if audio_dir and not os.path.isabs(audio_path):
        audio_path = os.path.join(audio_dir, os.path.basename(audio_path))

    # Build question template (matches R1-AQA)
    choice_str = f"Please choose the answer from the following options: {multi_choice}."
    if enable_think:
        question_template = (
            f"{question_text} {choice_str} "
            "Output the thinking process in <think> </think> "
            "and final answer in <answer></answer>."
        )
    else:
        question_template = (
            f"{question_text} {choice_str} "
            "Output the final answer in <answer></answer>."
        )

    # Chat-format prompt with audio content type (Qwen2-Audio format)
    prompt = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": audio_path},
                {"type": "text", "text": question_template},
            ],
        }
    ]

    # Correct answer string
    answer_str = multi_choice[answer_idx]
    solution = f" <answer>{answer_str}</answer> "

    return {
        "prompt": prompt,
        "solution": solution,
        "answer_str": answer_str,
        "audio_path": audio_path,
        "question_template": question_template,
    }


def preprocess_avqa(
    input_jsonl: str,
    local_save_dir: str,
    audio_dir: Optional[str] = None,
    enable_think: bool = False,
    val_ratio: float = 0.0,
) -> None:
    """
    Main preprocessing function for AVQA dataset.

    Reads the R1-AQA line-JSON file, converts each record, and saves as
    parquet for LightRFT consumption.

    :param input_jsonl: Path to the AVQA JSONL file (e.g., train_qa.data).
    :param local_save_dir: Directory to save the preprocessed parquet files.
    :param audio_dir: Optional base directory for audio files.
    :param enable_think: Whether to enable think-mode prompt template.
    :param val_ratio: Fraction of data to hold out as validation. 0 = no split.
    """
    # ---- Read JSONL ----
    records: List[Dict[str, Any]] = []
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARNING] Skipping line {line_no}: {e}")
                continue

            # Validate required fields
            required_fields = ["question_text", "multi_choice", "answer"]
            missing = [k for k in required_fields if k not in obj]
            if missing:
                print(f"[WARNING] Skipping line {line_no}: missing fields {missing}")
                continue

            records.append(obj)

    print(f"Loaded {len(records)} valid records from {input_jsonl}")

    # ---- Convert to LightRFT format ----
    processed: List[Dict[str, Any]] = []
    for idx, obj in enumerate(records):
        try:
            result = build_prompt_and_solution(obj, audio_dir, enable_think)
        except Exception as e:
            print(f"[WARNING] Skipping record {idx}: {e}")
            continue

        # Match LightRFT data format (see geo3k.py / gsm8k.py patterns)
        data = {
            "data_source": "r1-aqa-avqa",
            "prompt": result["prompt"],  # Chat-format messages with audio
            "reference": result["answer_str"],  # Ground truth answer text
            "label": "avqa_rule",  # Routes to RECIPE in reward_models_utils
            "audio_path": result["audio_path"],  # Audio file path
            "reward_model": {
                "ground_truth": result["answer_str"],
            },
            "extra_info": {
                "label": "avqa_rule",
                "reference": result["answer_str"],
                "solution": result["solution"],
                "index": idx,
                "id": obj.get("id", idx),
                "question_text": obj["question_text"],
                "multi_choice": obj["multi_choice"],
                "answer_idx": int(obj["answer"]),
                "dataset_name": obj.get("dataset_name", "AVQA"),
                "audio_path": result["audio_path"],
            },
        }
        processed.append(data)

    print(f"Processed {len(processed)} records")

    # ---- Create HuggingFace Dataset ----
    dataset = datasets.Dataset.from_list(processed)

    # ---- Optional validation split ----
    if val_ratio > 0.0:
        split = dataset.train_test_split(test_size=val_ratio, seed=42)
        train_dataset = split["train"]
        val_dataset = split["test"]
    else:
        train_dataset = dataset
        val_dataset = None

    # ---- Save ----
    local_save_dir = os.path.expanduser(local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    train_path = os.path.join(local_save_dir, "train.parquet")
    train_dataset.to_parquet(train_path)
    print(f"  Train set: {len(train_dataset)} examples saved to {train_path}")

    if val_dataset is not None:
        val_path = os.path.join(local_save_dir, "validation.parquet")
        val_dataset.to_parquet(val_path)
        print(f"  Val set:   {len(val_dataset)} examples saved to {val_path}")

    print(f"\nDataset format:")
    print(f"  - prompt: Chat messages with audio content type")
    print(f"  - reference: Ground truth answer text")
    print(f"  - label: 'avqa_rule' (for recipe-based reward)")
    print(f"  - audio_path: Path to audio .wav file")

    if len(train_dataset) > 0:
        ex = train_dataset[0]
        print(f"\nExample:")
        print(f"  prompt (first msg role): {ex['prompt'][0]['role']}")
        print(f"  reference: {ex['reference']}")
        print(f"  label: {ex['label']}")
        print(f"  audio_path: {ex['audio_path']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess AVQA dataset (R1-AQA format) for LightRFT training"
    )
    parser.add_argument(
        "--input_jsonl",
        required=True,
        type=str,
        help="Path to the R1-AQA AVQA JSONL file (e.g., data/AVQA/train_qa.data)",
    )
    parser.add_argument(
        "--audio_dir",
        default=None,
        type=str,
        help="Base directory containing the .wav audio files",
    )
    parser.add_argument(
        "--local_save_dir",
        default="./avqa_lightrft",
        type=str,
        help="Output directory for the preprocessed parquet dataset",
    )
    parser.add_argument(
        "--enable_think",
        action="store_true",
        default=False,
        help="Enable <think></think> mode in prompt template",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.0,
        help="Fraction of data to hold out as validation (0 = no split)",
    )
    args = parser.parse_args()
    preprocess_avqa(**vars(args))
