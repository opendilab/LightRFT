"""
MMAU Test-Mini Evaluation Script for R1-AQA models trained with LightRFT.

This script performs inference on the MMAU test-mini benchmark and outputs
results in the format expected by MMAU's official evaluation script.

Faithfully ported from R1-AQA's ``src/test_mmau.py``. You can find the original data of MMAU in the repository: https://github.com/Sakshi113/MMAU

Output format (per sample):
    {
        ...original_input_fields...,
        "model_prediction": "<extracted answer>"
    }

Usage:
    python examples/r1_aqa/eval_mmau.py \\
        --model_path /path/to/trained/model \\
        --data_file /path/to/mmau-test-mini.json \\
        --audio_dir /path/to/mmau/audio \\
        --out_file results/res_mmau_mini.json \\
        --batch_size 32

    # Then evaluate with MMAU's official script:
    python /path/to/mmau/evaluation.py --input results/res_mmau_mini.json
"""

import argparse
import json
import os
import re
from typing import Dict, List, Optional

import torch


def build_message(obj_dict: Dict, audio_dir: Optional[str] = None) -> list:
    """
    Build the chat message for MMAU evaluation.

    Ported from R1-AQA ``src/test_mmau.py::_get_message``.
    Note: MMAU uses 'question' and 'choices' fields (different from AVQA training).

    :param obj_dict: One sample from MMAU test-mini.
    :param audio_dir: Base directory for audio files.
    :return: Chat messages list.
    """
    choice_str = f"Please choose the answer from the following options: {obj_dict['choices']}."
    question_template = (
        f"{obj_dict['question']} {choice_str} "
        "Output the final answer in <answer></answer>."
    )

    audio_id = obj_dict.get("audio_id", "")
    if audio_dir and not os.path.isabs(audio_id):
        audio_path = os.path.join(audio_dir, audio_id)
    else:
        audio_path = audio_id

    message = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": audio_path},
                {"type": "text", "text": question_template},
            ],
        }
    ]
    return message


def extract_answer(output_str: str) -> str:
    """
    Extract the answer from model output using <answer>...</answer> tags.

    Ported from R1-AQA ``src/test_mmau.py::extract_answer``.

    :param output_str: Raw model output string.
    :return: Extracted answer string.
    """
    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.search(answer_pattern, output_str)
    if match:
        return match.group(1)
    return output_str


def run_inference_hf(
    model_path: str,
    data: List[Dict],
    audio_dir: Optional[str],
    batch_size: int = 32,
    max_new_tokens: int = 1024,
) -> List[str]:
    """
    Run inference using HuggingFace Transformers (non-batched for simplicity).

    :param model_path: Path to the trained model.
    :param data: List of MMAU samples.
    :param audio_dir: Base directory for audio files.
    :param batch_size: Not used for HF inference (processed sequentially).
    :param max_new_tokens: Maximum tokens to generate.
    :return: List of generated output strings.
    """
    from transformers import AutoProcessor

    try:
        from transformers import Qwen2AudioForConditionalGeneration
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to("cuda").eval()
    except (ImportError, OSError):
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to("cuda").eval()

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    try:
        import librosa
    except ImportError:
        raise ImportError("librosa is required for audio loading: pip install librosa")

    sr = getattr(processor.feature_extractor, "sampling_rate", 16000)

    all_outputs = []
    for i, sample in enumerate(data):
        if (i + 1) % 10 == 0:
            print(f"Processing {i + 1}/{len(data)}...")

        message = build_message(sample, audio_dir)

        # Apply chat template
        text = processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )

        # Load audio
        audio_id = sample.get("audio_id", "")
        if audio_dir and not os.path.isabs(audio_id):
            audio_path = os.path.join(audio_dir, audio_id)
        else:
            audio_path = audio_id

        try:
            audio, _ = librosa.load(audio_path, sr=sr)
            inputs = processor(
                text=text, audios=[audio], return_tensors="pt", padding=True
            )
        except Exception as e:
            print(f"[WARNING] Audio load failed for {audio_path}: {e}")
            inputs = processor(text=text, return_tensors="pt", padding=True)

        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        # Decode — skip input tokens
        input_len = inputs["input_ids"].shape[-1]
        generated = outputs[0][input_len:]
        output_text = processor.decode(generated, skip_special_tokens=True)
        all_outputs.append(output_text)

    return all_outputs


def run_inference_vllm(
    model_path: str,
    data: List[Dict],
    audio_dir: Optional[str],
    batch_size: int = 32,
    max_new_tokens: int = 1024,
) -> List[str]:
    """
    Run inference using vLLM for better throughput.

    :param model_path: Path to the trained model.
    :param data: List of MMAU samples.
    :param audio_dir: Base directory for audio files.
    :param batch_size: Batch size for vLLM.
    :param max_new_tokens: Maximum tokens to generate.
    :return: List of generated output strings.
    """
    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    sr = getattr(processor.feature_extractor, "sampling_rate", 16000)

    try:
        import librosa
    except ImportError:
        raise ImportError("librosa is required: pip install librosa")

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=4096,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_new_tokens,
    )

    # Prepare all inputs
    all_inputs = []
    for sample in data:
        message = build_message(sample, audio_dir)
        text = processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )

        audio_id = sample.get("audio_id", "")
        if audio_dir and not os.path.isabs(audio_id):
            audio_path = os.path.join(audio_dir, audio_id)
        else:
            audio_path = audio_id

        try:
            audio, _ = librosa.load(audio_path, sr=sr)
            all_inputs.append({
                "prompt": text,
                "multi_modal_data": {"audio": [(audio, sr)]},
            })
        except Exception as e:
            print(f"[WARNING] Audio load failed for {audio_path}: {e}")
            all_inputs.append({"prompt": text})

    # Batch generation
    all_outputs_raw = llm.generate(all_inputs, sampling_params)
    all_outputs = [out.outputs[0].text for out in all_outputs_raw]

    return all_outputs


def main():
    parser = argparse.ArgumentParser(description="MMAU Test-Mini Evaluation for R1-AQA")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model (HF format)")
    parser.add_argument("--data_file", type=str, required=True,
                        help="Path to mmau-test-mini.json")
    parser.add_argument("--audio_dir", type=str, default=None,
                        help="Base directory for MMAU audio files")
    parser.add_argument("--out_file", type=str, required=True,
                        help="Output JSON file for evaluation results")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for inference")
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                        help="Maximum new tokens to generate")
    parser.add_argument("--engine", type=str, default="hf",
                        choices=["hf", "vllm"],
                        help="Inference engine: hf (HuggingFace) or vllm")
    args = parser.parse_args()

    # Load MMAU data
    print(f"Loading data from {args.data_file}...")
    with open(args.data_file, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples")

    # Run inference
    print(f"Running inference with {args.engine} engine...")
    if args.engine == "vllm":
        all_outputs = run_inference_vllm(
            args.model_path, data, args.audio_dir,
            args.batch_size, args.max_new_tokens,
        )
    else:
        all_outputs = run_inference_hf(
            args.model_path, data, args.audio_dir,
            args.batch_size, args.max_new_tokens,
        )

    # Extract answers and build output
    final_output = []
    for input_example, model_output in zip(data, all_outputs):
        original_output = model_output
        model_answer = extract_answer(original_output).strip()

        result = dict(input_example)
        result["model_prediction"] = model_answer
        result["raw_output"] = original_output  # Keep for debugging
        final_output.append(result)

    # Save results
    os.makedirs(os.path.dirname(os.path.abspath(args.out_file)), exist_ok=True)
    with open(args.out_file, "w") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {args.out_file}")
    print(f"Total samples: {len(final_output)}")

    # Quick accuracy report
    correct = 0
    total = 0
    for item in final_output:
        if "answer" in item:
            total += 1
            if item["model_prediction"].strip().lower() == str(item["answer"]).strip().lower():
                correct += 1

    if total > 0:
        print(f"Quick accuracy: {correct}/{total} = {correct / total:.4f}")
    else:
        print("(No 'answer' field found in data — run MMAU evaluation.py for metrics)")

    print(f"\nTo evaluate with MMAU's official script:")
    print(f"  python /path/to/mmau/evaluation.py --input {args.out_file}")


if __name__ == "__main__":
    main()
