"""
MMAU and MMAR Evaluation Script for R1-AQA models trained with LightRFT.

This script performs inference on MMAU test-mini and/or MMAR benchmark and outputs
results in the format expected by each benchmark's official evaluation script.

- MMAU: https://github.com/Sakshi113/MMAU
  Output key: model_prediction
  Run: python /path/to/mmau/evaluation.py --input <out_file>

- MMAR: https://github.com/ddlBoJack/MMAR 
  Output key: answer_prediction
  Run: python /path/to/mmar/code/evaluation.py --input <out_file>

Output format (per sample):
    MMAU: { ...original_input_fields..., "model_prediction": "<extracted answer>" }
    MMAR: { ...original_input_fields..., "answer_prediction": "<extracted answer>" }

Usage:
    # MMAU
    python examples/r1_aqa/eval.py --benchmark mmau \\
        --model_path /path/to/trained/model \\
        --data_file /path/to/mmau-test-mini.json \\
        --audio_dir /path/to/mmau/audio \\
        --out_file results/res_mmau_mini.json

    # MMAR
    python examples/r1_aqa/eval.py --benchmark mmar \\
        --model_path /path/to/trained/model \\
        --data_file /path/to/MMAR-meta.jsonl \\
        --audio_dir /path/to/mmar/audio \\
        --out_file results/res_mmar.jsonl

    # Then run the official evaluation script for the chosen benchmark.
"""

import argparse
import json
import os
import re
from typing import Dict, List, Optional

import torch


# ---------------------------------------------------------------------------
# Message building (MMAU and MMAR share question/choices + <answer> format)
# ---------------------------------------------------------------------------

def build_message(obj_dict: Dict, audio_dir: Optional[str] = None) -> list:
    """
    Build the chat message for MMAU or MMAR evaluation.

    MMAU uses 'audio_id'; MMAR uses 'audio_path'. Both use 'question' and 'choices'.

    :param obj_dict: One sample from MMAU (JSON) or MMAR (JSONL).
    :param audio_dir: Base directory for audio files.
    :return: Chat messages list.
    """
    choice_str = f"Please choose the answer from the following options: {obj_dict['choices']}."
    question_template = (
        f"{obj_dict['question']} {choice_str} "
        "Output the final answer in <answer></answer>."
    )

    # MMAU uses audio_id; MMAR uses audio_path (e.g. ./audio/xxx.wav)
    raw_path = obj_dict.get("audio_path") or obj_dict.get("audio_id", "")
    if audio_dir and raw_path and not os.path.isabs(raw_path):
        # Allow audio_path to be relative to audio_dir (e.g. ./audio/xxx.wav -> audio_dir/audio/xxx.wav)
        audio_path = os.path.normpath(os.path.join(audio_dir, raw_path))
    else:
        audio_path = raw_path

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

    :param output_str: Raw model output string.
    :return: Extracted answer string.
    """
    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.search(answer_pattern, output_str)
    if match:
        return match.group(1)
    return output_str


# ---------------------------------------------------------------------------
# MMAR official evaluation uses token-based string_match; optional local use
# ---------------------------------------------------------------------------

def _string_match_mmar(answer: str, prediction: str, choices: List[str]) -> bool:
    """
    MMAR evaluation.py string_match: tokenize and check answer ⊆ prediction
    and prediction disjoint from incorrect-choice tokens.
    """
    def tokenize(text: str):
        return set(re.findall(r"\b\w+\b", text.lower()))

    pred_tokens = tokenize(prediction)
    ans_tokens = tokenize(answer)
    if not pred_tokens:
        return False
    incorrect_tokens = set()
    for choice in choices:
        ct = tokenize(choice)
        if ct != ans_tokens:
            incorrect_tokens.update(ct - ans_tokens)
    cond1 = ans_tokens.issubset(pred_tokens)
    cond2 = pred_tokens.isdisjoint(incorrect_tokens)
    return cond1 and cond2


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference_hf(
    model_path: str,
    data: List[Dict],
    audio_dir: Optional[str],
    batch_size: int = 32,
    max_new_tokens: int = 1024,
) -> List[str]:
    """
    Run inference using HuggingFace Transformers (sequential).
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

    def get_audio_path(sample: Dict) -> str:
        raw = sample.get("audio_path") or sample.get("audio_id", "")
        if audio_dir and raw and not os.path.isabs(raw):
            return os.path.normpath(os.path.join(audio_dir, raw))
        return raw

    all_outputs = []
    for i, sample in enumerate(data):
        if (i + 1) % 10 == 0:
            print(f"Processing {i + 1}/{len(data)}...")

        message = build_message(sample, audio_dir)
        text = processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
        audio_path = get_audio_path(sample)

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
    """
    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    sr = getattr(processor.feature_extractor, "sampling_rate", 16000)

    try:
        import librosa
    except ImportError:
        raise ImportError("librosa is required: pip install librosa")

    def get_audio_path(sample: Dict) -> str:
        raw = sample.get("audio_path") or sample.get("audio_id", "")
        if audio_dir and raw and not os.path.isabs(raw):
            return os.path.normpath(os.path.join(audio_dir, raw))
        return raw

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

    all_inputs = []
    for sample in data:
        message = build_message(sample, audio_dir)
        text = processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
        audio_path = get_audio_path(sample)

        try:
            audio, _ = librosa.load(audio_path, sr=sr)
            all_inputs.append({
                "prompt": text,
                "multi_modal_data": {"audio": [(audio, sr)]},
            })
        except Exception as e:
            print(f"[WARNING] Audio load failed for {audio_path}: {e}")
            all_inputs.append({"prompt": text})

    all_outputs_raw = llm.generate(all_inputs, sampling_params)
    all_outputs = [out.outputs[0].text for out in all_outputs_raw]

    return all_outputs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_data(data_file: str, benchmark: str) -> List[Dict]:
    """Load MMAU (JSON array) or MMAR (JSONL) data."""
    with open(data_file, "r", encoding="utf-8") as f:
        if benchmark == "mmar":
            data = [json.loads(line.strip()) for line in f if line.strip()]
        else:
            data = json.load(f)
    return data


def main():
    parser = argparse.ArgumentParser(
        description="MMAU and MMAR evaluation for R1-AQA"
    )
    parser.add_argument("--benchmark", type=str, required=True,
                        choices=["mmau", "mmar"],
                        help="Benchmark: mmau or mmar")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model (HF format)")
    parser.add_argument("--data_file", type=str, required=True,
                        help="Path to benchmark data (MMAU: .json, MMAR: .jsonl)")
    parser.add_argument("--audio_dir", type=str, default=None,
                        help="Base directory for audio files")
    parser.add_argument("--out_file", type=str, required=True,
                        help="Output file for evaluation results")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for inference")
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                        help="Maximum new tokens to generate")
    parser.add_argument("--engine", type=str, default="hf",
                        choices=["hf", "vllm"],
                        help="Inference engine: hf or vllm")
    args = parser.parse_args()

    # Load data
    print(f"Loading {args.benchmark.upper()} data from {args.data_file}...")
    data = load_data(args.data_file, args.benchmark)
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

    # Prediction key per benchmark
    pred_key = "answer_prediction" if args.benchmark == "mmar" else "model_prediction"

    # Build results
    final_output = []
    for input_example, model_output in zip(data, all_outputs):
        model_answer = extract_answer(model_output).strip()
        result = dict(input_example)
        result[pred_key] = model_answer
        result["raw_output"] = model_output
        final_output.append(result)

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(args.out_file)) or ".", exist_ok=True)

    if args.benchmark == "mmar":
        with open(args.out_file, "w", encoding="utf-8") as f:
            for item in final_output:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    else:
        with open(args.out_file, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {args.out_file}")
    print(f"Total samples: {len(final_output)}")

    # Quick accuracy
    correct = 0
    total = 0
    for item in final_output:
        gt = item.get("answer")
        if gt is None:
            continue
        total += 1
        pred = item[pred_key].strip()
        if args.benchmark == "mmar":
            match = _string_match_mmar(str(gt), pred, item.get("choices", []))
        else:
            match = pred.lower() == str(gt).strip().lower()
        if match:
            correct += 1

    if total > 0:
        print(f"Quick accuracy: {correct}/{total} = {correct / total:.4f}")
    else:
        print("(No 'answer' field in data — run official evaluation script for metrics)")

    if args.benchmark == "mmau":
        print(f"\nTo evaluate with MMAU's official script:")
        print(f"  python /path/to/mmau/evaluation.py --input {args.out_file}")
    else:
        print(f"\nTo evaluate with MMAR's official script:")
        print(f"  python /path/to/mmar/code/evaluation.py --input {args.out_file}")


if __name__ == "__main__":
    main()
