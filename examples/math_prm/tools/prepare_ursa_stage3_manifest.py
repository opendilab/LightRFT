"""
Prepare a LightRFT-compatible Stage 3 manifest from URSA-MATH raw data.

This script converts the raw `MMathCoT-1M` jsonl schema:

    {"image_url": "...", "instruction": "...", "output": "..."}

into a LightRFT prompt dataset schema:

    {
        "prompt": "...",
        "images": ["/abs/path/to/image.png"],
        "reference": "...",
        "label": "math_psgrpo"
    }

It also performs a lightweight `PromptDatasetVL` smoke validation on the
converted records so the output can be consumed directly by
`examples/math_prm/train_colocate.py`.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]

import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lightrft.datasets.prompts_dataset_vl import PromptDatasetVL


# --input-path and --image-root are intentionally required (no path default):
# the data lives outside the repo and varies per environment. The output paths
# resolve under REPO_ROOT/tmp/ so they always succeed locally.
DEFAULT_OUTPUT_PATH = str(REPO_ROOT / "tmp" / "ursa_stage3" / "mmathcot_stage3_math_psgrpo.jsonl")
DEFAULT_SUMMARY_PATH = str(REPO_ROOT / "tmp" / "ursa_stage3" / "mmathcot_stage3_math_psgrpo.summary.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert URSA-MATH MMathCoT-1M raw jsonl into a LightRFT "
            "Stage 3 prompt manifest and validate it with PromptDatasetVL."
        )
    )
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to MMathCoT-1M raw train.jsonl (e.g. /data/URSA-MATH/MMathCoT-1M/train.jsonl).",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        required=True,
        help="Root directory for URSA-MATH image assets (e.g. /data/URSA-MATH/images).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help="Output path for the converted LightRFT jsonl manifest.",
    )
    parser.add_argument(
        "--summary-path",
        type=str,
        default=DEFAULT_SUMMARY_PATH,
        help="Path to write the conversion/validation summary json.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="math_psgrpo",
        help="Label written into the converted manifest.",
    )
    parser.add_argument(
        "--prompt-mode",
        type=str,
        choices=["question_only", "instruction"],
        default="question_only",
        help="How to build the LightRFT prompt from raw instruction.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for the number of raw rows to process.",
    )
    parser.add_argument(
        "--smoke-samples",
        type=int,
        default=4,
        help="How many converted samples to use for PromptDatasetVL smoke validation.",
    )
    return parser.parse_args()


def extract_prompt(raw_instruction: str, prompt_mode: str) -> tuple[str, bool]:
    text = (raw_instruction or "").strip()
    if not text:
        return "", False

    if prompt_mode == "instruction":
        return text, False

    marker = "Question:"
    idx = text.find(marker)
    if idx == -1:
        return text, True

    question = text[idx + len(marker):].strip()
    # Some raw rows contain duplicated or malformed prefixes such as
    # "Question:estion: ...". Strip these repeatedly before returning.
    prefix_re = re.compile(r"^(?:(?:[Qq]uestion|[Qq]estion|[Ee]stion|[Uu]estion)\s*:)\s*")
    while True:
        cleaned = prefix_re.sub("", question)
        if cleaned == question:
            break
        question = cleaned.strip()
    if not question:
        return text, True
    return question, False


def extract_reference(raw_output: str) -> tuple[str, bool]:
    text = (raw_output or "").strip()
    if not text:
        return "", False

    marker = "†Answer:"
    idx = text.rfind(marker)
    if idx == -1:
        return text, True

    answer = text[idx + len(marker):].strip()
    if not answer:
        return text, True
    return answer, False


def build_record(
    raw: dict[str, Any],
    source_index: int,
    image_root: Path,
    prompt_mode: str,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    image_url = str(raw.get("image_url", "")).strip()
    instruction = str(raw.get("instruction", "")).strip()
    output = str(raw.get("output", "")).strip()

    prompt, used_prompt_fallback = extract_prompt(instruction, prompt_mode)
    reference, used_reference_fallback = extract_reference(output)

    image_path = (image_root / image_url).resolve()
    prefix = image_url.split("/", 1)[0] if image_url else ""

    record = {
        "data_source": "URSA-MATH/MMathCoT-1M",
        "prompt": prompt,
        "images": [str(image_path)],
        "reference": reference,
        "ground_truth": reference,
        "label": label,
        "reward_model": {
            "ground_truth": reference,
        },
        "extra_info": {
            "source_index": source_index,
            "raw_image_url": image_url,
            "image_prefix": prefix,
            "prompt_mode": prompt_mode,
        },
    }

    meta = {
        "image_path_exists": image_path.exists(),
        "prompt_empty": prompt == "",
        "reference_empty": reference == "",
        "used_prompt_fallback": used_prompt_fallback,
        "used_reference_fallback": used_reference_fallback,
        "image_prefix": prefix,
        "image_path": str(image_path),
    }
    return record, meta


def smoke_validate(converted_rows: list[dict[str, Any]], smoke_samples: int) -> dict[str, Any]:
    smoke_rows = converted_rows[: max(1, min(smoke_samples, len(converted_rows)))]
    strategy = SimpleNamespace(
        args=SimpleNamespace(
            input_key="prompt",
            images_key="images",
            reference_key="reference",
            label_key="label",
            apply_chat_template=False,
            system_prompt=None,
        )
    )
    dataset = PromptDatasetVL(
        smoke_rows,
        tokenizer=None,
        processor=None,
        max_length=0,
        strategy=strategy,
    )
    items = [dataset[i] for i in range(len(dataset))]
    prompts, images, references, labels = dataset.collate_fn(items)

    first_prompt, first_images, first_reference, first_label = items[0]
    return {
        "sample_count": len(dataset),
        "first_item": {
            "prompt_preview": first_prompt[:240],
            "image_count": len(first_images) if isinstance(first_images, list) else 0,
            "first_image": first_images[0] if isinstance(first_images, list) and first_images else None,
            "reference": first_reference,
            "label": first_label,
        },
        "collate_sizes": {
            "prompts": len(prompts),
            "images": len(images),
            "references": len(references),
            "labels": len(labels),
        },
    }


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_path).resolve()
    image_root = Path(args.image_root).resolve()
    output_path = Path(args.output_path).resolve()
    summary_path = Path(args.summary_path).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"input jsonl not found: {input_path}")
    if not image_root.exists():
        raise FileNotFoundError(f"image root not found: {image_root}")

    counters = Counter()
    prefix_counter: Counter[str] = Counter()
    smoke_rows: list[dict[str, Any]] = []

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as fp, output_path.open("w", encoding="utf-8") as out_fp:
        for source_index, line in enumerate(fp):
            if args.max_samples is not None and source_index >= args.max_samples:
                break

            counters["rows_seen"] += 1
            raw = json.loads(line)
            record, meta = build_record(
                raw=raw,
                source_index=source_index,
                image_root=image_root,
                prompt_mode=args.prompt_mode,
                label=args.label,
            )

            prefix_counter[meta["image_prefix"]] += 1
            if meta["used_prompt_fallback"]:
                counters["prompt_fallback_rows"] += 1
            if meta["used_reference_fallback"]:
                counters["reference_fallback_rows"] += 1
            if meta["prompt_empty"]:
                counters["empty_prompt_rows"] += 1
            if meta["reference_empty"]:
                counters["empty_reference_rows"] += 1
            if not meta["image_path_exists"]:
                raise FileNotFoundError(
                    f"missing image for row {source_index}: {meta['image_path']}"
                )

            out_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
            counters["rows_written"] += 1
            if len(smoke_rows) < max(1, args.smoke_samples):
                smoke_rows.append(record)

    if not smoke_rows:
        raise ValueError("No rows were converted. Check the input path and --max-samples.")

    smoke = smoke_validate(smoke_rows, args.smoke_samples)

    summary = {
        "input_path": str(input_path),
        "image_root": str(image_root),
        "output_path": str(output_path),
        "summary_path": str(summary_path),
        "label": args.label,
        "prompt_mode": args.prompt_mode,
        "rows_seen": counters["rows_seen"],
        "rows_written": counters["rows_written"],
        "prompt_fallback_rows": counters["prompt_fallback_rows"],
        "reference_fallback_rows": counters["reference_fallback_rows"],
        "empty_prompt_rows": counters["empty_prompt_rows"],
        "empty_reference_rows": counters["empty_reference_rows"],
        "image_prefix_counts": dict(prefix_counter),
        "images_per_sample_counts": {"1": counters["rows_written"]},
        "smoke_validation": smoke,
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
