#!/usr/bin/env python3
"""
Clean an audio dataset by dropping rows whose audio files are missing or unreadable.

This is intended for LightRFT R1-AQA parquet datasets, but it also works for any
dataset split that contains an ``audio_path`` column.

Recommended workflow:
1. Build parquet with ``examples/r1_aqa/data_preprocess/avqa.py``.
2. Run this script once on the parquet directory.
3. Train on the cleaned output directory, not on the raw parquet directory.

Why this exists:
- In audio GRPO training, the prompt text and the loaded audio payload must stay aligned.
- If a row still contains audio placeholders but its ``audio_path`` no longer exists on disk,
  one distributed rank can silently treat it as text-only while others still process audio.
- That mismatch often surfaces later as a hang during actor forward or replay/PPO processing.

Outputs:
- ``<split>.parquet``: cleaned split with only valid rows
- ``<split>.dropped.jsonl``: dropped rows with original index, ``audio_path``, and reason
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset


def can_decode_audio(audio_path: str) -> tuple[bool, str | None]:
    try:
        import soundfile as sf

        with sf.SoundFile(audio_path):
            return True, None
    except Exception as exc:  # pragma: no cover - best-effort validation helper
        return False, str(exc)


def clean_split(dataset, verify_decode: bool) -> tuple[list[int], list[dict[str, Any]]]:
    """Return kept row indices and dropped-row metadata for one dataset split."""
    keep_indices: list[int] = []
    dropped_rows: list[dict[str, Any]] = []

    for idx, row in enumerate(dataset):
        audio_path = row.get("audio_path")
        if not audio_path:
            dropped_rows.append({"index": idx, "audio_path": audio_path, "reason": "missing_audio_path"})
            continue

        path = Path(audio_path)
        if not path.exists():
            dropped_rows.append({"index": idx, "audio_path": audio_path, "reason": "missing_file"})
            continue

        if verify_decode:
            ok, error = can_decode_audio(audio_path)
            if not ok:
                dropped_rows.append(
                    {"index": idx, "audio_path": audio_path, "reason": "decode_error", "error": error}
                )
                continue

        keep_indices.append(idx)

    return keep_indices, dropped_rows


def main() -> None:
    """CLI entrypoint for dataset cleaning."""
    parser = argparse.ArgumentParser(description="Clean LightRFT audio parquet dataset by removing bad audio rows.")
    parser.add_argument("--input_dataset", required=True, help="Path to the input dataset directory or parquet file")
    parser.add_argument("--output_dir", required=True, help="Directory to write cleaned parquet split files")
    parser.add_argument(
        "--verify_decode",
        action="store_true",
        help="Also verify each existing audio file can be opened by soundfile",
    )
    args = parser.parse_args()

    input_path = Path(args.input_dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_dict = load_dataset(str(input_path))
    summary: dict[str, Any] = {}

    for split_name, split_dataset in dataset_dict.items():
        keep_indices, dropped_rows = clean_split(split_dataset, verify_decode=args.verify_decode)
        cleaned_dataset = split_dataset.select(keep_indices)
        split_out = output_dir / f"{split_name}.parquet"
        report_out = output_dir / f"{split_name}.dropped.jsonl"

        cleaned_dataset.to_parquet(str(split_out))
        with report_out.open("w", encoding="utf-8") as fout:
            for row in dropped_rows:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")

        summary[split_name] = {
            "total": len(split_dataset),
            "kept": len(keep_indices),
            "dropped": len(dropped_rows),
            "parquet": str(split_out),
            "report": str(report_out),
        }

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
