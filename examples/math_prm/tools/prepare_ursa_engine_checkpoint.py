#!/usr/bin/env python
"""
Build an engine-friendly local wrapper checkpoint for URSA-8B.

The upstream URSA checkpoints do not ship ``auto_map`` metadata or local model
code files, which prevents inference engines such as vLLM/SGLang from loading
the custom architecture via HuggingFace dynamic modules. This helper creates a
thin wrapper directory that:

1. symlinks the original checkpoint weights/tokenizer assets
2. symlinks the local ``examples/math_prm/ursa_model/*.py`` files
3. writes patched ``config.json`` / ``preprocessor_config.json`` /
   ``tokenizer_config.json`` with the required ``auto_map`` entries
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


MODEL_AUTO_MAP = {
    "AutoConfig": "configuration_ursa.UrsaConfig",
    "AutoModel": "modeling_ursa.UrsaForConditionalGeneration",
    "AutoModelForVision2Seq": "modeling_ursa.UrsaForConditionalGeneration",
}

PROCESSOR_AUTO_MAP = {
    "AutoProcessor": "processing_ursa.UrsaProcessor",
    "AutoImageProcessor": "image_processing_vlm.VLMImageProcessor",
}


def _safe_unlink(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _ensure_symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() and dst.resolve() == src.resolve():
            return
        _safe_unlink(dst)
    dst.symlink_to(src)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def build_wrapper(source_model_path: Path, output_path: Path, local_ursa_dir: Path) -> None:
    output_path.mkdir(parents=True, exist_ok=True)

    for src in source_model_path.iterdir():
        dst = output_path / src.name
        if src.name in {"config.json", "preprocessor_config.json", "tokenizer_config.json"}:
            continue
        _ensure_symlink(src, dst)

    for src in local_ursa_dir.glob("*.py"):
        _ensure_symlink(src, output_path / src.name)

    config = _load_json(source_model_path / "config.json")
    auto_map = dict(config.get("auto_map") or {})
    auto_map.update(MODEL_AUTO_MAP)
    config["auto_map"] = auto_map
    _write_json(output_path / "config.json", config)

    preprocessor_path = source_model_path / "preprocessor_config.json"
    if preprocessor_path.exists():
        preprocessor = _load_json(preprocessor_path)
        preprocessor["processor_class"] = "UrsaProcessor"
        preprocessor["image_processor_type"] = "VLMImageProcessor"
        preprocessor_auto_map = dict(preprocessor.get("auto_map") or {})
        preprocessor_auto_map.update(PROCESSOR_AUTO_MAP)
        preprocessor["auto_map"] = preprocessor_auto_map
        _write_json(output_path / "preprocessor_config.json", preprocessor)

    tokenizer_config_path = source_model_path / "tokenizer_config.json"
    if tokenizer_config_path.exists():
        tokenizer_config = _load_json(tokenizer_config_path)
        tokenizer_config["processor_class"] = "UrsaProcessor"
        tokenizer_auto_map = dict(tokenizer_config.get("auto_map") or {})
        tokenizer_auto_map.update(PROCESSOR_AUTO_MAP)
        tokenizer_config["auto_map"] = tokenizer_auto_map
        _write_json(output_path / "tokenizer_config.json", tokenizer_config)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-model-path", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()

    source_model_path = Path(args.source_model_path).resolve()
    output_path = Path(args.output_path).resolve()
    local_ursa_dir = Path(__file__).resolve().parents[1] / "ursa_model"

    build_wrapper(source_model_path, output_path, local_ursa_dir)
    print(str(output_path))


if __name__ == "__main__":
    main()
