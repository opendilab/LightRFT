import os
import sys
from pathlib import Path

import pytest
from PIL import Image
from transformers import AutoProcessor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from meme_dataset import MemeOnlineRLDataset  # noqa: E402
from meme_utils import extract_box_texts, normalize_detections, render_meme_image  # noqa: E402
from reward_model import (  # noqa: E402
    _load_reward_prompt,
    build_pairwise_judge_message,
    parse_pair_judge_response,
)


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        pytest.skip(f"{name} is not set; skipping real vLLM meme reward test")
    return value


def test_reward_model_vllm_real_data():
    if not os.getenv("RUN_GODSMEME_VLLM_TEST"):
        pytest.skip("Set RUN_GODSMEME_VLLM_TEST=1 to run the real vLLM integration test")
    if not Image:
        pytest.skip("PIL is unavailable")

    vllm = pytest.importorskip("vllm")
    if not os.getenv("CUDA_VISIBLE_DEVICES") and not os.path.exists("/dev/nvidia0"):
        pytest.skip("This integration test requires a CUDA-visible vLLM environment")

    model_path = _require_env("GODSMEME_REWARD_MODEL_PATH")
    annotation_path = _require_env("GODSMEME_ANNOTATION_PATH")
    image_root = _require_env("GODSMEME_IMAGE_ROOT")

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    dataset = MemeOnlineRLDataset(
        annotation_path=annotation_path,
        root_dir=image_root,
        processor=processor,
        shuffle=False,
    )
    prompt, image_paths, reference, _ = dataset[0]
    assert prompt
    assert image_paths
    assert reference["reference_output"]

    base_image = Image.open(image_paths[0]).convert("RGB")
    detections = normalize_detections(reference)
    reference_texts = extract_box_texts(reference["reference_output"])
    assert reference_texts, "reference_output must contain meme text boxes"

    bad_texts = ["lorem ipsum dolor sit amet" for _ in range(len(reference_texts))]
    good_image = render_meme_image(base_image, reference_texts, detections=detections, reference=reference)
    bad_image = render_meme_image(base_image, bad_texts, detections=detections, reference=reference)

    reward_prompt = _load_reward_prompt(os.getenv("GODSMEME_REWARD_PROMPT_PATH"))
    judge_message = build_pairwise_judge_message(
        reward_prompt=reward_prompt,
        image_a=good_image,
        image_b=bad_image,
        text_a="\\n".join(reference_texts),
        text_b="\\n".join(bad_texts),
        user_request=reference.get("user_request", ""),
    )
    rendered_prompt = processor.apply_chat_template(judge_message, tokenize=False, add_generation_prompt=True)

    llm = vllm.LLM(
        model=model_path,
        tensor_parallel_size=int(os.getenv("GODSMEME_VLLM_TP", "1")),
        trust_remote_code=True,
        max_model_len=int(os.getenv("GODSMEME_VLLM_MAX_MODEL_LEN", "4096")),
        limit_mm_per_prompt={"image": 2},
    )
    sampling_params = vllm.SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=int(os.getenv("GODSMEME_VLLM_MAX_TOKENS", "96")),
    )
    outputs = llm.generate(
        [{
            "prompt": rendered_prompt,
            "multi_modal_data": {
                "image": [good_image, bad_image]
            }
        }],
        sampling_params,
    )
    generated_text = outputs[0].outputs[0].text
    score_a, score_b = parse_pair_judge_response(generated_text)

    assert isinstance(score_a, float)
    assert isinstance(score_b, float)
    assert score_a >= score_b
