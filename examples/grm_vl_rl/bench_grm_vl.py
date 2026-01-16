import os
import json
import torch
from tqdm import tqdm
from typing import List, Dict
from loguru import logger
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod

from transformers import AutoProcessor, AutoTokenizer
from vllm import LLM, SamplingParams

from lightrft.datasets import RFTDatasetVL, extract_answer


def extract_response(text: str, media_type: str = "Image") -> str:
    """
    Extract the preference from the generated text.

    It first tries to extract the content from ``<answer>`` tags using :func:`extract_answer`.
    If no tags are found, it performs a heuristic search for key phrases (e.g., "Image 1 is better")
    at the end of the text.

    :param text: The generated text from the model
    :type text: str
    :param media_type: The type of media being evaluated ("Image", "Video", or "Audio"), defaults to "Image"
    :type media_type: str, optional

    :return: The extracted preference string (e.g., "Image 1 is better") or None if not found
    :rtype: str

    **Example:**

    .. code-block:: python

        resp = extract_response("<think>...</think><answer>Image 1 is better</answer>", media_type="Image")
    """
    # 1. Try extracting from <answer> tags
    ans = extract_answer(text)
    if ans:
        return ans

    # 2. Heuristic search if no tags found
    text_lower = text.lower()
    media_lower = media_type.lower()
    
    key_1 = f"{media_lower} 1 is better"
    key_2 = f"{media_lower} 2 is better"
    key_equal = f"both {media_lower}s are equally good"
    
    idx_1 = text_lower.rfind(key_1)
    idx_2 = text_lower.rfind(key_2)
    idx_equal = text_lower.rfind(key_equal)
    
    candidates = {}
    if idx_1 != -1: 
        candidates[f"{media_type} 1 is better"] = idx_1
    if idx_2 != -1: 
        candidates[f"{media_type} 2 is better"] = idx_2
    if idx_equal != -1: 
        candidates[f"Both {media_lower}s are equally good"] = idx_equal
    
    if not candidates:
        return None
        
    # Return the one that appears last in the text
    return max(candidates, key=candidates.get)


# Example Task Instruction for GRM Evaluation
TASK_INSTRUCTION_COT_T2I = """Given a caption and two images generated based on this caption, please analyze in detail the two provided images. 
Evaluate them on various dimensions such as semantic consistency (how closely the image content aligns with the caption), 
aesthetics (composition, color usage, artistic expression), authenticity (realism and attention to detail), 
and any other factors you deem relevant. For each evaluation dimension, 
provide a score between 1-10 for both images (e.g., Image 1: 8/10, Image 2: 6/10) and provide a concise rationale for the score. 
Calculate the total score for each image by summing all dimension scores. 
Use a chain-of-thought process to detail your reasoning steps, and enclose all your detailed reasoning within tags. 
Then, in the <answer> tag, output exactly one of the following strings: 'Image 1 is better' or 'Image 2 is better' or 'Both images are equally good' based on the total scores. 
No additional text is allowed in the <answer> section.
Example output format:
<think>
Semantic consistency: Image 1 (9/10) - ...; Image 2 (7/10) - ...
Aesthetics: Image 2 (8/10) - ...; Image 1 (8/10) - ...
Authenticity: Image 1 (8/10) - ...; Image 2 (5/10) - ...
[Additional dimensions if any]: Image 2 (8/10) - ...; Image 1 (6/10) - ...
Total score:
Image 1: 9+8+8+6=31
Image 2: 7+8+5+8=28
</think>
<answer>Image 1 is better</answer>
Note: In the example above, scores and the final answer are placeholders meant only to demonstrate the format. Your actual evaluation should be based on the quality of two given images.
Your task is provided as follows:
Text Caption: {prompt}
"""

TASK_INSTRUCTION_COT_T2V ="""
Given a caption and two videos generated based on this caption, please analyze in detail the two provided videos. 
Evaluate them on various dimensions such as semantic consistency (how closely the video content aligns with the caption), temporal coherence (smoothness and logical flow of motion across frames), authenticity (realism and attention to detail), and any other factors you deem relevant. 
For each evaluation dimension, provide a score between 1-10 for both videos (e.g., Video 1: 8/10, Video 2: 6/10) and provide a concise rationale for the score. 
Calculate the total score for each video by summing all dimension scores. 
Use a chain-of-thought process to detail your reasoning steps, and enclose all your detailed reasoning within <think> and </think> tags. Then, in the <answer> tag, output exactly one of the following strings:
'Video 1 is better' or 'Video 2 is better' or 'Both videos are equally good' based on the total scores. No additional text is allowed in the <answer> section.
Example output format:
<think>
1. Semantic consistency: Video 1 (9/10) - ...; Video 2 (7/10) - ...
2. Temporal coherence: Video 1 (8/10) - ...; Video 2 (6/10) - ...
3. Authenticity: Video 1 (7/10) - ...; Video 2 (5/10) - ...
...
[Additional dimensions if any]: Video 2 (8/10) - ...; Video 1 (6/10) - ...
Total score:
Video 1: 9+8+7+6=30
Video 2: 7+6+5+8=26
</think>
<answer>Video 1 is better</answer>

Note: In the example above, scores and the final answer are placeholders meant only to demonstrate the format. Your actual evaluation should be based on the quality of two given videos.
Your task is provided as follows:
Text Caption: **{prompt}**
"""

class BaseEvaluator(ABC):
    """
    Base class for evaluators responsible for parsing results and calculating metrics.
    """
    
    def __init__(self):
        """
        Initialize the evaluator with default metrics state.
        """
        self.correct = 0
        self.total = 0
        self.parse_failures = 0
        self.results = []

    @abstractmethod
    def evaluate_batch(self, gen_texts: List[str], extras: List[Dict]) -> None:
        """
        Process a batch of generation results and update metrics.

        :param gen_texts: List of texts generated by the model
        :type gen_texts: List[str]
        :param extras: List of metadata dictionaries from the dataset
        :type extras: List[Dict]
        """
        pass

    def get_accuracy(self) -> float:
        """
        Calculate the current accuracy.

        :return: Accuracy as a float between 0.0 and 1.0
        :rtype: float
        """
        return self.correct / self.total if self.total > 0 else 0.0

    def get_parse_failure_rate(self) -> float:
        """
        Calculate the current parsing failure rate.

        :return: Failure rate as a float between 0.0 and 1.0
        :rtype: float
        """
        return self.parse_failures / self.total if self.total > 0 else 0.0

    def get_results(self) -> List[Dict]:
        """
        Retrieve all evaluation results accumulated so far.

        :return: List of result dictionaries
        :rtype: List[Dict]
        """
        return self.results


class ImageGenCoTEvaluator(BaseEvaluator):
    """
    Evaluator for ImageGen-CoT-5K dataset.
    """
    def evaluate_batch(self, gen_texts: List[str], extras: List[Dict]) -> None:
        """
        Evaluate a batch of ImageGen-CoT-5K results.

        :param gen_texts: List of generated CoT responses
        :type gen_texts: List[str]
        :param extras: List of metadata including ground truth responses
        :type extras: List[Dict]
        """
        for i, (gen_text, other) in enumerate(zip(gen_texts, extras)):
            predicted_answer = extract_response(gen_text, media_type="Image")
            gt_answer = extract_answer(other["response"])   # 'Image 1 is better' or 'Image 2 is better'
            
            is_correct = False
            if gt_answer == predicted_answer:
                self.correct += 1
                is_correct = True
            elif predicted_answer is None:
                self.parse_failures += 1
                print(f"Could not extract answer from generated text: {gen_text}")
            self.total += 1
                        
            print(f"Batch Sample {i} | Pred: {predicted_answer} | GT: {gt_answer} | Correct: {is_correct}")

            self.results.append({
                "ground_truth": gt_answer,
                "predicted_answer": predicted_answer,
                "prompt": other.get("system_prompt", ""),
                "generated_text": gen_text,
            })
        

class OmniRewardBenchEvaluator(BaseEvaluator):
    """
    Generic evaluator for OmniRewardBench, supporting both Image and Video.
    """
    def __init__(self, media_type: str):
        """
        Initialize the OmniRewardBench evaluator.

        :param media_type: The media type being evaluated ("Image" or "Video")
        :type media_type: str
        """
        super().__init__()
        self.media_type = media_type  # "Image" or "Video"

    def evaluate_batch(self, gen_texts: List[str], extras: List[Dict]) -> None:
        """
        Evaluate a batch of OmniRewardBench results.

        :param gen_texts: List of generated responses
        :type gen_texts: List[str]
        :param extras: List of metadata including preferred choice (A/B/C)
        :type extras: List[Dict]
        """
        for i, (gen_text, other) in enumerate(zip(gen_texts, extras)):
            predicted_answer = extract_response(gen_text, media_type=self.media_type)
            gt_preference = other['preference'] # A, B, or C
            
            better_1 = f"{self.media_type} 1 is better"
            better_2 = f"{self.media_type} 2 is better"
            equal = f"Both {self.media_type.lower()}s are equally good"

            # Mapping logic: A -> 1, B -> 2, C -> Equal
            is_correct = False
            if gt_preference == "A" and predicted_answer == better_1:
                is_correct = True
            elif gt_preference == "B" and predicted_answer == better_2:
                is_correct = True
            elif gt_preference == "C" and predicted_answer == equal:
                is_correct = True

            if is_correct:
                self.correct += 1
            elif predicted_answer is None:
                self.parse_failures += 1
                print(f"Could not extract answer from generated text: {gen_text}")
            self.total += 1

            print(f"Sample {i} | Pred: {predicted_answer} | GT: {gt_preference} | Correct: {is_correct}")

            self.results.append({
                "id": other["id"],
                "prompt": other['prompt'],
                "criteria": other['criteria'],
                "ground_truth": gt_preference,
                "predicted_answer": predicted_answer,
                "is_correct": is_correct,
                "generated_text": gen_text,
            })


class HPDv3GRMEvaluator(BaseEvaluator):
    """
    Evaluator for HPDv3 Test set in GRM-style evaluation.
    """
    def evaluate_batch(self, gen_texts: List[str], extras: List[Dict]) -> None:
        """
        Evaluate a batch of HPDv3 results.

        :param gen_texts: List of generated responses
        :type gen_texts: List[str]
        :param extras: List of metadata including preference and paths
        :type extras: List[Dict]
        """
        for i, (gen_text, other) in enumerate(zip(gen_texts, extras)):
            predicted_answer = extract_response(gen_text, media_type="Image")
            gt_preference = other['preference'] # A, B, or C
            
            # Mapping logic: A -> Image 1, B -> Image 2
            is_correct = False
            if gt_preference == "A" and predicted_answer == "Image 1 is better":
                is_correct = True
            elif gt_preference == "B" and predicted_answer == "Image 2 is better":
                is_correct = True
            
            if is_correct:
                self.correct += 1
            elif predicted_answer is None:
                self.parse_failures += 1
                print(f"Could not extract answer from generated text: {gen_text}")
            self.total += 1

            print(f"Sample {i} | Pred: {predicted_answer} | GT: {gt_preference} | Correct: {is_correct}")

            self.results.append({
                "prompt": other['prompt'],
                "ground_truth": gt_preference,
                "predicted_answer": predicted_answer,
                "generated_text": gen_text,
                "is_correct": is_correct,
                "model_chosen": other['model_chosen'],
                "model_rejected": other['model_rejected'],
                "preferred_path": other['preferred_path'],
                "rejected_path": other['rejected_path'],
            })


class GenAIBenchEvaluator(BaseEvaluator):
    """
    Evaluator for GenAI-Bench dataset.
    """
    def __init__(self, media_type: str = "Image"):
        """
        Initialize the GenAI-Bench evaluator.

        :param media_type: The media type ("Image" or "Video"), defaults to "Image"
        :type media_type: str, optional
        """
        super().__init__()
        self.media_type = media_type  # "Image" or "Video"

    def evaluate_batch(self, gen_texts: List[str], extras: List[Dict]) -> None:
        """
        Evaluate a batch of GenAI-Bench results.

        :param gen_texts: List of generated responses
        :type gen_texts: List[str]
        :param extras: List of metadata including preference and model names
        :type extras: List[Dict]
        """
        for i, (gen_text, other) in enumerate(zip(gen_texts, extras)):
            predicted_answer = extract_response(gen_text, media_type=self.media_type)
            gt_preference = other['preference']  # 'A', 'B', or 'C' (Tie)
            
            better_1 = f"{self.media_type} 1 is better"
            better_2 = f"{self.media_type} 2 is better"
            equal = f"Both {self.media_type.lower()}s are equally good"

            is_correct = False
            if gt_preference == "A" and predicted_answer == better_1:
                is_correct = True
            elif gt_preference == "B" and predicted_answer == better_2:
                is_correct = True
            elif gt_preference == "C" and predicted_answer == equal:
                is_correct = True
            
            if is_correct:
                self.correct += 1
            elif predicted_answer is None:
                self.parse_failures += 1
                print(f"Could not extract answer from generated text: {gen_text}")
            self.total += 1

            print(f"Sample {i} | Pred: {predicted_answer} | GT: {gt_preference} | Correct: {is_correct}")

            self.results.append({
                "index": other.get("index", ""),
                "prompt": other.get('prompt', ""),
                "ground_truth": gt_preference,
                "predicted_answer": predicted_answer,
                "is_correct": is_correct,
                "model1": other.get("model1", ""),
                "model2": other.get("model2", ""),
                "generated_text": gen_text,
            })


class VideoGenRewardBenchEvaluator(BaseEvaluator):
    """
    Evaluator for VideoGen-RewardBench dataset.
    """
    def evaluate_batch(self, gen_texts: List[str], extras: List[Dict]) -> None:
        """
        Evaluate a batch of VideoGen-RewardBench results.

        :param gen_texts: List of generated responses
        :type gen_texts: List[str]
        :param extras: List of metadata including preference and model names
        :type extras: List[Dict]
        """
        for i, (gen_text, other) in enumerate(zip(gen_texts, extras)):
            predicted_answer = extract_response(gen_text, media_type="Video")
            gt_preference = other['preference']  # 'A', 'B', or 'C' (Tie)
            
            better_1 = "Video 1 is better"
            better_2 = "Video 2 is better"
            equal = "Both videos are equally good"

            is_correct = False
            if gt_preference == "A" and predicted_answer == better_1:
                is_correct = True
            elif gt_preference == "B" and predicted_answer == better_2:
                is_correct = True
            elif gt_preference == "C" and predicted_answer == equal:
                is_correct = True
            
            if is_correct:
                self.correct += 1
            elif predicted_answer is None:
                self.parse_failures += 1
                print(f"Could not extract answer from generated text: {gen_text}")
            self.total += 1

            print(f"Sample {i} | Pred: {predicted_answer} | GT: {gt_preference} | Correct: {is_correct}")

            self.results.append({
                "prompt": other.get('prompt', ""),
                "ground_truth": gt_preference,
                "predicted_answer": predicted_answer,
                "is_correct": is_correct,
                "A_model": other.get("A_model", ""),
                "B_model": other.get("B_model", ""),
                "generated_text": gen_text,
            })


@torch.no_grad()
def test_grm_vllm(
    model_path: str,
    data_path: List[str],
    evaluator: BaseEvaluator,
    llm: LLM,
    sampling_params: SamplingParams,
    config: dict = None,
    batch_size: int = 32,
    save_dir: str = "./test_results",
):
    """
    Run evaluation using vLLM on specified datasets.

    This function initializes a :class:`RFTDatasetVL`, performs batch inference using vLLM,
    delegates metric calculation to the evaluator, and saves the detailed results.

    :param model_path: Path to the model directory (to load processor and tokenizer)
    :type model_path: str
    :param data_path: List of dataset paths in "source:path" format
    :type data_path: List[str]
    :param evaluator: Evaluator instance to handle results
    :type evaluator: BaseEvaluator
    :param llm: The initialized vLLM engine instance
    :type llm: vllm.LLM
    :param sampling_params: Sampling parameters for text generation
    :type sampling_params: vllm.SamplingParams
    :param config: Task configuration including instruction and pixels, defaults to None
    :type config: dict, optional
    :param batch_size: Number of samples per batch, defaults to 32
    :type batch_size: int, optional
    :param save_dir: Directory to save results and logs, defaults to "./test_results"
    :type save_dir: str, optional

    **Example:**

    .. code-block:: python

        config = {"name": "HPDv3", "task_instruction": "Evaluate..."}
        test_grm_vllm(model_path, ["hpdv3:test.json"], evaluator, llm, params, config=config)
    """
    logger.info(f"Loading model from: {model_path}")
    
    logger.info(f"Model loaded successfully from {model_path}.")

    # Load Processor and Tokenizer for Dataset
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Load Dataset
    dataset = RFTDatasetVL(
        data_path,
        processor=processor,
        tokenizer=tokenizer,
        strategy=None,
        max_length=8192,
        config=config
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=dataset.collate_fn
    )

    logger.info(f"Starting inference with evaluator: {evaluator.__class__.__name__}")
    
    for batch_idx, batch in enumerate(tqdm(data_loader)):
        input_texts, image_inputs_list, video_inputs_list, extras, _ = batch
        
        inputs = []
        for i in range(len(input_texts)):
            prompt = input_texts[i]
            image_inputs = image_inputs_list[i]
            video_inputs = video_inputs_list[i]
            
            mm_data = {}
            if image_inputs is not None:
                mm_data["image"] = image_inputs
            if video_inputs is not None:
                mm_data["video"] = video_inputs
            
            inputs.append({
                "prompt": prompt,
                "multi_modal_data": mm_data
            })

        # Generate
        outputs = llm.generate(inputs, sampling_params=sampling_params)

        # Decode
        gen_texts = [output.outputs[0].text for output in outputs]

        # Delegate evaluation
        evaluator.evaluate_batch(gen_texts, extras)

    # Summary and Save
    accuracy = evaluator.get_accuracy()
    failure_rate = evaluator.get_parse_failure_rate()
    print(f"Evaluation completed. Accuracy: {accuracy*100:.2f}% ({evaluator.correct}/{evaluator.total})")
    print(f"Parse Failure Rate: {failure_rate*100:.2f}% ({evaluator.parse_failures}/{evaluator.total})")

    if save_dir:
        new_save_dir = os.path.join(save_dir, os.path.basename(model_path), config["name"])
        os.makedirs(new_save_dir, exist_ok=True)
        
        with open(os.path.join(new_save_dir, "evaluation_info.txt"), "w") as f:
            f.write(f"Dataset paths: {data_path}\n")
            f.write(f"Model path: {model_path}\n")
            f.write(f"Evaluator: {evaluator.__class__.__name__}\n")
            if config and "task_instruction" in config:
                f.write(f"Task Instruction:\n{config['task_instruction']}\n")
            f.write(f"Max new tokens: {sampling_params.max_tokens}\n")
            f.write(f"Accuracy: {accuracy*100:.2f}% ({evaluator.correct}/{evaluator.total})\n")
            f.write(f"Parse Failure Rate: {failure_rate*100:.2f}% ({evaluator.parse_failures}/{evaluator.total})\n")

        results = evaluator.get_results()
        with open(os.path.join(new_save_dir, "all_results.jsonl"), "w") as f:
            for item in results:
                f.write(json.dumps(item) + "\n")
        logger.info(f"Results saved to {new_save_dir}")


if __name__ == "__main__":
    model_path = "/path/to/your/grm_vl_model"

    benchmark_configs = [
        {
            "name": "HPDv3-Test",
            "evaluator": HPDv3GRMEvaluator(),
            "data_path": ["hpdv3:/path/to/HPDv3/test.json"],
            "task_instruction": TASK_INSTRUCTION_COT_T2I,
            "max_pixels": 768*480,
        },
        {
            "name": "OmniRewardBench-T2I", 
            "evaluator": OmniRewardBenchEvaluator(media_type="Image"), 
            "data_path": ["omnirewardbench-t2i:/path/to/OmniRewardBench/text_to_image/test.parquet"], 
            "task_instruction": TASK_INSTRUCTION_COT_T2I,
            "max_pixels": 768*480,
        },
        {
            "name": "OmniRewardBench-T2V", 
            "evaluator": OmniRewardBenchEvaluator(media_type="Video"), 
            "data_path": ["omnirewardbench-t2v:/path/to/OmniRewardBench/text_to_video/test.parquet"], 
            "task_instruction": TASK_INSTRUCTION_COT_T2V,
            "video_fps": 2.0,
            "max_pixels": 768*480,
        },
        {
            "name": "GenAI-Bench",
            "evaluator": GenAIBenchEvaluator(),
            "data_path": ["genai_bench:/path/to/GenAI-Bench/data"], 
            "task_instruction": TASK_INSTRUCTION_COT_T2I,
            "max_pixels": 768*480,
        },
        {
            "name": "GenAI-Bench-Video",
            "evaluator": GenAIBenchEvaluator(media_type="Video"),
            "data_path": ["genai_bench_video:/path/to/GenAI-Bench-Video"],
            "task_instruction": TASK_INSTRUCTION_COT_T2V,
            "video_fps": 2.0,
            "max_pixels": 768*480,
        },
        {
            "name": "VideoGen-RewardBench",
            "evaluator": VideoGenRewardBenchEvaluator(),
            "data_path": ["videogen-rewardbench:/path/to/VideoGen-RewardBench/test.csv"],
            "task_instruction": TASK_INSTRUCTION_COT_T2V,
            "video_fps": 2.0,
            "max_pixels": 768*480,
        },
        {
            "name": "ImageGen-CoT-Reward-5K",
            "evaluator": ImageGenCoTEvaluator(),
            "data_path": ["imagegen-cot-reward-5k:/path/to/ImageGen-CoT-Reward-5K/train_data.json"],
            "max_pixels": 768*480,
        },
    ]

    # Initialize vLLM
    tensor_parallel_size = 1    # Adjust based on your gpu hardware
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        gpu_memory_utilization=0.8,
        limit_mm_per_prompt={
            "image": 2, 
            "video": 2
        },
    )

    # Define sampling parameters
    sampling_params = SamplingParams(
        temperature=0.0,  # For deterministic output
        max_tokens=512,
    )

    for config in benchmark_configs:
        print(f">>> Running {config['name']} Evaluation")

        test_grm_vllm(
            model_path, 
            config["data_path"], 
            evaluator=config["evaluator"],
            llm=llm,
            sampling_params=sampling_params,
            config=config, 
            batch_size=128,
        )
