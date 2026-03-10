from typing import Callable, Dict, List, Tuple, Union, Sequence, Optional
import re
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoModel, PreTrainedModel, AutoConfig, Qwen2_5_VLForConditionalGeneration

from lightrft.utils import get_current_device


class AttentionPooling(nn.Module):
    """
    Overview:
        Attention pooling layer on the sequence dimension of LLM/VLM hidden states.
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 4,
        qkv_bias: bool = False,
        position_bias: bool = False,
        position_bias_scale: float = 3.0,
    ):
        super(AttentionPooling, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.position_bias = position_bias
        self.position_bias_scale = position_bias_scale

        self.k = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.v = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        # Using 0.02 for better initialization
        self.query = nn.Parameter(torch.randn(hidden_size) * 0.02)

    def forward(self, hidden_states):
        B, S, C = hidden_states.shape

        # Multi-head projection for key and value
        k = self.k(hidden_states).reshape(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # B, H, S, D
        v = self.v(hidden_states).reshape(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # B, H, S, D

        # Expand query for batch dimension
        q = self.query.unsqueeze(0).expand(B, -1, -1)  # B, H, C
        q = q.unsqueeze(2)  # B, H, 1, C
        q = q.reshape(B, self.num_heads, 1, self.head_dim)  # B, H, 1, C

        # Attention weights
        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, H, 1, S

        # Add position bias
        if self.position_bias:
            position_bias = torch.arange(S, device=k.device).float() / S * self.position_bias_scale
            attn = attn + position_bias.view(1, 1, 1, -1)  # Add position bias

        # Attention pooling
        attn = torch.softmax(attn, dim=-1)  # B, H, 1, S
        attn = attn.to(v.dtype)
        out = (attn @ v).squeeze(2)  # B, H, D
        out = out.reshape(B, -1)  # B, C

        return out


class Qwen2VLRewardModelMemeOutcome(PreTrainedModel):
    def __init__(self, pretrained_model):
        super().__init__(pretrained_model.config)
        self.pretrained_model = pretrained_model

        if hasattr(pretrained_model.config, 'hidden_size'):
            hidden_size = pretrained_model.config.hidden_size
        elif hasattr(pretrained_model.config, 'd_model'):
            hidden_size = pretrained_model.config.d_model
        else:
            raise ValueError("Cannot determine hidden size from model config")

        self.attention_pooling = AttentionPooling(
            hidden_size=hidden_size,
            num_heads=4,
            qkv_bias=False,
            position_bias=True,
            position_bias_scale=3.0,
        )

        self.classification_head = nn.Linear(hidden_size, 2)
        nn.init.normal_(self.classification_head.weight, std=0.02)
        nn.init.zeros_(self.classification_head.bias)

        self.attention_pooling.bfloat16()
        self.classification_head.bfloat16()

    @classmethod
    def from_pretrained(cls, pretrained_model: PreTrainedModel):
        """Create a binary classification model from a pretrained model."""
        return cls(pretrained_model)

    @torch.no_grad()
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs,
    ):
        """Forward pass for meme reward evaluation."""
        # Forward through base model
        prompt_and_outputs = kwargs.get('prompt_and_output')
        raw_images = kwargs.get('raw_images')
        outputs = self.pretrained_model(
            input_ids=input_ids.cuda(),
            attention_mask=attention_mask.cuda(),
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=True,
        )

        # Extract hidden states
        if hasattr(outputs, 'last_hidden_state'):
            hidden_states = outputs.last_hidden_state
        elif hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states[-1]
        else:
            raise ValueError("Cannot extract hidden states from model output")

        # Use attention pooling
        pooled_output = self.attention_pooling(hidden_states)

        # Get logits
        logits = self.classification_head(pooled_output)

        # Get logits [batch_size, 2], first dimension is probability of 0, second dimension is probability of 1
        logits = self.classification_head(pooled_output)

        # Calculate probabilities and binarize
        probabilities = F.softmax(logits, dim=-1)  # [batch_size, 2]
        # If the second dimension (probability of 1) is larger, output 1, otherwise output 0
        binary_scores = (probabilities[:, 1] > probabilities[:, 0]).float()

        return {
            'score': binary_scores,  # Binary result of 0/1
            'logits': logits  # Original logits, containing scores for two dimensions
        }


class Qwen2VLRewardModelMemeContent(PreTrainedModel):
    system_prompt = f"""
    You are a professional meme text generation evaluation expert who is good at evaluating the quality of the current reasoning process in combination with images.\n\n
    """
    eval_prompt = """
    You are a professional and strict-scoring expert in evaluating meme text generation, responsible for scoring the quality of the reasoning process (Chain of Thought, CoT) generated by the model.
    This reasoning process is a text generation reasoning conducted based on the first text-free base image and input parameters (i.e., the user's requirements for the meme).

    Details of the evaluation task:
    1. Input parameters:
    {input_params}

    2. Standard reasoning process (reference standard answer):
    {standard_cot}

    3. Actual reasoning process to be evaluated: (The generated text is after "Text on the Meme")
    {actual_cot}

    Please conduct the evaluation by combining the two provided images (the base image and the standard meme image with text), and the scoring must be strict. The evaluation criteria are as follows:

    1. Whether the chain of thought process includes an analysis of the expressions/actions/facial features/relationships/scenes of the entities in the image, and check its correctness: (Total 10 points)
    a. Rough description: For example, there is a woman in this picture; 1 point;
    b. With some details but no description of actions/facial expressions: For example, there is a woman in this picture, wearing a hat, sitting in a car; 4 points;
    c. With details and actions/facial expressions: For example, there is a woman in this picture, wearing a hat, sitting in a car, looking very happy; 7 points;
    d. Not only explaining the details such as the actions and expressions of the characters in the picture, but also immediately associating the character relationships/scenes where the actions occur; 10 points;

    2. Analysis of further scene associations based on the relationships between entities in the image: (Total 10 points)
    a. Only roughly describing possible scenes without specificity: For example, this may happen in daily life; 1 point;
    b. Describing a relatively specific scene: For example, this may be the scene when you went out with friends to drink and found yourself vomiting; 4 points;
    c. Describing multiple relatively specific scenes: For example, this may be the scene when you went out with friends to drink and found yourself vomiting, or the scene when the teacher checks homework but you find you haven't finished it; 10 points;

    Next, evaluate the content after [Specific analysis with user input]:

    1. Whether the chain of thought further specifies the scene based on the previously associated scenes combined with user needs, or re-associates a scene more in line with user needs: (5 points for each satisfied item, total 20 points)
    a. Whether the intention expressed in the sentence is consistent with the user's emotions;
    b. Whether the intention expressed in the sentence is consistent with the user's intentions and the theme;
    c. Whether the topic of the entire sentence is consistent with the keyword topic;
    d. Whether some humorous techniques are used, such as puns/homophones/semantic reversal/subverting expectations/exaggeration/role dislocation/suspenseful beginning/punchline reversal/rhyming structure/internet memes;

    2. Logical fluency and text length between the final generated text and the reasoning process: (Total 10 points)
    a. The reasoning is very forced, and the humor of the answer paired with this base image is much worse than that of the standard meme, and the text length is much longer than that of the standard meme; 1 point;
    b. The reasoning is roughly valid, and the length of the answer divided by boxes is roughly similar to that of the standard meme; 4 points;
    c. The reasoning is very fluent, the length of the answer divided by boxes is roughly similar to that of the standard meme, and it is humorous when paired with the image (for humorous techniques, refer to the rhetoric I just mentioned); 7 points;
    d. The reasoning is very fluent, the length of the answer divided by boxes is roughly similar to that of the standard meme, or even more interesting than the standard meme; 10 points;

    The total score is 50 points. It is necessary to provide the reasons and scores for each scoring criterion.
    Output a line similar to the following at the end:
    "Final score: 41 points"
    """

    def __init__(self, pretrained_model):
        super().__init__(pretrained_model.config)
        self.pretrained_model = pretrained_model
        self.processor = None

    def set_processor(self, processor):
        self.processor = processor

    @classmethod
    def from_pretrained(cls, pretrained_model: PreTrainedModel):
        return cls(pretrained_model)

    def _get_message(self, input_params, standard_cot, actual_cot, base_image, standard_meme_image):
        message = [
            {
                "role": "system",
                "content": [{
                    "type": "text",
                    "text": self.system_prompt
                }]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "The first image is the base map (no text, only a frame):"
                    },
                    {
                        "type": "image",
                        "image": base_image
                    },
                    #{"type": "text", "text": "The second image is a standard meme image (with the correct answer in text):"},
                    #{"type": "image", "image": standard_meme_image}
                    {
                        "type": "text",
                        "text": self.eval_prompt.format(
                            input_params=input_params, standard_cot=standard_cot, actual_cot=actual_cot
                        )
                    }
                ]
            }
        ]
        return message

    @torch.no_grad()
    def forward(
        self,
        input_ids,
        attention_mask,
        pixel_values,
        image_grid_thw,
        return_dict=None,
        output_attentions=None,
        output_hidden_states=None,
        **kwargs
    ):
        raw_images = kwargs.get('raw_images')
        references = kwargs.get('references')
        texts = self.processor.batch_decode(input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        texts = [t.split("user\n")[1] for t in texts]

        messages = []
        images = []
        for img, ref, text in zip(raw_images, references, texts):
            input_params_match = re.search(r'\*\*Input Parameters\*\*:\s*\[(.*?)\]', text, re.DOTALL)
            input_params = input_params_match.group(1) if input_params_match else "not found input params"
            message = self._get_message(
                input_params=input_params, standard_cot=ref, actual_cot=text, base_image=img, standard_meme_image=None
            )
            processed_img, _ = process_vision_info(message)
            messages.append(message)
            images.append(processed_img)

        messages = [
            self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            for message in messages
        ]
        if torch.distributed.get_rank() % 8 == 0:
            print(f"messages: {messages[0]}, {len(messages)}")
        inputs = self.processor(
            text=messages,
            images=images,
            max_length=5000,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        gen_ids = self.pretrained_model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            pixel_values=inputs.pixel_values,
            image_grid_thw=inputs.image_grid_thw,
            temperature=0.3,
            top_p=0.9,
            max_new_tokens=128,
            do_sample=False,
        )
        outputs_trim = [o[len(i):] for i, o in zip(inputs.input_ids, gen_ids)]
        outputs_text = self.processor.batch_decode(
            outputs_trim, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        if torch.distributed.get_rank() % 8 == 0:
            print(f"outputs_text: {outputs_text}, {inputs.input_ids.shape}")
        score = [self._extract_numeric_score(o) for o in outputs_text]
        return score

    def _extract_numeric_score(self, response: str) -> float:
        """Helper function: Extract score from model response and normalize to 0-1 range"""
        # Match numeric scores in 50-point scale (e.g., "Final score: 41 points" or "35/50")
        numeric_patterns = [
            r"Final score[:：]\s*([0-9.]+)\s*points?", r"Score[:：]\s*([0-9.]+)\s*points?", r"([0-9.]+)\s*/\s*50"
        ]

        for pattern in numeric_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                try:
                    score = float(match.group(1))
                    normalized_score = score * 0.02  # Convert 50-point scale to 0-1
                    return max(0.0, min(1.0, normalized_score))  # Clamp boundary values
                except ValueError:
                    continue

        # If no numeric score found, infer score from text description
        positive_indicators = ["excellent", "outstanding", "perfect", "very good", "meets requirements"]
        neutral_indicators = ["average", "acceptable", "basically meets", "partially meets"]
        negative_indicators = ["poor", "does not meet", "bad", "completely mismatched"]

        response_lower = response.lower()
        for idx, indicator in enumerate(positive_indicators):
            if indicator.lower() in response_lower:
                return 0.8 + (idx * 0.05)

        for idx, indicator in enumerate(neutral_indicators):
            if indicator.lower() in response_lower:
                return 0.5 + (idx * 0.05)

        for idx, indicator in enumerate(negative_indicators):
            if indicator.lower() in response_lower:
                return 0.2 + (idx * 0.05)

        # Return 0.0 when unable to infer
        return 0.0


def load_reward_models(
    reward_pretrain: str,
    strategy,
    use_engine: bool = False,
):
    if use_engine:
        raise NotImplementedError("Engine is not supported for reward model")

    model_list = []
    tokenizer_list = []
    processor_list = []
    with strategy.init_model_context() as _:
        cfg = json.loads(reward_pretrain)
        device = get_current_device()

        for key in cfg.keys():
            pretrain_path = cfg[key]
            model_config = AutoConfig.from_pretrained(
                pretrain_path,
                trust_remote_code=True,
            )
            base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                pretrain_path,
                config=model_config,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                trust_remote_code=True,
            )

            processor = AutoProcessor.from_pretrained(
                pretrain_path, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28
            )
            processor.tokenizer.padding_side = "left"

            if key == "outcome":
                model = Qwen2VLRewardModelMemeOutcome.from_pretrained(base)
            elif key == "content":
                model = Qwen2VLRewardModelMemeContent.from_pretrained(base)
                model.set_processor(processor)
            # for some case about meta device
            model.to_empty(device=device)
            model.eval()
            model_list.append(model)
            tokenizer_list.append(processor.tokenizer)
            processor_list.append(processor)
        return model_list, tokenizer_list, processor_list


def get_format_reward(response: str) -> float:
    """
    Evaluate the format compliance of model response, returns a score between 0-1
    
    Args:
        response: The model response content to be evaluated (string)
        
    Returns:
        Format compliance score (0-1), where 1 means fully compliant with format requirements, 0 means completely non-compliant
    """
    # Initialize score and total check items
    score = 0.0
    total_checks = 0

    # 1. Check if all required sections exist
    required_sections = [
        r'\[Comprehensive Description Section\]', r'\[Usage Scenarios Section\]', r'\[Text Analysis Section\]',
        r'\[Specific analysis with user input\]', r'Text on the Meme:'
    ]

    for section in required_sections:
        total_checks += 1
        if re.search(section, response, re.IGNORECASE):
            score += 1

    # 2. Check box format in 'Text on the Meme' section
    total_checks += 1
    meme_text_match = re.search(r'Text on the Meme:\s*(.*?)(?=\n\n|$)', response, re.DOTALL | re.IGNORECASE)
    if meme_text_match and re.search(r'box\d+:\s*[^\n]+', meme_text_match.group(1)):
        score += 1

    # 3. Check Step 1 and Step 2 in 'Specific analysis' section
    total_checks += 2
    specific_analysis_match = re.search(
        r'\[Specific analysis with user input\]\s*(.*?)(?=\n\[|Text on the Meme:|$)', response,
        re.DOTALL | re.IGNORECASE
    )
    if specific_analysis_match:
        analysis_content = specific_analysis_match.group(1)
        if re.search(r'Step 1:', analysis_content, re.IGNORECASE):
            score += 1
        if re.search(r'Step 2:', analysis_content, re.IGNORECASE):
            score += 1

    # Normalize to 0-1 range
    return round(score / total_checks, 4) if total_checks > 0 else 0.0


def reward_fn(
    model_reward_list: List[torch.Tensor],  # len = n_model , each shape=(B,)
    labels: Sequence[str],
    queries: Sequence[str],
    refs: Sequence[str],
    **kwargs,
) -> torch.Tensor:
    # outcome reward
    # model_reward_list: Shapes [2]
    outcome_reward = model_reward_list[0]
    dtype, device = outcome_reward.dtype, outcome_reward.device
    # rule reward
    format_reward = [get_format_reward(q) for q in queries]
    format_reward = torch.tensor(format_reward, dtype=dtype, device=device)
    if torch.distributed.get_rank() % 8 == 0:
        print(f"queries: {queries[0]}")
        print(f"model_reward_list: {model_reward_list}, format_reward: {format_reward}")
    return format_reward + outcome_reward
