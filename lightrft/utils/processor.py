from typing import Any, Callable, Dict, List
import torch
from tqdm import tqdm


def reward_normalization(objs: List[Dict[str, Any]]) -> None:
    """
    Normalize reward values across a list of objects using z-score normalization.

    This function applies standardization (z-score normalization) to reward values,
    transforming them to have zero mean and unit variance. This helps stabilize
    training by ensuring rewards are on a consistent scale.

    :param objs: List of dictionaries, each containing a 'reward' key.
    :type objs: List[Dict[str, Any]]
    :return: None (modifies objs in-place).
    :rtype: None
    """
    rewards = [float(obj["reward"]) for obj in objs]
    # Using float32 for efficiency; sufficient precision for reward normalization
    rewards = torch.tensor(rewards, dtype=torch.float32)
    rewards = (rewards - rewards.mean()) / rewards.std()
    for i, obj in enumerate(objs):
        obj["reward"] = rewards[i].item()


# Default reward prompt template for Conditional SFT
DEFAULT_REWARD_PROMPT = "{input} <rm_score>: {reward} "


def conditional_sft_processor(args: Any, objs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process data for Conditional SFT by prepending reward information to inputs.

    Implements the Conditional SFT approach from the paper:
    "Conditional Language Policy: A General Framework for Steerable Multi-Objective Finetuning"
    (https://arxiv.org/abs/2308.12050)

    This technique conditions the model on reward scores during training, allowing
    it to generate outputs of varying quality based on the specified reward threshold.

    :param args: Arguments object containing 'reward_template' and 'normalize_reward' flags.
    :type args: Any
    :param objs: List of training examples with 'input', 'output', and 'reward' keys.
    :type objs: List[Dict[str, Any]]
    :return: Processed list of training examples.
    :rtype: List[Dict[str, Any]]
    """
    if "reward_template" not in args or args.reward_template is None:
        reward_template = DEFAULT_REWARD_PROMPT
    else:
        reward_template = args.reward_template
    assert "{input}" in reward_template
    assert "{reward}" in reward_template

    if args.normalize_reward:
        reward_normalization(objs)

    for obj in tqdm(objs, desc="Conditional SFT process..."):
        input = obj["input"]
        reward = "{:.2f}".format(float(obj["reward"]))
        input = reward_template.replace("{reward}", reward).replace("{input}", input)
        obj["input"] = input

    return objs


def rejection_sampling_processor(args: Any, objs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process data using Rejection Sampling by selecting highest-reward output per input.

    Implements the Rejection Sampling approach from the paper:
    "Llama 2: Open Foundation and Fine-Tuned Chat Models" (https://arxiv.org/abs/2307.09288)

    This technique filters multiple candidate outputs per input, keeping only the one
    with the highest reward score. This creates a high-quality training dataset by
    rejecting lower-quality samples.

    :param args: Arguments object (unused but kept for API consistency).
    :type args: Any
    :param objs: List of examples with 'input', 'output', and 'reward' keys.
    :type objs: List[Dict[str, Any]]
    :return: List of examples with only the highest-reward output per unique input.
    :rtype: List[Dict[str, Any]]
    """
    out = {}
    for obj in tqdm(objs, desc="Rejection Sampling process...."):
        input = obj["input"]
        output = obj["output"]
        reward = float(obj["reward"])

        if input not in out:
            out[input] = {"output": output, "reward": reward}
        elif reward > out[input]["reward"]:
            out[input]["reward"] = reward
            out[input]["output"] = output

    return [{"input": k, "output": v["output"], "reward": v["reward"]} for k, v in out.items()]


def iterative_dpo_processor(args: Any, objs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process data for Iterative DPO by creating chosen/rejected pairs per input.

    Implements the Iterative DPO approach from:
    "Online Iterative Reinforcement Learning from Human Feedback with General Preference Model"
    (https://github.com/RLHFlow/Online-RLHF)

    For each unique input, this technique tracks the highest-reward (chosen) and
    lowest-reward (rejected) outputs to create preference pairs for Direct Preference
    Optimization (DPO) training. This enables iterative improvement through online RLHF.

    :param args: Arguments object (unused but kept for API consistency).
    :type args: Any
    :param objs: List of examples with 'input', 'output', and 'reward' keys.
    :type objs: List[Dict[str, Any]]
    :return: List of preference pairs with 'prompt', 'chosen', 'rejected', and reward values.
    :rtype: List[Dict[str, Any]]
    """
    out = {}
    for obj in tqdm(objs, desc="Iterative DPO process...."):
        input = obj["input"]
        output = obj["output"]
        reward = float(obj["reward"])

        if input not in out:
            out[input] = {
                "output": output,
                "chosen": output,
                "chosen_reward": reward,
                "rejected": output,
                "rejected_reward": reward,
            }
        elif reward > out[input]["chosen_reward"]:
            out[input]["chosen_reward"] = reward
            out[input]["chosen"] = output
        elif reward < out[input]["rejected_reward"]:
            out[input]["rejected_reward"] = reward
            out[input]["rejected"] = output

    return [{
        "prompt": k,
        "chosen": v["chosen"],
        "chosen_reward": v["chosen_reward"],
        "rejected": v["rejected"],
        "rejected_reward": v["rejected_reward"],
    } for k, v in out.items()]


PROCESSORS = {
    "rs": rejection_sampling_processor,
    "csft": conditional_sft_processor,
    "iter_dpo": iterative_dpo_processor,
}


def get_processor(name: str) -> Callable[[Any, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """
    Retrieve a data processor function by name.

    :param name: Name of the processor ('rs', 'csft', or 'iter_dpo').
    :type name: str
    :return: The corresponding processor function.
    :rtype: Callable[[Any, List[Dict[str, Any]]], List[Dict[str, Any]]]
    :raises ValueError: If the processor name doesn't exist.
    """
    if name in PROCESSORS:
        return PROCESSORS[name]
    else:
        raise ValueError(f"Processor {name} does not exist.")
