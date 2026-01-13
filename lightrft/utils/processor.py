import torch
from tqdm import tqdm


def reward_normalization(objs):
    """
    Normalize reward values across a list of objects using z-score normalization.

    :param objs: List of dictionaries, each containing a 'reward' key.
    :type objs: list
    :return: None (modifies objs in-place).
    :rtype: None
    """
    rewards = [float(obj["reward"]) for obj in objs]
    rewards = torch.tensor(rewards, dtype=torch.float64)
    rewards = (rewards - rewards.mean()) / rewards.std()
    for i, obj in enumerate(objs):
        obj["reward"] = rewards[i].item()


# Conditional SFT
# See https://arxiv.org/abs/2308.12050
DEFAULT_REWARD_PROMPT = "{input} <rm_score>: {reward} "


def conditional_sft_processor(args, objs):
    """
    Process data for Conditional SFT by prepending reward information to inputs.

    Implements the approach from https://arxiv.org/abs/2308.12050.

    :param args: Arguments object containing 'reward_template' and 'normalize_reward' flags.
    :type args: object
    :param objs: List of training examples with 'input', 'output', and 'reward' keys.
    :type objs: list
    :return: Processed list of training examples.
    :rtype: list
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


# Rejection Sampling
# See https://arxiv.org/abs/2307.09288
def rejection_sampling_processor(args, objs):
    """
    Process data using Rejection Sampling by selecting highest-reward output per input.

    Implements the approach from https://arxiv.org/abs/2307.09288.

    :param args: Arguments object (unused but kept for API consistency).
    :type args: object
    :param objs: List of examples with 'input', 'output', and 'reward' keys.
    :type objs: list
    :return: List of examples with only the highest-reward output per unique input.
    :rtype: list
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


# Iterative DPO
# See https://github.com/RLHFlow/Online-RLHF/blob/main/run_loop.sh
def iterative_dpo_processor(args, objs):
    """
    Process data for Iterative DPO by creating chosen/rejected pairs per input.

    For each unique input, tracks the highest-reward (chosen) and lowest-reward
    (rejected) outputs to create preference pairs.

    :param args: Arguments object (unused but kept for API consistency).
    :type args: object
    :param objs: List of examples with 'input', 'output', and 'reward' keys.
    :type objs: list
    :return: List of preference pairs with 'prompt', 'chosen', 'rejected', and reward values.
    :rtype: list
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


def get_processor(name):
    """
    Retrieve a data processor function by name.

    :param name: Name of the processor ('rs', 'csft', or 'iter_dpo').
    :type name: str
    :return: The corresponding processor function.
    :rtype: callable
    :raises ValueError: If the processor name doesn't exist.
    """
    if name in PROCESSORS:
        return PROCESSORS[name]
    else:
        raise ValueError(f"Processor {name} does not exist.")
