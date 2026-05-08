import argparse
import operator
import random
from pathlib import Path

from datasets import Dataset, DatasetDict


OPS = (
    ("+", operator.add),
    ("-", operator.sub),
    ("*", operator.mul),
)


def build_expression(rng: random.Random, depth: int) -> tuple[str, int]:
    if depth <= 0 or rng.random() < 0.35:
        value = rng.randint(0, 20)
        return str(value), value

    for _ in range(64):
        symbol, fn = rng.choice(OPS)
        left_expr, left_value = build_expression(rng, depth - 1)
        right_expr, right_value = build_expression(rng, depth - 1)

        if symbol == "-" and left_value < right_value:
            left_expr, right_expr = right_expr, left_expr
            left_value, right_value = right_value, left_value

        value = fn(left_value, right_value)
        if 0 <= value <= 200:
            return f"({left_expr} {symbol} {right_expr})", value

    value = rng.randint(0, 20)
    return str(value), value


def make_record(expr: str, answer: int, split: str, index: int) -> dict:
    question = (
        "Compute this Python-style arithmetic expression.\n"
        f"Expression: {expr}\n"
        "Return only the final result in the format \\boxed{answer}."
    )
    answer_str = str(answer)
    return {
        "data_source": "tiny_python_expr",
        "prompt": question,
        "ability": "math",
        "reward_model": {
            "ground_truth": answer_str,
        },
        "extra_info": {
            "label": "python_expr_rule",
            "reference": answer_str,
            "answer": answer_str,
            "expression": expr,
            "split": split,
            "index": index,
        },
    }


def build_split(rng: random.Random, size: int, split: str) -> Dataset:
    records = []
    seen = set()

    while len(records) < size:
        expr, answer = build_expression(rng, depth=rng.randint(1, 3))
        if expr in seen:
            continue
        seen.add(expr)
        records.append(make_record(expr, answer, split=split, index=len(records)))

    return Dataset.from_list(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a tiny arithmetic dataset for LightRFT.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="examples/tiny_python_expr/data/generated",
        help="Directory to save the generated DatasetDict.",
    )
    parser.add_argument("--train_size", type=int, default=128)
    parser.add_argument("--test_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    dataset = DatasetDict(
        {
            "train": build_split(rng, args.train_size, "train"),
            "test": build_split(rng, args.test_size, "test"),
        }
    )

    output_dir = Path(args.output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_dir))

    print(f"Saved dataset to: {output_dir}")
    print(dataset)
    print("Sample:", dataset["train"][0])


if __name__ == "__main__":
    main()
