#!/usr/bin/env python3
"""
Test script for math reasoning benchmarks dataset loading.

This script verifies that the benchmark datasets can be loaded correctly
and displays sample data for inspection.

Usage:
    python test_dataset_loading.py --data_root /path/to/eval_data
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from lightrft.datasets import load_math_reasoning_benchmark


class DummyTokenizer:
    """Dummy tokenizer for testing."""
    
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        """Simple chat template that concatenates messages."""
        result = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            result.append(f"[{role}] {content}")
        if add_generation_prompt:
            result.append("[assistant]")
        return "\n".join(result)


class DummyStrategy:
    """Dummy strategy for testing."""
    
    def __init__(self, apply_chat_template=True):
        self.args = type('Args', (), {
            'apply_chat_template': apply_chat_template,
        })()
    
    def is_rank_0(self):
        return True


def test_benchmark(benchmark_name, data_root, apply_chat_template=True):
    """
    Test loading a single benchmark.
    
    :param benchmark_name: Name of the benchmark to test
    :param data_root: Root directory containing benchmark data
    :param apply_chat_template: Whether to apply chat template
    """
    print(f"\n{'='*70}")
    print(f"Testing {benchmark_name}")
    print(f"{'='*70}\n")
    
    try:
        # Create dummy components
        tokenizer = DummyTokenizer()
        strategy = DummyStrategy(apply_chat_template=apply_chat_template)
        
        # Load dataset
        dataset = load_math_reasoning_benchmark(
            benchmark_name=benchmark_name,
            data_root=data_root,
            tokenizer=tokenizer,
            strategy=strategy,
        )
        
        print(f"✓ Successfully loaded {len(dataset)} examples")
        
        # Display first example
        if len(dataset) > 0:
            prompt, label, choices, raw_data = dataset[0]
            
            print(f"\nFirst example:")
            print(f"  Prompt length: {len(prompt)} characters")
            print(f"  Prompt preview: {prompt[:200]}...")
            print(f"  Ground truth: {label}")
            
            if choices:
                print(f"  Choices: {choices}")
                print(f"  (Multiple choice question)")
            else:
                print(f"  (Open-ended question)")
            
            # Show collate function
            batch = dataset.collate_fn([dataset[i] for i in range(min(3, len(dataset)))])
            prompts, labels, choices_list, raw_data_list = batch
            print(f"\nBatch test (size={len(prompts)}):")
            print(f"  ✓ Prompts: {len(prompts)}")
            print(f"  ✓ Labels: {len(labels)}")
            print(f"  ✓ Choices: {len(choices_list)}")
            print(f"  ✓ Raw data: {len(raw_data_list)}")
        
        return True
        
    except FileNotFoundError as e:
        print(f"✗ Data file not found: {e}")
        return False
    except Exception as e:
        print(f"✗ Error loading benchmark: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test math reasoning benchmarks dataset loading"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/mnt/shared-storage-user/sunjiaxuan/oct/Open-Reasoner-Zero/data/eval_data",
        help="Root directory containing benchmark data files",
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        default=["math500", "aime2024", "gpqa_diamond"],
        choices=["math500", "aime2024", "aime2025", "gpqa_diamond"],
        help="Benchmarks to test",
    )
    parser.add_argument(
        "--no_chat_template",
        action="store_true",
        help="Don't apply chat template",
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("Math Reasoning Benchmarks Dataset Loading Test")
    print("="*70)
    print(f"Data root: {args.data_root}")
    print(f"Benchmarks: {', '.join(args.benchmarks)}")
    print(f"Apply chat template: {not args.no_chat_template}")
    
    # Test each benchmark
    results = {}
    for benchmark in args.benchmarks:
        success = test_benchmark(
            benchmark_name=benchmark,
            data_root=args.data_root,
            apply_chat_template=not args.no_chat_template,
        )
        results[benchmark] = success
    
    # Print summary
    print(f"\n{'='*70}")
    print("Test Summary")
    print(f"{'='*70}\n")
    
    for benchmark, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {benchmark}: {status}")
    
    # Exit code
    all_passed = all(results.values())
    if all_passed:
        print(f"\n✓ All tests passed!")
        sys.exit(0)
    else:
        print(f"\n✗ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()

