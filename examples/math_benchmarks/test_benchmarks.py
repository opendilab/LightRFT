#!/usr/bin/env python3
"""
Comprehensive test script for math reasoning benchmarks.

This script provides two levels of testing:
1. Format validation: Quick check of JSON data format (no dependencies required)
2. Dataset loading: Full test of dataset loading with lightrft (requires dependencies)

Usage:
    # Quick format validation only
    python test_benchmarks.py --data_root /path/to/eval_data --format_only
    
    # Full dataset loading test only
    python test_benchmarks.py --data_root /path/to/eval_data --loading_only
    
    # Both tests (default)
    python test_benchmarks.py --data_root /path/to/eval_data
"""

import argparse
import json
import sys
from pathlib import Path


# ============================================================================
# Format Validation (No Dependencies Required)
# ============================================================================

def test_data_format(file_path, benchmark_name):
    """
    Test a single data file for correct JSON format.
    
    :param file_path: Path to the JSON data file
    :param benchmark_name: Name of the benchmark
    :return: True if test passes, False otherwise
    """
    print(f"\n{'='*70}")
    print(f"Format Test: {benchmark_name}")
    print(f"{'='*70}\n")
    
    try:
        # Check if file exists
        if not file_path.exists():
            print(f"✗ File not found: {file_path}")
            return False
        
        print(f"✓ File found: {file_path}")
        
        # Load JSON
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print(f"✗ Data should be a list, got {type(data)}")
            return False
        
        print(f"✓ Loaded {len(data)} examples")
        
        # Check first example
        if len(data) == 0:
            print(f"✗ Data is empty")
            return False
        
        example = data[0]
        
        # Check required fields
        if "prompt" not in example:
            print(f"✗ Missing 'prompt' field")
            return False
        
        if "final_answer" not in example:
            print(f"✗ Missing 'final_answer' field")
            return False
        
        print(f"✓ Required fields present: prompt, final_answer")
        
        # Check prompt format
        prompt = example["prompt"]
        if not isinstance(prompt, list):
            print(f"✗ 'prompt' should be a list, got {type(prompt)}")
            return False
        
        if len(prompt) > 0:
            msg = prompt[0]
            if not isinstance(msg, dict):
                print(f"✗ Prompt messages should be dicts, got {type(msg)}")
                return False
            
            if "from" not in msg and "role" not in msg:
                print(f"✗ Prompt message missing 'from' or 'role' field")
                return False
            
            if "value" not in msg and "content" not in msg:
                print(f"✗ Prompt message missing 'value' or 'content' field")
                return False
        
        print(f"✓ Prompt format is correct")
        
        # Check if it's multiple choice
        has_choices = "choices" in example
        if has_choices:
            choices = example["choices"]
            if not isinstance(choices, dict):
                print(f"✗ 'choices' should be a dict, got {type(choices)}")
                return False
            print(f"✓ Multiple choice question with choices: {list(choices.keys())}")
        else:
            print(f"✓ Open-ended question")
        
        # Show sample
        print(f"\nSample (first example):")
        print(f"  Prompt: {prompt[0].get('value', prompt[0].get('content', ''))[:100]}...")
        print(f"  Answer: {example['final_answer']}")
        if has_choices:
            print(f"  Choices: {choices}")
        
        # Statistics
        mc_count = sum(1 for ex in data if "choices" in ex)
        open_count = len(data) - mc_count
        
        print(f"\nStatistics:")
        print(f"  Total: {len(data)}")
        print(f"  Multiple choice: {mc_count}")
        print(f"  Open-ended: {open_count}")
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"✗ Invalid JSON: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# Dataset Loading Test (Requires lightrft)
# ============================================================================

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


def test_dataset_loading(benchmark_name, data_root, apply_chat_template=True):
    """
    Test loading a single benchmark with full dataset loader.
    
    :param benchmark_name: Name of the benchmark to test
    :param data_root: Root directory containing benchmark data
    :param apply_chat_template: Whether to apply chat template
    :return: True if test passes, False otherwise
    """
    print(f"\n{'='*70}")
    print(f"Loading Test: {benchmark_name}")
    print(f"{'='*70}\n")
    
    try:
        # Add parent directory to path for imports
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        
        from lightrft.datasets import load_math_reasoning_benchmark
        
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
        
    except ImportError as e:
        print(f"✗ Cannot import lightrft: {e}")
        print(f"  (lightrft dependencies not available, skipping loading test)")
        return False
    except FileNotFoundError as e:
        print(f"✗ Data file not found: {e}")
        return False
    except Exception as e:
        print(f"✗ Error loading benchmark: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive test for math reasoning benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run both format and loading tests
  python test_benchmarks.py --data_root ./eval_data
  
  # Quick format check only (no dependencies needed)
  python test_benchmarks.py --data_root ./eval_data --format_only
  
  # Full loading test only (requires lightrft)
  python test_benchmarks.py --data_root ./eval_data --loading_only
  
  # Test specific benchmarks
  python test_benchmarks.py --data_root ./eval_data --benchmarks math500 aime2024
        """
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./eval_data",
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
        "--format_only",
        action="store_true",
        help="Only run format validation (no lightrft dependencies needed)",
    )
    parser.add_argument(
        "--loading_only",
        action="store_true",
        help="Only run dataset loading test (requires lightrft)",
    )
    parser.add_argument(
        "--no_chat_template",
        action="store_true",
        help="Don't apply chat template (only for loading test)",
    )
    
    args = parser.parse_args()
    
    # Determine which tests to run
    run_format = not args.loading_only
    run_loading = not args.format_only
    
    data_root = Path(args.data_root)
    
    print("="*70)
    print("Math Reasoning Benchmarks Test Suite")
    print("="*70)
    print(f"Data root: {data_root}")
    print(f"Benchmarks: {', '.join(args.benchmarks)}")
    print(f"Tests to run: ", end="")
    tests = []
    if run_format:
        tests.append("Format Validation")
    if run_loading:
        tests.append("Dataset Loading")
    print(", ".join(tests))
    if run_loading:
        print(f"Apply chat template: {not args.no_chat_template}")
    
    # Map benchmark names to file names
    benchmark_files = {
        "math500": "math500.json",
        "aime2024": "aime2024.json",
        "aime2025": "aime2024.json",
        "gpqa_diamond": "gpqa_diamond.json",
    }
    
    # Test each benchmark
    format_results = {}
    loading_results = {}
    
    for benchmark in args.benchmarks:
        if benchmark not in benchmark_files:
            print(f"\n✗ Unknown benchmark: {benchmark}")
            if run_format:
                format_results[benchmark] = False
            if run_loading:
                loading_results[benchmark] = False
            continue
        
        # Format test
        if run_format:
            file_path = data_root / benchmark_files[benchmark]
            success = test_data_format(file_path, benchmark)
            format_results[benchmark] = success
        
        # Loading test
        if run_loading:
            success = test_dataset_loading(
                benchmark_name=benchmark,
                data_root=str(data_root),
                apply_chat_template=not args.no_chat_template,
            )
            loading_results[benchmark] = success
    
    # Print summary
    print(f"\n{'='*70}")
    print("Test Summary")
    print(f"{'='*70}\n")
    
    if run_format:
        print("Format Validation:")
        for benchmark, success in format_results.items():
            status = "✓ PASS" if success else "✗ FAIL"
            print(f"  {benchmark}: {status}")
        print()
    
    if run_loading:
        print("Dataset Loading:")
        for benchmark, success in loading_results.items():
            status = "✓ PASS" if success else "✗ FAIL"
            print(f"  {benchmark}: {status}")
        print()
    
    # Determine overall result
    all_results = list(format_results.values()) + list(loading_results.values())
    all_passed = all(all_results) if all_results else False
    
    if all_passed:
        print(f"✓ All tests passed!")
        if run_format and not run_loading:
            print(f"\nYou can now run the full loading test:")
            print(f"  python {Path(__file__).name} --data_root {data_root} --loading_only")
        elif run_format and run_loading:
            print(f"\nYou can now run the evaluation script:")
            print(f"  bash run_eval.sh /path/to/model")
        sys.exit(0)
    else:
        print(f"✗ Some tests failed!")
        if run_format and any(not v for v in format_results.values()):
            print(f"\nPlease check the data files in: {data_root}")
        sys.exit(1)


if __name__ == "__main__":
    main()

