#!/usr/bin/env python3
"""
Simple test script to verify data format without loading full dependencies.

This script checks the JSON data files to ensure they have the correct format.

Usage:
    python test_data_format.py --data_root /path/to/eval_data
"""

import argparse
import json
from pathlib import Path


def test_data_file(file_path, benchmark_name):
    """
    Test a single data file.
    
    :param file_path: Path to the JSON data file
    :param benchmark_name: Name of the benchmark
    :return: True if test passes, False otherwise
    """
    print(f"\n{'='*70}")
    print(f"Testing {benchmark_name}")
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


def main():
    parser = argparse.ArgumentParser(
        description="Test math reasoning benchmarks data format"
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
        help="Benchmarks to test",
    )
    
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    
    print("="*70)
    print("Math Reasoning Benchmarks Data Format Test")
    print("="*70)
    print(f"Data root: {data_root}")
    print(f"Benchmarks: {', '.join(args.benchmarks)}")
    
    # Map benchmark names to file names
    benchmark_files = {
        "math500": "math500.json",
        "aime2024": "aime2024.json",
        "aime2025": "aime2024.json",
        "gpqa_diamond": "gpqa_diamond.json",
    }
    
    # Test each benchmark
    results = {}
    for benchmark in args.benchmarks:
        if benchmark not in benchmark_files:
            print(f"\n✗ Unknown benchmark: {benchmark}")
            results[benchmark] = False
            continue
        
        file_path = data_root / benchmark_files[benchmark]
        success = test_data_file(file_path, benchmark)
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
        print(f"\nYou can now run the evaluation script:")
        print(f"  bash run_eval.sh /path/to/model")
        return 0
    else:
        print(f"\n✗ Some tests failed!")
        print(f"\nPlease check the data files in: {data_root}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

