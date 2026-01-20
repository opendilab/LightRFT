# Math Reasoning Benchmark Evaluation

This directory provides evaluation tools for math reasoning benchmarks, supporting the following datasets:

- **Math500**: 500 challenging math problems from the MATH dataset
- **AIME 2024/2025**: American Invitational Mathematics Examination problems
- **GPQA Diamond**: Graduate-level STEM multiple-choice questions

## Features

✅ Support for multiple benchmarks evaluation  
✅ vLLM acceleration support  
✅ Automatic answer extraction and evaluation  
✅ Detailed result logging  
✅ Batch inference support  

## Quick Start

### 0. Testing Data (Optional but Recommended)

Before running the full evaluation, you can verify that your data files are correctly formatted:

#### Quick Format Check (No Dependencies)

```bash
# Test all benchmarks (format validation only)
python test_benchmarks.py --data_root ./eval_data --format_only

# Test specific benchmarks
python test_benchmarks.py --data_root ./eval_data --format_only --benchmarks math500 aime2024
```

This validates the JSON structure and displays sample data without requiring any dependencies.

#### Full Dataset Loading Test (Requires lightrft)

```bash
# Test dataset loading with lightrft
python test_benchmarks.py --data_root ./eval_data --loading_only

# Test both format and loading (default)
python test_benchmarks.py --data_root ./eval_data
```

This verifies that the datasets can be loaded correctly with the actual data loading pipeline.

### 1. Data Preparation

The evaluation scripts read data from the local `eval_data` directory:

```bash
DATA_ROOT="./eval_data"
```

Data file structure:
```
eval_data/
├── math500.json          # 500 math problems
├── aime2024.json         # AIME 2024 problems
└── gpqa_diamond.json     # GPQA Diamond problems
```

### 2. Basic Evaluation

Evaluate a single benchmark:

```bash
bash run_eval.sh /path/to/your/model math500
```

Evaluate multiple benchmarks:

```bash
bash run_eval.sh /path/to/your/model math500 aime2024 gpqa_diamond
```

### 3. Using vLLM Acceleration

For large-scale evaluation, vLLM is recommended:

```bash
USE_VLLM=1 VLLM_TENSOR_PARALLEL_SIZE=4 bash run_eval.sh /path/to/your/model
```

### 4. Custom Configuration

You can adjust parameters via environment variables:

```bash
MAX_TOKENS=4096 \
TEMPERATURE=0.0 \
BATCH_SIZE=8 \
OUTPUT_DIR=./my_results \
bash run_eval.sh /path/to/your/model math500
```

## Detailed Usage

### Direct Python Script Invocation

```bash
python eval_math_benchmarks.py \
    --model_path /path/to/model \
    --benchmarks math500 aime2024 gpqa_diamond \
    --data_root /path/to/eval_data \
    --max_tokens 8192 \
    --temperature 0.0 \
    --batch_size 1 \
    --output_dir ./results
```

### Parameter Description

#### Model Parameters
- `--model_path`: Model checkpoint path (required)
- `--device`: Inference device, default `cuda`

#### Benchmark Parameters
- `--benchmarks`: List of benchmarks to evaluate, options:
  - `math500`: 500 math problems
  - `aime2024`: AIME 2024 problems
  - `aime2025`: AIME 2025 problems (if separate file exists)
  - `gpqa_diamond`: GPQA Diamond problems
- `--data_root`: Benchmark data root directory

#### Generation Parameters
- `--max_tokens`: Maximum generation tokens, default 8192
- `--temperature`: Sampling temperature, 0 for greedy decoding, default 0.0
- `--top_p`: Top-p sampling parameter, default 0.95
- `--batch_size`: Batch inference size, default 1

#### vLLM Parameters
- `--use_vllm`: Use vLLM for accelerated inference
- `--vllm_tensor_parallel_size`: vLLM tensor parallel size, default 1

#### Output Parameters
- `--output_dir`: Results save directory, default `./eval_results`

## Output Results

After evaluation, the following files will be generated in the output directory:

```
eval_results/
├── math500_20260116_143022.json      # Math500 detailed results
├── aime2024_20260116_143045.json     # AIME 2024 detailed results
├── gpqa_diamond_20260116_143102.json # GPQA Diamond detailed results
└── summary_20260116_143102.json      # Summary results
```

### Result Format

#### Detailed Results (separate file for each benchmark)

```json
{
  "benchmark": "math500",
  "total": 500,
  "correct": 423,
  "accuracy": 0.846,
  "detailed_results": [
    {
      "index": 0,
      "prompt": "Convert the point $(0,3)$ in rectangular...",
      "response": "<think>...</think> The answer is \\left( 3, \\frac{\\pi}{2} \\right)",
      "predicted_answer": "\\left( 3, \\frac{\\pi}{2} \\right)",
      "ground_truth": "\\left( 3, \\frac{\\pi}{2} \\right)",
      "is_correct": true,
      "is_multiple_choice": false
    },
    ...
  ]
}
```

#### Summary Results (summary file)

```json
{
  "model_path": "/path/to/model",
  "timestamp": "20260116_143102",
  "benchmarks": {
    "math500": {
      "total": 500,
      "correct": 423,
      "accuracy": 0.846
    },
    "aime2024": {
      "total": 30,
      "correct": 18,
      "accuracy": 0.6
    },
    "gpqa_diamond": {
      "total": 198,
      "correct": 156,
      "accuracy": 0.788
    }
  },
  "generation_config": {
    "max_tokens": 8192,
    "temperature": 0.0,
    "top_p": 0.95,
    "batch_size": 1
  }
}
```

## Answer Extraction Logic

The evaluation script supports multiple answer formats, attempting extraction in the following order:

1. **Boxed answer**: `\\boxed{answer}`
2. **Tagged answer**: `<answer>answer</answer>`
3. **Multiple choice**: Extract option letters (A/B/C/D)
4. **Numerical answer**: Extract numbers
5. **Fallback**: Last non-empty line of text

### Answer Comparison

- **Multiple choice**: Exact match (case-insensitive)
- **Mathematical expressions**: Symbolic comparison using SymPy
- **Numerical**: Float comparison (error < 1e-6)
- **Text**: Normalized string comparison

## Integration into Training Pipeline

You can use these benchmarks for periodic evaluation in your training scripts:

```python
from lightrft.datasets import load_math_reasoning_benchmark
from lightrft.evaluation import extract_answer, evaluate_predictions

# Load benchmark
dataset = load_math_reasoning_benchmark(
    benchmark_name="math500",
    data_root="/path/to/eval_data",
    tokenizer=tokenizer,
    strategy=strategy,
)

# Evaluate in training loop
for epoch in range(num_epochs):
    # ... training code ...
    
    # Evaluation
    predictions = []
    for prompt, label, choices, _ in dataset:
        response = model.generate(prompt)
        pred = extract_answer(response, is_multiple_choice=(choices is not None))
        predictions.append(pred)
    
    # Calculate accuracy
    results = evaluate_predictions(predictions, labels, is_mc_list)
    print(f"Epoch {epoch}: Accuracy = {results['accuracy']:.2%}")
```

## Performance Optimization Tips

1. **Use vLLM**: For large-scale evaluation (>100 samples), vLLM can significantly accelerate
2. **Batch inference**: Increasing `batch_size` can improve throughput
3. **Lower temperature**: For math problems, using `temperature=0` (greedy decoding) usually works better
4. **GPU memory**: If encountering OOM, reduce `max_tokens` or use smaller batch_size

## FAQ

### Q: Cannot find data files?
A: Ensure `data_root` points to the correct directory, or copy the data files to your project.

### Q: vLLM installation failed?
A: vLLM requires CUDA 11.8+. If installation fails, you can use standard HuggingFace inference without the `--use_vllm` flag.

### Q: How to add a new benchmark?
A: Add a new mapping in the `benchmark_files` dictionary in `lightrft/datasets/math_reasoning_benchmarks.py`. The data format should be consistent with existing benchmarks.

### Q: Answer extraction inaccurate?
A: You can customize the answer extraction logic in `lightrft/evaluation/math_eval_utils.py`.

## Citations

If you use these benchmarks, please cite the original papers:

**Math500 / MATH Dataset**:
```bibtex
@article{hendrycks2021measuring,
  title={Measuring mathematical problem solving with the math dataset},
  author={Hendrycks, Dan and Burns, Collin and Kadavath, Saurav and Arora, Akul and Basart, Steven and Tang, Eric and Song, Dawn and Steinhardt, Jacob},
  journal={arXiv preprint arXiv:2103.03874},
  year={2021}
}
```

**GPQA**:
```bibtex
@article{rein2023gpqa,
  title={GPQA: A Graduate-Level Google-Proof Q\&A Benchmark},
  author={Rein, David and Hou, Betty Li and Stickland, Asa Cooper and Petty, Jackson and Pang, Richard Yuanzhe and Dirani, Julien and Michael, Julian and Bowman, Samuel R},
  journal={arXiv preprint arXiv:2311.12022},
  year={2023}
}
```

## Contributing

Issues and Pull Requests are welcome to improve the evaluation tools!

## License

This project follows the LightRFT license.

