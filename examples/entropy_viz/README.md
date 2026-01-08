# Trajectory Token Entropy Visualization Tool

A web-based tool for visualizing trajectory data during reinforcement learning training, with a focus on high-entropy token visualization. This tool helps analyze which tokens have high entropy values during training, as these tokens typically correspond to critical decision points in the reasoning process.

## Features

- **High-Entropy Token Highlighting**: Automatically identifies and highlights high-entropy tokens (typically corresponding to critical decision points in reasoning)
- **Interactive Visualization**: Hover over high-entropy tokens to view detailed entropy scores
- **Statistics**: Displays statistics for each trajectory, including total token count, high-entropy token count, high-entropy ratio, and average entropy value
- **Metadata Display**: Shows training metadata such as training steps, advantages, returns, rewards, etc.
- **Flexible Display Controls**: Toggle visibility of different sections

## Usage

### 1. Generate Trajectory Data

First, enable trajectory saving during training. Add the following parameters to your training script:

```bash
--save_trajectories \
--num_trajectories_to_save 10 \
--mark_high_entropy_tokens \
--high_entropy_token_ratio 0.2
```

Or configure in your training script:

```python
# In train_colocate.py or similar files
args.save_trajectories = True
args.num_trajectories_to_save = 10
args.mark_high_entropy_tokens = True
args.high_entropy_token_ratio = 0.2
```

### 2. Open the Visualization Tool

1. Open the `render_trajectories.html` file in your browser
2. Click the "Choose File" button and select the saved trajectory JSON file
   - Trajectory files are typically saved at `results/<experiment_name>/trajectories/trajectories_step_<step>.json`

### 3. View Visualization Results

- **High-Entropy Tokens**: Highlighted with red background, hover to view entropy values
- **Normal Tokens**: Displayed normally
- **Statistics**: Token statistics displayed below each trajectory
- **Metadata**: Training-related metadata information displayed

## Data Format

The trajectory JSON file should contain the following fields:

```json
[
  {
    "global_step": 20,
    "experience_index": 0,
    "sample_in_exp": 0,
    "generated_tokens": [
      {
        "text": "token text",
        "high_entropy": true,
        "entropy_score": 0.1234
      }
    ],
    "pure_generated_tokens": [...],
    "advantages": 0.5,
    "return": 1.2,
    "info": {
      "reward": 0.8,
      "response_length": 100
    }
  }
]
```

## Display Options

- **Show Entropy Score**: Display entropy values on high-entropy tokens (on hover)
- **Show Metadata**: Display training steps, advantages, returns, etc.
- **Show Statistics**: Display token statistics
- **Show generated_text**: Display complete generated text (including prompt)
- **Show pure_generated_text**: Display only the generated response portion

## High-Entropy Token Explanation

High-entropy tokens are tokens with high uncertainty during model generation, typically corresponding to critical decision points (forking tokens) in the reasoning process. According to the research paper ["Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning"](https://arxiv.org/abs/2506.01939), training only on these high-entropy tokens can significantly improve training efficiency.

## Related Parameters

- `high_entropy_token_ratio`: Ratio of high-entropy tokens (e.g., 0.2 means top 20% highest entropy tokens)
- `mark_high_entropy_tokens`: Whether to mark high-entropy tokens when saving trajectories
- `num_trajectories_to_save`: Number of trajectories to save per training step

## Notes

1. Trajectory files can be large, so it's recommended to save only a small number of samples for analysis
2. Make sure to enable the `mark_high_entropy_tokens` option during training, otherwise high-entropy tokens cannot be identified
3. This tool is for local analysis only and does not upload any data to servers

## Technical Implementation

- Pure frontend implementation, no server required
- Uses vanilla JavaScript with no external dependencies
- Supports modern browsers (Chrome, Firefox, Safari, Edge)
