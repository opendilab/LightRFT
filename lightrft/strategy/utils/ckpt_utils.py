"""
Utility module for finding the latest checkpoint directory in machine learning training workflows.

This module provides functionality to locate the most recent checkpoint directory based on
a naming pattern that includes a step number. It's commonly used in deep learning training
scenarios where checkpoints are saved periodically with incremental step numbers.
"""

import os
import re
from typing import Optional


def find_latest_checkpoint_dir(load_dir: str, prefix: str = "global_step") -> Optional[str]:
    """
    Finds the latest subdirectory within the specified directory whose name
    matches the '<prefix><number>' format.

    This function is particularly useful in machine learning training scenarios where
    checkpoints are saved with incremental step numbers. It searches through all
    subdirectories in the given path and returns the one with the highest step number
    that matches the specified prefix pattern.

    If no matching subdirectory is found, returns the original `load_dir`.

    :param load_dir: The path to the parent directory containing checkpoint subdirectories.
    :type load_dir: str
    :param prefix: The expected prefix string at the beginning of checkpoint directory names. Defaults to "global_step".
    :type prefix: str, optional
    :return: The full path to the latest checkpoint subdirectory.
             Returns `load_dir` if no matching subdirectory is found.
             Returns `None` if `load_dir` is invalid (does not exist or is not a directory).
    :rtype: str or None

    Example::

        # Find latest checkpoint with default prefix "global_step"
        latest_dir = find_latest_checkpoint_dir("/path/to/checkpoints")
        # Returns: "/path/to/checkpoints/global_step1000" (if it's the highest numbered)

        # Find latest checkpoint with custom prefix
        latest_dir = find_latest_checkpoint_dir("/path/to/models", prefix="step_")
        # Returns: "/path/to/models/step_500" (if it's the highest numbered)

        # Handle case where directory doesn't exist
        result = find_latest_checkpoint_dir("/nonexistent/path")
        # Returns: None
    """
    # Check if load_dir exists and is a directory
    if not os.path.isdir(load_dir):
        print(f"Error: Directory '{load_dir}' not found or is not a valid directory.")
        return None

    latest_step = -1  # Initialize with a step number lower than any possible step
    # Default return value is the original path; it will be overwritten if a match is found
    latest_ckpt_path = load_dir

    try:
        # Regex: Matches start (^), escaped prefix, one or more digits (\d+), end ($)
        pattern_str = rf"^{re.escape(prefix)}(\d+)$"
        pattern = re.compile(pattern_str)
    except re.error as e:
        # Invalid regex prefix
        print(f"Error: Invalid prefix '{prefix}' resulted in a regex error: {e}")
        return None

    try:
        # Iterate through all entries in load_dir
        for item_name in os.listdir(load_dir):
            item_path = os.path.join(load_dir, item_name)

            # Check if the current item is a directory
            if os.path.isdir(item_path):
                match = pattern.match(item_name)
                if match:
                    try:
                        # Extract the numeric part
                        step_num = int(match.group(1))

                        # If the current step number is greater than the recorded latest step
                        if step_num > latest_step:
                            latest_step = step_num
                            # Update to the path of the latest checkpoint directory
                            latest_ckpt_path = item_path
                    except ValueError:
                        # If \d+ matched but couldn't be converted to int (should not happen theoretically)
                        print(f"Warning: Could not parse step number from directory name '{item_name}'.")
                        # Skip this directory
                        continue

    except OSError as e:
        # Catch potential OS errors during listdir (e.g., permission issues)
        print(f"Error: An OS error occurred while accessing directory '{load_dir}': {e}")
        return None

    # If a matching directory was found, latest_ckpt_path was updated to the latest one.
    # If none was found, latest_ckpt_path remains the initial load_dir.
    return latest_ckpt_path
