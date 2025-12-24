#!/usr/bin/env python3
"""
Interactive Chat Script for Qwen2.5-VL Model Testing

This script provides an efficient and user-friendly interface for testing
trained Qwen2.5-VL models with both text and image inputs.

Features:
    - Text-only conversations
    - Image + text conversations (supports multiple images)
    - Optimized inference with Flash Attention 2 and bfloat16
    - Interactive mode with command history
    - Batch testing from JSON file

Usage:
    # Interactive mode (text only)
    python test_chat.py --model_path <path_to_model>

    # Interactive mode with image support
    python test_chat.py --model_path <path_to_model> --image <path_to_image>

    # Batch mode from JSON
    python test_chat.py --model_path <path_to_model> --batch <path_to_json>

    # With custom generation parameters
    python test_chat.py --model_path <path_to_model> --max_tokens 2048 --temperature 0.7

Interactive Commands:
    - /image <path>  : Load an image for the next query
    - /clear        : Clear conversation history
    - /reset        : Reset image
    - /quit or /exit: Exit the program
    - /help         : Show help message
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

import torch
from PIL import Image
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)


class ChatBot:
    """
    Efficient chatbot wrapper for Qwen2.5-VL model.

    Optimized for inference with Flash Attention 2 and bfloat16 precision.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        max_tokens: int = 8192,
        temperature: float = 0.7,
        top_p: float = 0.95,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize the chatbot.

        Args:
            model_path: Path to the trained model checkpoint
            device: Device to run inference on (cuda/cpu)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            system_prompt: Optional system prompt for the model
        """
        print(f"Loading model from {model_path}...")

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28,
        )

        # Load model with optimizations
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=device,
        )
        self.model.eval()

        self.device = device
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.system_prompt = system_prompt or (
            "A conversation between the User and Assistant. "
            "The User asks a question, and the Assistant provides a solution. "
            "The Assistant first thinks through the reasoning process internally "
            "with self-reflection and consistency check and then gives the final "
            "analysis and answer. The reasoning process should be enclosed within "
            "<think></think>, followed directly by the final thought and answer, "
            "like this: <think> reasoning process here </think> final thought and answer here."
        )

        # Conversation history
        self.messages = []

        print(f"✓ Model loaded successfully on {device}")
        print(f"✓ Parameters: max_tokens={max_tokens}, temperature={temperature}, top_p={top_p}")

    def clear_history(self):
        """Clear conversation history."""
        self.messages = []
        print("✓ Conversation history cleared")

    def chat(
        self,
        query: str,
        images: Optional[List[str]] = None,
        add_to_history: bool = True,
    ) -> str:
        """
        Generate a response for the given query.

        Args:
            query: User query text
            images: Optional list of image paths
            add_to_history: Whether to add this exchange to history

        Returns:
            Generated response text
        """
        # Prepare image inputs
        image_inputs = []
        if images:
            for img_path in images:
                if not os.path.exists(img_path):
                    print(f"Warning: Image not found: {img_path}")
                    continue
                try:
                    img = Image.open(img_path).convert("RGB")
                    image_inputs.append(img)
                except Exception as e:
                    print(f"Warning: Failed to load image {img_path}: {e}")

        # Build messages
        current_messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        current_messages.extend(self.messages)

        # Add current query
        if image_inputs:
            # For vision queries, add images to the message
            content = []
            for img in image_inputs:
                content.append({"type": "image", "image": img})
            content.append({"type": "text", "text": query})
            current_messages.append({"role": "user", "content": content})
        else:
            current_messages.append({"role": "user", "content": query})

        # Apply chat template and generate
        text = self.processor.apply_chat_template(
            current_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Prepare inputs
        inputs = self.processor(
            text=[text],
            images=image_inputs if image_inputs else None,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True if self.temperature > 0 else False,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )

        # Decode response
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        response = self.processor.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        # Add to history if requested
        if add_to_history:
            self.messages.append({"role": "user", "content": query})
            self.messages.append({"role": "assistant", "content": response})

        return response


def interactive_mode(chatbot: ChatBot):
    """
    Run interactive chat mode with command support.

    Args:
        chatbot: Initialized ChatBot instance
    """
    print("\n" + "="*70)
    print("Interactive Chat Mode")
    print("="*70)
    print("Commands:")
    print("  /image <path>  - Load an image for the next query")
    print("  /clear        - Clear conversation history")
    print("  /reset        - Reset loaded image")
    print("  /quit, /exit  - Exit the program")
    print("  /help         - Show this help message")
    print("="*70 + "\n")

    current_images = []

    while True:
        try:
            user_input = input("\n[You] ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.startswith("/"):
                cmd_parts = user_input.split(maxsplit=1)
                cmd = cmd_parts[0].lower()

                if cmd in ["/quit", "/exit"]:
                    print("Goodbye!")
                    break

                elif cmd == "/clear":
                    chatbot.clear_history()
                    current_images = []
                    continue

                elif cmd == "/reset":
                    current_images = []
                    print("✓ Image reset")
                    continue

                elif cmd == "/image":
                    if len(cmd_parts) < 2:
                        print("Usage: /image <path_to_image>")
                        continue
                    img_path = cmd_parts[1].strip()
                    if os.path.exists(img_path):
                        current_images.append(img_path)
                        print(f"✓ Image loaded: {img_path}")
                    else:
                        print(f"✗ Image not found: {img_path}")
                    continue

                elif cmd == "/help":
                    print("\nCommands:")
                    print("  /image <path>  - Load an image for the next query")
                    print("  /clear        - Clear conversation history")
                    print("  /reset        - Reset loaded image")
                    print("  /quit, /exit  - Exit the program")
                    print("  /help         - Show this help message")
                    continue

                else:
                    print(f"Unknown command: {cmd}. Type /help for available commands.")
                    continue

            # Generate response
            print("\n[Assistant] ", end="", flush=True)
            response = chatbot.chat(user_input, images=current_images if current_images else None)
            print(response)

            # Reset images after use (single-turn image mode)
            if current_images:
                current_images = []

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()


def batch_mode(chatbot: ChatBot, batch_file: str, output_file: Optional[str] = None):
    """
    Run batch testing from JSON file.

    JSON format:
    [
        {
            "query": "What is this?",
            "images": ["path/to/image.jpg"],  // optional
            "expected": "..."  // optional, for comparison
        },
        ...
    ]

    Args:
        chatbot: Initialized ChatBot instance
        batch_file: Path to batch JSON file
        output_file: Optional path to save results
    """
    print(f"\nRunning batch mode from {batch_file}...")

    with open(batch_file, 'r', encoding='utf-8') as f:
        batch_data = json.load(f)

    results = []

    for i, item in enumerate(batch_data, 1):
        query = item.get("query", "")
        images = item.get("images", [])
        expected = item.get("expected")

        print(f"\n{'='*70}")
        print(f"Test {i}/{len(batch_data)}")
        print(f"{'='*70}")
        print(f"Query: {query}")
        if images:
            print(f"Images: {', '.join(images)}")

        response = chatbot.chat(query, images=images, add_to_history=False)
        print(f"\nResponse:\n{response}")

        result = {
            "query": query,
            "images": images,
            "response": response,
        }

        if expected:
            result["expected"] = expected
            print(f"\nExpected:\n{expected}")

        results.append(result)

    # Save results if output file specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Results saved to {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Interactive chat script for Qwen2.5-VL model testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive text chat
  python test_chat.py --model_path ./ckpt_20251212_pyoy_step160_hf

  # Interactive with initial image
  python test_chat.py --model_path ./ckpt_20251212_pyoy_step160_hf --image test.jpg

  # Batch testing
  python test_chat.py --model_path ./ckpt_20251212_pyoy_step160_hf --batch tests.json

  # Custom generation parameters
  python test_chat.py --model_path ./ckpt_20251212_pyoy_step160_hf --max_tokens 4096 --temperature 0.5
        """
    )

    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (cuda/cpu)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=8192,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (0 for greedy)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p sampling parameter",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None,
        help="Custom system prompt",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Initial image path for interactive mode",
    )
    parser.add_argument(
        "--batch",
        type=str,
        default=None,
        help="JSON file for batch testing",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for batch results",
    )

    args = parser.parse_args()

    # Initialize chatbot
    chatbot = ChatBot(
        model_path=args.model_path,
        device=args.device,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        system_prompt=args.system_prompt,
    )

    # Run appropriate mode
    if args.batch:
        batch_mode(chatbot, args.batch, args.output)
    else:
        # Interactive mode
        if args.image:
            print(f"Initial image loaded: {args.image}")
        interactive_mode(chatbot)


if __name__ == "__main__":
    main()
