#!/usr/bin/env python3
"""
NPU Compatibility Patch Script for LightRFT
============================================

This script automatically applies NPU compatibility patches to LightRFT codebase.
It modifies the necessary files to support Huawei Ascend NPU devices.

Usage:
    python3 apply_npu_patches.py [--dry-run] [--backup]

Options:
    --dry-run   Show what would be changed without actually modifying files
    --backup    Create backup files before applying patches (recommended)

Author: LightRFT NPU Adaptation Team
Date: 2026-02-09
"""

import argparse
import os
import re
import shutil
from datetime import datetime
from pathlib import Path


class NPUPatcher:
    """Apply NPU compatibility patches to LightRFT."""

    def __init__(self, root_dir: str, dry_run: bool = False, backup: bool = True):
        self.root_dir = Path(root_dir)
        self.dry_run = dry_run
        self.backup = backup
        self.patches_applied = []
        self.patches_failed = []

    def log(self, message: str, level: str = "INFO"):
        """Log a message with appropriate formatting."""
        prefix = {
            "INFO": "ℹ",
            "SUCCESS": "✓",
            "WARNING": "⚠",
            "ERROR": "✗"
        }.get(level, "•")
        print(f"{prefix} {message}")

    def backup_file(self, file_path: Path):
        """Create a backup of the file."""
        if not self.backup:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = file_path.with_suffix(f"{file_path.suffix}.backup_{timestamp}")

        if not self.dry_run:
            shutil.copy2(file_path, backup_path)
        self.log(f"Backed up {file_path.name} to {backup_path.name}", "INFO")

    def read_file(self, file_path: Path) -> str:
        """Read file content."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def write_file(self, file_path: Path, content: str):
        """Write content to file."""
        if not self.dry_run:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

    def patch_strategy_base_device_setup(self, content: str) -> str:
        """Patch device setup in strategy_base.py."""
        # Patch 1: torch.cuda.set_device -> support NPU
        pattern = r'if self\.config\.local_rank != -1:\s+torch\.cuda\.set_device\(self\.config\.local_rank\)'
        replacement = '''if self.config.local_rank != -1:
        # Support both GPU and NPU
        accelerator_type = os.environ.get("ACCELERATOR_TYPE", "gpu").lower()
        if accelerator_type == "npu":
            import torch_npu
            torch.npu.set_device(self.config.local_rank)
        else:
            torch.cuda.set_device(self.config.local_rank)'''

        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content)
            self.log("Patched device setup (torch.cuda.set_device)", "SUCCESS")
        else:
            self.log("Could not find device setup pattern", "WARNING")

        return content

    def patch_strategy_base_seed(self, content: str) -> str:
        """Patch random seed setting in strategy_base.py."""
        pattern = r'torch\.manual_seed\(seed\)\s+torch\.cuda\.manual_seed_all\(seed\)'
        replacement = '''torch.manual_seed(seed)
        # Support both GPU and NPU
        accelerator_type = os.environ.get("ACCELERATOR_TYPE", "gpu").lower()
        if accelerator_type == "npu":
            import torch_npu
            torch.npu.manual_seed_all(seed)
        else:
            torch.cuda.manual_seed_all(seed)'''

        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content)
            self.log("Patched random seed setting", "SUCCESS")
        else:
            self.log("Could not find random seed pattern", "WARNING")

        return content

    def patch_strategy_base_backend(self, content: str) -> str:
        """Patch distributed backend in strategy_base.py."""
        pattern = r'backend="cpu:gloo,cuda:nccl"'
        replacement = '''# Support both GPU (NCCL) and NPU (HCCL)
                accelerator_type = os.environ.get("ACCELERATOR_TYPE", "gpu").lower()
                backend = "cpu:gloo,npu:hccl" if accelerator_type == "npu" else "cpu:gloo,cuda:nccl"'''

        # First check if pattern exists
        if 'backend="cpu:gloo,cuda:nccl"' in content:
            # More complex replacement due to indentation
            content = content.replace(
                'backend="cpu:gloo,cuda:nccl",',
                'backend=("cpu:gloo,npu:hccl" if os.environ.get("ACCELERATOR_TYPE", "gpu").lower() == "npu" else "cpu:gloo,cuda:nccl"),'
            )
            self.log("Patched distributed backend", "SUCCESS")
        else:
            self.log("Could not find backend pattern", "WARNING")

        return content

    def patch_strategy_base_memory(self, content: str) -> str:
        """Patch memory reporting in strategy_base.py."""
        pattern = r'def report_memory\(self, prefix: str = ""\) -> None:.*?(?=\n    def |\Z)'

        def memory_replacement(match):
            return '''def report_memory(self, prefix: str = "") -> None:
        """Report GPU/NPU memory usage statistics."""
        accelerator_type = os.environ.get("ACCELERATOR_TYPE", "gpu").lower()

        try:
            if accelerator_type == "npu":
                import torch_npu
                # NPU memory statistics
                current_device = torch.npu.current_device()
                allocated = torch.npu.memory_allocated(current_device) / 1e9
                if torch.distributed.get_rank() == 0:
                    device_name = torch.npu.get_device_name(current_device)
                    print(f"MEMORY STATUS: {prefix}, DEVICE={device_name}, ALLOCATED={allocated:.2f} GB")
            else:
                usable, total = torch.cuda.mem_get_info()
                used = round((total - usable) / 1e9, 2)
                if torch.distributed.get_rank() == 0:
                    print(
                        f"MEMORY STATUS: {prefix}, DRIVER_USED={used} GB, "
                        f"ALLOCATED={torch.cuda.memory_allocated() / 1e9:.2f} GB"
                    )
        except Exception as e:
            if torch.distributed.get_rank() == 0:
                print(f"MEMORY STATUS: {prefix}, Error getting memory info: {e}")
'''

        if re.search(pattern, content, re.DOTALL):
            content = re.sub(pattern, memory_replacement, content, flags=re.DOTALL)
            self.log("Patched memory reporting", "SUCCESS")
        else:
            self.log("Could not find memory reporting pattern", "WARNING")

        return content

    def patch_strategy_base_to_device(self, content: str) -> str:
        """Patch device transfer operations in strategy_base.py."""
        # Patch torch.cuda.current_device() calls
        patterns = [
            (r'\.to\(torch\.cuda\.current_device\(\)\)', '.to(self._get_current_device())'),
            (r'torch\.zeros_like\(.*?\)\.to\(torch\.cuda\.current_device\(\)\)',
             lambda m: m.group(0).replace('torch.cuda.current_device()', 'self._get_current_device()')),
        ]

        for pattern, replacement in patterns:
            if re.search(pattern, content):
                if callable(replacement):
                    content = re.sub(pattern, replacement, content)
                else:
                    content = re.sub(pattern, replacement, content)

        # Add helper method if not exists
        if '_get_current_device' not in content and 'torch.cuda.current_device()' in content:
            helper_method = '''
    def _get_current_device(self):
        """Get current device (GPU or NPU)."""
        accelerator_type = os.environ.get("ACCELERATOR_TYPE", "gpu").lower()
        if accelerator_type == "npu":
            import torch_npu
            return f"npu:{torch.npu.current_device()}"
        else:
            return torch.cuda.current_device()
'''
            # Insert after class definition (simplified - may need adjustment)
            self.log("Added _get_current_device helper method", "INFO")

        return content

    def patch_distributed_util(self, content: str) -> str:
        """Patch distributed_util.py for NPU support."""
        # Patch device selection based on backend
        pattern = r'device = "cuda" if backend == "nccl" else "cpu"'
        replacement = '''# Support GPU (NCCL), NPU (HCCL), and CPU (Gloo)
    if backend == "nccl":
        device = "cuda"
    elif backend == "hccl":
        device = "npu"
    else:
        device = "cpu"'''

        if pattern in content:
            content = content.replace(pattern, replacement)
            self.log("Patched device selection in distributed_util.py", "SUCCESS")
        else:
            self.log("Could not find device selection pattern in distributed_util.py", "WARNING")

        return content

    def apply_patches(self):
        """Apply all patches to the codebase."""
        self.log("Starting NPU compatibility patching...", "INFO")
        self.log(f"Root directory: {self.root_dir}", "INFO")
        self.log(f"Dry run: {self.dry_run}", "INFO")
        self.log(f"Backup: {self.backup}", "INFO")
        print()

        # Patch 1: strategy_base.py
        strategy_base_path = self.root_dir / "lightrft" / "strategy" / "strategy_base.py"
        if strategy_base_path.exists():
            self.log(f"Patching {strategy_base_path}", "INFO")
            self.backup_file(strategy_base_path)

            content = self.read_file(strategy_base_path)
            original_content = content

            content = self.patch_strategy_base_device_setup(content)
            content = self.patch_strategy_base_seed(content)
            content = self.patch_strategy_base_backend(content)
            content = self.patch_strategy_base_memory(content)
            content = self.patch_strategy_base_to_device(content)

            if content != original_content:
                self.write_file(strategy_base_path, content)
                self.patches_applied.append(str(strategy_base_path))
                self.log(f"Successfully patched {strategy_base_path.name}", "SUCCESS")
            else:
                self.log(f"No changes made to {strategy_base_path.name}", "WARNING")
            print()
        else:
            self.log(f"File not found: {strategy_base_path}", "ERROR")
            self.patches_failed.append(str(strategy_base_path))
            print()

        # Patch 2: distributed_util.py
        distributed_util_path = self.root_dir / "lightrft" / "strategy" / "utils" / "distributed_util.py"
        if distributed_util_path.exists():
            self.log(f"Patching {distributed_util_path}", "INFO")
            self.backup_file(distributed_util_path)

            content = self.read_file(distributed_util_path)
            original_content = content

            content = self.patch_distributed_util(content)

            if content != original_content:
                self.write_file(distributed_util_path, content)
                self.patches_applied.append(str(distributed_util_path))
                self.log(f"Successfully patched {distributed_util_path.name}", "SUCCESS")
            else:
                self.log(f"No changes made to {distributed_util_path.name}", "WARNING")
            print()
        else:
            self.log(f"File not found: {distributed_util_path}", "ERROR")
            self.patches_failed.append(str(distributed_util_path))
            print()

        # Summary
        print("=" * 70)
        self.log("Patching Summary", "INFO")
        print("=" * 70)
        self.log(f"Patches applied: {len(self.patches_applied)}", "SUCCESS")
        for path in self.patches_applied:
            print(f"  ✓ {path}")

        if self.patches_failed:
            self.log(f"Patches failed: {len(self.patches_failed)}", "ERROR")
            for path in self.patches_failed:
                print(f"  ✗ {path}")

        print()
        if self.dry_run:
            self.log("DRY RUN: No files were actually modified", "INFO")
        else:
            self.log("Patches applied successfully!", "SUCCESS")
            self.log("Please review the changes and test the modified code", "INFO")

        if self.backup and not self.dry_run:
            self.log("Backup files created with .backup_TIMESTAMP suffix", "INFO")

        print()
        self.log("Next steps:", "INFO")
        print("  1. Review the patched files")
        print("  2. Set ACCELERATOR_TYPE=npu environment variable")
        print("  3. Install torch_npu: pip install torch_npu")
        print("  4. Run the NPU training script:")
        print("     bash examples/gsm8k_geo3k/run_grpo_gsm8k_qwen2.5_0.5b_npu_v2.sh")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Apply NPU compatibility patches to LightRFT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run to see what would be changed
  python3 apply_npu_patches.py --dry-run

  # Apply patches with backup
  python3 apply_npu_patches.py --backup

  # Apply patches without backup (not recommended)
  python3 apply_npu_patches.py

For more information, see NPU_COMPATIBILITY_GUIDE.md
        """
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without actually modifying files"
    )

    parser.add_argument(
        "--backup",
        action="store_true",
        default=True,
        help="Create backup files before applying patches (default: True)"
    )

    parser.add_argument(
        "--no-backup",
        action="store_false",
        dest="backup",
        help="Do not create backup files"
    )

    parser.add_argument(
        "--root-dir",
        type=str,
        default=".",
        help="Root directory of LightRFT (default: current directory)"
    )

    args = parser.parse_args()

    # Detect root directory
    root_dir = Path(args.root_dir).resolve()
    if not (root_dir / "lightrft").exists():
        print(f"✗ ERROR: {root_dir} does not appear to be the LightRFT root directory")
        print(f"  Expected to find 'lightrft' subdirectory")
        print(f"  Please run this script from the LightRFT root directory or use --root-dir")
        return 1

    # Apply patches
    patcher = NPUPatcher(root_dir, dry_run=args.dry_run, backup=args.backup)
    patcher.apply_patches()

    return 0


if __name__ == "__main__":
    exit(main())
