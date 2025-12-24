import sys
from pathlib import Path
from safetensors import safe_open
import torch
import torch.distributed.checkpoint as DCP

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

# Adapated from https://github.com/pytorch/torchtitan/issues/305#issuecomment-2129251951


@torch.inference_mode()
def convert_hf_checkpoint(
    checkpoint_dir: Path,
    output_dir: Path,
) -> None:
    # Load the json file containing weight mapping

    safetensor_files = sorted(list(checkpoint_dir.glob("*.safetensors")))

    if not safetensor_files:
        print(f"Warning: No *.safetensors files found in {checkpoint_dir}. Nothing to convert.")
        return

    print(f"Found {len(safetensor_files)} safetensors file(s) in {checkpoint_dir}:")

    merged_result = {}
    for file in sorted(safetensor_files):
        with safe_open(file, framework="pt", device="cpu") as f:
            for k in f.keys():
                merged_result[k] = f.get_tensor(k)

    output_dir.mkdir(parents=True, exist_ok=True)
    storage_writer = DCP.filesystem.FileSystemWriter(output_dir)
    DCP.save(merged_result, storage_writer=storage_writer)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert HuggingFace checkpoint.')
    parser.add_argument('--hf_checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)

    args = parser.parse_args()
    convert_hf_checkpoint(
        checkpoint_dir=args.hf_checkpoint,
        output_dir=args.output,
    )
