"""Export fine-tuned LoRA checkpoints to merged safetensors.

Merges LoRA adapters back into the base model and writes full-precision
safetensors to models/<name>/.  The downstream GGUF conversion and
quantization are handled by the Makefile targets convert-gguf and
quantize-gguf (which use our local llama.cpp build and avoid Unsloth's
brittle subprocess calls).
"""

import argparse
import sys
from pathlib import Path


def export_merged(checkpoint_dir: str, output_name: str):
    """Merge LoRA adapters and save as safetensors."""
    checkpoint = Path(checkpoint_dir)
    if not checkpoint.exists():
        print(f"Checkpoint not found: {checkpoint_dir}")
        print("Run fine-tuning first: make finetune-ef")
        sys.exit(1)

    output_dir = Path("models") / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Merging LoRA weights: {checkpoint_dir} → {output_dir}")

    try:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(checkpoint),
            load_in_4bit=True,
        )

        # Save merged model as 16-bit safetensors (no GGUF – that's a
        # separate make target so we don't depend on Unsloth's subprocess
        # plumbing which breaks on systems where `python` ≠ `python3`).
        model.save_pretrained_merged(
            str(output_dir),
            tokenizer,
            save_method="merged_16bit",
        )

        print(f"\n{'=' * 60}")
        print(f"Merge complete!  Safetensors written to {output_dir}")
        print()
        print("Next: make convert-gguf  →  make quantize-gguf  →  make ollama-load")
        print("  Or just run 'make ollama-load' (triggers the full chain).")
        print(f"{'=' * 60}")

    except ImportError:
        print("ERROR: unsloth not installed.")
        print("Run on training machine: bash scripts/setup_training_wsl.sh")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapters into base model (safetensors)"
    )
    parser.add_argument("--checkpoint", help="Path to checkpoint dir")
    parser.add_argument("--name", help="Output model name")
    args = parser.parse_args()

    if not args.checkpoint:
        for name, ckpt in [
            ("executive-helper-ef", "checkpoints/ef"),
            ("executive-helper-auto", "checkpoints/auto"),
        ]:
            if Path(ckpt).exists():
                export_merged(ckpt, name)
        if not any(Path(p).exists() for p in ["checkpoints/ef", "checkpoints/auto"]):
            print("No checkpoints found. Run fine-tuning first:")
            print("  make finetune-ef")
            print("  make finetune-auto")
    else:
        name = args.name or Path(args.checkpoint).name
        export_merged(args.checkpoint, name)


if __name__ == "__main__":
    main()
