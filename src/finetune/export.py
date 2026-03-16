"""Export fine-tuned LoRA models to GGUF for Ollama serving."""

import argparse
import sys
from pathlib import Path


def export_to_gguf(checkpoint_dir: str, output_name: str, quantization: str = "q4_k_m"):
    """Export a fine-tuned model to GGUF format."""
    checkpoint = Path(checkpoint_dir)
    if not checkpoint.exists():
        print(f"Checkpoint not found: {checkpoint_dir}")
        print("Run fine-tuning first: make finetune-ef")
        sys.exit(1)

    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)

    print(f"Exporting {checkpoint_dir} → models/{output_name}")
    print(f"Quantization: {quantization}")

    try:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(checkpoint),
            load_in_4bit=True,
        )

        # Save as GGUF
        model.save_pretrained_gguf(
            str(output_dir / output_name),
            tokenizer,
            quantization_method=quantization,
        )

        gguf_path = output_dir / f"{output_name}-unsloth.Q4_K_M.gguf"
        # Find the actual GGUF file (name may vary)
        gguf_files = list(output_dir.glob(f"{output_name}*.gguf"))
        if gguf_files:
            gguf_path = gguf_files[0]

        # Generate Ollama Modelfile
        modelfile_path = output_dir / f"Modelfile.{output_name}"
        modelfile_content = f"""FROM {gguf_path.name}
PARAMETER temperature 0.7
PARAMETER num_predict 512
PARAMETER stop <|end|>
PARAMETER stop <|endoftext|>

SYSTEM You are a compassionate, neuroaffirmative executive function support assistant.
"""
        modelfile_path.write_text(modelfile_content)

        print(f"\n{'=' * 60}")
        print(f"Export complete!")
        print(f"  GGUF file:  {gguf_path}")
        print(f"  Modelfile:  {modelfile_path}")
        print(f"")
        print(f"To deploy on your HA/inference server:")
        print(f"")
        print(f"  1. Copy the GGUF + Modelfile to the server:")
        print(f"     scp {gguf_path} {modelfile_path} user@ha-server:~/models/")
        print(f"")
        print(f"  2. On the server, create the Ollama model:")
        print(f"     cd ~/models")
        print(f"     ollama create {output_name} -f Modelfile.{output_name}")
        print(f"")
        print(f"  3. Test it:")
        print(f'     ollama run {output_name} "I need to clean but I can\'t start"')
        print(f"")
        print(f"  4. Update your Executive Helper config:")
        print(f"     Set EH_EF_MODEL={output_name} in .env")
        print(f"{'=' * 60}")

    except ImportError:
        print("ERROR: unsloth not installed.")
        print("Run on training machine: bash scripts/setup_training_wsl.sh")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Export fine-tuned models to GGUF")
    parser.add_argument("--checkpoint", help="Path to checkpoint dir")
    parser.add_argument("--name", help="Output model name")
    parser.add_argument("--quantization", default="q4_k_m", help="GGUF quantization method")
    args = parser.parse_args()

    # Default: export both models if no specific checkpoint given
    if not args.checkpoint:
        for name, ckpt in [
            ("executive-helper-ef", "checkpoints/ef"),
            ("executive-helper-auto", "checkpoints/auto"),
        ]:
            if Path(ckpt).exists():
                export_to_gguf(ckpt, name, args.quantization)
        if not any(Path(p).exists() for p in ["checkpoints/ef", "checkpoints/auto"]):
            print("No checkpoints found. Run fine-tuning first:")
            print("  make finetune-ef")
            print("  make finetune-auto")
    else:
        name = args.name or Path(args.checkpoint).name
        export_to_gguf(args.checkpoint, name, args.quantization)


if __name__ == "__main__":
    main()
