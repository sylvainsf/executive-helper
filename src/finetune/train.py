"""QLoRA fine-tuning via Unsloth."""

import argparse
import json
import sys
from pathlib import Path

import yaml


def train(config_path: str):
    """Run QLoRA fine-tuning with the given config."""
    config = yaml.safe_load(Path(config_path).read_text())

    print("=" * 60)
    print("Executive Helper — QLoRA Fine-Tuning")
    print("=" * 60)
    print(f"  Config:     {config_path}")
    print(f"  Base model: {config['model']['base']}")
    print(f"  Dataset:    {config['data']['path']}")
    print(f"  Output:     {config['output']['dir']}")
    print(f"  LoRA rank:  {config.get('lora', {}).get('r', 16)}")
    print(f"  Epochs:     {config.get('training', {}).get('epochs', 3)}")
    print(f"  BF16:       {config['model'].get('bf16', False)}")
    print()

    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("ERROR: unsloth not installed.")
        print("On WSL with RTX 4090, run: bash scripts/setup_training_wsl.sh")
        print("Or install manually: pip install unsloth")
        sys.exit(1)

    # Check GPU
    import torch
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU: {gpu} ({vram:.1f} GB VRAM)")
    else:
        print("  WARNING: No CUDA GPU detected — training will be very slow")
    print()

    # Load base model with 4-bit quantization
    print("Loading base model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model"]["base"],
        max_seq_length=config["model"].get("max_seq_length", 2048),
        load_in_4bit=True,
        dtype=None,  # auto-detect
    )

    # Add LoRA adapters
    lora_cfg = config.get("lora", {})
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("alpha", 32),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=lora_cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    # Load and filter dataset
    from datasets import load_dataset

    dataset_path = config["data"]["path"]
    if Path(dataset_path).exists():
        dataset = load_dataset("json", data_files=dataset_path, split="train")
    else:
        dataset = load_dataset(dataset_path, split="train")

    # Filter: only train on positive examples (negatives are for evaluation)
    total_before = len(dataset)
    dataset = dataset.filter(lambda ex: ex.get("metadata", {}).get("quality", "positive") == "positive")
    print(f"Dataset: {total_before} total → {len(dataset)} positive examples (filtered {total_before - len(dataset)} negative)")

    # Format for chat template
    def format_example(example):
        messages = example.get("messages", [])
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": text}

    dataset = dataset.map(format_example)

    # Training
    from trl import SFTTrainer
    from transformers import TrainingArguments

    train_cfg = config.get("training", {})
    output_dir = config["output"]["dir"]

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=config["model"].get("max_seq_length", 2048),
        args=TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=train_cfg.get("batch_size", 4),
            gradient_accumulation_steps=train_cfg.get("gradient_accumulation", 2),
            num_train_epochs=train_cfg.get("epochs", 5),
            learning_rate=train_cfg.get("learning_rate", 2e-4),
            warmup_steps=train_cfg.get("warmup_steps", 5),
            logging_steps=train_cfg.get("logging_steps", 5),
            save_steps=train_cfg.get("save_steps", 50),
            save_total_limit=3,
            fp16=not config["model"].get("bf16", False),
            bf16=config["model"].get("bf16", False),
            optim="adamw_8bit",
            seed=42,
            report_to="none",
        ),
    )

    print()
    print("Starting training...")
    print(f"  Steps per epoch: ~{len(dataset) // (train_cfg.get('batch_size', 4) * train_cfg.get('gradient_accumulation', 2))}")
    print(f"  Total steps: ~{len(dataset) // (train_cfg.get('batch_size', 4) * train_cfg.get('gradient_accumulation', 2)) * train_cfg.get('epochs', 5)}")
    print()
    trainer.train()

    # Save final checkpoint
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\n{'=' * 60}")
    print(f"Training complete!")
    print(f"LoRA adapter saved to: {output_dir}")
    print(f"\nNext steps:")
    print(f"  Export to GGUF:  python -m src.finetune.export --checkpoint {output_dir} --name executive-helper-ef")
    print(f"  Or use make:     make export")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune model with QLoRA")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
