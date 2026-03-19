"""Export fine-tuned LoRA checkpoints to merged safetensors.

Merges LoRA adapters back into the base model and writes full-precision
safetensors to models/<name>/.  The downstream GGUF conversion and
quantization are handled by the Makefile targets convert-gguf and
quantize-gguf (which use our local llama.cpp build).

NOTE: We deliberately bypass Unsloth's save_pretrained_merged because it
is known to produce corrupt weights for text models (see
https://github.com/unslothai/unsloth/issues/2374).  Instead we reload
the base model in bf16 via plain transformers, apply the LoRA adapter
via PEFT, merge_and_unload(), and save — the standard, reliable path.
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml


def _resolve_base_model(checkpoint_dir: str) -> str:
    """Read the base model identifier from adapter_config.json.

    Unsloth trains against pre-quantized 4-bit variants (e.g.
    ``unsloth/phi-4-mini-instruct-unsloth-bnb-4bit``).  For the merge
    we need the *original* full-precision model so that
    merge_and_unload() produces real bf16 weights, not uint8 BNB blobs.
    """
    import json

    # Known Unsloth BNB-4bit → original model mappings
    UNSLOTH_TO_ORIGINAL = {
        "unsloth/phi-4-mini-instruct-unsloth-bnb-4bit": "microsoft/Phi-4-mini-instruct",
        "unsloth/Phi-4-mini-instruct":                   "microsoft/Phi-4-mini-instruct",
        "unsloth/Phi-3.5-mini-instruct":                 "microsoft/Phi-3.5-mini-instruct",
    }

    base = None
    adapter_cfg = Path(checkpoint_dir) / "adapter_config.json"
    if adapter_cfg.exists():
        base = json.loads(adapter_cfg.read_text()).get("base_model_name_or_path")

    if not base:
        for cfg_path in ["configs/finetune_ef.yaml", "configs/finetune_auto.yaml"]:
            p = Path(cfg_path)
            if p.exists():
                cfg = yaml.safe_load(p.read_text())
                base = cfg.get("model", {}).get("base")
                if base:
                    break

    if not base:
        base = "microsoft/Phi-4-mini-instruct"

    original = UNSLOTH_TO_ORIGINAL.get(base, base)
    if original != base:
        print(f"  Remapping {base} → {original} (need full-precision for merge)")
    return original


def export_merged(checkpoint_dir: str, output_name: str):
    """Merge LoRA adapters and save as safetensors."""
    checkpoint = Path(checkpoint_dir)
    if not checkpoint.exists():
        print(f"Checkpoint not found: {checkpoint_dir}")
        print("Run fine-tuning first: make finetune-ef")
        sys.exit(1)

    output_dir = Path("models") / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean up stale artifacts from previous exports (e.g. Unsloth's
    # sharded index file) so convert_hf_to_gguf doesn't get confused.
    for stale in output_dir.glob("model-*.safetensors"):
        stale.unlink()
    stale_index = output_dir / "model.safetensors.index.json"
    if stale_index.exists():
        stale_index.unlink()

    base_model_id = _resolve_base_model(checkpoint_dir)
    print(f"Base model: {base_model_id}")
    print(f"Merging LoRA weights: {checkpoint_dir} → {output_dir}")

    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # 1. Load tokenizer from the base model (not from checkpoint,
        #    which may have Unsloth's modified tokenizer files)
        print("Loading tokenizer from base model...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        print(f"  Vocab size: {tokenizer.vocab_size}")

        # 2. Load base model in bf16 — NOT 4-bit, NOT via Unsloth.
        #    This avoids Unsloth's internal Phi→Llama weight remapping
        #    that corrupts the merge.
        print("Loading base model in bf16 (this takes ~30s)...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            device_map="cpu",  # CPU to avoid OOM; merge is fast
        )
        print(f"  Architecture: {base_model.__class__.__name__}")

        # 3. Attach LoRA adapter and merge
        print("Applying LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, str(checkpoint))
        print("Merging weights...")
        model = model.merge_and_unload()

        # 4. Save merged model + tokenizer
        print(f"Saving to {output_dir}...")
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        # 5. Post-save fixups for config.json and tokenizer_config.json
        import json

        # 5a. Strip quantization_config — training used 4-bit BNB via
        #     Unsloth, but the merged weights are bf16.
        cfg_path = output_dir / "config.json"
        cfg = json.loads(cfg_path.read_text())
        changed = False
        if "quantization_config" in cfg:
            del cfg["quantization_config"]
            changed = True
            print("  Stripped stale quantization_config from config.json")
        if changed:
            cfg_path.write_text(json.dumps(cfg, indent=2))

        # 5b. Ensure tokenizer_config.json has model_type — newer
        #     transformers requires it for AutoTokenizer.from_pretrained
        #     on local dirs; save_pretrained doesn't always include it.
        #     Also fix tokenizer_class — Unsloth/PEFT saves it as
        #     "TokenizersBackend" but llama.cpp's convert_hf_to_gguf
        #     needs "GPT2Tokenizer" to pick the correct BPE path.
        tok_cfg_path = output_dir / "tokenizer_config.json"
        tok_cfg = json.loads(tok_cfg_path.read_text())
        tok_changed = False
        if "model_type" not in tok_cfg:
            tok_cfg["model_type"] = cfg.get("model_type", "phi3")
            tok_changed = True
            print(f"  Added model_type={tok_cfg['model_type']!r} to tokenizer_config.json")
        if tok_cfg.get("tokenizer_class") != "GPT2Tokenizer":
            tok_cfg["tokenizer_class"] = "GPT2Tokenizer"
            tok_changed = True
            print("  Fixed tokenizer_class → 'GPT2Tokenizer' in tokenizer_config.json")
        if tok_changed:
            tok_cfg_path.write_text(json.dumps(tok_cfg, indent=2))

        print(f"\n{'=' * 60}")
        print(f"Merge complete!  Safetensors written to {output_dir}")
        print()
        print("Next steps (run each in order):")
        print("  make convert-gguf")
        print("  make quantize-gguf")
        print("  make ollama-load")
        print("  make ollama-test")
        print(f"{'=' * 60}")

    except ImportError as e:
        print(f"ERROR: missing dependency: {e}")
        print("Ensure transformers and peft are installed:")
        print("  pip install transformers peft")
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
