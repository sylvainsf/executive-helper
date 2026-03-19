"""Validate the exported model directory before GGUF conversion.

Checks every failure mode we've encountered in the export → convert
pipeline.  Run this after `make export` and before `make convert-gguf`.

Usage:
    python -m scripts.validate_export [model_dir]
    make validate-export

Exit code 0 = all checks pass, 1 = failures found.
"""

import json
import sys
from pathlib import Path

EXPECTED_ARCHITECTURE = "Phi3ForCausalLM"
EXPECTED_MODEL_TYPE = "phi3"
EXPECTED_VOCAB_SIZE = 200064


def check(name: str, ok: bool, detail: str = "") -> bool:
    status = "✓" if ok else "✗"
    msg = f"  {status} {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return ok


def validate(model_dir: Path) -> bool:
    print(f"Validating export: {model_dir}\n")
    passed = True

    # ── 1. Directory exists ──────────────────────────────────────────────
    if not model_dir.is_dir():
        print(f"  ✗ Directory does not exist: {model_dir}")
        return False

    # ── 2. Safetensors files ─────────────────────────────────────────────
    single = model_dir / "model.safetensors"
    index = model_dir / "model.safetensors.index.json"
    shards = sorted(model_dir.glob("model-*.safetensors"))

    if single.exists() and index.exists():
        # Stale index file from a previous sharded export
        passed &= check(
            "No stale shard index",
            False,
            f"model.safetensors.index.json exists alongside model.safetensors — "
            f"delete it: rm {index}",
        )
    elif index.exists() and not single.exists():
        # Sharded export — verify all parts exist
        shard_meta = json.loads(index.read_text())
        expected_files = set(shard_meta.get("weight_map", {}).values())
        missing = [f for f in expected_files if not (model_dir / f).exists()]
        passed &= check(
            "All shard files present",
            len(missing) == 0,
            f"missing: {missing}" if missing else f"{len(expected_files)} shards OK",
        )
    elif single.exists():
        size_gb = single.stat().st_size / (1024**3)
        passed &= check(
            "model.safetensors exists",
            True,
            f"{size_gb:.1f} GB",
        )
    else:
        passed &= check("Safetensors files exist", False, "no model files found")

    # ── 3. config.json ───────────────────────────────────────────────────
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        passed &= check("config.json exists", False)
    else:
        cfg = json.loads(cfg_path.read_text())

        arch = cfg.get("architectures", [None])[0]
        passed &= check(
            "Architecture is Phi3ForCausalLM",
            arch == EXPECTED_ARCHITECTURE,
            f"got {arch!r}",
        )

        model_type = cfg.get("model_type")
        passed &= check(
            "model_type is phi3",
            model_type == EXPECTED_MODEL_TYPE,
            f"got {model_type!r}",
        )

        vocab = cfg.get("vocab_size")
        passed &= check(
            f"vocab_size is {EXPECTED_VOCAB_SIZE}",
            vocab == EXPECTED_VOCAB_SIZE,
            f"got {vocab}",
        )

        has_quant = "quantization_config" in cfg
        passed &= check(
            "No quantization_config in config.json",
            not has_quant,
            "found bitsandbytes config — weights may be uint8 not bf16"
            if has_quant
            else "",
        )

    # ── 4. tokenizer_config.json ─────────────────────────────────────────
    tok_cfg_path = model_dir / "tokenizer_config.json"
    if not tok_cfg_path.exists():
        passed &= check("tokenizer_config.json exists", False)
    else:
        tok_cfg = json.loads(tok_cfg_path.read_text())

        has_model_type = "model_type" in tok_cfg
        passed &= check(
            "tokenizer_config.json has model_type",
            has_model_type,
            f"got {tok_cfg.get('model_type')!r}" if has_model_type else "MISSING — will crash convert_hf_to_gguf",
        )

        eos = tok_cfg.get("eos_token")
        passed &= check(
            "eos_token is set",
            eos is not None,
            f"got {eos!r}",
        )

    # ── 5. tokenizer.json vocab size ─────────────────────────────────────
    tok_path = model_dir / "tokenizer.json"
    if tok_path.exists():
        tok_data = json.loads(tok_path.read_text())
        vocab_entries = len(tok_data.get("model", {}).get("vocab", {}))
        # Phi-4-mini has 200064 in config but tokenizer.json has 200019
        # base entries + added_tokens. Just check it's in the right ballpark.
        passed &= check(
            "tokenizer.json has reasonable vocab",
            vocab_entries > 190000,
            f"{vocab_entries} entries",
        )

    # ── 6. Tensor dtype spot-check (requires safetensors) ────────────────
    try:
        from safetensors import safe_open

        st_file = single if single.exists() else (shards[0] if shards else None)
        if st_file:
            with safe_open(str(st_file), framework="pt") as f:
                keys = list(f.keys())
                bad_dtypes = []
                for key in keys[:20]:  # spot-check first 20
                    tensor = f.get_tensor(key)
                    dtype = str(tensor.dtype)
                    # Norms can be float32, everything else should be bfloat16
                    if "norm" not in key and dtype not in ("torch.bfloat16", "torch.float32"):
                        bad_dtypes.append(f"{key}: {dtype}")
                passed &= check(
                    "Tensor dtypes are bf16/f32 (not uint8/int8)",
                    len(bad_dtypes) == 0,
                    f"bad: {bad_dtypes}" if bad_dtypes else f"checked {min(20, len(keys))} tensors",
                )
    except ImportError:
        print("  ? Skipped tensor dtype check (safetensors not installed)")

    # ── 7. AutoTokenizer.from_pretrained smoke test ──────────────────────
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(model_dir))
        passed &= check(
            "AutoTokenizer.from_pretrained works",
            True,
            f"vocab_size={tok.vocab_size}",
        )
    except Exception as e:
        passed &= check(
            "AutoTokenizer.from_pretrained works",
            False,
            f"{type(e).__name__}: {e}",
        )

    # ── Summary ──────────────────────────────────────────────────────────
    print()
    if passed:
        print("All checks passed ✓ — safe to run: make convert-gguf")
    else:
        print("FAILURES FOUND ✗ — fix the issues above before converting.")
    return passed


def main():
    model_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("models/executive-helper-ef")
    ok = validate(model_dir)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
