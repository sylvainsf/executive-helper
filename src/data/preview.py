"""Preview and inspect generated training data."""

import argparse
import json
import sys
from pathlib import Path


def preview_dataset(path: str, count: int = 5):
    """Preview samples from a JSONL training dataset."""
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        # Try finding train.jsonl in common locations
        for candidate in [
            Path(f"data/generated/ef/train.jsonl"),
            Path(f"data/generated/auto/train.jsonl"),
        ]:
            if candidate.exists():
                jsonl_path = candidate
                break
        else:
            print(f"No training data found at {path}")
            print("Run 'make gen-data-ef' first.")
            sys.exit(1)

    examples = []
    with open(jsonl_path) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    print(f"\nDataset: {jsonl_path}")
    print(f"Total examples: {len(examples)}")
    print(f"Showing first {min(count, len(examples))}:\n")
    print("=" * 70)

    for i, ex in enumerate(examples[:count]):
        msgs = ex.get("messages", [])
        meta = ex.get("metadata", {})

        print(f"\n--- Example {i + 1} ---")
        if meta:
            print(f"  Source: {meta.get('source', '?')} | ID: {meta.get('case_id', '?')}")

        for msg in msgs:
            role = msg["role"].upper()
            content = msg["content"]
            if len(content) > 300:
                content = content[:300] + "..."
            print(f"  [{role}]: {content}")

        print()

    # Stats
    user_lengths = []
    asst_lengths = []
    for ex in examples:
        for msg in ex.get("messages", []):
            if msg["role"] == "user":
                user_lengths.append(len(msg["content"]))
            elif msg["role"] == "assistant":
                asst_lengths.append(len(msg["content"]))

    if user_lengths:
        print(f"User message lengths: min={min(user_lengths)}, max={max(user_lengths)}, avg={sum(user_lengths)//len(user_lengths)}")
    if asst_lengths:
        print(f"Assistant message lengths: min={min(asst_lengths)}, max={max(asst_lengths)}, avg={sum(asst_lengths)//len(asst_lengths)}")


def main():
    parser = argparse.ArgumentParser(description="Preview training data")
    parser.add_argument("--path", default="data/generated", help="Path to dataset or directory")
    parser.add_argument("--count", type=int, default=5, help="Number of examples to show")
    args = parser.parse_args()

    path = Path(args.path)
    if path.is_dir():
        # Preview all datasets found
        for jsonl in sorted(path.rglob("train.jsonl")):
            preview_dataset(str(jsonl), args.count)
    else:
        preview_dataset(args.path, args.count)


if __name__ == "__main__":
    main()
