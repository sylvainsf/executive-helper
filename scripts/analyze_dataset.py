"""Quick dataset analysis script."""
import json
from collections import Counter

with open("data/generated/ef/train.jsonl") as f:
    examples = [json.loads(line) for line in f if line.strip()]

print(f"Total examples: {len(examples)}")
quality = Counter(ex["metadata"]["quality"] for ex in examples)
print(f"  Positive: {quality['positive']}")
print(f"  Negative: {quality['negative']}")
print()

tech_counts = Counter()
tech_quality = {}
for ex in examples:
    for tid in ex["metadata"]["technique_ids"]:
        tech_counts[tid] += 1
        key = (tid, ex["metadata"]["quality"])
        tech_quality[key] = tech_quality.get(key, 0) + 1

print(f"Techniques covered: {len(tech_counts)}")
print()
for tid, count in sorted(tech_counts.items()):
    pos = tech_quality.get((tid, "positive"), 0)
    neg = tech_quality.get((tid, "negative"), 0)
    print(f"  {tid:30s}  {pos:2d} pos  {neg:2d} neg  ({count} total)")

cats = Counter()
for ex in examples:
    cats[ex["metadata"]["category"]] += 1
print()
print("By category:")
for cat, count in sorted(cats.items()):
    print(f"  {cat:25s}  {count}")

combos = sum(1 for ex in examples if len(ex["metadata"]["technique_ids"]) > 1)
print(f"\nMulti-technique combos: {combos}")

lengths = [len(ex["messages"][1]["content"]) for ex in examples]
avg = sum(lengths) // len(lengths)
print(f"Avg response length: {avg} chars")
print(f"Min: {min(lengths)}, Max: {max(lengths)}")
