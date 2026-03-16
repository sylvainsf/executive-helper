"""Evaluation runner — tests baseline and fine-tuned models against eval cases."""

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.eval.cases import AUTO_CASES, EF_CASES
from src.gateway.models import chat


async def evaluate_case(case: dict, model_role: str) -> dict:
    """Run a single eval case and return the result."""
    print(f"  [{case['id']}] {case['name']}...")

    messages = [{"role": "user", "content": case["input"]}]

    try:
        response = await chat(model_role, messages, temperature=0.3, max_tokens=1024)
    except Exception as e:
        return {
            **case,
            "response": f"ERROR: {e}",
            "model_role": model_role,
            "status": "error",
        }

    return {
        **case,
        "response": response,
        "model_role": model_role,
        "status": "completed",
    }


async def run_eval(model_role: str) -> list[dict]:
    """Run all eval cases for a model role."""
    cases = EF_CASES if model_role == "ef" else AUTO_CASES
    print(f"\nRunning {len(cases)} eval cases for '{model_role}' model\n")

    results = []
    for case in cases:
        result = await evaluate_case(case, model_role)
        results.append(result)

        if result["status"] == "completed":
            print(f"    Response: {result['response'][:200]}...")
        else:
            print(f"    {result['response']}")
        print()

    return results


def print_results(results: list[dict]):
    """Print eval results summary."""
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    for r in results:
        status = "✓" if r["status"] == "completed" else "✗"
        print(f"\n{status} [{r['id']}] {r['name']}")
        if r["status"] == "completed":
            print(f"  Response (first 300 chars):")
            print(f"    {r['response'][:300]}")
            print(f"  Criteria to check manually:")
            for criterion in r["criteria"]:
                print(f"    [ ] {criterion}")
        else:
            print(f"  Error: {r['response']}")

    completed = sum(1 for r in results if r["status"] == "completed")
    print(f"\n{completed}/{len(results)} cases completed successfully")


def save_results(results: list[dict], output_dir: str = "data/eval"):
    """Save eval results to JSON for later comparison."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_role = results[0]["model_role"] if results else "unknown"
    path = Path(output_dir) / f"eval_{model_role}_{ts}.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {path}")


async def run_comparison():
    """Run eval for both models and display side by side."""
    for role in ("ef", "auto"):
        results = await run_eval(role)
        print_results(results)
        save_results(results)


async def main():
    parser = argparse.ArgumentParser(description="Run model evaluation")
    parser.add_argument(
        "--model",
        choices=["ef", "auto"],
        help="Which model to evaluate (omit for both)",
    )
    parser.add_argument("--compare", action="store_true", help="Run both models")
    args = parser.parse_args()

    if args.compare or args.model is None:
        await run_comparison()
    else:
        results = await run_eval(args.model)
        print_results(results)
        save_results(results)


if __name__ == "__main__":
    asyncio.run(main())
    sys.exit(0)
