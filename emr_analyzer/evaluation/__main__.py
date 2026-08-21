from __future__ import annotations

import argparse
import json

from .runner import EvaluationRunner


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Valuta eventi clinici su split patient-level JSONL"
    )
    parser.add_argument("gold_jsonl")
    parser.add_argument("--output")
    args = parser.parse_args()
    report = EvaluationRunner().evaluate_jsonl(args.gold_jsonl)
    if args.output:
        EvaluationRunner.save_report(report, args.output)
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
