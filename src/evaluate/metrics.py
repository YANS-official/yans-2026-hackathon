"""Evaluation script that computes metrics for the citation correctness benchmark."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def _read_jsonl(path: str) -> list[dict]:
    """Read a JSONL file and return a list of dicts."""
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def evaluate(gold_path: str, pred_path: str) -> dict:
    """Compute evaluation metrics by comparing gold labels to predictions.

    Parameters
    ----------
    gold_path : str
        Path to the gold-standard JSONL file.
        Required fields: id, label.  Optional field: negative_type.
    pred_path : str
        Path to the prediction JSONL file.
        Required fields: id, prediction.

    Returns
    -------
    dict
        Dictionary containing overall metrics and, when negative_type is
        present, a per-type breakdown.
    """
    gold_records = _read_jsonl(gold_path)
    pred_records = _read_jsonl(pred_path)

    # Build lookup from id -> prediction
    pred_by_id: dict[str, int] = {r["id"]: int(r["prediction"]) for r in pred_records}

    # Align gold and predictions by id
    y_true: list[int] = []
    y_pred: list[int] = []
    gold_with_neg_type: list[dict] = []

    for g in gold_records:
        gid = g["id"]
        if gid not in pred_by_id:
            raise ValueError(f"Prediction missing for id: {gid}")
        y_true.append(int(g["label"]))
        y_pred.append(pred_by_id[gid])
        gold_with_neg_type.append(g)

    # Overall metrics
    results: dict = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }

    # Per negative_type breakdown (only when negative_type field is present)
    has_neg_type = any(
        g.get("negative_type") is not None for g in gold_with_neg_type
    )

    if has_neg_type:
        # Collect the set of cited_paper_ids (or record ids) that are positive
        # and associated negative types
        neg_types: set[str] = set()
        for g in gold_with_neg_type:
            nt = g.get("negative_type")
            if nt is not None:
                neg_types.add(nt)

        # For each negative_type, evaluate on the subset consisting of:
        #   - all negative examples of that type
        #   - all positive examples (label == 1)
        positive_indices = [
            i for i, g in enumerate(gold_with_neg_type) if int(g["label"]) == 1
        ]

        per_type: dict[str, dict] = {}
        for nt in sorted(neg_types):
            neg_indices = [
                i
                for i, g in enumerate(gold_with_neg_type)
                if g.get("negative_type") == nt
            ]
            subset_indices = sorted(set(positive_indices + neg_indices))
            sub_true = [y_true[i] for i in subset_indices]
            sub_pred = [y_pred[i] for i in subset_indices]
            per_type[nt] = {
                "accuracy": accuracy_score(sub_true, sub_pred),
                "count_negative": len(neg_indices),
                "count_total": len(subset_indices),
            }

        results["per_type"] = per_type

    return results


def _format_table(results: dict) -> str:
    """Format results dict as a human-readable table string."""
    lines: list[str] = []

    # Overall metrics
    lines.append("=" * 50)
    lines.append("Overall Metrics")
    lines.append("=" * 50)
    lines.append(f"  {'Metric':<15} {'Value':>10}")
    lines.append(f"  {'-' * 15} {'-' * 10}")
    for key in ("accuracy", "f1", "precision", "recall"):
        lines.append(f"  {key:<15} {results[key]:>10.4f}")

    # Per-type breakdown
    if "per_type" in results:
        lines.append("")
        lines.append("=" * 50)
        lines.append("Per Negative-Type Breakdown")
        lines.append("=" * 50)
        lines.append(
            f"  {'Type':<25} {'Accuracy':>10} {'#Neg':>6} {'#Total':>8}"
        )
        lines.append(f"  {'-' * 25} {'-' * 10} {'-' * 6} {'-' * 8}")
        for nt, info in results["per_type"].items():
            lines.append(
                f"  {nt:<25} {info['accuracy']:>10.4f}"
                f" {info['count_negative']:>6}"
                f" {info['count_total']:>8}"
            )

    return "\n".join(lines)


def main() -> None:
    """CLI entry point for evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate predictions against gold labels."
    )
    parser.add_argument(
        "--gold",
        type=str,
        required=True,
        help="Path to gold-standard JSONL file.",
    )
    parser.add_argument(
        "--pred",
        type=str,
        required=True,
        help="Path to prediction JSONL file.",
    )
    args = parser.parse_args()

    results = evaluate(args.gold, args.pred)
    print(_format_table(results))


if __name__ == "__main__":
    main()
