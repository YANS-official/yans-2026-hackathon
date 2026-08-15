"""Baseline predictor using embedding cosine similarity.

Encodes citation contexts and paper abstracts with a Japanese embedding model
(cl-nagoya/ruri-v3-70m, 70M parameters; within the 100M-parameter regulation),
computes cosine similarity, optimises a threshold on the dev set, and
produces predictions for the test set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score


MODEL_NAME = "cl-nagoya/ruri-v3-70m"


def _read_jsonl(path: str) -> list[dict]:
    """Read a JSONL file and return a list of dicts."""
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_abstracts(papers_path: str) -> dict[str, str]:
    """Load paper abstracts from a papers.jsonl file.

    Returns a mapping from paper_id to abstract text.
    """
    abstracts: dict[str, str] = {}
    with open(papers_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                paper_id = entry.get("paper_id", "")
                abstract = entry.get("abstract", "")
                if paper_id and abstract:
                    abstracts[paper_id] = abstract
    return abstracts


def _compute_similarities(
    model: SentenceTransformer,
    records: list[dict],
    abstracts: dict[str, str],
) -> tuple[list[str], np.ndarray, list[int]]:
    """Compute cosine similarities between citation contexts and abstracts.

    Returns
    -------
    ids : list[str]
        Record IDs in the same order.
    similarities : np.ndarray
        Cosine similarity scores.
    labels : list[int]
        Gold labels (may be empty if label field is absent).
    """
    contexts: list[str] = []
    abstract_texts: list[str] = []
    ids: list[str] = []
    labels: list[int] = []

    for rec in records:
        cited_id = rec["cited_paper_id"]
        abstract = abstracts.get(cited_id, "")
        contexts.append(rec["citation_context"])
        abstract_texts.append(abstract)
        ids.append(rec["id"])
        if "label" in rec:
            labels.append(int(rec["label"]))

    # Encode in batches
    ctx_embs = model.encode(contexts, normalize_embeddings=True, show_progress_bar=True)
    abs_embs = model.encode(abstract_texts, normalize_embeddings=True, show_progress_bar=True)

    # Cosine similarity (embeddings are already L2-normalised)
    similarities = np.sum(ctx_embs * abs_embs, axis=1)

    return ids, similarities, labels


def _optimise_threshold(
    similarities: np.ndarray,
    labels: list[int],
    steps: int = 200,
) -> float:
    """Find the similarity threshold that maximises F1 on the given labels."""
    best_f1 = -1.0
    best_threshold = 0.5
    lo, hi = float(similarities.min()), float(similarities.max())

    for i in range(steps + 1):
        threshold = lo + (hi - lo) * i / steps
        preds = (similarities >= threshold).astype(int)
        score = f1_score(labels, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    return best_threshold


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Embedding-similarity baseline for citation correctness."
    )
    parser.add_argument(
        "--papers",
        type=str,
        required=True,
        help="Path to papers.jsonl file.",
    )
    parser.add_argument(
        "--dev",
        type=str,
        required=True,
        help="Path to dev JSONL file (used for threshold optimisation).",
    )
    parser.add_argument(
        "--test",
        type=str,
        required=True,
        help="Path to test JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to write prediction JSONL.",
    )
    args = parser.parse_args()

    print(f"Loading abstracts from {args.papers} ...")
    abstracts = _load_abstracts(args.papers)
    print(f"  Loaded {len(abstracts)} abstracts.")

    print(f"Loading model: {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    # --- Dev set: compute similarities and optimise threshold ---
    print("Processing dev set ...")
    dev_records = _read_jsonl(args.dev)
    dev_ids, dev_sims, dev_labels = _compute_similarities(model, dev_records, abstracts)
    threshold = _optimise_threshold(dev_sims, dev_labels)
    print(f"  Optimised threshold: {threshold:.4f}")

    # Dev set performance
    dev_preds = (dev_sims >= threshold).astype(int)
    dev_f1 = f1_score(dev_labels, dev_preds, zero_division=0)
    print(f"  Dev F1: {dev_f1:.4f}")

    # --- Test set: predict ---
    print("Processing test set ...")
    test_records = _read_jsonl(args.test)
    test_ids, test_sims, _ = _compute_similarities(model, test_records, abstracts)
    test_preds = (test_sims >= threshold).astype(int)

    # Write predictions
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec_id, pred in zip(test_ids, test_preds):
            f.write(json.dumps({"id": rec_id, "prediction": int(pred)}, ensure_ascii=False) + "\n")

    print(f"Predictions written to {args.output} ({len(test_preds)} records).")


if __name__ == "__main__":
    main()
