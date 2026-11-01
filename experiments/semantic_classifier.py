"""Semantic router classifier (real, CPU).

Task: map an incoming MMLU question to one of four subject groups
(STEM / Humanities / Social / Other). This is the signal a semantic router uses to
pick a backend pool. Training data: MMLU validation+dev (disjoint from the test
split we evaluate on). Model: TF-IDF (word 1-2 grams) + logistic regression.

Writes:
  results/semantic/classifier_metrics.json
  data/processed/semantic_predictions.json   (predicted group for each eval question)
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "results" / "semantic"

GROUPS = ["STEM", "Humanities", "Social", "Other"]


def load_jsonl(p):
    return [json.loads(l) for l in open(p)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train = load_jsonl(PROC / "mmlu_clf_train.jsonl")
    test = load_jsonl(PROC / "mmlu_questions.jsonl")
    # the eval questions carry the question text in the same "Answer:" format as train
    X_tr = [r["text"] for r in train]
    y_tr = [r["category"] for r in train]
    X_te = [r["question"] for r in test]
    y_te = [r["category"] for r in test]

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    Xtr = vec.fit_transform(X_tr)
    clf = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")
    clf.fit(Xtr, y_tr)

    Xte = vec.transform(X_te)
    pred = clf.predict(Xte)

    # per-request latency on a single CPU core, one request at a time (router-style)
    sample = X_te[:500]
    t0 = time.perf_counter()
    for s in sample:
        clf.predict(vec.transform([s]))
    lat_ms = (time.perf_counter() - t0) / len(sample) * 1000

    majority = Counter(y_tr).most_common(1)[0][0]
    metrics = {
        "train_rows": len(train),
        "test_rows": len(test),
        "accuracy": accuracy_score(y_te, pred),
        "macro_f1": f1_score(y_te, pred, average="macro", labels=GROUPS),
        "majority_baseline_accuracy": accuracy_score(y_te, [majority] * len(y_te)),
        "per_request_latency_ms_cpu": lat_ms,
        "report": classification_report(y_te, pred, labels=GROUPS, output_dict=True, zero_division=0),
    }
    (OUT / "classifier_metrics.json").write_text(json.dumps(metrics, indent=2))

    # predictions aligned with data/processed/mmlu_questions.jsonl
    (PROC / "semantic_predictions.json").write_text(json.dumps(list(map(str, pred))))
    print(f"accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} "
          f"majority={metrics['majority_baseline_accuracy']:.4f} latency={lat_ms:.3f} ms/req")


if __name__ == "__main__":
    main()
