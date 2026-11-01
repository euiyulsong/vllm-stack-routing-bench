"""Download open-source datasets and convert them into workload files.

Sources (all public, Hugging Face):
  * ShareGPT_V3_unfiltered_cleaned_split  (anon8231489123/ShareGPT_Vicuna_unfiltered)
      -> multi-turn chat sessions; every later turn re-sends the earlier history,
         so it exercises prefix / KV-cache reuse exactly like real chat traffic.
  * MMLU (cais/mmlu, "all" config)
      -> 57-subject multiple choice. Each subject gets a shared 5-shot prompt
         header (from the `dev` split), so requests of one subject share a long
         prefix. Subject groups (STEM / Humanities / Social / Other) are the
         labels for the semantic-router experiment.
  * Qwen2.5-0.5B-Instruct tokenizer (open) used only to count/encode tokens.
         Llama-3-style tokenizers differ slightly; token counts are approximate.

Outputs (data/processed/):
  chat_sessions.jsonl.gz   list of conversations with user/assistant turns
  mmlu_questions.jsonl     labelled eval questions (test split, subsampled)
  mmlu_shots.json          per-subject 5-shot header text
  mmlu_clf_train.jsonl     validation+dev split for the semantic classifier
  data/cache/*.pkl         token streams (gitignored, rebuilt on demand)
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
CACHE = ROOT / "data" / "cache"

SHAREGPT_REPO = "anon8231489123/ShareGPT_Vicuna_unfiltered"
SHAREGPT_FILE = "ShareGPT_V3_unfiltered_cleaned_split.json"
TOK_REPO = "Qwen/Qwen2.5-0.5B-Instruct"

# MMLU official 4-way grouping (Hendrycks et al., 2021).
MMLU_CATEGORY = {}
for s in """abstract_algebra anatomy astronomy college_biology college_chemistry
college_computer_science college_mathematics college_physics computer_security
conceptual_physics electrical_engineering elementary_mathematics high_school_biology
high_school_chemistry high_school_computer_science high_school_mathematics
high_school_physics high_school_statistics machine_learning""".split():
    MMLU_CATEGORY[s] = "STEM"
for s in """formal_logic high_school_european_history high_school_us_history
high_school_world_history international_law jurisprudence logical_fallacies
moral_disputes moral_scenarios philosophy prehistory professional_law
world_religions""".split():
    MMLU_CATEGORY[s] = "Humanities"
for s in """econometrics high_school_geography high_school_government_and_politics
high_school_macroeconomics high_school_microeconomics high_school_psychology
human_sexuality professional_psychology public_relations security_studies sociology
us_foreign_policy""".split():
    MMLU_CATEGORY[s] = "Social"
for s in """business_ethics clinical_knowledge college_medicine global_facts human_aging
management marketing medical_genetics miscellaneous nutrition professional_accounting
professional_medicine virology""".split():
    MMLU_CATEGORY[s] = "Other"


def download() -> None:
    from huggingface_hub import hf_hub_download, snapshot_download

    RAW.mkdir(parents=True, exist_ok=True)
    hf_hub_download(SHAREGPT_REPO, SHAREGPT_FILE, repo_type="dataset",
                    local_dir=str(RAW / "sharegpt"))
    snapshot_download("cais/mmlu", repo_type="dataset", allow_patterns=["all/*"],
                      local_dir=str(RAW / "mmlu"))
    snapshot_download(TOK_REPO, allow_patterns=["tokenizer*", "vocab.json", "merges.txt"],
                      local_dir=str(RAW / "tokenizer"))


def is_mostly_english(text: str) -> bool:
    if not text:
        return False
    ascii_frac = sum(c.isascii() for c in text) / len(text)
    return ascii_frac > 0.97


def build_chat(max_convs: int, seed: int) -> list[dict]:
    raw = json.loads((RAW / "sharegpt" / SHAREGPT_FILE).read_text())
    rng = random.Random(seed)
    rng.shuffle(raw)
    convs = []
    for item in raw:
        msgs = item.get("conversations", [])
        if len(msgs) < 4 or msgs[0].get("from") != "human":
            continue
        turns = []
        ok = True
        for i in range(0, len(msgs) - 1, 2):
            u, a = msgs[i], msgs[i + 1]
            if u.get("from") != "human" or a.get("from") != "gpt":
                ok = False
                break
            ut, at = u["value"].strip(), a["value"].strip()
            if not ut or not at:
                ok = False
                break
            turns.append({"user": ut, "assistant": at})
        if not ok or len(turns) < 3:
            continue
        if not is_mostly_english(turns[0]["user"]):
            continue
        # keep a bounded history so one session stays within ~4k tokens
        total = 0
        kept = []
        for t in turns:
            total += len(t["user"]) + len(t["assistant"])
            if total > 12000:
                break
            kept.append(t)
        if len(kept) < 3:
            continue
        convs.append({"session_id": item.get("id", str(len(convs))), "turns": kept[:8]})
        if len(convs) >= max_convs:
            break
    return convs


def build_mmlu(eval_per_subject: int, seed: int) -> tuple[dict, list[dict], list[dict]]:
    import pyarrow.parquet as pq

    base = RAW / "mmlu" / "all"
    test = pq.read_table(base / "test-00000-of-00001.parquet").to_pylist()
    dev = pq.read_table(base / "dev-00000-of-00001.parquet").to_pylist()
    val = pq.read_table(base / "validation-00000-of-00001.parquet").to_pylist()

    letters = "ABCD"

    def fmt(row, with_answer: bool) -> str:
        s = row["question"].strip() + "\n"
        for i, c in enumerate(row["choices"]):
            s += f"{letters[i]}. {c}\n"
        s += "Answer:"
        if with_answer:
            s += f" {letters[row['answer']]}\n\n"
        return s

    shots = {}
    for row in dev:
        subj = row["subject"]
        shots.setdefault(subj, []).append(row)
    headers = {}
    for subj, rows in shots.items():
        pretty = subj.replace("_", " ")
        head = (f"The following are multiple choice questions (with answers) about {pretty}.\n\n")
        head += "".join(fmt(r, True) for r in rows[:5])
        headers[subj] = head

    rng = random.Random(seed)
    by_subj = {}
    for row in test:
        by_subj.setdefault(row["subject"], []).append(row)
    eval_rows = []
    for subj, rows in sorted(by_subj.items()):
        rng.shuffle(rows)
        for row in rows[:eval_per_subject]:
            eval_rows.append({
                "subject": subj,
                "category": MMLU_CATEGORY[subj],
                "question": fmt(row, False),
                "answer": letters[row["answer"]],
            })
    clf_train = []
    for row in val + dev:
        clf_train.append({"text": fmt(row, False), "category": MMLU_CATEGORY[row["subject"]]})
    return headers, eval_rows, clf_train


def build_token_cache(chat: list[dict], headers: dict, eval_rows: list[dict]) -> None:
    """Pre-tokenize everything once so simulator runs do not re-tokenize."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(RAW / "tokenizer" / "tokenizer.json"))
    enc = lambda s: tok.encode(s, add_special_tokens=False).ids  # noqa: E731

    chat_tok = []
    for c in chat:
        segs = []  # (text, token_ids) alternating user/assistant
        for t in c["turns"]:
            u_text = f"User: {t['user']}\n"
            a_text = f"Assistant: {t['assistant']}\n"
            segs.append((u_text, enc(u_text)))
            segs.append((a_text, enc(a_text)))
        chat_tok.append({"session_id": c["session_id"], "segs": segs})

    mmlu_tok = {"headers": {s: enc(h) for s, h in headers.items()},
                "questions": [enc(r["question"]) for r in eval_rows]}
    CACHE.mkdir(parents=True, exist_ok=True)
    with open(CACHE / "tokens.pkl", "wb") as f:
        pickle.dump({"chat": chat_tok, "mmlu": mmlu_tok}, f, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--max-convs", type=int, default=1500)
    ap.add_argument("--mmlu-per-subject", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not args.skip_download:
        download()
    PROC.mkdir(parents=True, exist_ok=True)

    chat = build_chat(args.max_convs, args.seed)
    with gzip.open(PROC / "chat_sessions.jsonl.gz", "wt") as f:
        for c in chat:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    headers, eval_rows, clf_train = build_mmlu(args.mmlu_per_subject, args.seed)
    (PROC / "mmlu_shots.json").write_text(json.dumps(headers, ensure_ascii=False))
    with open(PROC / "mmlu_questions.jsonl", "w") as f:
        for r in eval_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(PROC / "mmlu_clf_train.jsonl", "w") as f:
        for r in clf_train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    build_token_cache(chat, headers, eval_rows)
    n_turns = sum(len(c["turns"]) for c in chat)
    print(f"chat sessions: {len(chat)} (turns: {n_turns}, avg turns {n_turns/len(chat):.2f})")
    print(f"mmlu eval questions: {len(eval_rows)} across {len(headers)} subjects")
    print(f"mmlu classifier train rows: {len(clf_train)}")


if __name__ == "__main__":
    main()
