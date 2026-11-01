"""Turn the processed open-source datasets into timed request streams.

Two keys are produced per request:
  * block_keys : chained hashes of 16-token KV blocks (vLLM-style prefix cache).
                 Used by replica caches, CPU offload and the shared KV store.
  * char_keys  : chained hashes of 128-character chunks of the prompt text.
                 Used by the prefix-aware router, which mirrors the production
                 stack's `HashTrie` (chunk_size=128, xxhash there, blake2b here).
"""
from __future__ import annotations

import hashlib
import json
import pickle
import random
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
CACHE = ROOT / "data" / "cache" / "tokens.pkl"

BLOCK = 16
CHUNK = 128
THINK_MEAN_S = 30.0  # mean user think time between chat turns (open-loop)


def chain_block_keys(tokens: list[int]) -> list[int]:
    keys, prev = [], 0
    for b in range(len(tokens) // BLOCK):
        prev = hash((prev, tuple(tokens[b * BLOCK:(b + 1) * BLOCK])))
        keys.append(prev)
    return keys


def chain_char_keys(text: str) -> list[int]:
    keys, prev = [], b"\x00" * 8
    for i in range(0, len(text), CHUNK):
        h = hashlib.blake2b(prev + text[i:i + CHUNK].encode("utf-8", "replace"),
                            digest_size=8).digest()
        keys.append(int.from_bytes(h, "little"))
        prev = h
    return keys


@dataclass
class Req:
    rid: int
    arrival: float
    kind: str                      # "chat" | "mmlu"
    session: str | None
    prompt_len: int
    out_len: int
    n_prompt_blocks: int
    keys: list                     # block keys covering prompt (+ reply for chat)
    char_keys: list
    category: str | None = None    # true semantic category (mmlu only)
    qidx: int = -1                 # index into mmlu_questions.jsonl
    # ---- runtime state (filled by the simulator) ----
    replica: int = -1
    dreplica: int = -1
    phase: str = "wait"
    held: list = field(default_factory=list)
    matched: int = 0
    load_ready: float = 0.0
    rem_prefill: int = 0
    generated: int = 0
    first_t: float = -1.0
    finish_t: float = -1.0
    tok_gpu: int = 0
    tok_cpu: int = 0
    tok_remote: int = 0
    pred_category: str | None = None

    @property
    def n_total_blocks(self) -> int:
        return len(self.keys)


def _load_tokens():
    with open(CACHE, "rb") as f:
        return pickle.load(f)


def chat_requests(rate: float, duration: float, seed: int, rid0: int = 0,
                  think_mean: float = THINK_MEAN_S) -> list[Req]:
    """Open-loop multi-turn chat sessions targeting `rate` requests/s overall."""
    tok = _load_tokens()["chat"]
    rng = random.Random(seed)
    pool = list(range(len(tok)))
    rng.shuffle(pool)
    avg_turns = sum(len(tok[i]["segs"]) // 2 for i in pool) / len(pool)
    session_rate = rate / avg_turns
    reqs, t, rid, s_idx = [], 0.0, rid0, 0
    while True:
        t += rng.expovariate(session_rate)
        if t >= duration:
            break
        conv = tok[pool[s_idx % len(pool)]]
        s_idx += 1
        segs = conv["segs"]
        stream = [tid for _, ids in segs for tid in ids]
        stream_keys = chain_block_keys(stream)
        texts = [txt for txt, _ in segs]
        turn_t = t
        prompt_len = 0
        text_so_far = ""
        for i in range(len(segs) // 2):
            u_ids = segs[2 * i][1]
            a_ids = segs[2 * i + 1][1]
            prompt_len += len(u_ids)
            text_so_far += texts[2 * i]
            full_len = prompt_len + len(a_ids)  # tokens incl. this turn's reply
            if i > 0:
                turn_t += rng.expovariate(1.0 / think_mean)
            if turn_t >= duration:
                break
            # stream position of the start of this turn's user segment
            n_total = full_len // BLOCK
            n_prompt = prompt_len // BLOCK
            reqs.append(Req(
                rid=rid, arrival=turn_t, kind="chat",
                session=f"s{seed}-{conv['session_id']}-{s_idx}",
                prompt_len=prompt_len, out_len=len(a_ids),
                n_prompt_blocks=n_prompt,
                keys=stream_keys[:n_total],
                char_keys=chain_char_keys(text_so_far),
            ))
            rid += 1
            # reply becomes part of the next turn's prompt
            prompt_len = full_len
            text_so_far += texts[2 * i + 1]
    reqs.sort(key=lambda r: r.arrival)
    return reqs


def mmlu_requests(rate: float, duration: float, seed: int, rid0: int = 0) -> list[Req]:
    """Poisson stream of 5-shot MMLU prompts; prompts of one subject share a prefix."""
    data = _load_tokens()["mmlu"]
    shots_text = json.loads((PROC / "mmlu_shots.json").read_text())
    qrows = [json.loads(l) for l in open(PROC / "mmlu_questions.jsonl")]
    subjects = sorted(shots_text)
    rng = random.Random(seed + 1000)
    preds_path = PROC / "semantic_predictions.json"
    preds = json.loads(preds_path.read_text()) if preds_path.exists() else None
    reqs, t, rid = [], 0.0, rid0
    order = list(range(len(qrows)))
    rng.shuffle(order)
    q_idx = 0
    while True:
        t += rng.expovariate(rate)
        if t >= duration:
            break
        qi = order[q_idx % len(order)]
        q_idx += 1
        row = qrows[qi]
        head_ids = data["headers"][row["subject"]]
        q_ids = data["questions"][qi]
        ids = head_ids + q_ids
        text = shots_text[row["subject"]] + row["question"]
        req = Req(
            rid=rid, arrival=t, kind="mmlu", session=None,
            prompt_len=len(ids), out_len=1,
            n_prompt_blocks=len(ids) // BLOCK,
            keys=chain_block_keys(ids),
            char_keys=chain_char_keys(text),
            category=row["category"], qidx=qi,
        )
        req.pred_category = preds[qi] if preds is not None else None
        reqs.append(req)
        rid += 1
    return reqs


def build_workload(name: str, rate: float, duration: float, seed: int) -> list[Req]:
    """name: 'chat' | 'mmlu' | 'mixed' (50/50 request split at the same total rate)."""
    if name == "chat":
        return chat_requests(rate, duration, seed)
    if name == "mmlu":
        return mmlu_requests(rate, duration, seed)
    if name == "mixed":
        a = chat_requests(rate / 2, duration, seed)
        b = mmlu_requests(rate / 2, duration, seed, rid0=len(a))
        reqs = a + b
        reqs.sort(key=lambda r: r.arrival)
        for i, r in enumerate(reqs):
            r.rid = i
        return reqs
    raise ValueError(name)
