"""Summaries computed over the measurement window [warmup, duration)."""
from __future__ import annotations

import math


def _pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = min(len(xs) - 1, max(0, int(math.ceil(p * len(xs))) - 1))
    return xs[k]


def summarize(reqs, done, warmup: float, duration: float, n_entry: int,
              ttft_slo: float = 2.0, tpot_slo: float = 0.1) -> dict:
    window = [r for r in reqs if warmup <= r.arrival < duration]
    fin = [r for r in window if r.finish_t > 0]
    ttft = [r.first_t - r.arrival for r in fin]
    e2e = [r.finish_t - r.arrival for r in fin]
    tpot = [(r.finish_t - r.first_t) / (r.out_len - 1) for r in fin if r.out_len > 1]
    prompt_tok = sum(r.prompt_len for r in fin) or 1
    gpu = sum(r.tok_gpu for r in fin)
    cpu = sum(r.tok_cpu for r in fin)
    rem = sum(r.tok_remote for r in fin)
    slo_ok = sum(
        1 for r in fin
        if (r.first_t - r.arrival) <= ttft_slo
        and (r.out_len <= 1 or (r.finish_t - r.first_t) / (r.out_len - 1) <= tpot_slo)
    )
    per_rep = [0] * n_entry
    for r in window:
        if 0 <= r.replica < n_entry:
            per_rep[r.replica] += 1
    mean_rep = sum(per_rep) / max(1, len(per_rep))
    cv = (math.sqrt(sum((c - mean_rep) ** 2 for c in per_rep) / len(per_rep)) / mean_rep
          if mean_rep > 0 else float("nan"))
    span = max(1e-9, duration - warmup)
    return {
        "n_window": len(window),
        "completed_frac": len(fin) / max(1, len(window)),
        "ttft_p50_s": _pct(ttft, 0.50),
        "ttft_p90_s": _pct(ttft, 0.90),
        "ttft_p99_s": _pct(ttft, 0.99),
        "ttft_mean_s": sum(ttft) / max(1, len(ttft)),
        "tpot_p50_ms": _pct(tpot, 0.50) * 1000 if tpot else float("nan"),
        "tpot_p99_ms": _pct(tpot, 0.99) * 1000 if tpot else float("nan"),
        "e2e_p99_s": _pct(e2e, 0.99),
        "slo_attainment": slo_ok / max(1, len(window)),
        "req_per_s": len(fin) / span,
        "out_tok_per_s": sum(r.out_len for r in fin) / span,
        "hit_ratio": (gpu + cpu + rem) / prompt_tok,
        "hit_gpu": gpu / prompt_tok,
        "hit_cpu": cpu / prompt_tok,
        "hit_remote": rem / prompt_tok,
        "load_cv": cv,
    }
