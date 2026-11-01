"""Run every experiment suite and write one CSV row per (configuration, seed).

Suites
  routing   : router policies on a 4-replica colocated cluster            (E1)
  beta      : load-aware weight sweep                                     (E1b)
  features  : CPU offload / shared KV store / disaggregated prefill        (E2)
  semantic  : semantic-router pools vs cache-only routing on MMLU          (E3)

Usage:
  python experiments/run_experiments.py --suite all --seeds 5 --workers 24
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import itertools
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.cluster import HW, Cluster  # noqa: E402
from sim.metrics import summarize  # noqa: E402
from sim.workload import build_workload  # noqa: E402

DURATION = 600.0   # simulated seconds of arrivals
WARMUP = 120.0     # discard the cold-cache transient
DRAIN_END = 1800.0 # keep stepping until queues drain
CPU_TOKENS = 128_000        # 16 GB host-memory KV tier per replica
REMOTE_TOKENS = 1_000_000   # 128 GB shared KV store (cluster-wide)

RATES = {
    "chat": [4, 8, 12, 16],
    "mmlu": [40, 80, 120, 160],
    "mixed": [8, 16, 24, 32],
}

ROUTING_POLICIES = ["round_robin", "session", "qps", "prefix", "kvaware", "loadaware"]


@dataclasses.dataclass(frozen=True)
class Job:
    suite: str
    name: str            # human-readable configuration name
    workload: str
    rate: float
    seed: int
    policy: str
    n_mixed: int = 0
    n_prefill: int = 0
    n_decode: int = 0
    cpu: bool = False
    remote: bool = False
    beta: float = 1.0
    lag: float = 0.0
    sem_bonus: float = 0.0


def run_job(job: Job) -> dict:
    hw = HW(loadaware_beta=job.beta,
            cpu_kv_tokens=CPU_TOKENS if job.cpu else 0,
            remote_kv_tokens=REMOTE_TOKENS if job.remote else 0)
    reqs = build_workload(job.workload, job.rate, DURATION, job.seed)
    cl = Cluster(hw, n_mixed=job.n_mixed, n_prefill=job.n_prefill, n_decode=job.n_decode,
                 policy=job.policy, use_remote=job.remote, cpu_offload=job.cpu,
                 index_lag=job.lag, sem_bonus=job.sem_bonus)
    t0 = time.time()
    done = cl.run(reqs, DRAIN_END)
    n_entry = len(cl.entry)
    m = summarize(reqs, done, WARMUP, DURATION, n_entry)
    row = {
        "suite": job.suite, "config": job.name, "workload": job.workload,
        "rate_req_s": job.rate, "seed": job.seed, "policy": job.policy,
        "topology": (f"{job.n_mixed}mixed" if job.n_mixed else
                     f"{job.n_prefill}P+{job.n_decode}D"),
        "cpu_offload": int(job.cpu), "shared_store": int(job.remote),
        "beta": job.beta, "index_lag_s": job.lag, "sem_bonus": job.sem_bonus,
        "sim_wall_s": round(time.time() - t0, 2),
    }
    row.update({k: (round(v, 6) if isinstance(v, float) else v) for k, v in m.items()})
    return row


def jobs_for(suite: str, seeds: int) -> list[Job]:
    S = range(seeds)
    out: list[Job] = []
    if suite == "routing":
        for wl in RATES:
            for rate in RATES[wl]:
                for p in ROUTING_POLICIES:
                    for s in S:
                        out.append(Job("routing", f"colocated-4x-{p}", wl, rate, s, p, n_mixed=4))
    elif suite == "beta":
        for wl, rate in [("chat", 12), ("mixed", 24), ("mmlu", 160)]:
            for beta in [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]:
                for s in S:
                    out.append(Job("beta", f"loadaware-beta{beta}", wl, rate, s,
                                   "loadaware", n_mixed=4, beta=beta))
    elif suite == "features":
        configs = [
            ("colocated_base",      dict(n_mixed=4)),
            ("colocated+cpu_offload", dict(n_mixed=4, cpu=True)),
            ("colocated+shared_store", dict(n_mixed=4, remote=True)),
            ("colocated+offload+shared", dict(n_mixed=4, cpu=True, remote=True)),
            ("disagg_3P+1D",        dict(n_prefill=3, n_decode=1)),
            ("disagg_2P+2D",        dict(n_prefill=2, n_decode=2)),
            ("disagg_1P+3D",        dict(n_prefill=1, n_decode=3)),
            ("disagg_2P+2D+offload+shared", dict(n_prefill=2, n_decode=2, cpu=True, remote=True)),
        ]
        for wl in ["chat", "mixed"]:
            for rate in RATES[wl]:
                for name, kw in configs:
                    for s in S:
                        out.append(Job("features", name, wl, rate, s, "kvaware", **kw))
    elif suite == "semantic":
        for pol in ["qps", "prefix", "kvaware", "loadaware", "semantic", "semantic_oracle"]:
            for rate in RATES["mmlu"]:
                for s in S:
                    out.append(Job("semantic", f"colocated-4x-{pol}", "mmlu", rate, s, pol,
                                   n_mixed=4))
    elif suite == "lag":
        # Q2: KV-aware routing with a stale cache index (controller update delay)
        for wl, rates in [("chat", [8, 12, 16]), ("mixed", [16, 24]), ("mmlu", [120, 160])]:
            for rate in rates:
                for lag in [0.0, 0.1, 0.5, 2.0]:
                    for s in S:
                        out.append(Job("lag", f"kvaware-lag{lag}s", wl, rate, s, "kvaware",
                                       n_mixed=4, lag=lag))
    elif suite == "softsem":
        # Q3: load-aware routing with the predicted semantic group as a soft hint
        for rate in [80, 120, 160]:
            for s in S:
                out.append(Job("softsem", "loadaware", "mmlu", rate, s, "loadaware", n_mixed=4))
                out.append(Job("softsem", "kvaware", "mmlu", rate, s, "kvaware", n_mixed=4))
                out.append(Job("softsem", "semantic-hard", "mmlu", rate, s, "semantic", n_mixed=4))
                for g in [0.25, 0.5, 1.0]:
                    out.append(Job("softsem", f"loadaware_sem-bonus{g}", "mmlu", rate, s,
                                   "loadaware_sem", n_mixed=4, sem_bonus=g))
    else:
        raise ValueError(suite)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="all",
                    help="routing|beta|features|semantic|all (comma separated allowed)")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=min(24, os.cpu_count() or 4))
    ap.add_argument("--out", default=str(ROOT / "results" / "raw"))
    args = ap.parse_args()

    suites = ["routing", "beta", "features", "semantic"] if args.suite == "all" \
        else args.suite.split(",")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for suite in suites:
        jobs = jobs_for(suite, args.seeds)
        print(f"[{suite}] {len(jobs)} runs", flush=True)
        t0 = time.time()
        with mp.get_context("fork").Pool(args.workers) as pool:
            rows = []
            for i, row in enumerate(pool.imap_unordered(run_job, jobs, chunksize=1), 1):
                rows.append(row)
                if i % 50 == 0 or i == len(jobs):
                    print(f"  {i}/{len(jobs)}  {time.time()-t0:.0f}s", flush=True)
        rows.sort(key=lambda r: (r["workload"], r["rate_req_s"], r["config"], r["seed"]))
        path = out_dir / f"{suite}.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"  wrote {path} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
