# Queue

## Q1. Run the experiments on real vLLM + production-stack (queued)

Status: queued, not started. Blocked on GPU availability.

- Blocker: the only GPU (RTX 4090) was held by another user's job
  (`build_heads.py`, ~24 GB, 100% util) at the time of the simulated run.
- Steps:
  1. Check `nvidia-smi` for a free GPU with >= 20 GB.
  2. Install vLLM and the production-stack router (Helm chart or Docker, per docs).
  3. Serve an 8B fp16 model (same family the cost model assumes).
  4. Replay the same workloads (`sim/workload.py` request streams) against the
     router for each policy: round_robin, session, qps, prefix, kvaware, loadaware.
  5. Compare measured TTFT/TPOT/hit ratio with the simulated tables in
     `results/tables/`, and recalibrate `sim/cluster.py::HW`.
  6. Save raw results under `results/real/` and update README.

## Q2. Controller lag for the KV-aware router (DONE in simulator, see README E4)

- Done: lagged index with lag 0, 0.1, 0.5, 2 s (`results/raw/lag.csv`). Small effect
  except on MMLU at 160 req/s, where lag 2 s is unstable. Real LMCache lag is still
  unmeasured.

## Q3. Load-aware + predicted semantic group (DONE in simulator, see README E4)

- Done: soft bonus 0.25/0.5/1.0 on load-aware (`results/raw/softsem.csv`). Bonus
  0.25–0.5 gives about 5–9% lower p99; 1.0 is unstable.
