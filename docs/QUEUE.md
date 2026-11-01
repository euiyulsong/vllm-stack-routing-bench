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

## Q2. Controller lag for the KV-aware router (queued)

- Add a configurable update delay to the KV-aware cache index, to match the
  LMCache controller. Re-run E1 and E2 with lags of 0, 100 ms and 500 ms.

## Q3. Load-aware + predicted semantic group (queued)

- Use the predicted group as a soft hint (score bonus) rather than a hard
  replica pool. Compare with E3.
