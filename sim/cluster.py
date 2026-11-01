"""Discrete-event simulator of a vLLM cluster behind a production-stack style router.

What is modelled
  * Each replica: paged KV pool (16-token blocks) with refcounted prefix cache
    (LRU of idle blocks), continuous batching, chunked prefill that shares each
    step with decode (the source of prefill/decode interference).
  * CPU offload tier: idle GPU blocks evicted to host memory; reloaded over PCIe.
  * Shared KV store: one cluster-wide LRU store (LMCache-style) reachable from
    every replica over the network; every computed block is written to it.
  * Disaggregated prefill: prefill-only replicas compute KV, ship it to a decode
    replica, which generates the remaining tokens.
  * Routing policies copied from production-stack semantics (see routing_logic.py):
    round_robin, session (consistent hash on session id), qps (least recent
    arrivals), prefix (128-char chunk trie), kvaware (max cached prefix),
    loadaware (score = hit_fraction - beta * relative_load).

What is NOT modelled
  * Exact GPU kernels, network congestion, controller message latency (kvaware
    reads cache state with zero lag), tokenizer/HTTP overhead, preemption.
  * Latency constants are a roofline estimate for an 8B fp16 model on an RTX 4090
    (see HW below and README). They are assumptions until measured on a free GPU.
"""
from __future__ import annotations

import bisect
import hashlib
import heapq
from collections import OrderedDict, deque
from dataclasses import dataclass, replace

from sim.workload import BLOCK, Req

GROUPS = ["STEM", "Humanities", "Social", "Other"]  # semantic routing pools


def _h(s: str) -> int:
    """Deterministic 48-bit hash (Python's str hash is salted per process)."""
    return int.from_bytes(hashlib.md5(s.encode()).digest()[:6], "little")


@dataclass(frozen=True)
class HW:
    kv_bytes_per_token: int = 131072      # 32 layers x 8 KV heads x 128 dim x (K,V) x fp16
    gpu_kv_tokens: int = 40_000           # 24 GB card - 16 GB weights - activations
    cpu_kv_tokens: int = 0                # host-memory offload tier (0 = off)
    remote_kv_tokens: int = 0             # shared KV store capacity (0 = off)
    prefill_tok_per_s: float = 6000.0     # ~2*8e9 FLOP/token at ~100 TFLOPS effective
    decode_base_s: float = 0.015          # weight streaming per decode step (16 GB / 1 TB/s)
    decode_ctx_s_per_tok: float = 1.3e-7  # KV read per context token per step
    chunk_budget: int = 2048              # prefill tokens admitted into one step
    max_running: int = 256
    pcie_GBps: float = 25.0               # host <-> GPU for offloaded blocks
    net_GBps: float = 25.0                # shared store and prefill->decode transfer
    net_lat_s: float = 2e-4
    qps_window_s: float = 60.0
    loadaware_beta: float = 1.0           # production-stack DEFAULT_LOADAWARE_BETA

    @property
    def block_bytes(self) -> int:
        return BLOCK * self.kv_bytes_per_token


class SharedStore:
    """Cluster-wide LRU of block keys (LMCache-like remote KV storage)."""

    def __init__(self, capacity_blocks: int):
        self.cap = capacity_blocks
        self.d: OrderedDict = OrderedDict()

    def __contains__(self, k):
        return k in self.d

    def touch(self, k):
        self.d.move_to_end(k)

    def put(self, k):
        if k in self.d:
            self.d.move_to_end(k)
            return
        self.d[k] = None
        if len(self.d) > self.cap:
            self.d.popitem(last=False)


class Replica:
    def __init__(self, idx: int, role: str, hw: HW):
        self.idx = idx
        self.role = role            # "mixed" | "prefill" | "decode"
        self.hw = hw
        self.gpu_cap = hw.gpu_kv_tokens // BLOCK
        self.cpu_cap = hw.cpu_kv_tokens // BLOCK
        self.idle: OrderedDict = OrderedDict()   # refcount 0 cached blocks, LRU order
        self.pinned: dict = {}                   # key -> refcount (>0)
        self.cpu: OrderedDict = OrderedDict()    # offloaded blocks
        self.reserved = 0                        # slots promised to admitted requests
        self.waiting: deque = deque()
        self.running: list = []
        self.inflight = 0
        self.scheduled = False
        self.arrivals: deque = deque()           # for qps routing
        self.busy_s = 0.0
        self.snap = frozenset()                  # lagged view of cached keys (controller index)
        self.snap_t = float("-inf")

    # ---------- cache primitives ----------
    def _make_room(self, need: int) -> bool:
        while len(self.idle) + len(self.pinned) + self.reserved + need > self.gpu_cap:
            if not self.idle:
                return False
            victim, _ = self.idle.popitem(last=False)
            if self.cpu_cap:
                self.cpu[victim] = None
                if len(self.cpu) > self.cpu_cap:
                    self.cpu.popitem(last=False)
        return True

    def _acquire(self, k):
        if k in self.idle:
            del self.idle[k]
            self.pinned[k] = 1
        else:
            self.pinned[k] += 1

    def _release(self, k):
        rc = self.pinned[k] - 1
        if rc == 0:
            del self.pinned[k]
            self.idle[k] = None
        else:
            self.pinned[k] = rc

    def _place(self, k):
        """Convert one reserved slot into a (pinned) cache entry for key k."""
        self.reserved -= 1
        if k in self.pinned:
            self.pinned[k] += 1
        elif k in self.idle:
            del self.idle[k]
            self.pinned[k] = 1
        else:
            self.pinned[k] = 1

    def _in_gpu(self, k) -> bool:
        return k in self.pinned or k in self.idle

    # ---------- routing helpers (no side effects) ----------
    def peek_match(self, r: Req, remote) -> int:
        """Longest cached prefix (in blocks) this replica can serve for r."""
        n = 0
        for k in r.keys[:r.n_prompt_blocks]:
            if self._in_gpu(k) or k in self.cpu or (remote is not None and k in remote):
                n += 1
            else:
                break
        return n

    def match_lagged(self, r: Req, remote, now: float, lag: float) -> int:
        """Matched blocks as seen by a controller index that refreshes every `lag` s."""
        if lag <= 0:
            return self.peek_match(r, remote)
        if now - self.snap_t >= lag:
            self.snap = frozenset(self.idle) | frozenset(self.pinned) | frozenset(self.cpu)
            self.snap_t = now
        n = 0
        for k in r.keys[:r.n_prompt_blocks]:
            if k in self.snap or (remote is not None and k in remote):
                n += 1
            else:
                break
        return n

    # ---------- admission ----------
    def try_admit(self, r: Req, now: float, cl: "Cluster") -> bool:
        hw = self.hw
        if self.role == "decode":
            need = r.n_total_blocks
            if not self._make_room(need):
                return False
            self.reserved += need
            r.held = []
            r.phase = "decode"
            return True

        keys_len = r.n_prompt_blocks if self.role == "prefill" else r.n_total_blocks
        matched = []  # (key, tier)
        for k in r.keys[:min(r.n_prompt_blocks, keys_len)]:
            if self._in_gpu(k):
                matched.append((k, "gpu"))
            elif k in self.cpu:
                matched.append((k, "cpu"))
            elif cl.remote is not None and k in cl.remote:
                matched.append((k, "remote"))
            else:
                break
        if matched and len(matched) * BLOCK >= r.prompt_len:
            matched.pop()  # always compute at least one prompt token
        n_gpu = sum(1 for _, t in matched if t == "gpu")
        reserve = keys_len - n_gpu

        pinned_now = [k for k, t in matched if t == "gpu"]
        claimed = [k for k, t in matched if t == "cpu"]
        for k in pinned_now:
            self._acquire(k)
        for k in claimed:          # protect matched CPU blocks from eviction below
            del self.cpu[k]
        if not self._make_room(reserve):
            for k in pinned_now:
                self._release(k)
            for k in claimed:
                self.cpu[k] = None
            return False
        self.reserved += reserve

        n_cpu = n_rem = 0
        for k, t in matched:
            if t == "gpu":
                continue
            if t == "cpu":
                n_cpu += 1
            else:
                cl.remote.touch(k)
                n_rem += 1
            self._place(k)
        r.held = [k for k, _ in matched]
        r.matched = len(matched)
        r.tok_gpu = n_gpu * BLOCK
        r.tok_cpu = n_cpu * BLOCK
        r.tok_remote = n_rem * BLOCK
        load = n_cpu * hw.block_bytes / (hw.pcie_GBps * 1e9)
        if n_rem:
            load += hw.net_lat_s + n_rem * hw.block_bytes / (hw.net_GBps * 1e9)
        r.load_ready = now + load
        r.rem_prefill = r.prompt_len - (r.tok_gpu + r.tok_cpu + r.tok_remote)
        r.phase = "prefill"
        r.generated = 0
        return True

    def _prefill_done(self, r: Req, cl: "Cluster"):
        for k in r.keys[r.matched:r.n_prompt_blocks]:
            self._place(k)
            r.held.append(k)
        if cl.remote is not None:
            for k in r.keys[:r.n_prompt_blocks]:
                cl.remote.put(k)

    def _finish_mixed(self, r: Req, end: float, cl: "Cluster"):
        for k in r.keys[r.n_prompt_blocks:]:
            self._place(k)
            r.held.append(k)
        if cl.remote is not None:
            for k in r.keys[r.n_prompt_blocks:]:
                cl.remote.put(k)
        for k in r.held:
            self._release(k)
        r.held = []

    # ---------- one engine step ----------
    def step(self, now: float, cl: "Cluster"):
        hw = self.hw
        while self.waiting and len(self.running) < hw.max_running:
            if not self.try_admit(self.waiting[0], now, cl):
                break
            self.running.append(self.waiting.popleft())
        if not self.running:
            return None

        budget = hw.chunk_budget
        ptok = 0
        decoders, prefilling = [], []
        next_wake = None
        for r in self.running:
            if r.phase == "prefill":
                if r.load_ready > now:
                    next_wake = r.load_ready if next_wake is None else min(next_wake, r.load_ready)
                    continue
                if budget > 0 and r.rem_prefill > 0:
                    take = min(budget, r.rem_prefill)
                    r.rem_prefill -= take
                    budget -= take
                    ptok += take
                    prefilling.append(r)
            elif r.phase == "decode":
                decoders.append(r)

        if ptok == 0 and not decoders:
            return next_wake

        ctx = sum(r.prompt_len + r.generated for r in decoders)
        dt = hw.decode_base_s + hw.decode_ctx_s_per_tok * ctx + ptok / hw.prefill_tok_per_s
        end = now + dt
        self.busy_s += dt

        done = set()
        for r in decoders:
            r.generated += 1
            if r.generated >= r.out_len:
                r.finish_t = end
                self._release_decode(r, cl)
                cl.complete(r, end, self)
                done.add(id(r))

        for r in prefilling:
            if r.rem_prefill > 0:
                continue
            self._prefill_done(r, cl)
            r.first_t = end
            r.generated = 1
            if self.role == "prefill" and r.out_len > 1:
                done.add(id(r))
                cl.handoff(r, end, self)
            elif r.out_len <= 1:
                done.add(id(r))
                r.finish_t = end
                if self.role == "prefill":
                    self._finish_mixed(r, end, cl)
                else:
                    self._finish_mixed(r, end, cl)
                cl.complete(r, end, self)
            else:
                # mixed replica: first token emitted now, rest decoded in later steps
                r.phase = "decode"

        self.running = [r for r in self.running if id(r) not in done]
        return end

    def _release_decode(self, r: Req, cl: "Cluster"):
        if self.role == "decode":
            self.reserved -= r.n_total_blocks
        else:
            self._finish_mixed(r, r.finish_t, cl)


class Cluster:
    def __init__(self, hw: HW, n_mixed: int = 0, n_prefill: int = 0, n_decode: int = 0,
                 policy: str = "kvaware", use_remote: bool = False,
                 cpu_offload: bool = False, index_lag: float = 0.0,
                 sem_bonus: float = 0.0):
        hw = replace(hw, cpu_kv_tokens=hw.cpu_kv_tokens if cpu_offload else 0)
        self.hw = hw
        self.policy = policy
        self.index_lag = index_lag      # Q2: KV index staleness (s); 0 = live
        self.sem_bonus = sem_bonus      # Q3: soft bonus for the predicted semantic pool
        self.remote = SharedStore(hw.remote_kv_tokens // BLOCK) if use_remote else None
        self.replicas: list[Replica] = []
        for i in range(n_mixed):
            self.replicas.append(Replica(i, "mixed", hw))
        for i in range(n_prefill):
            self.replicas.append(Replica(len(self.replicas), "prefill", hw))
        for i in range(n_decode):
            self.replicas.append(Replica(len(self.replicas), "decode", hw))
        self.entry = [r for r in self.replicas if r.role in ("mixed", "prefill")]
        self.decoders = [r for r in self.replicas if r.role == "decode"]
        self.rr = 0
        self.events: list = []
        self.seq = 0
        self.done: list[Req] = []
        # prefix-aware trie: prefix-chunk-key -> replica index (last writer wins)
        self.trie: dict = {}
        self.ring = self._build_ring(len(self.entry))

    # ---------- event plumbing ----------
    def _push(self, t, kind, obj):
        self.seq += 1
        heapq.heappush(self.events, (t, self.seq, kind, obj))

    def _ensure_step(self, rep: Replica, t: float):
        if not rep.scheduled:
            rep.scheduled = True
            self._push(t, "step", rep)

    def handoff(self, r: Req, end: float, src: Replica):
        """Disaggregated prefill: ship KV from a prefill replica to a decode replica."""
        src.inflight -= 1
        for k in r.held:
            src._release(k)
        r.held = []
        dst = min(self.decoders, key=lambda d: d.inflight)
        dst.inflight += 1
        r.dreplica = dst.idx
        xfer = r.n_prompt_blocks * self.hw.block_bytes / (self.hw.net_GBps * 1e9) + self.hw.net_lat_s
        self._push(end + xfer, "deliver", (r, dst))

    def complete(self, r: Req, end: float, rep: Replica):
        rep.inflight -= 1
        r.finish_t = end
        self.done.append(r)

    # ---------- routing ----------
    @staticmethod
    def _build_ring(n: int, vnodes: int = 100):
        ring = []
        for i in range(n):
            for v in range(vnodes):
                h = _h(f"node-{i}-{v}")
                ring.append((h, i))
        ring.sort()
        return ring

    def _ring_lookup(self, key: str, n: int) -> int:
        h = _h(f"key-{key}")
        j = bisect.bisect(self.ring, (h, 10 ** 9))
        return self.ring[j % len(self.ring)][1]

    def _qps_count(self, rep: Replica, t: float) -> int:
        win = self.hw.qps_window_s
        while rep.arrivals and rep.arrivals[0] < t - win:
            rep.arrivals.popleft()
        return len(rep.arrivals)

    def _qps_pick(self, t: float) -> Replica:
        best, best_q = None, None
        for rep in self.entry:
            q = self._qps_count(rep, t)
            if best is None or q < best_q:
                best, best_q = rep, q
        return best

    def _fallback(self, r: Req, t: float) -> Replica:
        if r.session is not None:
            return self.entry[self._ring_lookup(r.session, len(self.entry))]
        return self._qps_pick(t)

    def route(self, r: Req, t: float) -> Replica:
        pol = self.policy
        entry = self.entry
        if pol == "round_robin":
            rep = entry[self.rr % len(entry)]
            self.rr += 1
            return rep
        if pol == "session":
            return self._fallback(r, t)
        if pol == "qps":
            return self._qps_pick(t)
        if pol == "prefix":
            hit = None
            keys = r.char_keys
            for i, k in enumerate(keys):
                if k in self.trie:
                    hit = self.trie[k]
                else:
                    break
            rep = entry[hit] if hit is not None else self._fallback(r, t)
            for k in keys:
                self.trie[k] = rep.idx
            return rep
        if pol in ("semantic", "semantic_oracle"):
            grp = r.pred_category if pol == "semantic" else r.category
            if grp is None:
                return self._fallback(r, t)
            return entry[GROUPS.index(grp) % len(entry)]
        matched = {rep.idx: rep.match_lagged(r, self.remote, t, self.index_lag) for rep in entry}
        if pol == "kvaware":
            best = max(matched.values())
            if best == 0:
                return self._fallback(r, t)
            cands = [entry[i] for i, m in matched.items() if m == best]
            return min(cands, key=lambda rep: (rep.inflight, self._qps_count(rep, t)))
        if pol in ("loadaware", "loadaware_sem"):
            if max(matched.values()) == 0:
                return self._fallback(r, t)
            loads = {rep.idx: rep.inflight for rep in entry}
            mean = sum(loads.values()) / len(loads)
            beta = self.hw.loadaware_beta
            target = None
            if pol == "loadaware_sem" and r.pred_category in GROUPS:
                target = GROUPS.index(r.pred_category) % len(entry)
            best_rep, best_key = None, None
            for rep in entry:
                frac = min(matched[rep.idx] * BLOCK, r.prompt_len) / max(r.prompt_len, 1)
                rel = (loads[rep.idx] - mean) / max(1.0, mean)
                bonus = self.sem_bonus if rep.idx == target else 0.0
                key = (frac - beta * rel + bonus, -self._qps_count(rep, t))  # ties -> lower QPS
                if best_key is None or key > best_key:
                    best_rep, best_key = rep, key
            return best_rep
        raise ValueError(pol)

    # ---------- main loop ----------
    def run(self, reqs: list[Req], t_end: float):
        for r in reqs:
            self._push(r.arrival, "arr", r)
        while self.events:
            t, _, kind, obj = heapq.heappop(self.events)
            if t > t_end:
                break
            if kind == "arr":
                rep = self.route(obj, t)
                obj.replica = rep.idx
                rep.inflight += 1
                rep.arrivals.append(t)
                rep.waiting.append(obj)
                self._ensure_step(rep, t)
            elif kind == "step":
                rep = obj
                rep.scheduled = False
                nt = rep.step(t, self)
                if nt is not None:
                    rep.scheduled = True
                    self._push(max(nt, t + 1e-9), "step", rep)
            elif kind == "deliver":
                r, dst = obj
                r.phase = "decode"
                r.generated = 1
                dst.waiting.append(r)
                self._ensure_step(dst, t)
        return self.done
