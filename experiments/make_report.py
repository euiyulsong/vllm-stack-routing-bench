"""Aggregate raw CSVs into markdown tables and figures.

Reads  results/raw/{routing,beta,features,semantic}.csv
Writes results/tables/*.md and results/figures/*.png

Colors: categorical slots 1-6 of the validated reference palette (light mode),
see the validator run recorded in README.md. Each line also carries a direct
end label and a distinct marker, because three of those slots are below 3:1
contrast on the light surface (relief rule).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"
TAB = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"

PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MARK = ["o", "s", "^", "D", "v", "P"]
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e6e5e1"

POLICY_LABEL = {
    "round_robin": "round-robin",
    "session": "session hash",
    "qps": "least QPS",
    "prefix": "prefix-aware",
    "kvaware": "KV-cache-aware",
    "loadaware": "load-aware",
    "semantic": "semantic (predicted)",
    "semantic_oracle": "semantic (oracle)",
}
POLICY_ORDER = ["round_robin", "session", "qps", "prefix", "kvaware", "loadaware"]
SEM_ORDER = ["qps", "prefix", "kvaware", "loadaware", "semantic", "semantic_oracle"]


def style(ax):
    ax.set_facecolor("white")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def agg(df, keys, metrics):
    g = df.groupby(keys)[metrics]
    m = g.mean()
    s = g.std()
    return m, s


def fmt_pm(m, s, nd=2):
    return f"{m:.{nd}f} ± {0 if pd.isna(s) else s:.{nd}f}"


def line_panel(ax, df, x, y, series, order, labels, ylog=False):
    """Plot one line per series with end labels; overlapping labels are pushed apart."""
    import math

    ends = []
    for i, name in enumerate(order):
        sub = df[df[series] == name].sort_values(x)
        if sub.empty:
            continue
        ax.plot(sub[x], sub[y], color=PAL[i % 6], marker=MARK[i % 6], markersize=5,
                linewidth=1.8)
        ends.append((name, sub[x].iloc[-1], sub[y].iloc[-1], i))
    tf = (lambda v: math.log10(v)) if ylog else (lambda v: v)
    ends.sort(key=lambda e: tf(e[2]))
    ys = [tf(e[2]) for e in ends]
    if ys:
        span = (max(ys) - min(ys)) or 1.0
        gap = 0.075 * max(span, 1e-9) if span > 0 else 0.0
        placed = []
        for k, e in enumerate(ends):
            ly = ys[k]
            if placed and ly - placed[-1] < gap:
                ly = placed[-1] + gap
            placed.append(ly)
        for (name, xe, ye, i), ly in zip(ends, placed):
            val = 10 ** ly if ylog else ly
            ax.annotate(labels[name], xy=(xe, ye), xytext=(xe, val),
                        textcoords="data", xycoords="data", fontsize=8, color=TEXT,
                        va="center", ha="left",
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5,
                                        shrinkA=0, shrinkB=2) if abs(tf(ye) - ly) > 1e-9 else None)
    if ylog:
        ax.set_yscale("log")


def routing_tables_and_figs(df):
    m, s = agg(df, ["workload", "rate_req_s", "policy"],
               ["ttft_p99_s", "hit_ratio", "slo_attainment", "ttft_p50_s", "load_cv"])
    rows = []
    for (wl, rate, pol), r in m.iterrows():
        sr = s.loc[(wl, rate, pol)]
        rows.append({
            "workload": wl, "rate_req_s": rate, "policy": POLICY_LABEL[pol],
            "TTFT p50 (s)": fmt_pm(r.ttft_p50_s, sr.ttft_p50_s),
            "TTFT p99 (s)": fmt_pm(r.ttft_p99_s, sr.ttft_p99_s),
            "cache hit (token %)": fmt_pm(100 * r.hit_ratio, 100 * sr.hit_ratio, 1),
            "SLO attainment (%)": fmt_pm(100 * r.slo_attainment, 100 * sr.slo_attainment, 1),
            "replica load CV": fmt_pm(r.load_cv, sr.load_cv, 3),
        })
    out = pd.DataFrame(rows)
    (TAB / "routing_policies.md").write_text(out.to_markdown(index=False))

    # figure: TTFT p99 and hit ratio per workload vs rate
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharex="col")
    labels = {p: POLICY_LABEL[p] for p in POLICY_ORDER}
    for j, wl in enumerate(["chat", "mmlu", "mixed"]):
        sub = df[df.workload == wl].groupby(["rate_req_s", "policy"], as_index=False).mean(numeric_only=True)
        sub["policy_order"] = sub.policy
        ax = axes[0, j]
        style(ax)
        line_panel(ax, sub, "rate_req_s", "ttft_p99_s", "policy", POLICY_ORDER, labels, ylog=True)
        ax.set_title(f"{wl}: TTFT p99", color=TEXT, fontsize=11, loc="left")
        ax.set_ylabel("seconds (log)" if j == 0 else "", color=MUTED)
        ax2 = axes[1, j]
        style(ax2)
        line_panel(ax2, sub, "rate_req_s", "hit_ratio", "policy", POLICY_ORDER, labels)
        ax2.set_title(f"{wl}: prompt tokens served from cache", color=TEXT, fontsize=11, loc="left")
        ax2.set_ylim(0, 1)
        ax2.set_xlabel("offered load (req/s, 4 replicas)", color=MUTED)
        ax2.set_ylabel("hit ratio" if j == 0 else "", color=MUTED)
        ax.set_xlabel("")
    fig.suptitle("Router policies on a 4-replica cluster (mean of 5 seeds)", color=TEXT, fontsize=13,
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 0.97, 0.96))
    fig.savefig(FIG / "routing_policies.png", dpi=150)
    plt.close(fig)


def beta_outputs(df):
    m, s = agg(df, ["workload", "rate_req_s", "beta"], ["ttft_p99_s", "load_cv", "hit_ratio"])
    rows = []
    for (wl, rate, beta), r in m.iterrows():
        sr = s.loc[(wl, rate, beta)]
        rows.append({"workload": wl, "rate_req_s": rate, "beta": beta,
                     "TTFT p99 (s)": fmt_pm(r.ttft_p99_s, sr.ttft_p99_s),
                     "replica load CV": fmt_pm(r.load_cv, sr.load_cv, 3),
                     "cache hit (%)": fmt_pm(100 * r.hit_ratio, 100 * sr.hit_ratio, 1)})
    (TAB / "loadaware_beta.md").write_text(pd.DataFrame(rows).to_markdown(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, wl in zip(axes, ["chat", "mixed", "mmlu"]):
        sub = df[df.workload == wl].groupby("beta", as_index=False).mean(numeric_only=True)
        style(ax)
        ax.plot(sub.beta, sub.ttft_p99_s, color=PAL[0], marker="o", linewidth=1.8, label="TTFT p99 (s)")
        ax.set_yscale("log")
        ax.set_xscale("symlog", linthresh=0.25)
        ax.set_title(f"{wl}", color=TEXT, fontsize=11, loc="left")
        ax.set_xlabel("beta (load weight)", color=MUTED)
        ax.set_ylabel("TTFT p99 (s, log)", color=PAL[0])
        ax2 = ax.twinx()
        ax2.plot(sub.beta, sub.load_cv, color=PAL[1], marker="s", linewidth=1.8)
        ax2.set_ylabel("replica load CV", color=PAL[1])
        ax2.spines["top"].set_visible(False)
        ax2.tick_params(colors=TEXT, labelsize=9)
    fig.suptitle("Load-aware weight sweep: TTFT p99 (blue) vs load imbalance (orange)",
                 color=TEXT, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG / "loadaware_beta.png", dpi=150)
    plt.close(fig)


FEATURE_ORDER = [
    "colocated_base", "colocated+cpu_offload", "colocated+shared_store",
    "colocated+offload+shared", "disagg_3P+1D", "disagg_2P+2D", "disagg_1P+3D",
    "disagg_2P+2D+offload+shared",
]


def features_outputs(df):
    m, s = agg(df, ["workload", "rate_req_s", "config"],
               ["ttft_p50_s", "ttft_p99_s", "tpot_p50_ms", "hit_ratio", "slo_attainment",
                "req_per_s", "out_tok_per_s", "completed_frac"])
    rows = []
    for (wl, rate, cfg), r in m.iterrows():
        sr = s.loc[(wl, rate, cfg)]
        rows.append({
            "workload": wl, "rate_req_s": rate, "config": cfg,
            "TTFT p50 (s)": fmt_pm(r.ttft_p50_s, sr.ttft_p50_s),
            "TTFT p99 (s)": fmt_pm(r.ttft_p99_s, sr.ttft_p99_s),
            "TPOT p50 (ms)": fmt_pm(r.tpot_p50_ms, sr.tpot_p50_ms, 1),
            "cache hit (%)": fmt_pm(100 * r.hit_ratio, 100 * sr.hit_ratio, 1),
            "SLO attainment (%)": fmt_pm(100 * r.slo_attainment, 100 * sr.slo_attainment, 1),
            "requests/s": fmt_pm(r.req_per_s, sr.req_per_s),
            "completed (%)": fmt_pm(100 * r.completed_frac, 100 * sr.completed_frac, 1),
        })
    (TAB / "features.md").write_text(pd.DataFrame(rows).to_markdown(index=False))

    # figure: at each workload's 3rd rate level, bars of TTFT p99 and TPOT p50
    fig, axes = plt.subplots(1, 4, figsize=(17, 5.2), gridspec_kw={"width_ratios": [1.3, 1, 1.3, 1]})
    panels = [("chat", 12, "ttft_p99_s", "TTFT p99 (s)"),
              ("chat", 12, "tpot_p50_ms", "TPOT p50 (ms)"),
              ("mixed", 24, "ttft_p99_s", "TTFT p99 (s)"),
              ("mixed", 24, "tpot_p50_ms", "TPOT p50 (ms)")]
    for ax, (wl, rate, col, title) in zip(axes, panels):
        sub = df[(df.workload == wl) & (df.rate_req_s == rate)].groupby("config").mean(numeric_only=True)
        sub = sub.reindex([c for c in FEATURE_ORDER if c in sub.index])
        ys = range(len(sub))
        style(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.grid(axis="y", visible=False)
        ax.barh(list(ys), sub[col], color=PAL[0], height=0.6)
        for y, v in zip(ys, sub[col]):
            ax.text(v, y, f" {v:.2f}" if col == "ttft_p99_s" else f" {v:.1f}", va="center",
                    fontsize=8, color=TEXT)
        ax.set_yticks(list(ys))
        ax.set_yticklabels(list(sub.index) if ax is axes[0] or ax is axes[2] else [""] * len(sub),
                           fontsize=8, color=TEXT)
        ax.invert_yaxis()
        ax.set_title(f"{wl}, {rate} req/s\n{title}", color=TEXT, fontsize=10, loc="left")
    fig.suptitle("Feature configurations on 4 GPUs (mean of 5 seeds)", color=TEXT, fontsize=13,
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 0.99, 0.93))
    fig.savefig(FIG / "features_bars.png", dpi=150)
    plt.close(fig)


def semantic_outputs(df, clf):
    m, s = agg(df, ["rate_req_s", "policy"], ["ttft_p99_s", "hit_ratio", "load_cv", "slo_attainment"])
    rows = []
    for (rate, pol), r in m.iterrows():
        sr = s.loc[(rate, pol)]
        rows.append({"rate_req_s": rate, "policy": POLICY_LABEL[pol],
                     "TTFT p99 (s)": fmt_pm(r.ttft_p99_s, sr.ttft_p99_s),
                     "cache hit (%)": fmt_pm(100 * r.hit_ratio, 100 * sr.hit_ratio, 1),
                     "replica load CV": fmt_pm(r.load_cv, sr.load_cv, 3),
                     "SLO attainment (%)": fmt_pm(100 * r.slo_attainment, 100 * sr.slo_attainment, 1)})
    out = pd.DataFrame(rows)
    body = out.to_markdown(index=False)
    cls = (f"Classifier (MMLU test split, n={clf['test_rows']}): accuracy {clf['accuracy']:.3f}, "
           f"macro-F1 {clf['macro_f1']:.3f}, majority baseline {clf['majority_baseline_accuracy']:.3f}, "
           f"CPU latency {clf['per_request_latency_ms_cpu']:.3f} ms/request.\n\n")
    (TAB / "semantic.md").write_text(cls + body)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    labels = {p: POLICY_LABEL[p] for p in SEM_ORDER}
    sub = df.groupby(["rate_req_s", "policy"], as_index=False).mean(numeric_only=True)
    for ax, col, title in [(axes[0], "ttft_p99_s", "TTFT p99 (s, log)"),
                           (axes[1], "hit_ratio", "cache hit ratio")]:
        style(ax)
        line_panel(ax, sub, "rate_req_s", col, "policy", SEM_ORDER, labels,
                   ylog=(col == "ttft_p99_s"))
        ax.set_title(f"MMLU, 4 replicas: {title}", color=TEXT, fontsize=11, loc="left")
        ax.set_xlabel("offered load (req/s)", color=MUTED)
    axes[1].set_ylim(0, 1)
    fig.tight_layout(rect=(0, 0, 0.9, 1))
    fig.savefig(FIG / "semantic_routing.png", dpi=150)
    plt.close(fig)


def main():
    import json

    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    clf = json.loads((ROOT / "results" / "semantic" / "classifier_metrics.json").read_text())
    routing = pd.read_csv(RAW / "routing.csv")
    routing_tables_and_figs(routing)
    beta_outputs(pd.read_csv(RAW / "beta.csv"))
    features_outputs(pd.read_csv(RAW / "features.csv"))
    semantic_outputs(pd.read_csv(RAW / "semantic.csv"), clf)
    print("tables:", sorted(p.name for p in TAB.iterdir()))
    print("figures:", sorted(p.name for p in FIG.iterdir()))


if __name__ == "__main__":
    main()
