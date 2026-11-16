"""
CSC 4501 Computer Networks
QuantumNet: Routing and Congestion in a Distributed Quantum Network

STUDENT STARTER FILE

No prior quantum-computing knowledge is required. In this project:
- QPUs are network nodes.
- Quantum links are graph edges.
- An entanglement request is treated as an end-to-end network request.
- The focus is routing, reliability, capacity, queueing, congestion, and throughput.

Complete the TODO sections. Do not rename required functions unless instructed.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

POLICIES = ["shortest_hop", "min_latency", "max_reliability", "student_policy"]
OFFERED_LOADS = [0.02, 0.05, 0.10, 0.15]  # requests per ms

# Student policy parameters
STUDENT_ALPHA = 1.0   # weight on delay (ms per ms)
STUDENT_BETA = 30.0   # ms of delay traded for one unit of -ln(p)


@dataclass
class Request:
    request_id: int
    arrival_ms: float
    src: str
    dst: str


@dataclass
class RequestResult:
    request_id: int
    policy: str
    src: str
    dst: str
    arrival_ms: float
    finish_ms: float
    latency_ms: float
    waiting_ms: float
    hop_count: int
    success: bool
    path: str


# TASK 1: Build and validate the network topology - 10 points


def load_topology(path: str | Path) -> nx.Graph:
    """
    Read data/topology.json and return an undirected NetworkX graph.

    Required checks:
      1. 0 < success_prob <= 1
      2. capacity >= 1
      3. the graph is connected
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    g = nx.Graph()
    g.add_nodes_from(data["nodes"])

    for idx, link in enumerate(data["links"], start=1):
        u, v = link["u"], link["v"]
        if u not in g or v not in g:
            raise ValueError(f"Link {u}-{v} references an unknown node")

        delay = float(link["delay_ms"])
        p = float(link["success_prob"])
        capacity = int(link["capacity"])

        if not (0.0 < p <= 1.0):
            raise ValueError(f"Link {u}-{v}: success_prob must be in (0, 1], got {p}")
        if capacity < 1:
            raise ValueError(f"Link {u}-{v}: capacity must be >= 1, got {capacity}")
        if delay < 0:
            raise ValueError(f"Link {u}-{v}: delay_ms must be >= 0, got {delay}")

        # Keep every attribute from the JSON 
        attrs = {k: val for k, val in link.items() if k not in ("u", "v")}
        attrs.update(delay_ms=delay, success_prob=p, capacity=capacity)
        attrs.setdefault("link_id", f"L{idx}")  # L1-L9 in file order
        g.add_edge(u, v, **attrs)

    if g.number_of_nodes() == 0 or not nx.is_connected(g):
        raise ValueError("Topology must be a non-empty connected graph")

    return g


def draw_topology(g: nx.Graph, output_path: str | Path) -> None:
    """Provided visualization helper. You may improve the layout if desired."""
    pos = {
        "Q1": (0.0, 1.0),
        "Q2": (2.0, 1.8),
        "Q3": (2.0, 0.2),
        "Q4": (4.2, 1.8),
        "Q5": (4.2, 0.2),
        "Q6": (6.2, 1.0),
    }

    ordered_edges = [
        ("Q1", "Q2"), ("Q1", "Q3"), ("Q2", "Q3"),
        ("Q2", "Q4"), ("Q3", "Q4"), ("Q3", "Q5"),
        ("Q4", "Q5"), ("Q4", "Q6"), ("Q5", "Q6"),
    ]
    labels = {}
    for i, (u, v) in enumerate(ordered_edges, start=1):
        a = g[u][v]
        labels[(u, v)] = (
            f"L{i}\n{a['delay_ms']:g} ms, p={a['success_prob']:.2f}, c={a['capacity']}"
        )

    plt.figure(figsize=(9.6, 4.4))
    nx.draw_networkx_nodes(g, pos=pos, node_size=1050)
    nx.draw_networkx_edges(g, pos=pos, width=1.4)
    nx.draw_networkx_labels(g, pos=pos, font_size=10)
    nx.draw_networkx_edge_labels(
        g, pos=pos, edge_labels=labels, font_size=7, rotate=False,         bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9),
    )
    plt.title("Six-QPU Distributed Network Topology")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close()


# ============================================================
# TASK 2 — Generate network traffic [10 points]
# ============================================================

def generate_requests(
    g: nx.Graph,
    n_requests: int,
    arrival_rate_per_ms: float,
    rng: random.Random,
) -> List[Request]:
    """
    Generate requests using exponentially distributed inter-arrival times.

    Hint:
        rng.expovariate(arrival_rate_per_ms)

    Every request needs:
        request_id, arrival_ms, src, dst

    Source and destination must be different.
    """
    if arrival_rate_per_ms <= 0:
        raise ValueError("arrival_rate_per_ms must be positive")

    nodes = sorted(g.nodes())
    if len(nodes) < 2:
        raise ValueError("Need at least two nodes to generate requests")

    requests: List[Request] = []
    t = 0.0
    for i in range(n_requests):
        # Poisson arrivals <=> exponential inter-arrival times.
        t += rng.expovariate(arrival_rate_per_ms)
        # Sampling 2 distinct nodes guarantees src != dst (uniform over pairs).
        src, dst = rng.sample(nodes, 2)
        requests.append(Request(request_id=i, arrival_ms=t, src=src, dst=dst))

    return requests


# ============================================================
# TASK 3 — Implement routing policies [30 points]
# ============================================================

def edge_weight(
    policy: str,
    attrs: dict,
    alpha: float = STUDENT_ALPHA,
    beta: float = STUDENT_BETA,
) -> float:
    """
    Required policies:

    shortest_hop:
        Minimum number of links.

    min_latency:
        Minimum total delay.

    max_reliability:
        Maximum product of link success probabilities.
        Convert this into an additive Dijkstra edge cost.

    student_policy:
        Your own policy using at least TWO link properties.
        Explain the formula and parameters in your report.
    """
    delay = attrs["delay_ms"]
    p = attrs["success_prob"]
    capacity = attrs["capacity"]

    if policy == "shortest_hop":
        # Every link costs 1, so path cost = number of links.
        return 1.0

    if policy == "min_latency":
        # Path cost = sum of link delays.
        return float(delay)

    if policy == "max_reliability":
        # max prod(p_e)  <=>  max sum(ln p_e)  <=>  min sum(-ln p_e).
        # ln is strictly increasing, so the ordering of paths is preserved,
        # and -ln p_e >= 0 for 0 < p_e <= 1, so Dijkstra is valid.
        return -math.log(p)

    if policy == "student_policy":
        # TODO 3D
               # Reliability-penalized latency (uses delay AND success probability):
        #     w_e = alpha * d_e + beta * (-ln p_e)
        # Summed over a path: alpha * total_delay + beta * (-ln P_success).
        # beta is an exchange rate in ms per unit of -ln(p). A detour that
        # adds Delta_d ms is taken only if it multiplies the end-to-end
        # success probability by more than exp(Delta_d / beta).
        # beta = 30: +5 ms must buy >18% relative reliability, +7 ms >26%.
        return alpha * float(delay) + beta * (-math.log(p))

    raise ValueError(f"Unknown policy: {policy}")


def choose_path(
    g: nx.Graph,
    src: str,
    dst: str,
    policy: str,
    alpha: float = STUDENT_ALPHA,
    beta: float = STUDENT_BETA,
) -> List[str]:
    """Select a path using Dijkstra's algorithm and edge_weight(...)."""
    if src == dst:
        return [src]

    def weight(u: str, v: str, attrs: dict) -> float:
        return edge_weight(policy, attrs, alpha=alpha, beta=beta)

    return nx.dijkstra_path(g, src, dst, weight=weight)

# ============================================================
# TASK 4 — Model link capacity and queueing [15 points]
# ============================================================

class LinkState:
    """
    A link has a finite number of simultaneous service slots.

    Recommended representation:
        store one next-available time per slot in a min-heap.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity

        # TODO 4A:
        # One next free at for time per slot; all slots start free at t = 0
        self.available_times = [0.0] * capacity
        heapq.heapify(self.available_times)
        self.busy_time = 0.0
        self.jobs = 0

    def reserve(self, ready_ms: float, service_ms: float) -> Tuple[float, float]:
        """
        Return (start_time, finish_time).

        start_time = max(ready_ms, earliest_slot_free_time)
        finish_time = start_time + service_ms
        """
        # TODO 4B:
        earliest_free = heapq.heappop(self.available_times)
        start = max(ready_ms, earliest_free)
        finish = start + service_ms
        heapq.heappush(self.available_times, finish)

        self.busy_time += service_ms
        self.jobs += 1
        return start, finish


def canonical_edge(u: str, v: str) -> Tuple[str, str]:
    return tuple(sorted((u, v)))


def make_link_states(g: nx.Graph) -> Dict[Tuple[str, str], LinkState]:
    """Create one LinkState object for each network link."""
    # TODO 4C
        return {
        canonical_edge(u, v): LinkState(int(attrs["capacity"]))
        for u, v, attrs in g.edges(data=True)
    }

# ============================================================
# TASK 5 — End-to-end simulation and metrics [20 points]
# ============================================================

def simulate(
    g: nx.Graph,
    requests: List[Request],
    policy: str,
    seed: int,
    alpha: float = STUDENT_ALPHA,
    beta: float = STUDENT_BETA,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each request:
      1. choose a route;
      2. traverse links in path order;
      3. reserve link capacity;
      4. accumulate queue waiting time;
      5. sample link success/failure;
      6. stop after the first failed link;
      7. record the result.

    A link succeeds when:
        rng.random() <= success_prob
    """
    # TODO 5A:
    # Initialize random generator and link states.
    # Fresh RNG + empty links for every policy and trace run
    rng = random.Random(seed)
    links = make_link_states(g)
    path_cache: Dict[Tuple[str, str], List[str]] = {}

    # TODO 5B:
    # Process requests in increasing arrival-time order
    results: List[RequestResult] = []
    for req in sorted(requests, key=lambda r: (r.arrival_ms, r.request_id)):
        key = (req.src, req.dst)
        if key not in path_cache:  # static weights -> path never changes
            path_cache[key] = choose_path(g, req.src, req.dst, policy, alpha, beta)
        path = path_cache[key]

    # TODO 5C:
    t = req.arrival_ms
        waiting = 0.0
        success = True
        hops_used = 0

        for u, v in zip(path[:-1], path[1:]):
            attrs = g[u][v]
            start, finish = links[canonical_edge(u, v)].reserve(t, attrs["delay_ms"])
            waiting += start - t
            t = finish
            hops_used += 1
            if rng.random() > attrs["success_prob"]:
                success = False
                break  # first failed link ends the request

        results.append(
            RequestResult(
                request_id=req.request_id,
                policy=policy,
                src=req.src,
                dst=req.dst,
                arrival_ms=req.arrival_ms,
                finish_ms=t,
                latency_ms=t - req.arrival_ms,
                waiting_ms=waiting,
                hop_count=len(path) - 1,
                success=success,
                path="-".join(path),
            )
        )

    # 5C: per-request and per-link tables.
    request_df = pd.DataFrame([asdict(r) for r in results])
    sim_time_ms = float(request_df["finish_ms"].max()) if len(request_df) else 0.0

    link_rows = []
    for u, v, attrs in sorted(g.edges(data=True), key=lambda e: e[2]["link_id"]):
        state = links[canonical_edge(u, v)]
        denom = state.capacity * sim_time_ms
        link_rows.append(
            {
                "policy": policy,
                "link": attrs["link_id"],
                "endpoints": f"{u}-{v}",
                "capacity": state.capacity,
                "jobs": state.jobs,
                "busy_ms": state.busy_time,
                # fraction of total slot-time the link was in service
                "utilization": state.busy_time / denom if denom > 0 else 0.0,
            }
        )
    link_df = pd.DataFrame(link_rows)

    return request_df, link_df

def summarize(request_df: pd.DataFrame, offered_load: float) -> dict:
    """
    Compute:
      policy
      offered_load
      success_rate
      avg_latency_ms
      avg_waiting_ms
      avg_hops
      throughput_req_per_s
    """
    # TODO 5D
    n_total = len(request_df)
    ok = request_df[request_df["success"]]
    n_success = len(ok)
    # Simulation starts at t = 0 (first arrival is at t > 0) and ends when the
    # last request finishes (successfully or not).
    sim_time_ms = float(request_df["finish_ms"].max()) if n_total else 0.0

    return {
        "policy": request_df["policy"].iloc[0] if n_total else "",
        "offered_load": offered_load,
        "n_requests": n_total,
        "n_success": n_success,
        "success_rate": n_success / n_total if n_total else 0.0,
        # End-to-end latency of delivered requests (arrival -> final link done).
        "avg_latency_ms": float(ok["latency_ms"].mean()) if n_success else float("nan"),
        # Queueing delay averaged over all requests.
        "avg_waiting_ms": float(request_df["waiting_ms"].mean()) if n_total else 0.0,
        "avg_hops": float(request_df["hop_count"].mean()) if n_total else 0.0,
        "sim_time_ms": sim_time_ms,
        "throughput_req_per_s": n_success / (sim_time_ms / 1000.0) if sim_time_ms > 0 else 0.0,
    }


# ============================================================
# TASK 6 — Experimental evaluation and report [15 points]
# ============================================================

def run_experiments(
    g: nx.Graph,
    output_dir: Path,
    n_requests: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Evaluate:
        shortest_hop
        min_latency
        max_reliability
        student_policy

    Offered loads:
        0.02, 0.05, 0.10, 0.15 requests/ms

    Important:
    At one offered-load value, ALL policies must receive the SAME
    generated request trace for a fair comparison.
    """
    # TODO 6A:
    # - Generate one request trace per load.
    # - Run all four policies on the same trace.
    # - Save per-request and per-link CSV files.
    # - Save and return summary.csv.
      output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    all_requests = []
    all_links = []

    for load in OFFERED_LOADS:
        # One trace per load, seeded from the base seed and the load value.
        load_key = int(round(load * 10000))
        trace_rng = random.Random(seed * 100003 + load_key)
        trace = generate_requests(g, n_requests, load, trace_rng)

        for policy in POLICIES:
            # Same trace and same link-outcome seed for every policy.
            req_df, link_df = simulate(g, trace, policy, seed=seed * 7919 + load_key)
            req_df.insert(1, "offered_load", load)
            link_df.insert(1, "offered_load", load)
            all_requests.append(req_df)
            all_links.append(link_df)
            summaries.append(summarize(req_df, load))

    pd.concat(all_requests, ignore_index=True).to_csv(
        output_dir / "requests.csv", index=False
    )
    pd.concat(all_links, ignore_index=True).to_csv(
        output_dir / "link_utilization.csv", index=False
    )
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(output_dir / "summary.csv", index=False)

    return summary_df


def plot_metric(
    summary_df: pd.DataFrame,
    metric: str,
    ylabel: str,
    path: Path,
) -> None:
    """Provided plotting helper."""
    display_names = {
        "shortest_hop": "Minimum Hop",
        "min_latency": "Minimum Latency",
        "max_reliability": "Maximum Reliability",
        "student_policy": "Student Policy",
    }

    plt.figure(figsize=(6.8, 4.4))
    for policy, group in summary_df.groupby("policy"):
        group = group.sort_values("offered_load")
        offered_load_per_s = 1000.0 * group["offered_load"]
        plt.plot(
            offered_load_per_s,
            group[metric],
            marker="o",
            linewidth=1.6,
            markersize=5,
            label=display_names.get(policy, policy),
        )

    plt.xlabel("Offered Load (requests/s)")
    plt.ylabel(ylabel)
    plt.grid(axis="y", linewidth=0.5, alpha=0.35)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=240, bbox_inches="tight")
    plt.close()


def make_plots(summary_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Required figures:
      1. success rate vs offered load
      2. average latency vs offered load
      3. throughput vs offered load
      4. average waiting time vs offered load
    """
    # TODO 6B:
    # Call plot_metric(...) for all four required metrics.
       output_dir = Path(output_dir)
    plot_metric(summary_df, "success_rate", "Request Success Rate",
                output_dir / "success_rate_vs_load.png")
    plot_metric(summary_df, "avg_latency_ms", "Average End-to-End Latency (ms)",
                output_dir / "latency_vs_load.png")
    plot_metric(summary_df, "throughput_req_per_s", "Throughput (successful requests/s)",
                output_dir / "throughput_vs_load.png")
    plot_metric(summary_df, "avg_waiting_ms", "Average Queue Waiting Time (ms)",
                output_dir / "waiting_vs_load.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", default="../data/topology.json")
    parser.add_argument("--output", default="../outputs/student_run")
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    g = load_topology(Path(args.topology))
    draw_topology(g, output_dir / "topology.png")

    summary_df = run_experiments(
        g,
        output_dir,
        n_requests=args.requests,
        seed=args.seed,
    )
    make_plots(summary_df, output_dir)

    print(summary_df.to_string(index=False))
    print(f"\nResults saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
