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
    alpha: float = 1.0,
    beta: float = 15.0,
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
        # Use at least TWO of: delay, p, capacity.
        raise NotImplementedError("Complete Task 3D")

    raise ValueError(f"Unknown policy: {policy}")


def choose_path(
    g: nx.Graph,
    src: str,
    dst: str,
    policy: str,
    alpha: float = 1.0,
    beta: float = 15.0,
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
        # One "next free at" time per slot; all slots start free at t = 0.
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
        # Update the slot heap, busy_time, and jobs.
        raise NotImplementedError("Complete Task 4B")


def canonical_edge(u: str, v: str) -> Tuple[str, str]:
    return tuple(sorted((u, v)))


def make_link_states(g: nx.Graph) -> Dict[Tuple[str, str], LinkState]:
    """Create one LinkState object for each network link."""
    # TODO 4C
    raise NotImplementedError("Complete Task 4C")


# ============================================================
# TASK 5 — End-to-end simulation and metrics [20 points]
# ============================================================

def simulate(
    g: nx.Graph,
    requests: List[Request],
    policy: str,
    seed: int,
    alpha: float = 1.0,
    beta: float = 15.0,
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

    # TODO 5B:
    # Process requests in increasing arrival-time order.

    # TODO 5C:
    # Return:
    #   request_df with per-request results
    #   link_df with policy, link, jobs, utilization
    raise NotImplementedError("Complete Task 5")


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
    raise NotImplementedError("Complete Task 5D")


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
    raise NotImplementedError("Complete Task 6A")


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
    raise NotImplementedError("Complete Task 6B")


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
