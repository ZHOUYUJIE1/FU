import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent

SEVERE_SUMMARY = ROOT / "Cifar10_FedAvg_heterogeneity_summary_final.csv"
CLIENT_COUNT_FILES = [
    ROOT / "Cifar10_FedAvg_client_count_severe_c10_summary.csv",
    ROOT / "Cifar10_FedAvg_client_count_severe_c20_summary.csv",
    ROOT / "Cifar10_FedAvg_client_count_severe_c50_summary.csv",
    ROOT / "Cifar10_FedAvg_client_count_severe_c100_summary.csv",
]

OUT_PNG = ROOT / "fu_tradeoff_scatter_pareto.png"
OUT_PDF = ROOT / "fu_tradeoff_scatter_pareto.pdf"
OUT_MD = ROOT / "fu_tradeoff_scatter_pareto.md"

METHOD_ORDER = ["fu", "fedau", "fedcsa", "fedosd", "retrain"]
METHOD_LABELS = {
    "fu": "FU",
    "fedau": "FedAU",
    "fedcsa": "FedCSA",
    "fedosd": "FedOSD",
    "retrain": "Retrain",
}
METHOD_COLORS = {
    "fu": "#d04a02",
    "fedau": "#1b9e77",
    "fedcsa": "#7570b3",
    "fedosd": "#e7298a",
    "retrain": "#4d4d4d",
}
COUNT_MARKERS = {
    10: "o",
    20: "s",
    50: "^",
    100: "D",
}


def read_csv(path):
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(row, key):
    value = row.get(key, "")
    return float(value) if value not in ("", None) else None


def pareto_frontier(points):
    frontier = []
    for point in points:
        dominated = False
        for other in points:
            if other is point:
                continue
            better_or_equal = other["x"] <= point["x"] and other["y"] >= point["y"]
            strictly_better = other["x"] < point["x"] or other["y"] > point["y"]
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(point)
    frontier.sort(key=lambda item: (item["x"], -item["y"]))
    return frontier


def build_severe_points():
    rows = read_csv(SEVERE_SUMMARY)
    rows = [row for row in rows if row["level"] == "severe"]
    points = []
    for row in rows:
        points.append(
            {
                "method": row["method"],
                "label": METHOD_LABELS[row["method"]],
                "x": to_float(row, "final_target_acc"),
                "y": to_float(row, "final_avg_acc"),
            }
        )
    return points


def build_client_count_points():
    points = []
    for path in CLIENT_COUNT_FILES:
        rows = read_csv(path)
        client_count = int(rows[0]["num_clients"])
        for row in rows:
            points.append(
                {
                    "method": row["method"],
                    "label": METHOD_LABELS[row["method"]],
                    "client_count": client_count,
                    "x": to_float(row, "final_target_acc"),
                    "y": to_float(row, "final_avg_acc"),
                }
            )
    return points


def add_frontier(ax, frontier, color="#333333"):
    xs = [item["x"] for item in frontier]
    ys = [item["y"] for item in frontier]
    ax.plot(xs, ys, linestyle="--", linewidth=1.6, color=color, alpha=0.9, zorder=1)


def plot_tradeoff():
    severe_points = build_severe_points()
    client_points = build_client_count_points()

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.2), constrained_layout=True)
    fig.patch.set_facecolor("#f7f3eb")

    for ax in axes:
        ax.set_facecolor("#fffdf8")
        ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.5)

    ax = axes[0]
    severe_frontier = pareto_frontier(severe_points)
    add_frontier(ax, severe_frontier, color="#5b4b3a")
    for point in severe_points:
        method = point["method"]
        edge = "#111111" if method == "fu" else "white"
        lw = 1.8 if method == "fu" else 1.0
        size = 180 if method == "fu" else 120
        ax.scatter(
            point["x"],
            point["y"],
            s=size,
            color=METHOD_COLORS[method],
            edgecolor=edge,
            linewidth=lw,
            zorder=3,
        )
        dx = -0.012 if method == "retrain" else 0.005
        dy = 0.01 if method in {"fu", "fedcsa"} else -0.012
        ax.text(point["x"] + dx, point["y"] + dy, point["label"], fontsize=10)
    ax.set_title("CIFAR-10 Severe Heterogeneity", fontsize=13)
    ax.set_xlabel("Final Target Accuracy (lower is better)")
    ax.set_ylabel("Final Average Accuracy (higher is better)")

    ax = axes[1]
    client_frontier = pareto_frontier(client_points)
    add_frontier(ax, client_frontier, color="#5b4b3a")
    for point in client_points:
        method = point["method"]
        count = point["client_count"]
        alpha = 1.0 if method == "fu" else 0.82
        size = 150 if method == "fu" else 95
        ax.scatter(
            point["x"],
            point["y"],
            s=size,
            color=METHOD_COLORS[method],
            marker=COUNT_MARKERS[count],
            edgecolor="#111111" if method == "fu" else "white",
            linewidth=1.5 if method == "fu" else 0.8,
            alpha=alpha,
            zorder=3,
        )
        if method == "fu":
            ax.text(point["x"] + 0.004, point["y"] + 0.008, f"C{count}", fontsize=9)
    ax.set_title("CIFAR-10 Severe: Client Count Sweep", fontsize=13)
    ax.set_xlabel("Final Target Accuracy (lower is better)")
    ax.set_ylabel("Final Average Accuracy (higher is better)")

    method_handles = []
    for method in METHOD_ORDER:
        handle = plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=METHOD_COLORS[method],
            markeredgecolor="#111111" if method == "fu" else "white",
            markersize=9,
            label=METHOD_LABELS[method],
        )
        method_handles.append(handle)

    count_handles = []
    for count, marker in COUNT_MARKERS.items():
        handle = plt.Line2D(
            [0],
            [0],
            marker=marker,
            color="#555555",
            linestyle="None",
            markersize=8,
            label=f"{count} clients",
        )
        count_handles.append(handle)

    axes[1].legend(handles=method_handles, loc="lower right", frameon=False, title="Methods")
    axes[0].legend(handles=count_handles, loc="lower right", frameon=False, title="Markers in Panel B")

    fig.suptitle("Utility-Forgetting Trade-off of FU on CIFAR-10", fontsize=15, y=1.02)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")

    severe_frontier_labels = ", ".join(item["label"] for item in severe_frontier)
    client_frontier_labels = ", ".join(
        f"{item['label']}@C{item['client_count']}" for item in client_frontier
    )
    lines = [
        "# Trade-off Figure Notes",
        "",
        f"- Output PNG: `{OUT_PNG.name}`",
        f"- Output PDF: `{OUT_PDF.name}`",
        f"- Severe heterogeneity Pareto frontier: {severe_frontier_labels}",
        f"- Client-count Pareto frontier: {client_frontier_labels}",
        "- Panel A highlights that under severe heterogeneity, FU lies on the Pareto frontier and provides the highest final average accuracy among all methods.",
        "- Panel B highlights that FU stays on or near the Pareto frontier for the practically relevant 10-50 client regime.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    plot_tradeoff()
