import os
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_results(methods: List[str], dataset: str, base_dir: str = ".") -> pd.DataFrame:
    """Load per-cycle results from text logs produced by main training script."""
    all_data = []
    for method in methods:
        filename = f"results_{method}_{dataset}_main10False.txt"
        filepath = os.path.join(base_dir, filename)
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue

        with open(filepath, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = [p.strip("'") for p in line.strip().split()]
            if len(parts) == 8:
                (
                    method_name,
                    trial,
                    total_trial,
                    cycle,
                    total_cycle,
                    samples,
                    test_acc,
                    best_acc,
                ) = parts
                all_data.append(
                    {
                        "method": method_name,
                        "trial": int(trial),
                        "total_trial": int(total_trial),
                        "cycle": int(cycle),
                        "total_cycle": int(total_cycle),
                        "samples": int(samples),
                        "test_acc": float(test_acc),
                        "best_acc": float(best_acc),
                    }
                )

    return pd.DataFrame(all_data)


def plot_results(df: pd.DataFrame, dataset: str, save_dir: str = "assets") -> str:
    """Plot average test accuracy vs number of labeled samples for all methods."""
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(6, 4.5))

    grouped = df.groupby(["method", "samples"])["test_acc"]
    stats_df = grouped.agg(["mean", "std"]).reset_index()

    legend_order = [
        "Random",
        "CoreSet",
        "Lloss",
        "VAAL",
        "TA-VAAL",
        "UncertainGCN",
        "CoreGCN",
        "FDAL",
    ]
    method_markers = {
        "Random": "o",
        "CoreSet": "s",
        "Lloss": "D",
        "VAAL": "^",
        "TA-VAAL": "v",
        "UncertainGCN": "<",
        "CoreGCN": ">",
        "FDAL": "s",
    }
    base_colors = sns.color_palette("tab10", n_colors=8)
    method_colors = {
        "Random": base_colors[7],
        "CoreSet": base_colors[1],
        "Lloss": base_colors[2],
        "VAAL": base_colors[3],
        "TA-VAAL": base_colors[4],
        "UncertainGCN": base_colors[5],
        "CoreGCN": base_colors[6],
        "FDAL": base_colors[0],
    }

    for method in legend_order:
        method_data = stats_df[stats_df["method"] == method]
        if method_data.empty:
            continue
        x = method_data["samples"]
        y = method_data["mean"]
        std = method_data["std"]

        plt.plot(
            x,
            y,
            label="FDAL (Ours)" if method == "FDAL" else method,
            marker=method_markers[method],
            linewidth=2,
            markersize=7,
            color=method_colors[method],
        )

        plt.fill_between(x, y - std, y + std, alpha=0.2, color=method_colors[method])

    data_name = "CIFAR-10" if dataset == "cifar10" else "CIFAR-100"
    plt.title(f"Average Test Accuracy on {data_name}", fontsize=16)
    plt.xlabel("Number of Samples", fontsize=14)
    plt.ylabel("Average Test Accuracy (%)", fontsize=14)
    xtick_vals = range(1000, 11000, 1000)
    xtick_labels = [f"{x//1000}k" for x in xtick_vals]
    plt.xticks(xtick_vals, xtick_labels, fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(800, 10200)
    plt.legend(
        title="Sampling Method",
        fontsize=12,
        title_fontsize=13,
        loc="lower right",
    )
    plt.tight_layout()

    out_path = os.path.join(save_dir, f"FDAL_sampling_plot_{dataset}.png")
    plt.savefig(out_path, dpi=400)
    plt.close()
    return out_path


if __name__ == "__main__":
    methods = [
        "Random",
        "CoreSet",
        "Lloss",
        "VAAL",
        "TA-VAAL",
        "UncertainGCN",
        "CoreGCN",
        "FDAL",
    ]
    dataset = "cifar10"
    df_combined = load_results(methods, dataset)
    out = plot_results(df_combined, dataset)
    print(f"Saved line graph to {out}")

