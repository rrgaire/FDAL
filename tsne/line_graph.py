import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_results(methods, dataset, base_dir="."):
    all_data = []
    for method in methods:
        filename = f"results_{method}_{dataset}_main10False.txt"
        filepath = os.path.join(base_dir, filename)
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue

        # Read the file
        with open(filepath, "r") as f:
            lines = f.readlines()

        # Parse lines
        for line in lines:
            parts = [p.strip("'") for p in line.strip().split()]
            if len(parts) == 8:
                print(parts)
                method_name, trial, total_trial, cycle, total_cycle, samples, test_acc, best_acc = parts
                all_data.append({
                    "method": method_name,
                    "trial": int(trial),
                    "total_trial": int(total_trial),
                    "cycle": int(cycle),
                    "total_cycle": int(total_cycle),
                    "samples": int(samples),
                    "test_acc": float(test_acc),
                    "best_acc": float(best_acc)
                })

    # Return combined DataFrame
    return pd.DataFrame(all_data)


def plot_results_custom(df, save_path="sampling_comparison_plot.png"):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # sns.set(context="paper", font_scale=1.4)
    plt.figure(figsize=(10, 6))

    # Compute mean and std
    grouped = df.groupby(['method', 'samples'])['test_acc']
    stats_df = grouped.agg(['mean', 'std']).reset_index()

    # Define fixed legend order and custom styles
    legend_order = ['Random', 'CoreSet', 'lloss', 'VAAL', 'TA-VAAL', 'UncertainGCN', 'CoreGCN', 'FDAL']
    method_markers = {
        'Random': 'o', 'CoreSet': 's', 'lloss': 'D', 'VAAL': '^',
        'TA-VAAL': 'v', 'UncertainGCN': '<', 'CoreGCN': '>', 'FDAL': 's'
    }
    base_colors = sns.color_palette("tab10", n_colors=8)
    method_colors = {
        'Random': base_colors[7], 'CoreSet': base_colors[1], 'lloss': base_colors[2], 'VAAL': base_colors[3],
        'TA-VAAL': base_colors[4], 'UncertainGCN': base_colors[5], 'CoreGCN': base_colors[6], 'FDAL': base_colors[0]
    }

    for method in legend_order:
        method_data = stats_df[stats_df['method'] == method]
        if method_data.empty:
            continue
        x = method_data['samples']
        y = method_data['mean']
        std = method_data['std']

        plt.plot(x, y,
                 label='Ours' if method == 'FDAL' else method,
                 marker=method_markers[method],
                 linewidth=2,
                 markersize=7,
                 color=method_colors[method])

        plt.fill_between(x, y - std, y + std, alpha=0.1, color=method_colors[method])

    plt.title("Average Test Accuracy on CIFAR-10", fontsize=16)
    plt.xlabel("Number of Samples", fontsize=14)
    plt.ylabel("Average Test Accuracy (%)", fontsize=14)
    plt.xticks(range(1000, 11000, 1000), fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(800, 10200)
    plt.legend(title="Sampling Method", fontsize=12, title_fontsize=13, loc='lower right')
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

# Example usage
methods = ['Random', 'UncertainGCN', 'CoreGCN', 'CoreSet', 'lloss', 'VAAL', 'TA-VAAL', 'FDAL']
dataset = 'cifar10'
df_combined = load_results(methods, dataset)
plot_results_custom(df_combined, save_path=f"sampling_plot_{dataset}.png")
