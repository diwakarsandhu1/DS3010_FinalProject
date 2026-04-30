from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"

mpl.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "serif",
    "axes.labelcolor": "#333333",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
})

def plot_model_comparison(results):
    results = results.sort_values("rmse").reset_index(drop=True)
    models = results["model"]
    x = np.arange(len(models))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Regression Model Comparison", fontsize=14, y=1.02)

    # MAE + RMSE grouped bars
    bars1 = ax1.bar(x - width/2, results["mae"], width, label="MAE", color="#2d6a9f")
    bars2 = ax1.bar(x + width/2, results["rmse"], width, label="RMSE", color="#52b788")

    ax1.set_title("Error Metrics", fontsize=11)
    ax1.set_ylabel("Score")
    ax1.set_xlabel("Model")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=9)
    ax1.legend(title="Metric", fontsize=9)
    ax1.set_ylim(0, results[["mae", "rmse"]].max().max() * 1.2)
    ax1.spines["left"].set_color("#cccccc")
    ax1.spines["bottom"].set_color("#cccccc")
    ax1.yaxis.grid(True, color="#e0e0e0", linewidth=0.5)
    ax1.set_axisbelow(True)

    # R² bars with green/red coloring. 
    colors = ["#2d6a4f" if v > 0 else "#c1121f" for v in results["r2"]]
    ax2.bar(x, results["r2"], width=0.5, color=colors)
    ax2.axhline(0, color="#333333", linewidth=0.8)

    ax2.set_title("R² Score ", fontsize=11)
    ax2.set_ylabel("R² Value")
    ax2.set_xlabel("Model")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=9)
    ax2.spines["left"].set_color("#cccccc")
    ax2.spines["bottom"].set_color("#cccccc")
    ax2.yaxis.grid(True, color="#e0e0e0", linewidth=0.5)
    ax2.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_comparison_final.png", bbox_inches="tight", dpi=150)
    plt.show()
    print("Saved model_comparison_final.png")

def main():
    results = pd.read_csv(OUTPUT_DIR / "regression_metrics.csv")
    plot_model_comparison(results)

if __name__ == "__main__":
    main()