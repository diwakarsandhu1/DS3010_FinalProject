from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import classification_report, precision_recall_curve

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

MODEL_COLS = {
    "Dummy Baseline":      "pred_dummy_baseline",
    "Logistic Regression": "pred_logistic_regression",
    "Random Forest":       "pred_random_forest",
    "Gradient Boosting":   "pred_gradient_boosting",
}
# heatmap per class
def plot_heatmap(preds):
    classes = ["Down/Flat", "Up"]
    metrics = ["precision", "recall", "f1-score"]
    model_names = list(MODEL_COLS.keys())

    fig, axes = plt.subplots(1, len(model_names), figsize=(14, 3.5))
    fig.suptitle("Classification Report Heatmap (Per Class)", fontsize=13, y=1.05)

    for ax, (name, col) in zip(axes, MODEL_COLS.items()):
        report = classification_report(
            preds["target_direction"],
            preds[col],
            target_names=classes,
            output_dict=True,
            zero_division=0
        )
        data = np.array([
            [report[c][m] for m in metrics]
            for c in classes
        ])

        im = ax.imshow(data, vmin=0, vmax=1, cmap="Blues", aspect="auto")

        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(["Precision", "Recall", "F1"], fontsize=8)
        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels(classes, fontsize=8)
        ax.set_title(name, fontsize=9, pad=8)

        for i in range(len(classes)):
            for j in range(len(metrics)):
                val = data[i, j]
                color = "white" if val > 0.6 else "#333333"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=10, color=color, fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "classification_heatmap.png", bbox_inches="tight", dpi=150)
    plt.show()
    print("Saved classification_heatmap.png")

# cruve plot 
def plot_pr_curve(preds):
    real_models = {k: v for k, v in MODEL_COLS.items() if k != "Dummy Baseline"}
    colors = ["#2d6a9f", "#52b788", "#e76f51"]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_title("Precision-Recall Curve (Up class)", fontsize=12)

    for (name, col), color in zip(real_models.items(), colors):
        precision, recall, _ = precision_recall_curve(
            preds["target_direction"], preds[col]
        )
        ax.plot(recall, precision, label=name, color=color, linewidth=2)

    # Baseline = proportion of positives
    baseline = preds["target_direction"].mean()
    ax.axhline(baseline, color="#999999", linestyle="--",
               linewidth=1, label=f"Dummy baseline ({baseline:.2f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.yaxis.grid(True, color="#e0e0e0", linewidth=0.5)
    ax.xaxis.grid(True, color="#e0e0e0", linewidth=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "classification_pr_curve.png", bbox_inches="tight", dpi=150)
    plt.show()
    print("Saved classification_pr_curve.png")

#accuracy bar graph 

def plot_accuracy_comparison(results):
    # Use Gradient Boosting as "Final Model" since it's the best
    display = results.copy()
    display.loc[display["model"] == "Gradient Boosting", "model"] = "Gradiant Boosting"
    display = display[display["model"].isin(["Dummy Baseline", "Logistic Regression", "Gradiant Boosting"])]
    display = display.sort_values("accuracy").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#3d405b", "#2d6a9f", "#52b788"]

    bars = ax.bar(display["model"], display["accuracy"], color=colors, width=0.5)

    # Values on top, no y-axis
    for bar, val in zip(bars, display["accuracy"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.2f}",
                ha="center", va="bottom", fontsize=11, color="#333333")

    ax.set_title("Comparison of Model Accuracy", fontsize=13)
    ax.set_xlabel("Model")
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 1])
    ax.yaxis.set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_color("#cccccc")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "classification_accuracy.png", bbox_inches="tight", dpi=150)
    plt.show()
    print("Saved classification_accuracy.png")

def main():
    preds = pd.read_csv(OUTPUT_DIR / "classification_predictions.csv")
    results = pd.read_csv(OUTPUT_DIR / "classification_metrics.csv")
    plot_heatmap(preds)
    plot_accuracy_comparison(results)
    plot_pr_curve(preds)

if __name__ == "__main__":
    main()

