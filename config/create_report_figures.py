#Create only the figures used in the final written report.

#Run this after:

#    python3 config/prepare_data.py
#    python3 config/train_and_evaluate_model.py --cross-validate

#The script writes the four PNG files referenced by the LaTeX report to
#outputs/figures/.


import json
import os
import tempfile
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
DATE_FORMAT = "%m/%d/%Y %I:%M:%S %p"

# Keep Matplotlib cache out of the submitted project folder.
MPLCONFIG_DIR = Path(tempfile.gettempdir()) / "nyc311_mlp_matplotlib"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def setup_style():
    """Use one consistent visual style for all report figures."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 180,
        "axes.titlesize": 13,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "font.family": "DejaVu Sans",
    })


def reset_figures_dir():
    """Remove stale report images before writing the current report figures."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for old_file in FIGURES_DIR.glob("*"):
        if old_file.is_file():
            old_file.unlink()


def save_figure(fig, filename, title, description, registry):
    """Save one figure and record it in the figure index."""
    path = FIGURES_DIR / filename
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    registry.append({
        "file": str(path.relative_to(PROJECT_ROOT)),
        "title": title,
        "description": description,
    })


def add_bar_labels(ax, total=None):
    """Add readable labels to bar charts."""
    for patch in ax.patches:
        height = patch.get_height()
        if height == 0:
            continue
        if total is None:
            label = f"{int(height):,}"
        else:
            label = f"{int(height):,}\n({height / total * 100:.1f}%)"
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            height,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
        )


def make_target(train_df):
    """Create the report target: 1 if closed within 24 hours, else 0."""
    created = pd.to_datetime(
        train_df["Created Date"],
        format=DATE_FORMAT,
        errors="coerce",
    )
    closed = pd.to_datetime(
        train_df["Closed Date"],
        format=DATE_FORMAT,
        errors="coerce",
    )
    hours_to_close = (closed - created).dt.total_seconds() / 3600
    return ((hours_to_close >= 0) & (hours_to_close <= 24)).astype(int)


def plot_target_distribution(train_df, registry):
    """Plot the class balance used to motivate the majority-class baseline."""
    target = make_target(train_df)
    counts = target.value_counts().reindex([0, 1], fill_value=0)
    labels = ["Not closed\nwithin 24h", "Closed\nwithin 24h"]

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    sns.barplot(
        x=labels,
        y=counts.values,
        hue=labels,
        palette=["#4C78A8", "#59A14F"],
        legend=False,
        ax=ax,
    )
    ax.set_title("Target Distribution")
    ax.set_xlabel("Target class")
    ax.set_ylabel("Number of requests")
    ax.set_ylim(0, counts.max() * 1.18)
    add_bar_labels(ax, total=len(target))

    save_figure(
        fig,
        "01_target_distribution.png",
        "Target distribution",
        "Class balance for the 24-hour closure target.",
        registry,
    )


def plot_creation_hour(train_df, registry):
    """Plot request volume by creation hour for the EDA section."""
    created = pd.to_datetime(
        train_df["Created Date"],
        format=DATE_FORMAT,
        errors="coerce",
    )
    hour_counts = created.dt.hour.value_counts().reindex(range(24), fill_value=0)

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    sns.barplot(x=hour_counts.index, y=hour_counts.values, color="#F28E2B", ax=ax)
    ax.set_title("Requests By Creation Hour")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Number of requests")

    save_figure(
        fig,
        "06_requests_by_creation_hour.png",
        "Requests by creation hour",
        "Temporal request pattern used to justify creation-time features.",
        registry,
    )


def plot_feature_set_summary(registry):
    """Plot raw input, final feature, and categorical feature counts."""
    metadata_path = OUTPUTS_DIR / "processed" / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            "Missing outputs/processed/metadata.json. Run "
            "`python3 config/prepare_data.py` before generating figures."
        )

    metadata = json.loads(metadata_path.read_text())
    summary = pd.Series({
        "Raw selected inputs": len(metadata["input_columns"]),
        "Final model features": len(metadata["output_features"]),
        "Categorical features": len(metadata["categorical_cols"]),
    })

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    sns.barplot(
        x=summary.index,
        y=summary.values,
        hue=summary.index,
        palette="mako",
        legend=False,
        ax=ax,
    )
    ax.set_title("Feature Set Summary")
    ax.set_xlabel("")
    ax.set_ylabel("Number of features")
    ax.set_ylim(0, summary.max() * 1.2)
    ax.tick_params(axis="x", rotation=15)
    add_bar_labels(ax)

    save_figure(
        fig,
        "09_feature_set_summary.png",
        "Feature set summary",
        "Counts selected inputs, final features, and categorical features.",
        registry,
    )


def plot_confusion_matrix(registry):
    """Plot the validation confusion matrix from the final model run."""
    confusion_path = OUTPUTS_DIR / "confusion_matrix.csv"
    if not confusion_path.exists():
        raise FileNotFoundError(
            "Missing outputs/confusion_matrix.csv. Run "
            "`python3 config/train_and_evaluate_model.py --cross-validate` first."
        )

    confusion = pd.read_csv(confusion_path, index_col=0)

    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    sns.heatmap(
        confusion,
        annot=True,
        fmt=",d",
        cmap="Blues",
        cbar=False,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )
    ax.set_title("Validation Confusion Matrix")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")

    save_figure(
        fig,
        "13_confusion_matrix_heatmap.png",
        "Validation confusion matrix",
        "Validation-set correct predictions and error types.",
        registry,
    )


def main():
    setup_style()
    reset_figures_dir()
    registry = []

    train_df = pd.read_csv(DATA_DIR / "train.csv")

    plot_target_distribution(train_df, registry)
    plot_creation_hour(train_df, registry)
    plot_feature_set_summary(registry)
    plot_confusion_matrix(registry)

    pd.DataFrame(registry).to_csv(FIGURES_DIR / "figure_index.csv", index=False)
    print(f"Generated {len(registry)} report figures in {FIGURES_DIR}")


if __name__ == "__main__":
    main()
