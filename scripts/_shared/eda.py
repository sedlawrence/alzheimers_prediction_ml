from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import runpy
import seaborn as sns


_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
_TASK1_EDA = runpy.run_path(str(_SCRIPTS_DIR / "01_describe_array.py"))

human_bytes = _TASK1_EDA["human_bytes"]
decode_if_bytes = _TASK1_EDA["decode_if_bytes"]
preview_dataset = _TASK1_EDA["preview_dataset"]
numeric_summary = _TASK1_EDA["numeric_summary"]
summarise_3d_array = _TASK1_EDA["summarise_3d_array"]
string_summary = _TASK1_EDA["string_summary"]
describe_dataset = _TASK1_EDA["describe_dataset"]
get_top_variable_cpgs = _TASK1_EDA["get_top_variable_cpgs"]
save_cpg_correlation_clustermap = _TASK1_EDA["save_cpg_correlation_clustermap"]
TOP_N_CPGS = _TASK1_EDA["TOP_N_CPGS"]


def save_subject_clustermap(
    X_2d,
    y,
    cpg_ids,
    out_png,
    title,
    class0_label="class 0",
    class1_label="class 1",
):
    """
    Save clustered subject x CpG methylation heatmap with configurable labels.
    """
    df = pd.DataFrame(X_2d, columns=cpg_ids)

    row_labels = [
        f"sample_{i:03d}_{class0_label.replace(' ', '_').replace('->', 'to')}"
        if label == 0
        else f"sample_{i:03d}_{class1_label.replace(' ', '_').replace('->', 'to')}"
        for i, label in enumerate(y)
    ]
    df.index = row_labels

    row_colors = pd.Series(y, index=df.index).map({
        0: "lightgrey",
        1: "firebrick",
    })

    g = sns.clustermap(
        df,
        cmap="viridis",
        row_cluster=True,
        col_cluster=True,
        row_colors=row_colors,
        figsize=(16, 12),
        xticklabels=False,
        yticklabels=False,
        dendrogram_ratio=(0.10, 0.10),
        cbar_pos=(0.02, 0.82, 0.03, 0.12),
    )

    g.fig.suptitle(title, y=1.02)

    for label, color in [
        (f"{class0_label}, y=0", "lightgrey"),
        (f"{class1_label}, y=1", "firebrick"),
    ]:
        g.ax_col_dendrogram.bar(0, 0, color=color, label=label, linewidth=0)

    g.ax_col_dendrogram.legend(
        loc="center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.15),
        frameon=False,
    )

    g.fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(g.fig)
