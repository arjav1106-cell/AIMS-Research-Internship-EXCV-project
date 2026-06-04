"""
Correlation analysis of explainability metrics for Grad-CAM and Integrated Gradients.
Saves Pearson and Spearman correlation matrices as CSVs and visualizes them as heatmaps.
"""
from __future__ import annotations

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import OUTPUTS_DIR, PLOTS_DIR

CERS_RESULTS_CSV = OUTPUTS_DIR / "metrics" / "cers_results.csv"
ANALYSIS_DIR = OUTPUTS_DIR / "analysis"


def load_data() -> pd.DataFrame:
    """Load results from cers_results.csv, falling back if not in metrics subdirectory."""
    path = CERS_RESULTS_CSV
    if not path.is_file():
        # Try root outputs directory
        path = OUTPUTS_DIR / "cers_results.csv"
        if not path.is_file():
            raise FileNotFoundError(f"CERS results file not found at {CERS_RESULTS_CSV} or {OUTPUTS_DIR / 'cers_results.csv'}")
    return pd.read_csv(path)


def compute_correlations(
    df: pd.DataFrame,
    method_prefix: str,
    metric_map: dict[str, str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute Pearson and Spearman correlation matrices for specified method."""
    columns = [f"{method_prefix}_{col}" for col in metric_map]
    
    # Filter columns that exist
    columns = [col for col in columns if col in df.columns]
    
    subset = df[columns].copy()
    
    # Rename columns to friendly names
    rename_dict = {f"{method_prefix}_{key}": val for key, val in metric_map.items()}
    subset.rename(columns=rename_dict, inplace=True)
    
    pearson = subset.corr(method="pearson")
    spearman = subset.corr(method="spearman")
    
    return pearson, spearman


def save_matrix_to_csv(df: pd.DataFrame, filename: str) -> Path:
    """Save correlation matrix to outputs/analysis/."""
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ANALYSIS_DIR / filename
    df.to_csv(out_path)
    print(f"Saved correlation matrix -> {out_path}")
    return out_path


def save_heatmap(df: pd.DataFrame, title: str, filename: str) -> Path:
    """Generate and save correlation heatmap to outputs/plots/."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOTS_DIR / filename
    
    fig, ax = plt.subplots(figsize=(8, 6.5))
    
    # Mask the upper triangle for cleaner look
    mask = np.triu(np.ones_like(df, dtype=bool))
    
    sns.heatmap(
        df,
        mask=mask,
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        annot=True,
        fmt=".3f",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"},
        ax=ax
    )
    
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved heatmap plot -> {out_path}")
    return out_path


def find_key_relationships(corr_df: pd.DataFrame, method_name: str, corr_type: str) -> None:
    """Find and print strongest correlations and conflicts."""
    print(f"\n=== Key Relationships for {method_name} ({corr_type}) ===")
    
    # Flatten the matrix and extract values
    corr_matrix = corr_df.values
    indices = np.triu_indices_from(corr_matrix, k=1)
    
    pairs = []
    for i, j in zip(*indices):
        m1 = corr_df.index[i]
        m2 = corr_df.columns[j]
        val = corr_matrix[i, j]
        pairs.append((m1, m2, val))
        
    # Sort by absolute correlation
    pairs_sorted = sorted(pairs, key=lambda x: abs(x[2]), reverse=True)
    
    print("Top 3 Strongest Correlations:")
    count = 0
    for m1, m2, val in pairs_sorted:
        if count >= 3:
            break
        print(f"  * {m1} <--> {m2}: {val:+.4f}")
        count += 1
        
    print("Strongest Negative Correlations / Conflicts:")
    # Filter negative correlations
    neg_pairs = [p for p in pairs if p[2] < 0]
    neg_sorted = sorted(neg_pairs, key=lambda x: x[2])  # Most negative first
    
    for m1, m2, val in neg_sorted[:3]:
        print(f"  * {m1} <--> {m2}: {val:.4f}")
        
    # Specifically check for expected conflicts
    print("Conflict Audits (Theoretical vs Empirical):")
    
    # 1. Entropy vs. Lung Focus Score
    # Theoretical expectation: lower entropy (more focused) should correlate with higher lung focus (better alignment)
    # Since lower entropy is better, if we look at raw entropy, a positive correlation with LFS represents a conflict
    # (more entropy correlates with more lung focus).
    val_ent_lfs = corr_df.loc["Entropy", "Lung Focus Score"]
    print(f"  * Entropy vs. Lung Focus Score: {val_ent_lfs:+.4f}")
    if val_ent_lfs > 0:
        print("    [CONFLICT] Positive correlation indicates more diffuse maps (higher entropy) have more lung focus.")
    else:
        print("    [ALIGN] Negative correlation indicates more focused maps (lower entropy) align better with the lungs.")
        
    # 2. Deletion AUC vs. Lung Focus Score
    # Theoretical expectation: lower deletion AUC (faster drop when deleting) correlates with higher lung focus (better alignment)
    # So we expect a negative correlation between Deletion AUC and Lung Focus Score.
    val_del_lfs = corr_df.loc["Deletion AUC", "Lung Focus Score"]
    print(f"  * Deletion AUC vs. Lung Focus Score: {val_del_lfs:+.4f}")
    if val_del_lfs > 0:
        print("    [CONFLICT] Positive correlation indicates that maps relying on non-lung features can still yield faster drops when deleted.")
    else:
        print("    [ALIGN] Negative correlation indicates that focusing on lungs aligns with faithful model performance drops.")
        
    # 3. AOPC vs. Insertion AUC
    # Theoretical expectation: both measure positive faithfulness, so they should be positively correlated.
    val_aopc_ins = corr_df.loc["AOPC", "Insertion AUC"]
    print(f"  * AOPC vs. Insertion AUC: {val_aopc_ins:+.4f}")
    if val_aopc_ins < 0.1:
        print("    [CONFLICT/WEAK] Very weak or negative correlation suggests insertion and deletion curves evaluate explanation order inconsistently.")
    else:
        print("    [ALIGN] Positive correlation suggests consistency in feature ordering.")


def main():
    print("=" * 60)
    print("RUNNING EXPLAINABILITY METRICS CORRELATION PIPELINE")
    print("=" * 60)
    
    df = load_data()
    print(f"Loaded results data: {len(df)} samples")
    
    # Metrics to analyze and their column suffixes
    metric_map = {
        "aopc": "AOPC",
        "insertion_auc": "Insertion AUC",
        "deletion_auc": "Deletion AUC",
        "entropy": "Entropy",
        "lfs": "Lung Focus Score",
        "robustness": "Robustness",
        "cers": "CERS"
    }
    
    # Compute correlations for Grad-CAM
    gc_pearson, gc_spearman = compute_correlations(df, "gradcam", metric_map)
    save_matrix_to_csv(gc_pearson, "pearson_gradcam.csv")
    save_matrix_to_csv(gc_spearman, "spearman_gradcam.csv")
    save_heatmap(gc_pearson, "Grad-CAM Metrics Correlation (Pearson)", "gradcam_correlation_heatmap.png")
    
    # Compute correlations for Integrated Gradients
    ig_pearson, ig_spearman = compute_correlations(df, "integrated_gradients", metric_map)
    save_matrix_to_csv(ig_pearson, "pearson_ig.csv")
    save_matrix_to_csv(ig_spearman, "spearman_ig.csv")
    save_heatmap(ig_pearson, "Integrated Gradients Metrics Correlation (Pearson)", "ig_correlation_heatmap.png")
    
    # Print relationship audits
    find_key_relationships(gc_spearman, "Grad-CAM", "Spearman")
    find_key_relationships(ig_spearman, "Integrated Gradients", "Spearman")
    
    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
