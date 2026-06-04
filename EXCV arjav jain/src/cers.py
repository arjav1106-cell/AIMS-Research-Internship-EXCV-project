"""
CERS (Comprehensive Explanation Reliability Score) core engine.
Calculates unified explainability metrics, aggregates results, and generates plots.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import METRICS_DIR, PLOTS_DIR

XAI_METRICS_PATH = METRICS_DIR / "xai_metrics_with_robustness.json"
CERS_RESULTS_CSV = METRICS_DIR / "cers_results.csv"
CERS_SUMMARY_JSON = METRICS_DIR / "cers_summary.json"

# Image size for entropy normalization
H_W_LOG = np.log(256 * 256)


def calculate_fidelity(aopc: float, insertion_auc: float, deletion_auc: float) -> float:
    """Fidelity = (AOPC + Insertion_AUC + (1 - Deletion_AUC)) / 3"""
    return float((aopc + insertion_auc + (1.0 - deletion_auc)) / 3.0)


def calculate_localization(lfs: float, entropy: float) -> float:
    """Localization = 0.7 * LFS + 0.3 * (1 - Entropy / ln(H*W))"""
    norm_focus = 1.0 - (entropy / H_W_LOG)
    norm_focus = max(0.0, min(1.0, norm_focus))  # Clamp to [0,1]
    return float(0.7 * lfs + 0.3 * norm_focus)


def calculate_concordance(spearman_rho: float, top_k_overlap: float) -> float:
    """Concordance = 0.5 * (max(0, spearman_rho) + top_k_overlap)"""
    clipped_rho = max(0.0, spearman_rho)
    return float(0.5 * (clipped_rho + top_k_overlap))


def compute_class_method_robustness_averages(records: list[dict]) -> dict[str, dict[str, float]]:
    """Compute average robustness per class and method from the 30 calculated samples."""
    sums: dict[str, dict[str, list[float]]] = {}
    
    for rec in records:
        true_class = rec["true_class"]
        if true_class not in sums:
            sums[true_class] = {"gradcam": [], "integrated_gradients": []}
            
        for method in ("gradcam", "integrated_gradients"):
            r_mean = rec[method].get("robustness_mean")
            if r_mean is not None and not np.isnan(r_mean):
                sums[true_class][method].append(float(r_mean))
                
    # Compute averages
    averages: dict[str, dict[str, float]] = {}
    global_sums = {"gradcam": [], "integrated_gradients": []}
    
    for cls, methods in sums.items():
        averages[cls] = {}
        for method, vals in methods.items():
            if vals:
                averages[cls][method] = float(np.mean(vals))
                global_sums[method].extend(vals)
            else:
                # Fallback value if no class-specific robustness exists
                averages[cls][method] = 0.0
                
    # Global fallbacks
    global_avg = {
        "gradcam": float(np.mean(global_sums["gradcam"])) if global_sums["gradcam"] else 0.6394,
        "integrated_gradients": float(np.mean(global_sums["integrated_gradients"])) if global_sums["integrated_gradients"] else 0.5395
    }
    
    # Fill in any missing class averages with the global averages
    for cls in averages:
        for method in ("gradcam", "integrated_gradients"):
            if averages[cls][method] == 0.0:
                averages[cls][method] = global_avg[method]
                
    averages["__global__"] = global_avg
    return averages


def process_cers_pipeline(
    w_fid: float = 0.4,
    w_loc: float = 0.3,
    w_rob: float = 0.3
) -> tuple[list[dict], dict]:
    """Run the CERS evaluation pipeline."""
    if not XAI_METRICS_PATH.is_file():
        raise FileNotFoundError(f"Source XAI metrics file not found: {XAI_METRICS_PATH}")
        
    with open(XAI_METRICS_PATH, encoding="utf-8") as f:
        data = json.load(f)
        
    records = data.get("per_sample", [])
    if not records:
        raise ValueError("No per-sample records found in metrics file.")
        
    # Get robustness averages for backfilling
    rob_averages = compute_class_method_robustness_averages(records)
    print(f"Computed robustness averages for backfilling: {json.dumps(rob_averages, indent=2)}")
    
    cers_records = []
    
    for rec in records:
        true_class = rec["true_class"]
        
        # Calculate cross-method concordance
        concordance_val = calculate_concordance(
            rec["concordance"]["spearman_rho"],
            rec["concordance"]["top5pct_overlap"]
        )
        
        row = {
            "path": rec["path"],
            "true_class": true_class,
            "pred_class": rec["pred_class"],
            "correct": rec["correct"],
            "confidence": rec["confidence"],
            "concordance_score": concordance_val,
            "spearman_rho": rec["concordance"]["spearman_rho"],
            "top5pct_overlap": rec["concordance"]["top5pct_overlap"]
        }
        
        for method in ("gradcam", "integrated_gradients"):
            block = rec[method]
            
            # Fidelity
            fid = calculate_fidelity(block["aopc"], block["insertion_auc"], block["deletion_auc"])
            
            # Localization
            loc = calculate_localization(block["lfs"], block["entropy"])
            
            # Robustness (backfill if missing)
            r_mean = block.get("robustness_mean")
            was_backfilled = False
            if r_mean is None or np.isnan(r_mean):
                r_mean = rob_averages.get(true_class, rob_averages["__global__"])[method]
                was_backfilled = True
                
            # CERS and C-CERS
            cers_val = w_fid * fid + w_loc * loc + w_rob * r_mean
            ccers_val = cers_val * concordance_val
            
            row.update({
                f"{method}_fidelity": fid,
                f"{method}_localization": loc,
                f"{method}_robustness": r_mean,
                f"{method}_robustness_backfilled": was_backfilled,
                f"{method}_cers": cers_val,
                f"{method}_ccers": ccers_val,
                f"{method}_aopc": block["aopc"],
                f"{method}_insertion_auc": block["insertion_auc"],
                f"{method}_deletion_auc": block["deletion_auc"],
                f"{method}_entropy": block["entropy"],
                f"{method}_lfs": block["lfs"],
            })
            
        cers_records.append(row)
        
    # Write detailed CSV results
    fieldnames = list(cers_records[0].keys())
    with open(CERS_RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cers_records)
        
    # Compute aggregates
    summary = {
        "n_samples": len(cers_records),
        "weights": {
            "fidelity": w_fid,
            "localization": w_loc,
            "robustness": w_rob
        },
        "concordance_mean": float(np.mean([r["concordance_score"] for r in cers_records])),
        "spearman_rho_mean": float(np.mean([r["spearman_rho"] for r in cers_records])),
        "top5pct_overlap_mean": float(np.mean([r["top5pct_overlap"] for r in cers_records])),
    }
    
    for method in ("gradcam", "integrated_gradients"):
        summary[method] = {
            "fidelity_mean": float(np.mean([r[f"{method}_fidelity"] for r in cers_records])),
            "localization_mean": float(np.mean([r[f"{method}_localization"] for r in cers_records])),
            "robustness_mean": float(np.mean([r[f"{method}_robustness"] for r in cers_records])),
            "cers_mean": float(np.mean([r[f"{method}_cers"] for r in cers_records])),
            "cers_std": float(np.std([r[f"{method}_cers"] for r in cers_records])),
            "ccers_mean": float(np.mean([r[f"{method}_ccers"] for r in cers_records])),
            "ccers_std": float(np.std([r[f"{method}_ccers"] for r in cers_records])),
        }
        
    with open(CERS_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    print(f"\nSaved detailed results -> {CERS_RESULTS_CSV}")
    print(f"Saved summary metrics -> {CERS_SUMMARY_JSON}")
    
    return cers_records, summary


def generate_visualizations(records: list[dict], summary: dict) -> None:
    """Generate CERS distribution comparison and conflict scatter plots."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # 1. Box/Violin plot for CERS and C-CERS
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    methods = ["Grad-CAM", "Integrated Gradients"]
    cers_data = {
        "CERS": [],
        "C-CERS": [],
        "Method": []
    }
    
    for r in records:
        cers_data["CERS"].extend([r["gradcam_cers"], r["integrated_gradients_cers"]])
        cers_data["C-CERS"].extend([r["gradcam_ccers"], r["integrated_gradients_ccers"]])
        cers_data["Method"].extend(methods)
        
    # Violin plot for CERS
    sns.violinplot(
        x="Method", y="CERS", data=cers_data, ax=axes[0],
        palette={"Grad-CAM": "#1f77b4", "Integrated Gradients": "#ff7f0e"},
        inner="quartile", cut=0
    )
    axes[0].set_title("CERS Distribution Comparison", fontsize=13, fontweight="bold", pad=12)
    axes[0].set_ylabel("CERS Score (Unified)", fontsize=11)
    axes[0].set_xlabel("")
    axes[0].set_ylim(-0.05, 1.05)
    
    # Violin plot for C-CERS
    sns.violinplot(
        x="Method", y="C-CERS", data=cers_data, ax=axes[1],
        palette={"Grad-CAM": "#1f77b4", "Integrated Gradients": "#ff7f0e"},
        inner="quartile", cut=0
    )
    axes[1].set_title("C-CERS (Consensus-Adjusted) Distribution", fontsize=13, fontweight="bold", pad=12)
    axes[1].set_ylabel("C-CERS Score", fontsize=11)
    axes[1].set_xlabel("")
    axes[1].set_ylim(-0.05, 1.05)
    
    # Annotate summary means on plots
    for idx, method_key in enumerate(["gradcam", "integrated_gradients"]):
        m_cers = summary[method_key]["cers_mean"]
        m_ccers = summary[method_key]["ccers_mean"]
        
        # CERS mean text annotation
        axes[0].text(
            idx, m_cers + 0.03, f"Mean: {m_cers:.4f}",
            ha="center", va="bottom", color="black", fontweight="bold", fontsize=10
        )
        # C-CERS mean text annotation
        axes[1].text(
            idx, m_ccers + 0.03, f"Mean: {m_ccers:.4f}",
            ha="center", va="bottom", color="black", fontweight="bold", fontsize=10
        )
        
    plt.tight_layout()
    plot_path1 = PLOTS_DIR / "cers_distributions.png"
    fig.savefig(plot_path1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved distribution plot -> {plot_path1}")
    
    # 2. Scatter plot for Fidelity vs. Localization (Metric Conflict Frontier)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    gc_fid = [r["gradcam_fidelity"] for r in records]
    gc_loc = [r["gradcam_localization"] for r in records]
    ig_fid = [r["integrated_gradients_fidelity"] for r in records]
    ig_loc = [r["integrated_gradients_localization"] for r in records]
    
    ax.scatter(
        gc_fid, gc_loc, label="Grad-CAM", alpha=0.6,
        color="#1f77b4", edgecolor="w", s=50, marker="o"
    )
    ax.scatter(
        ig_fid, ig_loc, label="Integrated Gradients", alpha=0.6,
        color="#ff7f0e", edgecolor="w", s=50, marker="s"
    )
    
    # Draw centroids
    gc_fid_mean = summary["gradcam"]["fidelity_mean"]
    gc_loc_mean = summary["gradcam"]["localization_mean"]
    ig_fid_mean = summary["integrated_gradients"]["fidelity_mean"]
    ig_loc_mean = summary["integrated_gradients"]["localization_mean"]
    
    ax.scatter(
        [gc_fid_mean], [gc_loc_mean], color="#0d47a1", s=150,
        marker="X", edgecolor="black", label="Grad-CAM Centroid", zorder=5
    )
    ax.scatter(
        [ig_fid_mean], [ig_loc_mean], color="#e65100", s=150,
        marker="P", edgecolor="black", label="IG Centroid", zorder=5
    )
    
    ax.set_title("Explainability Conflict Frontier: Fidelity vs. Localization", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Fidelity (AOPC, Insertion, Deletion drops)", fontsize=11)
    ax.set_ylabel("Localization (Lung Focus, Saliency Entropy)", fontsize=11)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=True, facecolor="white", edgecolor="none")
    
    plot_path2 = PLOTS_DIR / "metric_conflicts_scatter.png"
    fig.savefig(plot_path2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved conflict scatter plot -> {plot_path2}")


def main():
    print("=" * 60)
    print("RUNNING CERS EVALUATION AND METRIC CONFLICT RESOLUTION PIPELINE")
    print("=" * 60)
    records, summary = process_cers_pipeline()
    generate_visualizations(records, summary)
    
    print("\n" + "=" * 60)
    print("CERS PIPELINE SUMMARY RESULTS")
    print("=" * 60)
    print(f"Total processed samples: {summary['n_samples']}")
    print(f"Overall Cross-Method Concordance (concordance_score): {summary['concordance_mean']:.4f}")
    print(f"Mean Spearman Rho: {summary['spearman_rho_mean']:.4f}")
    print(f"Mean Top-5% Overlap: {summary['top5pct_overlap_mean']:.4f}")
    print()
    
    for method in ("gradcam", "integrated_gradients"):
        m_summary = summary[method]
        name = "Grad-CAM" if method == "gradcam" else "Integrated Gradients"
        print(f"--- {name} ---")
        print(f"  Fidelity (Mean):     {m_summary['fidelity_mean']:.4f}")
        print(f"  Localization (Mean): {m_summary['localization_mean']:.4f}")
        print(f"  Robustness (Mean):   {m_summary['robustness_mean']:.4f}")
        print(f"  CERS (Mean):         {m_summary['cers_mean']:.4f} (std={m_summary['cers_std']:.4f})")
        print(f"  C-CERS (Mean):       {m_summary['ccers_mean']:.4f} (std={m_summary['ccers_std']:.4f})")
        print()
    print("=" * 60)


if __name__ == "__main__":
    main()
