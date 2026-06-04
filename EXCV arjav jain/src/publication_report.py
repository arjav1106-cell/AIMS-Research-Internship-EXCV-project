"""Build publication-quality evaluation reports from classification and XAI results."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.config import METRICS_DIR

METHOD_GRADCAM = "Grad-CAM"
METHOD_IG = "Integrated Gradients"

# metric_key -> (gradcam_summary_key, ig_summary_key, higher_is_better, display_name)
METRIC_SPECS: dict[str, tuple[str, str, bool, str]] = {
    "aopc": ("gradcam_aopc_mean", "ig_aopc_mean", True, "AOPC"),
    "insertion": ("gradcam_insertion_auc_mean", "ig_insertion_auc_mean", True, "Insertion AUC"),
    "deletion": ("gradcam_deletion_auc_mean", "ig_deletion_auc_mean", False, "Deletion AUC"),
    "entropy": ("gradcam_entropy_mean", "ig_entropy_mean", False, "Entropy"),
    "lfs": ("gradcam_lfs_mean", "ig_lfs_mean", True, "Lung Focus Score"),
}


def _winner(cam: float, ig: float, higher_is_better: bool) -> str:
    if abs(cam - ig) < 1e-12:
        return "Tie"
    if higher_is_better:
        return METHOD_GRADCAM if cam > ig else METHOD_IG
    return METHOD_GRADCAM if cam < ig else METHOD_IG


def _fmt(value: float | None, decimals: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}"


def load_classification_metrics(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_xai_metrics(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def aggregate_method_metrics(
    per_sample: list[dict],
    method_key: str,
) -> dict[str, float]:
    if not per_sample:
        return {}
    keys = ["aopc", "insertion_auc", "deletion_auc", "entropy", "lfs"]
    out: dict[str, float] = {}
    for k in keys:
        out[k] = float(np.mean([s[method_key][k] for s in per_sample]))
    from src.robustness import aggregate_robustness_summary

    rob_agg = aggregate_robustness_summary(per_sample, method_key)
    out.update(rob_agg)
    return out


def aggregate_consistency(per_sample: list[dict]) -> dict[str, float]:
    if not per_sample:
        return {"spearman_concordance": 0.0, "top_k_overlap": 0.0}
    return {
        "spearman_concordance": float(np.mean([s["concordance"]["spearman_rho"] for s in per_sample])),
        "top_k_overlap": float(np.mean([s["concordance"]["top5pct_overlap"] for s in per_sample])),
    }


def build_method_block(
    method_name: str,
    metrics: dict[str, float],
) -> dict[str, Any]:
    faithfulness = {
        "aopc": metrics.get("aopc"),
        "insertion_auc": metrics.get("insertion_auc"),
        "deletion_auc": metrics.get("deletion_auc"),
    }
    localization = {
        "entropy": metrics.get("entropy"),
        "lung_focus_score": metrics.get("lfs"),
    }
    def _perturb_block(prefix: str) -> dict[str, float | None]:
        return {
            "mean": metrics.get(f"robustness_{prefix}"),
            "spearman": metrics.get(f"robustness_{prefix}_spearman"),
            "ssim": metrics.get(f"robustness_{prefix}_ssim"),
            "iou": metrics.get(f"robustness_{prefix}_iou"),
        }

    robustness = {
        "noise": _perturb_block("noise"),
        "brightness": _perturb_block("brightness"),
        "rotation": _perturb_block("rotation"),
        "mean": metrics.get("robustness_mean"),
    }
    return {
        "method": method_name,
        "faithfulness": faithfulness,
        "localization": localization,
        "robustness": robustness,
    }


def build_rankings(summary: dict) -> list[dict[str, Any]]:
    rankings = []
    for key, (cam_k, ig_k, higher, display) in METRIC_SPECS.items():
        cam_v = summary.get(cam_k)
        ig_v = summary.get(ig_k)
        if cam_v is None or ig_v is None:
            continue
        rankings.append({
            "metric": display,
            "metric_key": key,
            "grad_cam": float(cam_v),
            "integrated_gradients": float(ig_v),
            "higher_is_better": higher,
            "winner": _winner(float(cam_v), float(ig_v), higher),
        })

    cam_r = summary.get("gradcam_robustness_mean")
    ig_r = summary.get("ig_robustness_mean")
    if cam_r is not None and ig_r is not None:
        rankings.append({
            "metric": "Robustness (mean)",
            "metric_key": "robustness",
            "grad_cam": float(cam_r),
            "integrated_gradients": float(ig_r),
            "higher_is_better": True,
            "winner": _winner(float(cam_r), float(ig_r), True),
        })
    return rankings


GAP_EXPLANATIONS: dict[frozenset[str], str] = {
    frozenset({"aopc", "lfs"}): (
        "AOPC measures whether removing highlighted pixels reduces model confidence (faithfulness), "
        "while LFS measures whether mass falls inside an automatic lung mask (anatomical plausibility). "
        "Integrated Gradients often spread attribution across lung fields (higher AOPC potential but "
        "diffuse maps), whereas Grad-CAM peaks on discriminative conv features (sharper heatmaps, "
        "sometimes higher LFS when peaks align with lungs)."
    ),
    frozenset({"aopc", "insertion"}): (
        "AOPC reflects average confidence drop under deletion, while insertion AUC measures how "
        "quickly confidence rises when revealing salient pixels from a blurred baseline. A method "
        "can delete convincingly yet insert slowly if attributions are ordered differently than "
        "true causal pixels."
    ),
    frozenset({"deletion", "lfs"}): (
        "Low deletion AUC (confidence falls fast when masking) does not guarantee high LFS: the model "
        "may rely on non-lung cues (devices, borders) that are faithfully important to the classifier "
        "but clinically implausible."
    ),
    frozenset({"entropy", "lfs"}): (
        "Lower entropy indicates a peaked saliency distribution; higher LFS indicates mass inside lungs. "
        "A peaked map on a collar marker yields low entropy but low LFS; a diffuse lung-wide map can "
        "have higher entropy yet higher LFS."
    ),
    frozenset({"aopc", "entropy"}): (
        "Faithfulness (AOPC) and focus (entropy) diverge when the model uses few decisive pixels "
        "(low entropy, high AOPC) versus many weak contributors (higher entropy, moderate AOPC)."
    ),
    frozenset({"insertion", "deletion"}): (
        "Insertion and deletion probe opposite perturbation directions. Asymmetric performance suggests "
        "saliency ordering is better at revealing than suppressing evidence, or vice versa."
    ),
    frozenset({"lfs", "robustness"}): (
        "LFS depends on anatomical alignment; robustness measures stability under image perturbations. "
        "Grad-CAM hook activations can shift abruptly under rotation, while IG attributions may drift "
        "without changing lung overlap scores markedly."
    ),
}


def analyze_research_gaps(rankings: list[dict]) -> list[dict[str, Any]]:
    winners = {r["metric_key"]: r["winner"] for r in rankings if r["winner"] != "Tie"}
    gaps: list[dict[str, Any]] = []

    def method_short(name: str) -> str:
        return "Grad-CAM" if name == METHOD_GRADCAM else "IG"

    pairs = [
        ("aopc", "lfs"),
        ("aopc", "insertion"),
        ("deletion", "lfs"),
        ("entropy", "lfs"),
        ("aopc", "entropy"),
        ("insertion", "deletion"),
        ("lfs", "robustness"),
    ]
    for k1, k2 in pairs:
        if k1 not in winners or k2 not in winners:
            continue
        w1, w2 = winners[k1], winners[k2]
        if w1 == w2:
            continue
        r1 = next(r for r in rankings if r["metric_key"] == k1)
        r2 = next(r for r in rankings if r["metric_key"] == k2)
        explanation = GAP_EXPLANATIONS.get(
            frozenset({k1, k2}),
            f"{r1['metric']} favors {method_short(w1)} while {r2['metric']} favors {method_short(w2)}. "
            "This indicates the methods optimize different evaluation criteria.",
        )
        gaps.append({
            "metrics": [r1["metric"], r2["metric"]],
            "metric_keys": [k1, k2],
            f"{k1}_winner": w1,
            f"{k2}_winner": w2,
            "summary": f"{r1['metric']} favors {method_short(w1)}; {r2['metric']} favors {method_short(w2)}.",
            "explanation": explanation,
        })

    return gaps


def build_full_report(
    classification: dict,
    xai_data: dict,
    xai_summary: dict | None = None,
) -> dict[str, Any]:
    per_sample = xai_data.get("per_sample", [])
    summary = xai_summary or xai_data.get("summary", {})

    gradcam_m = aggregate_method_metrics(per_sample, "gradcam") if per_sample else {
        "aopc": summary.get("gradcam_aopc_mean"),
        "insertion_auc": summary.get("gradcam_insertion_auc_mean"),
        "deletion_auc": summary.get("gradcam_deletion_auc_mean"),
        "entropy": summary.get("gradcam_entropy_mean"),
        "lfs": summary.get("gradcam_lfs_mean"),
        "robustness_noise": summary.get("gradcam_robustness_noise_mean"),
        "robustness_brightness": summary.get("gradcam_robustness_brightness_mean"),
        "robustness_rotation": summary.get("gradcam_robustness_rotation_mean"),
        "robustness_mean": summary.get("gradcam_robustness_mean"),
    }
    ig_m = aggregate_method_metrics(per_sample, "integrated_gradients") if per_sample else {
        "aopc": summary.get("ig_aopc_mean"),
        "insertion_auc": summary.get("ig_insertion_auc_mean"),
        "deletion_auc": summary.get("ig_deletion_auc_mean"),
        "entropy": summary.get("ig_entropy_mean"),
        "lfs": summary.get("ig_lfs_mean"),
        "robustness_noise": summary.get("ig_robustness_noise_mean"),
        "robustness_brightness": summary.get("ig_robustness_brightness_mean"),
        "robustness_rotation": summary.get("ig_robustness_rotation_mean"),
        "robustness_mean": summary.get("ig_robustness_mean"),
    }

    # Fill from CSV aggregates if per-sample robustness missing
    from src.robustness import load_robustness_aggregates_from_csv

    csv_agg = load_robustness_aggregates_from_csv()
    if csv_agg.get("Grad-CAM"):
        for k, v in csv_agg["Grad-CAM"].items():
            if v is not None and gradcam_m.get(k) is None:
                gradcam_m[k] = v
    if csv_agg.get("Integrated Gradients"):
        for k, v in csv_agg["Integrated Gradients"].items():
            if v is not None and ig_m.get(k) is None:
                ig_m[k] = v

    for prefix, target in [("gradcam", gradcam_m), ("ig", ig_m)]:
        for rk in ["noise", "brightness", "rotation", "mean"]:
            sk = f"{prefix}_robustness_{rk}_mean" if rk != "mean" else f"{prefix}_robustness_mean"
            if target.get(f"robustness_{rk}") is None and summary.get(sk) is not None:
                target[f"robustness_{rk}"] = summary[sk]

    consistency = aggregate_consistency(per_sample) if per_sample else {
        "spearman_concordance": summary.get("concordance_spearman_mean", 0.0),
        "top_k_overlap": summary.get("concordance_top5pct_overlap_mean", 0.0),
    }

    merged_summary = dict(summary)
    merged_summary.update({
        "gradcam_aopc_mean": gradcam_m.get("aopc"),
        "gradcam_insertion_auc_mean": gradcam_m.get("insertion_auc"),
        "gradcam_deletion_auc_mean": gradcam_m.get("deletion_auc"),
        "gradcam_entropy_mean": gradcam_m.get("entropy"),
        "gradcam_lfs_mean": gradcam_m.get("lfs"),
        "ig_aopc_mean": ig_m.get("aopc"),
        "ig_insertion_auc_mean": ig_m.get("insertion_auc"),
        "ig_deletion_auc_mean": ig_m.get("deletion_auc"),
        "ig_entropy_mean": ig_m.get("entropy"),
        "ig_lfs_mean": ig_m.get("lfs"),
        "gradcam_robustness_mean": gradcam_m.get("robustness_mean"),
        "ig_robustness_mean": ig_m.get("robustness_mean"),
        "concordance_spearman_mean": consistency["spearman_concordance"],
        "concordance_top5pct_overlap_mean": consistency["top_k_overlap"],
    })

    rankings = build_rankings(merged_summary)
    gaps = analyze_research_gaps(rankings)

    if consistency["spearman_concordance"] < 0.3:
        gaps.append({
            "metrics": ["Spearman Concordance", "Top-k Overlap"],
            "metric_keys": ["concordance", "overlap"],
            "summary": (
                f"Low cross-method agreement (Spearman={consistency['spearman_concordance']:.3f}, "
                f"top-5% overlap={consistency['top_k_overlap']:.3f})."
            ),
            "explanation": (
                "Grad-CAM uses gradient-weighted class activation hooks on convolutional features; "
                "Integrated Gradients integrate gradients along a path from a baseline. They often "
                "highlight different regions even when both are faithful, which undermines "
                "interchangeable clinical interpretation."
            ),
        })

    return {
        "meta": {
            "n_xai_samples": len(per_sample) or summary.get("n_samples"),
            "xai_subset_accuracy": summary.get("accuracy_subset"),
        },
        "classification": {
            "accuracy": classification.get("accuracy"),
            "precision": classification.get("precision_macro"),
            "recall": classification.get("recall_macro"),
            "f1": classification.get("f1_macro"),
            "per_class": {
                "class_names": classification.get("class_names", []),
                "precision": classification.get("precision_per_class", []),
                "recall": classification.get("recall_per_class", []),
                "f1": classification.get("f1_per_class", []),
            },
        },
        "grad_cam": build_method_block(METHOD_GRADCAM, gradcam_m),
        "integrated_gradients": build_method_block(METHOD_IG, ig_m),
        "consistency": consistency,
        "method_ranking": rankings,
        "research_gap_analysis": gaps,
    }


def format_text_report(report: dict) -> str:
    c = report["classification"]
    gc = report["grad_cam"]
    ig = report["integrated_gradients"]
    con = report["consistency"]
    lines: list[str] = []

    def section(title: str) -> None:
        bar = "=" * 48
        lines.append(bar)
        lines.append(title)
        lines.append(bar)
        lines.append("")

    def method_section(block: dict, title: str) -> None:
        section(title)
        lines.append("## Faithfulness Metrics")
        lines.append("")
        lines.append(f"AOPC:           {_fmt(block['faithfulness']['aopc'])}")
        lines.append(f"Insertion AUC:  {_fmt(block['faithfulness']['insertion_auc'])}")
        lines.append(f"Deletion AUC:   {_fmt(block['faithfulness']['deletion_auc'])}")
        lines.append("")
        lines.append("## Localization Metrics")
        lines.append("")
        lines.append(f"Entropy:            {_fmt(block['localization']['entropy'])}")
        lines.append(f"Lung Focus Score:   {_fmt(block['localization']['lung_focus_score'])}")
        lines.append("")
        lines.append("## Robustness Metrics")
        lines.append("")
        rob = block["robustness"]
        for label, key in [
            ("Noise", "noise"),
            ("Brightness", "brightness"),
            ("Rotation", "rotation"),
        ]:
            p = rob.get(key) or {}
            if p.get("mean") is None:
                lines.append(f"{label} Robustness:       N/A")
                continue
            lines.append(f"{label} Robustness (mean):  {_fmt(p.get('mean'))}")
            lines.append(f"  Spearman: {_fmt(p.get('spearman'))}  "
                         f"SSIM: {_fmt(p.get('ssim'))}  IoU: {_fmt(p.get('iou'))}")
        if rob.get("mean") is not None:
            lines.append(f"Overall Mean Robustness: {_fmt(rob.get('mean'))}")
        lines.append("")

    section("CLASSIFICATION PERFORMANCE")
    lines.append(f"Accuracy:   {_fmt(c['accuracy'])}")
    lines.append(f"Precision:  {_fmt(c['precision'])}")
    lines.append(f"Recall:     {_fmt(c['recall'])}")
    lines.append(f"F1:         {_fmt(c['f1'])}")
    lines.append("")

    method_section(gc, "GRAD-CAM EVALUATION")
    method_section(ig, "INTEGRATED GRADIENTS EVALUATION")

    section("CONSISTENCY ANALYSIS")
    lines.append(f"Spearman Concordance:  {_fmt(con['spearman_concordance'])}")
    lines.append(f"Top-k Overlap:         {_fmt(con['top_k_overlap'])}")
    lines.append("")

    section("METHOD RANKING")
    lines.append(f"{'Metric':<22} {'Winner':<22} {'Grad-CAM':>12} {'IG':>12}")
    lines.append("-" * 70)
    for r in report["method_ranking"]:
        lines.append(
            f"{r['metric']:<22} {r['winner']:<22} "
            f"{_fmt(r['grad_cam']):>12} {_fmt(r['integrated_gradients']):>12}"
        )
    lines.append("")

    section("RESEARCH GAP ANALYSIS")
    gaps = report["research_gap_analysis"]
    if not gaps:
        lines.append("No major cross-metric disagreements detected among ranked winners.")
    else:
        for i, g in enumerate(gaps, 1):
            lines.append(f"{i}. {g['summary']}")
            lines.append(f"   {g['explanation']}")
            lines.append("")
    lines.append("")

    return "\n".join(lines)


def export_csvs(report: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Summary comparison
    comp_path = out_dir / "publication_method_comparison.csv"
    gc, ig = report["grad_cam"], report["integrated_gradients"]
    with open(comp_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["category", "metric", "grad_cam", "integrated_gradients"])
        for cat in ["faithfulness", "localization"]:
            for key in gc[cat]:
                w.writerow([cat, key, gc[cat].get(key), ig[cat].get(key)])
        for perturb in ["noise", "brightness", "rotation", "mean"]:
            p_gc = gc["robustness"].get(perturb)
            p_ig = ig["robustness"].get(perturb)
            if isinstance(p_gc, dict):
                for metric in ["mean", "spearman", "ssim", "iou"]:
                    w.writerow([
                        "robustness",
                        f"{perturb}_{metric}",
                        (p_gc or {}).get(metric),
                        (p_ig or {}).get(metric),
                    ])
            else:
                w.writerow(["robustness", perturb, p_gc, p_ig])

    # Rankings
    rank_path = out_dir / "publication_method_ranking.csv"
    with open(rank_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["metric", "winner", "grad_cam", "integrated_gradients", "higher_is_better"],
        )
        w.writeheader()
        for r in report["method_ranking"]:
            w.writerow({
                "metric": r["metric"],
                "winner": r["winner"],
                "grad_cam": r["grad_cam"],
                "integrated_gradients": r["integrated_gradients"],
                "higher_is_better": r["higher_is_better"],
            })

    # Classification
    cls_path = out_dir / "publication_classification.csv"
    with open(cls_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        c = report["classification"]
        for k in ["accuracy", "precision", "recall", "f1"]:
            w.writerow([k, c[k]])

    # Research gaps
    gap_path = out_dir / "publication_research_gaps.csv"
    with open(gap_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "summary", "explanation"])
        for i, g in enumerate(report["research_gap_analysis"], 1):
            w.writerow([i, g["summary"], g["explanation"]])

    # Flat summary (one row)
    flat_path = out_dir / "publication_metrics_summary.csv"
    row: dict[str, Any] = {}
    row["accuracy"] = report["classification"]["accuracy"]
    row["precision"] = report["classification"]["precision"]
    row["recall"] = report["classification"]["recall"]
    row["f1"] = report["classification"]["f1"]
    row["spearman_concordance"] = report["consistency"]["spearman_concordance"]
    row["top_k_overlap"] = report["consistency"]["top_k_overlap"]
    for prefix, block in [("gradcam", gc), ("ig", ig)]:
        for cat in ["faithfulness", "localization"]:
            for k, v in block[cat].items():
                row[f"{prefix}_{cat}_{k}"] = v
        for perturb, pdata in block["robustness"].items():
            if isinstance(pdata, dict):
                for mk, mv in pdata.items():
                    row[f"{prefix}_robustness_{perturb}_{mk}"] = mv
            else:
                row[f"{prefix}_robustness_{perturb}"] = pdata
    with open(flat_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)


def save_publication_outputs(report: dict, out_dir: Path | None = None) -> dict[str, Path]:
    out_dir = out_dir or METRICS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "publication_evaluation_report.json"
    txt_path = out_dir / "publication_evaluation_report.txt"
    table_path = out_dir / "publication_evaluation_tables.txt"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    text = format_text_report(report)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)

    with open(table_path, "w", encoding="utf-8") as f:
        f.write(text)

    export_csvs(report, out_dir)

    return {
        "json": json_path,
        "txt": txt_path,
        "tables": table_path,
    }
