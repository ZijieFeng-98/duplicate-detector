#!/usr/bin/env python3
"""
Local Test Runner & Scorer (No UI)
Runs the pipeline with parameter overrides, scores results, and suggests improvements.
"""

import argparse
import subprocess
import sys
import json
from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "ai_pdf_panel_duplicate_check_AUTO.py"


def _page_from_path(p):
    """Extract page number from path string."""
    s = str(p)
    # Expects segments like "page_19_" or "page-19-"
    for tok in s.replace("-", "_").split("_"):
        if tok.isdigit():
            return int(tok)
    return 0


def score_tsv(tsv: Path, focus_pages=None):
    """
    Score a final_merged_report.tsv file and compute metrics.
    
    Returns dict with:
    - pairs_total, tierA, tierB
    - cross_page_pct
    - fp_proxy_count, fp_proxy_pct
    - anchor_precision
    - focus_pages breakdown (if provided)
    """
    df = pd.read_csv(tsv, sep="\t")
    out = {}
    out["pairs_total"] = len(df)
    out["tierA"] = int((df.get("Tier") == "A").sum()) if "Tier" in df else 0
    out["tierB"] = int((df.get("Tier") == "B").sum()) if "Tier" in df else 0

    # Cross-page percentage
    if {"Path_A", "Path_B"}.issubset(df.columns):
        cross = sum(_page_from_path(a) != _page_from_path(b)
                    for a, b in zip(df["Path_A"], df["Path_B"]))
        out["cross_page_pct"] = 100.0 * cross / max(len(df), 1)
    else:
        out["cross_page_pct"] = None

    # FP proxy: high CLIP + low SSIM and weak pHash (confocal pattern)
    clip = pd.to_numeric(df.get("Cosine_Similarity", 0), errors="coerce").fillna(0)
    ssim = pd.to_numeric(df.get("SSIM", 0), errors="coerce").fillna(0)
    phash = pd.to_numeric(df.get("Hamming_Distance", 999), errors="coerce").fillna(999)
    fp_mask = (clip >= 0.96) & (ssim < 0.60) & (phash >= 10)
    out["fp_proxy_count"] = int(fp_mask.sum())
    out["fp_proxy_pct"] = 100.0 * out["fp_proxy_count"] / max(len(df), 1)

    # Anchor precision (Tier A among top-6 by CLIP)
    top = df.sort_values("Cosine_Similarity", ascending=False).head(6)
    out["anchor_precision"] = 100.0 * (top.get("Tier", pd.Series()).eq("A").sum()) / max(len(top), 1)

    # Focus pages breakdown
    if focus_pages:
        focus = []
        for pg in focus_pages:
            mask = df.apply(
                lambda r: _page_from_path(r["Path_A"]) == pg or _page_from_path(r["Path_B"]) == pg,
                axis=1
            )
            sub = df[mask]
            focus.append({
                "page": pg,
                "pairs": int(len(sub)),
                "tierA": int((sub.get("Tier") == "A").sum()),
                "best": None if sub.empty else {
                    "A": Path(sub.iloc[0]["Image_A"]).name if "Image_A" in sub else "",
                    "B": Path(sub.iloc[0]["Image_B"]).name if "Image_B" in sub else "",
                    "CLIP": float(pd.to_numeric(sub.iloc[0].get("Cosine_Similarity", 0), errors="coerce") or 0),
                    "SSIM": float(pd.to_numeric(sub.iloc[0].get("SSIM", 0), errors="coerce") or 0),
                    "Tier": sub.iloc[0].get("Tier", "")
                }
            })
        out["focus_pages"] = focus

    return out


def suggest(sim, ssim_thr, phash):
    """
    Suggest parameter improvements based on current values.
    
    - Tighten CLIP if FP proxy is high
    - Keep SSIM and pHash stable unless specific issues detected
    """
    rec = {}
    # Tighten CLIP if below recommended threshold
    rec["sim_threshold"] = 0.985 if sim < 0.985 else sim
    rec["ssim_threshold"] = ssim_thr
    rec["phash_max_dist"] = phash
    return rec


def main():
    ap = argparse.ArgumentParser(
        description="Local test runner and scorer (no UI)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--sim-threshold", type=float, default=0.96, help="CLIP similarity threshold")
    ap.add_argument("--ssim-threshold", type=float, default=0.90, help="SSIM threshold")
    ap.add_argument("--phash-max-dist", type=int, default=4, help="pHash max Hamming distance")
    ap.add_argument("--enable-orb-relax", action="store_true", help="Enable relaxed ORB path for tough partials")
    ap.add_argument("--suffix", type=str, default="", help="Custom suffix for output directory")
    ap.add_argument("--focus-pages", nargs="*", type=int, default=[19, 30], help="Pages to highlight in metrics")
    ap.add_argument("--pdf", type=str, help="Path to PDF file (optional override)")
    ap.add_argument("--output-dir", type=str, help="Base output directory (optional override)")
    args = ap.parse_args()

    # Generate suffix if not provided
    suffix = args.suffix or f"loc_{str(args.sim_threshold).replace('.', '')}"
    
    # Build command to run pipeline
    run_cmd = [
        sys.executable, str(APP),
        "--sim-threshold", str(args.sim_threshold),
        "--ssim-threshold", str(args.ssim_threshold),
        "--phash-max-dist", str(args.phash_max_dist),
        "--out-suffix", suffix,
        "--save-metrics-json"
    ]
    
    if args.enable_orb_relax:
        run_cmd.append("--enable-orb-relax")
    
    if args.pdf:
        run_cmd.extend(["--pdf", args.pdf])
    
    if args.output_dir:
        run_cmd.extend(["--output", args.output_dir])

    print("="*70)
    print("🔬 LOCAL TEST RUNNER (No UI)")
    print("="*70)
    print("▶ Running pipeline:")
    print("  " + " ".join(run_cmd))
    print()
    
    rc = subprocess.call(run_cmd)
    if rc != 0:
        print(f"\n❌ Pipeline failed with exit code {rc}")
        sys.exit(rc)

    # Locate output TSV
    # The app appends suffix to ai_clip_output, so we need to find it
    base_out = Path(APP).parent / "ai_clip_output"
    if args.output_dir:
        base_out = Path(args.output_dir)
    
    out_dir = base_out.parent / f"{base_out.name}_{suffix}" if suffix else base_out
    tsv = out_dir / "final_merged_report.tsv"
    
    if not tsv.exists():
        print(f"✖ No TSV produced: {tsv}")
        sys.exit(2)

    # Score the results
    report = score_tsv(tsv, focus_pages=args.focus_pages)
    
    print("\n" + "="*70)
    print("📊 RESULTS")
    print("="*70)
    print(json.dumps(report, indent=2))
    print("="*70)

    # Suggest improvements
    rec = suggest(args.sim_threshold, args.ssim_threshold, args.phash_max_dist)
    
    print("\n💡 SUGGESTED NEXT PARAMS:")
    print(json.dumps(rec, indent=2))
    
    # Evaluate against policy
    print("\n🎯 PERFORMANCE POLICY:")
    print(f"  FP Proxy: {report['fp_proxy_pct']:.1f}% {'✅ PASS' if report['fp_proxy_pct'] <= 35 else '❌ FAIL'} (target ≤35%)")
    if report['cross_page_pct'] is not None:
        print(f"  Cross-page: {report['cross_page_pct']:.1f}% {'✅ PASS' if report['cross_page_pct'] >= 40 else '❌ FAIL'} (target ≥40%)")
    tier_a_pct = 100.0 * report['tierA'] / max(report['pairs_total'], 1)
    print(f"  Tier A: {tier_a_pct:.1f}% {'✅ PASS' if tier_a_pct >= 5 else '❌ FAIL'} (target ≥5%)")
    print(f"  Anchor Precision: {report['anchor_precision']:.1f}% {'✅ PASS' if report['anchor_precision'] >= 90 else '❌ FAIL'} (target ≥90%)")
    
    # Write summary JSON
    summary = {
        "report": report,
        "suggest": rec,
        "policy_pass": {
            "fp_proxy": report['fp_proxy_pct'] <= 35,
            "cross_page": report['cross_page_pct'] >= 40 if report['cross_page_pct'] is not None else None,
            "tier_a": tier_a_pct >= 5,
            "anchor_precision": report['anchor_precision'] >= 90
        }
    }
    
    (out_dir / "local_score.json").write_text(json.dumps(summary, indent=2))
    print(f"\n✓ Saved score → {out_dir/'local_score.json'}")
    print("="*70)


if __name__ == "__main__":
    main()


