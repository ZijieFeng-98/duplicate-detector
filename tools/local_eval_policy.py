#!/usr/bin/env python3
"""
Local Model Performance Policy Evaluator
Unbiased calculation-only test harness (no page suppression)
"""

import sys
import json
import re
import pandas as pd
import numpy as np
from pathlib import Path


def _page_num(p: str):
    """Extract page number from path"""
    if not isinstance(p, str):
        return None
    m = re.search(r'page[_-]?(\d+)', p, flags=re.I)
    return int(m.group(1)) if m else None


def evaluate(tsv_path: str, current_sim=0.96, current_ssim=0.90, current_phash=4):
    """
    Evaluate a detection run against unbiased performance policy.
    
    Policy:
    - FP proxy ≤ 35% (high-CLIP/low-SSIM/weak-geometry)
    - Cross-page ratio ≥ 40% (no page suppression)
    - Tier-A share ≥ 5%
    - Anchor precision ≥ 90% (exact pHash or strong ORB → Tier A)
    
    Returns suggestions for sim_threshold, ssim_threshold, phash_max_dist
    """
    
    if not Path(tsv_path).exists():
        print(f"ERROR: File not found: {tsv_path}", file=sys.stderr)
        return None
    
    df = pd.read_csv(tsv_path, sep="\t", low_memory=False)
    n = len(df)
    
    if n == 0:
        print("ERROR: Empty TSV file", file=sys.stderr)
        return None
    
    # Coerce signals
    clip = pd.to_numeric(df.get("Cosine_Similarity", 0), errors="coerce").fillna(0.0)
    ssim = pd.to_numeric(df.get("SSIM", np.nan), errors="coerce")
    phash = pd.to_numeric(df.get("Hamming_Distance", np.nan), errors="coerce")
    orb_in = pd.to_numeric(df.get("ORB_Inliers", 0), errors="coerce").fillna(0)
    inlier_ratio = pd.to_numeric(df.get("Inlier_Ratio", 0), errors="coerce").fillna(0.0)
    reproj = pd.to_numeric(df.get("Reproj_Error", np.nan), errors="coerce")
    tier = df.get("Tier", pd.Series(["Other"]*n, dtype=str)).astype(str)
    confocal_flag = df.get("Confocal_FP", pd.Series([False]*n)).astype(bool)
    
    # Cross-page ratio (no page suppression, purely diagnostic)
    pa = df["Path_A"].astype(str).map(_page_num)
    pb = df["Path_B"].astype(str).map(_page_num)
    cross_page = (pa != pb)
    cross_page_ratio = float(cross_page.mean()) if n else 0.0
    
    # FP proxy: high semantic but low structural & no exact/weak geometry
    low_ssim = ssim.lt(0.60) if ssim.notna().any() else pd.Series([False]*n)
    no_phash = phash.isna() | phash.gt(8) if phash.notna().any() else pd.Series([True]*n)
    weak_orb = orb_in.lt(15)
    fp_proxy = confocal_flag | (low_ssim & no_phash & weak_orb)
    fp_rate = float(fp_proxy.mean()) if n else 0.0
    
    # Anchor precision: exact pHash or strong ORB should almost always be Tier A
    strong_orb = (orb_in.ge(40) & inlier_ratio.ge(0.35) & (reproj.fillna(999).le(4.0)))
    exact_phash = phash.le(3) if phash.notna().any() else pd.Series([False]*n)
    anchors = exact_phash | strong_orb
    anchor_total = int(anchors.sum())
    anchor_tierA = int(((tier == "A") & anchors).sum())
    anchor_precision = (anchor_tierA / anchor_total) if anchor_total > 0 else None
    
    tierA_ratio = float((tier == "A").mean()) if n else 0.0
    tierB_ratio = float((tier == "B").mean()) if n else 0.0
    
    # Performance policy thresholds
    policy = dict(
        max_fp_rate=0.35,          # ≤35% FP proxy in discovery runs
        min_cross_page_ratio=0.40, # ≥40% cross-page coverage
        min_tierA_ratio=0.05,      # ≥5% Tier-A overall
        min_anchor_precision=0.90  # ≥90% of anchors must be Tier-A
    )
    
    # Pass/fail checks
    passes = dict(
        fp_rate = fp_rate <= policy["max_fp_rate"],
        cross_page = cross_page_ratio >= policy["min_cross_page_ratio"],
        tierA_ratio = tierA_ratio >= policy["min_tierA_ratio"],
        anchor_precision = True if anchor_precision is None else (anchor_precision >= policy["min_anchor_precision"])
    )
    
    overall_pass = all(passes.values())
    
    # Suggestions: push just above 95th percentile of the FP cohort
    margin_sim, margin_ssim = 0.003, 0.01
    
    suggested_sim = None
    if fp_proxy.any():
        clip_fp = clip[fp_proxy]
        if clip_fp.notna().sum() >= 10:
            pct95 = float(clip_fp.quantile(0.95))
            suggested_sim = round(max(current_sim, pct95 + margin_sim), 3)
    
    suggested_ssim = None
    if ssim.notna().any() and fp_proxy.any():
        ssim_fp = ssim[fp_proxy]
        if ssim_fp.notna().sum() >= 10:
            pct95 = float(ssim_fp.quantile(0.95))
            suggested_ssim = round(max(current_ssim, pct95 + margin_ssim), 3)
    
    suggested_phash = None
    if anchor_total > 0 and phash.notna().any():
        missed = (phash[anchors].fillna(999) > current_phash).sum()
        if missed > 0:
            # Relax slightly if anchors exceeded current max
            suggested_phash = max(current_phash, 4)
    
    # Build output
    out = dict(
        file=str(tsv_path),
        counts=dict(
            total_pairs=n,
            anchors=anchor_total,
            anchors_tierA=anchor_tierA,
            tierA=int((tier == "A").sum()),
            tierB=int((tier == "B").sum()),
            fp_proxy=int(fp_proxy.sum())
        ),
        metrics=dict(
            fp_rate=round(fp_rate, 3),
            cross_page_ratio=round(cross_page_ratio, 3),
            tierA_ratio=round(tierA_ratio, 3),
            tierB_ratio=round(tierB_ratio, 3),
            anchor_precision=round(anchor_precision, 3) if anchor_precision is not None else None
        ),
        policy=policy,
        pass_fail=passes,
        overall_pass=overall_pass,
        current_params=dict(
            sim_threshold=current_sim,
            ssim_threshold=current_ssim,
            phash_max_dist=current_phash
        ),
        suggestions=dict(
            sim_threshold=suggested_sim,
            ssim_threshold=suggested_ssim,
            phash_max_dist=suggested_phash
        )
    )
    
    return out


def print_report(result):
    """Pretty print the evaluation result"""
    if result is None:
        return
    
    print("="*80)
    print("🧪 MODEL PERFORMANCE EVALUATION")
    print("="*80)
    
    print(f"\n📊 Dataset: {result['file']}")
    print(f"   Total pairs: {result['counts']['total_pairs']}")
    print(f"   Tier A: {result['counts']['tierA']}")
    print(f"   Tier B: {result['counts']['tierB']}")
    print(f"   FP proxy: {result['counts']['fp_proxy']}")
    print(f"   Anchors: {result['counts']['anchors']} ({result['counts']['anchors_tierA']} in Tier A)")
    
    print("\n📈 Metrics:")
    print(f"   FP rate: {result['metrics']['fp_rate']:.1%} {'✅' if result['pass_fail']['fp_rate'] else '❌'} (target: ≤{result['policy']['max_fp_rate']:.0%})")
    print(f"   Cross-page ratio: {result['metrics']['cross_page_ratio']:.1%} {'✅' if result['pass_fail']['cross_page'] else '❌'} (target: ≥{result['policy']['min_cross_page_ratio']:.0%})")
    print(f"   Tier A share: {result['metrics']['tierA_ratio']:.1%} {'✅' if result['pass_fail']['tierA_ratio'] else '❌'} (target: ≥{result['policy']['min_tierA_ratio']:.0%})")
    
    anchor_prec = result['metrics']['anchor_precision']
    if anchor_prec is not None:
        print(f"   Anchor precision: {anchor_prec:.1%} {'✅' if result['pass_fail']['anchor_precision'] else '❌'} (target: ≥{result['policy']['min_anchor_precision']:.0%})")
    else:
        print(f"   Anchor precision: N/A (no anchors)")
    
    print(f"\n{'✅ PASS' if result['overall_pass'] else '❌ FAIL'}")
    
    print("\n⚙️ Current Parameters:")
    print(f"   sim_threshold: {result['current_params']['sim_threshold']}")
    print(f"   ssim_threshold: {result['current_params']['ssim_threshold']}")
    print(f"   phash_max_dist: {result['current_params']['phash_max_dist']}")
    
    print("\n💡 Suggested Parameters:")
    sugg = result['suggestions']
    if sugg['sim_threshold']:
        print(f"   sim_threshold: {sugg['sim_threshold']} (↑ from {result['current_params']['sim_threshold']})")
    else:
        print(f"   sim_threshold: {result['current_params']['sim_threshold']} (no change)")
    
    if sugg['ssim_threshold']:
        print(f"   ssim_threshold: {sugg['ssim_threshold']} (↑ from {result['current_params']['ssim_threshold']})")
    else:
        print(f"   ssim_threshold: {result['current_params']['ssim_threshold']} (no change)")
    
    if sugg['phash_max_dist']:
        print(f"   phash_max_dist: {sugg['phash_max_dist']} (→ from {result['current_params']['phash_max_dist']})")
    else:
        print(f"   phash_max_dist: {result['current_params']['phash_max_dist']} (no change)")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/local_eval_policy.py <final_merged_report.tsv> [current_sim] [current_ssim] [current_phash]")
        print("\nExample:")
        print("  python tools/local_eval_policy.py output/20241017_143052/final_merged_report.tsv 0.96 0.90 4")
        sys.exit(2)
    
    tsv = sys.argv[1]
    cur_sim = float(sys.argv[2]) if len(sys.argv) > 2 else 0.96
    cur_ssim = float(sys.argv[3]) if len(sys.argv) > 3 else 0.90
    cur_ph = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    
    result = evaluate(tsv, cur_sim, cur_ssim, cur_ph)
    
    if result:
        print_report(result)
        
        # Also output JSON for programmatic use
        json_out = Path(tsv).parent / "evaluation_result.json"
        with open(json_out, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n💾 JSON saved to: {json_out}")


