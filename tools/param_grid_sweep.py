#!/usr/bin/env python3
"""
Parameter Grid Sweep for Fully Automatic Tuning
Runs multiple parameter combinations and ranks by policy compliance.
"""

import argparse
import subprocess
import sys
import json
import time
from pathlib import Path
from itertools import product
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "local_run_and_score.py"


def run_config(sim, ssim, phash, orb_relax, suffix, pdf=None, output_dir=None):
    """
    Run one configuration and return its score.
    
    Returns:
        dict with policy_pass metrics and summary, or None if failed
    """
    cmd = [
        sys.executable, str(RUNNER),
        "--sim-threshold", str(sim),
        "--ssim-threshold", str(ssim),
        "--phash-max-dist", str(phash),
        "--suffix", suffix
    ]
    
    if orb_relax:
        cmd.append("--enable-orb-relax")
    
    if pdf:
        cmd.extend(["--pdf", pdf])
    
    if output_dir:
        cmd.extend(["--output-dir", output_dir])
    
    print(f"\n{'='*70}")
    print(f"🧪 Testing: sim={sim}, ssim={ssim}, phash={phash}, orb_relax={orb_relax}")
    print(f"{'='*70}")
    
    start = time.time()
    rc = subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    elapsed = time.time() - start
    
    if rc != 0:
        print(f"❌ Failed (exit code {rc})")
        return None
    
    # Read results
    base_out = Path(output_dir) if output_dir else (ROOT / "ai_clip_output")
    out_dir = base_out.parent / f"{base_out.name}_{suffix}"
    score_file = out_dir / "local_score.json"
    
    if not score_file.exists():
        print(f"❌ No score file: {score_file}")
        return None
    
    with open(score_file) as f:
        data = json.load(f)
    
    # Extract key metrics
    report = data.get("report", {})
    policy = data.get("policy_pass", {})
    
    result = {
        "config": {
            "sim_threshold": sim,
            "ssim_threshold": ssim,
            "phash_max_dist": phash,
            "enable_orb_relax": orb_relax
        },
        "metrics": {
            "pairs_total": report.get("pairs_total", 0),
            "tierA": report.get("tierA", 0),
            "tierB": report.get("tierB", 0),
            "fp_proxy_pct": report.get("fp_proxy_pct", 100.0),
            "cross_page_pct": report.get("cross_page_pct", 0.0),
            "anchor_precision": report.get("anchor_precision", 0.0),
        },
        "policy_pass": policy,
        "score": sum(1 for v in policy.values() if v is True),  # Count passes
        "elapsed_sec": elapsed,
        "output_dir": str(out_dir)
    }
    
    print(f"✅ Complete in {elapsed:.1f}s")
    print(f"   FP: {result['metrics']['fp_proxy_pct']:.1f}% "
          f"XP: {result['metrics']['cross_page_pct']:.1f}% "
          f"AP: {result['metrics']['anchor_precision']:.1f}% "
          f"Policy: {result['score']}/4")
    
    return result


def main():
    ap = argparse.ArgumentParser(
        description="Parameter grid sweep for automatic tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--sim-range", nargs="+", type=float, default=[0.96, 0.98, 0.985, 0.99],
                    help="CLIP thresholds to test")
    ap.add_argument("--ssim-range", nargs="+", type=float, default=[0.90],
                    help="SSIM thresholds to test")
    ap.add_argument("--phash-range", nargs="+", type=int, default=[4],
                    help="pHash max distances to test")
    ap.add_argument("--test-orb-relax", action="store_true",
                    help="Also test each config with ORB relax enabled")
    ap.add_argument("--pdf", type=str, help="Path to PDF file")
    ap.add_argument("--output-dir", type=str, help="Base output directory")
    ap.add_argument("--max-configs", type=int, default=20,
                    help="Max number of configurations to test (safety limit)")
    args = ap.parse_args()
    
    # Generate parameter grid
    orb_relax_options = [False, True] if args.test_orb_relax else [False]
    grid = list(product(args.sim_range, args.ssim_range, args.phash_range, orb_relax_options))
    
    if len(grid) > args.max_configs:
        print(f"⚠️  Grid has {len(grid)} configs, limiting to {args.max_configs}")
        grid = grid[:args.max_configs]
    
    print("="*70)
    print("🔬 PARAMETER GRID SWEEP")
    print("="*70)
    print(f"Total configurations to test: {len(grid)}")
    print(f"SIM range: {args.sim_range}")
    print(f"SSIM range: {args.ssim_range}")
    print(f"pHash range: {args.phash_range}")
    print(f"ORB relax: {orb_relax_options}")
    print("="*70)
    
    # Run all configurations
    results = []
    for i, (sim, ssim, phash, orb_relax) in enumerate(grid, 1):
        suffix = f"grid_{i:02d}_s{str(sim).replace('.', '')}_ss{str(ssim).replace('.', '')}_p{phash}{'_orb' if orb_relax else ''}"
        
        result = run_config(sim, ssim, phash, orb_relax, suffix, 
                           pdf=args.pdf, output_dir=args.output_dir)
        
        if result:
            results.append(result)
    
    # Sort by score (descending), then by FP proxy (ascending)
    results.sort(key=lambda r: (-r['score'], r['metrics']['fp_proxy_pct']))
    
    # Display ranked results
    print("\n" + "="*70)
    print("📊 RANKED RESULTS (Best → Worst)")
    print("="*70)
    
    for i, r in enumerate(results, 1):
        cfg = r['config']
        m = r['metrics']
        p = r['policy_pass']
        
        status = "✅ OPTIMAL" if r['score'] == 4 else f"⚠️  {r['score']}/4 pass"
        
        print(f"\n#{i}: {status}")
        print(f"  Config: sim={cfg['sim_threshold']}, ssim={cfg['ssim_threshold']}, "
              f"phash={cfg['phash_max_dist']}, orb_relax={cfg['enable_orb_relax']}")
        print(f"  Metrics:")
        print(f"    FP Proxy: {m['fp_proxy_pct']:.1f}% {'✅' if p.get('fp_proxy') else '❌'}")
        print(f"    Cross-page: {m['cross_page_pct']:.1f}% {'✅' if p.get('cross_page') else '❌'}")
        print(f"    Tier A: {100.0 * m['tierA'] / max(m['pairs_total'], 1):.1f}% {'✅' if p.get('tier_a') else '❌'}")
        print(f"    Anchor Precision: {m['anchor_precision']:.1f}% {'✅' if p.get('anchor_precision') else '❌'}")
        print(f"  Output: {r['output_dir']}")
    
    # Save summary
    summary_file = Path("param_grid_summary.json")
    with open(summary_file, 'w') as f:
        json.dump({
            "grid_size": len(grid),
            "tested": len(results),
            "results": results,
            "best_config": results[0]['config'] if results else None,
            "best_metrics": results[0]['metrics'] if results else None
        }, f, indent=2)
    
    print(f"\n✓ Saved summary → {summary_file}")
    
    # Recommendation
    if results:
        best = results[0]
        if best['score'] == 4:
            print("\n" + "="*70)
            print("🎯 RECOMMENDED CONFIG (all policy gates pass):")
            print("="*70)
            print(f"  sim_threshold = {best['config']['sim_threshold']}")
            print(f"  ssim_threshold = {best['config']['ssim_threshold']}")
            print(f"  phash_max_dist = {best['config']['phash_max_dist']}")
            print(f"  enable_orb_relax = {best['config']['enable_orb_relax']}")
            print("="*70)
        else:
            print("\n⚠️  No configuration passed all 4 policy gates.")
            print("   Best config passed {}/{} gates.".format(best['score'], 4))
            print("   Consider:")
            print("   - Expanding SIM range (try 0.975, 0.98)")
            print("   - Testing with --test-orb-relax")
            print("   - Adjusting SSIM range (try 0.85, 0.88)")


if __name__ == "__main__":
    main()


