#!/usr/bin/env python3
"""
Diagnostic script to analyze Force OFF results
Run this in the same directory as your Streamlit app
"""

import json
from pathlib import Path
import sys

def analyze_results(output_dir):
    """
    Extract diagnostic information from duplicate detection results
    
    Args:
        output_dir: Path to the output directory (e.g., /tmp/duplicate_detector/output/20251019_004631/)
    """
    
    output_path = Path(output_dir)
    
    if not output_path.exists():
        print(f"❌ Error: Output directory not found: {output_path}")
        print("\n💡 Tip: Check /tmp/duplicate_detector/output/ for the latest run")
        return None
    
    print("="*70)
    print("🔍 DIAGNOSTIC REPORT - Force OFF Analysis")
    print("="*70)
    
    results = {
        "panels_extracted": 0,
        "panel_files": [],
        "pages_extracted": 0,
        "clip_candidates": 0,
        "ssim_validated": 0,
        "phash_matches": 0,
        "orb_matches": 0,
        "final_pairs": 0,
        "tier_a": 0,
        "tier_b": 0,
        "panels_folder_exists": False,
        "metadata_exists": False,
        "tsv_exists": False
    }
    
    # 1. Check RUN_METADATA.json
    print("\n📊 STEP 1: Checking RUN_METADATA.json")
    print("-"*70)
    
    metadata_path = output_path / "RUN_METADATA.json"
    if metadata_path.exists():
        results["metadata_exists"] = True
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            res = metadata.get('results', {})
            results["panels_extracted"] = res.get('panels', 0)
            results["pages_extracted"] = res.get('pages', 0)
            results["clip_candidates"] = res.get('stage1_clip', 0)
            results["ssim_validated"] = res.get('stage2_ssim', 0)
            results["phash_matches"] = res.get('stage3_phash', 0)
            results["orb_matches"] = res.get('stage4_orb', 0)
            results["final_pairs"] = res.get('merged_pairs', 0)
            results["tier_a"] = res.get('tier_a', 0)
            results["tier_b"] = res.get('tier_b', 0)
            
            print(f"✅ Metadata found")
            print(f"   Pages extracted: {results['pages_extracted']}")
            print(f"   Panels extracted: {results['panels_extracted']}")
            print(f"   Stage 1 (CLIP candidates): {results['clip_candidates']}")
            print(f"   Stage 2 (SSIM validated): {results['ssim_validated']}")
            print(f"   Stage 3 (pHash matches): {results['phash_matches']}")
            print(f"   Stage 4 (ORB matches): {results['orb_matches']}")
            print(f"   Final pairs: {results['final_pairs']}")
            if results['tier_a'] > 0 or results['tier_b'] > 0:
                print(f"   Tier A: {results['tier_a']}")
                print(f"   Tier B: {results['tier_b']}")
        except Exception as e:
            print(f"❌ Error reading metadata: {e}")
    else:
        print("❌ RUN_METADATA.json not found")
    
    # 2. Check panels folder
    print("\n📁 STEP 2: Checking panels/ folder")
    print("-"*70)
    
    panels_dir = output_path / "panels"
    if panels_dir.exists():
        results["panels_folder_exists"] = True
        
        # Find all panel PNG files
        panel_files = list(panels_dir.rglob("*.png"))
        results["panel_files"] = [str(p.relative_to(output_path)) for p in panel_files]
        
        print(f"✅ Panels folder exists")
        print(f"   Total panel files: {len(panel_files)}")
        
        # Group by page
        pages = {}
        for pf in panel_files:
            page = pf.parent.name
            if page not in pages:
                pages[page] = []
            pages[page].append(pf.name)
        
        print(f"   Pages with panels: {len(pages)}")
        
        # Show first few pages
        for page_name in sorted(pages.keys())[:5]:
            print(f"   • {page_name}: {len(pages[page_name])} panels")
            for panel in sorted(pages[page_name])[:3]:
                print(f"     - {panel}")
            if len(pages[page_name]) > 3:
                print(f"     ... and {len(pages[page_name])-3} more")
        
        if len(pages) > 5:
            print(f"   ... and {len(pages)-5} more pages")
    else:
        print("❌ Panels folder not found")
    
    # 3. Check TSV report
    print("\n📋 STEP 3: Checking final_merged_report.tsv")
    print("-"*70)
    
    tsv_path = output_path / "final_merged_report.tsv"
    if tsv_path.exists():
        results["tsv_exists"] = True
        try:
            import pandas as pd
            df = pd.read_csv(tsv_path, sep='\t')
            
            print(f"✅ TSV report found")
            print(f"   Total rows: {len(df)}")
            
            if len(df) > 0:
                print(f"\n   First 3 pairs:")
                for idx, row in df.head(3).iterrows():
                    print(f"   • {row.get('Image_A', 'N/A')} ↔ {row.get('Image_B', 'N/A')}")
                    if 'Cosine_Similarity' in row:
                        print(f"     CLIP: {row['Cosine_Similarity']:.3f}")
                    if 'SSIM' in row:
                        print(f"     SSIM: {row['SSIM']:.3f}")
                    if 'Tier' in row:
                        print(f"     Tier: {row['Tier']}")
            else:
                print("   ⚠️  Report is empty (0 pairs found)")
        except Exception as e:
            print(f"❌ Error reading TSV: {e}")
    else:
        print("❌ TSV report not found")
    
    # 4. Diagnosis
    print("\n" + "="*70)
    print("🎯 DIAGNOSIS")
    print("="*70)
    
    if results["panels_extracted"] == 0:
        print("\n❌ PROBLEM: No panels were extracted from PDF")
        print("\n💡 POSSIBLE CAUSES:")
        print("   1. PDF file is corrupted or unreadable")
        print("   2. Images are too small (< MIN_PANEL_AREA)")
        print("   3. Panel detection failed (no clear borders)")
        print("   4. All images are vector graphics (not raster)")
        print("\n🔧 SOLUTION:")
        print("   • Check if your PDF opens normally in Preview/Acrobat")
        print("   • Verify images have clear borders/separation")
        print("   • Try exporting PDF with higher quality raster images")
    
    elif results["clip_candidates"] == 0:
        print("\n❌ PROBLEM: Panels extracted but no CLIP candidates found")
        print(f"\n✅ Good news: {results['panels_extracted']} panels were extracted")
        print("❌ Bad news: None were semantically similar enough")
        print("\n💡 POSSIBLE CAUSES:")
        print("   1. CLIP threshold too high (current settings)")
        print("   2. Your duplicates are visually very different")
        print("   3. Same-page suppression filtered them out")
        print("\n🔧 SOLUTION:")
        print("   • Lower CLIP threshold to 0.80 (currently likely 0.85-0.94)")
        print("   • Disable same-page suppression if duplicates on same page")
        print("   • Verify your duplicates are actually in the panels/ folder")
    
    elif results["final_pairs"] == 0 and results["clip_candidates"] > 0:
        print("\n❌ PROBLEM: CLIP found candidates but they were filtered out")
        print(f"\n✅ Stage 1 (CLIP): {results['clip_candidates']} candidates")
        print(f"❌ Final pairs: 0")
        print("\n💡 FILTERING BREAKDOWN:")
        print(f"   • After SSIM: {results['ssim_validated']} pairs")
        print(f"   • After pHash: {results['phash_matches']} pairs")
        print(f"   • After ORB: {results['orb_matches']} pairs")
        print("\n🔧 SOLUTION:")
        if results['ssim_validated'] == 0:
            print("   • SSIM filtered everything! Lower SSIM to 0.50-0.60")
        if results['phash_matches'] < results['clip_candidates']:
            print("   • pHash is too strict! Increase max distance to 10-12")
        print("   • Try ultra-relaxed settings: CLIP=0.80, SSIM=0.50, pHash=12")
    
    else:
        print(f"\n✅ SUCCESS: {results['final_pairs']} pairs found")
        if results['tier_a'] > 0:
            print(f"   🚨 Tier A (Review Required): {results['tier_a']}")
        if results['tier_b'] > 0:
            print(f"   ⚠️  Tier B (Manual Check): {results['tier_b']}")
    
    # 5. Next steps
    print("\n" + "="*70)
    print("📤 NEXT STEPS")
    print("="*70)
    
    if results["panels_extracted"] > 0 and results["panels_folder_exists"]:
        print("\n1️⃣  Check if your duplicate images are in panels/ folder:")
        print(f"   • Location: {panels_dir}")
        print("   • Look for the 2 images you know are duplicates")
        print("   • Verify they were extracted as separate panel files")
        
        print("\n2️⃣  If your duplicates ARE in the panels/ folder:")
        print("   • The extraction worked correctly")
        print("   • Detection thresholds are too strict")
        print("   • Try these settings:")
        print("     CLIP: 0.80")
        print("     SSIM: 0.50")
        print("     pHash: 12")
        
        print("\n3️⃣  If your duplicates are NOT in the panels/ folder:")
        print("   • Panel detection split/merged them incorrectly")
        print("   • You may need manual extraction")
        print("   • Or try adjusting MIN_PANEL_AREA in backend")
    
    print("\n" + "="*70)
    
    return results


def find_latest_run():
    """Find the most recent output directory"""
    base_dir = Path("/tmp/duplicate_detector/output")
    
    if not base_dir.exists():
        print(f"❌ Output directory not found: {base_dir}")
        print("\n💡 Tip: Make sure you've run the detector at least once")
        return None
    
    # Find all subdirectories (format: YYYYMMDD_HHMMSS)
    runs = [d for d in base_dir.iterdir() if d.is_dir()]
    
    if not runs:
        print(f"❌ No runs found in: {base_dir}")
        return None
    
    # Sort by modification time (most recent first)
    latest = sorted(runs, key=lambda d: d.stat().st_mtime, reverse=True)[0]
    
    return latest


if __name__ == "__main__":
    print("🔬 Duplicate Detection Diagnostic Tool")
    print("="*70)
    
    # Check if output directory provided
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    else:
        # Try to find latest run
        print("\n🔍 Looking for latest run...")
        output_dir = find_latest_run()
        
        if output_dir is None:
            print("\n❌ Could not find output directory")
            print("\nUsage:")
            print("  python diagnostic.py [output_dir]")
            print("\nExample:")
            print("  python diagnostic.py /tmp/duplicate_detector/output/20251019_004631")
            sys.exit(1)
        
        print(f"✅ Found: {output_dir.name}")
    
    # Run analysis
    results = analyze_results(output_dir)
    
    if results:
        print("\n" + "="*70)
        print("✅ Diagnostic complete! Review the findings above.")
        print("="*70)



