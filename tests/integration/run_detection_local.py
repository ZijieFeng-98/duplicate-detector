#!/usr/bin/env python3
"""
Run duplicate detection locally on test panels.

This script runs the detection pipeline on the created test duplicates.
"""

import sys
from pathlib import Path
import subprocess

def main():
    """Run detection locally."""
    print(f"{'='*70}")
    print("Running Duplicate Detection Locally")
    print(f"{'='*70}\n")
    
    # Check if test panels exist
    test_panels_dir = Path("test_duplicate_detection/test_panels")
    if not test_panels_dir.exists():
        print(f"ERROR: Test panels directory not found: {test_panels_dir}")
        print("\nPlease run create_duplicates_standalone.py first:")
        print("  python tests/integration/create_duplicates_standalone.py")
        return False
    
    panels = list(test_panels_dir.glob("*.png"))
    if not panels:
        print(f"ERROR: No panels found in {test_panels_dir}")
        return False
    
    print(f"✓ Found {len(panels)} test panels")
    
    # Check if we can use the main pipeline
    pdf_path = Path("/Users/zijiefeng/Desktop/Guo's lab/My_Research/Dr_Zhong/STM-Combined Figures.pdf")
    
    if not pdf_path.exists():
        print(f"ERROR: PDF not found: {pdf_path}")
        print("\nWe'll need the PDF to run the full pipeline.")
        return False
    
    output_dir = Path("test_duplicate_detection/detection_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nPDF: {pdf_path}")
    print(f"Output: {output_dir}")
    print(f"Test panels: {len(panels)} images\n")
    
    # Run detection
    print("Running duplicate detection pipeline...")
    print("This may take a few minutes...\n")
    
    cmd = [
        sys.executable,
        "ai_pdf_panel_duplicate_check_AUTO.py",
        "--pdf", str(pdf_path),
        "--output", str(output_dir),
        "--preset", "balanced",
        "--sim-threshold", "0.94",  # Lower threshold for testing
        "--phash-max-dist", "5",
        "--use-phash-bundles",
        "--use-orb",
        "--use-tier-gating",
        "--dpi", "150"
    ]
    
    try:
        result = subprocess.run(cmd, check=False, text=True)
        
        if result.returncode == 0:
            print(f"\n{'='*70}")
            print("✓ Detection Complete!")
            print(f"{'='*70}\n")
            
            # Check results
            results_file = output_dir / "final_merged_report.tsv"
            if results_file.exists():
                print(f"Results saved to: {results_file}")
                
                # Try to read and show summary
                try:
                    import pandas as pd
                    df = pd.read_csv(results_file, sep='\t')
                    print(f"\nFound {len(df)} duplicate pairs")
                    
                    # Check for our test duplicates
                    print("\nChecking for test duplicates...")
                    exact_found = False
                    rotated_found = False
                    partial_found = False
                    wb_found = False
                    confocal_found = False
                    ihc_found = False
                    
                    for _, row in df.iterrows():
                        img_a = str(row.get('Image_A', ''))
                        img_b = str(row.get('Image_B', ''))
                        
                        if 'exact' in img_a.lower() or 'exact' in img_b.lower():
                            exact_found = True
                        if 'rotated' in img_a.lower() or 'rotated' in img_b.lower():
                            rotated_found = True
                        if 'partial' in img_a.lower() or 'partial' in img_b.lower():
                            partial_found = True
                        if 'wb' in img_a.lower() or 'wb' in img_b.lower():
                            wb_found = True
                        if 'confocal' in img_a.lower() or 'confocal' in img_b.lower():
                            confocal_found = True
                        if 'ihc' in img_a.lower() or 'ihc' in img_b.lower():
                            ihc_found = True
                    
                    print(f"\nDetection Summary:")
                    print(f"  Exact duplicates: {'✓' if exact_found else '✗'}")
                    print(f"  Rotated duplicates: {'✓' if rotated_found else '✗'}")
                    print(f"  Partial duplicates: {'✓' if partial_found else '✗'}")
                    print(f"  WB panels: {'✓' if wb_found else '✗'}")
                    print(f"  Confocal panels: {'✓' if confocal_found else '✗'}")
                    print(f"  IHC panels: {'✓' if ihc_found else '✗'}")
                    
                except Exception as e:
                    print(f"Could not parse results: {e}")
                    print("Please check the TSV file manually")
            else:
                print(f"Results file not found: {results_file}")
                print("Check the output directory for other files")
            
            print(f"\nView results:")
            print(f"  - TSV report: {results_file}")
            print(f"  - Panel manifest: {output_dir / 'panel_manifest.tsv'}")
            print(f"  - Comparisons: {output_dir / 'duplicate_comparisons'}")
            
        else:
            print(f"\n⚠ Detection completed with exit code {result.returncode}")
            print("Check the output above for errors")
            return False
        
        return True
        
    except FileNotFoundError:
        print(f"\nERROR: Could not find ai_pdf_panel_duplicate_check_AUTO.py")
        print("Make sure you're in the project root directory")
        return False
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return False
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

