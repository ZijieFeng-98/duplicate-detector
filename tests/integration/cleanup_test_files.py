#!/usr/bin/env python3
"""
Clean up test files and temporary directories.
Keeps essential test files and removes temporary outputs.
"""

import shutil
from pathlib import Path

def cleanup():
    """Clean up temporary files."""
    print("="*70)
    print("Cleaning Up Test Files")
    print("="*70 + "\n")
    
    # Directories to clean (but keep structure)
    base_dir = Path("test_duplicate_detection")
    
    if not base_dir.exists():
        print("No test_duplicate_detection directory found.")
        return
    
    # Keep these directories
    keep_dirs = [
        "intentional_duplicates",  # Keep duplicates
        "test_panels",              # Keep test panels
        "pages"                     # Keep original pages
    ]
    
    # Remove these directories (temporary outputs)
    remove_dirs = [
        "detection_results",        # Temporary detection results
        "initial_run",              # Temporary initial run
        "cache"                     # Cache files
    ]
    
    print("Removing temporary directories...")
    for dir_name in remove_dirs:
        dir_path = base_dir / dir_name
        if dir_path.exists():
            try:
                shutil.rmtree(dir_path)
                print(f"  ✓ Removed: {dir_name}/")
            except Exception as e:
                print(f"  ✗ Failed to remove {dir_name}: {e}")
    
    # Clean up Python cache
    print("\nCleaning Python cache files...")
    for pycache in base_dir.rglob("__pycache__"):
        try:
            shutil.rmtree(pycache)
            print(f"  ✓ Removed: {pycache}")
        except:
            pass
    
    for pyc in base_dir.rglob("*.pyc"):
        try:
            pyc.unlink()
            print(f"  ✓ Removed: {pyc}")
        except:
            pass
    
    # Clean up log files
    print("\nCleaning log files...")
    for log_file in base_dir.rglob("*.log"):
        try:
            log_file.unlink()
            print(f"  ✓ Removed: {log_file}")
        except:
            pass
    
    # Summary
    print("\n" + "="*70)
    print("Cleanup Summary")
    print("="*70)
    print("\nKept:")
    for dir_name in keep_dirs:
        dir_path = base_dir / dir_name
        if dir_path.exists():
            file_count = len(list(dir_path.rglob("*"))) - len(list(dir_path.rglob("*/")))
            print(f"  ✓ {dir_name}/ ({file_count} files)")
    
    print("\nRemoved:")
    for dir_name in remove_dirs:
        print(f"  ✓ {dir_name}/")
    
    print("\n" + "="*70)
    print("Cleanup Complete!")
    print("="*70)

if __name__ == "__main__":
    cleanup()

