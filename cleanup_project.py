#!/usr/bin/env python3
"""
Clean up the entire duplicate detector project.
Removes temporary files, cache, and organizes the project structure.
"""

import shutil
from pathlib import Path
import os

def cleanup_project():
    """Clean up the entire project."""
    print("="*70)
    print("Cleaning Up Duplicate Detector Project")
    print("="*70 + "\n")
    
    base_dir = Path(".")
    
    # Remove Python cache
    print("1. Removing Python cache files...")
    cache_count = 0
    for pycache in base_dir.rglob("__pycache__"):
        try:
            shutil.rmtree(pycache)
            cache_count += 1
        except:
            pass
    
    for pyc in base_dir.rglob("*.pyc"):
        try:
            pyc.unlink()
            cache_count += 1
        except:
            pass
    
    print(f"   ✓ Removed {cache_count} cache files/directories")
    
    # Remove system files
    print("\n2. Removing system files...")
    system_files = list(base_dir.rglob(".DS_Store"))
    system_files.extend(base_dir.rglob("*.tmp"))
    for f in system_files:
        try:
            f.unlink()
        except:
            pass
    print(f"   ✓ Removed {len(system_files)} system files")
    
    # Remove log files
    print("\n3. Removing log files...")
    log_files = list(base_dir.rglob("*.log"))
    for f in log_files:
        try:
            f.unlink()
        except:
            pass
    print(f"   ✓ Removed {len(log_files)} log files")
    
    # Remove temporary output directories
    print("\n4. Removing temporary output directories...")
    temp_dirs = [
        "test_output",
        "duplicate_detector_output",
        "output",
        "results",
        ".pytest_cache",
        ".coverage",
        "htmlcov",
        "dist",
        "build"
    ]
    
    removed_dirs = []
    for dir_name in temp_dirs:
        dir_path = base_dir / dir_name
        if dir_path.exists() and dir_path.is_dir():
            try:
                shutil.rmtree(dir_path)
                removed_dirs.append(dir_name)
                print(f"   ✓ Removed: {dir_name}/")
            except Exception as e:
                print(f"   ✗ Failed to remove {dir_name}: {e}")
    
    # Remove egg-info if exists
    egg_info_dirs = list(base_dir.glob("*.egg-info"))
    for egg_info in egg_info_dirs:
        try:
            shutil.rmtree(egg_info)
            print(f"   ✓ Removed: {egg_info.name}/")
        except:
            pass
    
    # Summary
    print("\n" + "="*70)
    print("Cleanup Summary")
    print("="*70)
    print(f"✓ Python cache: {cache_count} files/directories")
    print(f"✓ System files: {len(system_files)} files")
    print(f"✓ Log files: {len(log_files)} files")
    print(f"✓ Temporary directories: {len(removed_dirs)} directories")
    
    print("\n" + "="*70)
    print("Cleanup Complete!")
    print("="*70)
    print("\nNote: Test data in test_duplicate_detection/ was preserved.")
    print("      Essential project files were kept.")

if __name__ == "__main__":
    cleanup_project()

