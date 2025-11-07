#!/usr/bin/env python3
"""
Quick setup script for local duplicate detection testing.
"""

import subprocess
import sys

def install_dependencies():
    """Install required dependencies."""
    print("Installing required dependencies...")
    print("This may take a few minutes...\n")
    
    dependencies = [
        "opencv-python-headless",
        "pillow",
        "pandas",
        "numpy",
        "imagehash",
        "scikit-image",
        "scikit-learn",
        "tqdm",
        "scipy",
        "pymupdf",
    ]
    
    for dep in dependencies:
        print(f"Installing {dep}...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", dep, "--quiet"], check=True)
            print(f"  ✓ {dep}")
        except subprocess.CalledProcessError:
            print(f"  ✗ Failed to install {dep}")
            return False
    
    print("\n✓ Basic dependencies installed")
    print("\nNote: PyTorch and CLIP will be installed automatically when needed")
    return True

if __name__ == "__main__":
    print("="*70)
    print("Local Setup for Duplicate Detection")
    print("="*70 + "\n")
    
    success = install_dependencies()
    
    if success:
        print("\n" + "="*70)
        print("Setup Complete!")
        print("="*70)
        print("\nYou can now run:")
        print("  python tests/integration/run_detection_local.py")
    else:
        print("\nSome dependencies failed to install.")
        print("Try installing manually:")
        print("  pip install -r requirements.txt")

