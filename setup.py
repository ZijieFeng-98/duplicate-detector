"""
Setup script for duplicate-detector package.

This file provides backward compatibility for older build tools.
Modern builds should use pyproject.toml directly.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="duplicate-detector",
    version="1.0.0",
    description="Professional scientific figure duplicate detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Duplicate Detector Contributors",
    license="MIT",
    python_requires=">=3.12",
    packages=find_packages(exclude=["tests", "tests.*", "docs", "docs.*"]),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "open-clip-torch>=2.20.0",
        "opencv-python>=4.8.0",
        "pillow>=10.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-image>=0.21.0",
        "scipy>=1.11.0",
        "imagehash>=4.3.1",
        "pymupdf>=1.23.0",
        "pydantic>=2.0.0",
        "pyyaml>=6.0.0",
        "python-dotenv>=1.0.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
            "mypy>=1.5.0",
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    include_package_data=True,
    zip_safe=False,
)

