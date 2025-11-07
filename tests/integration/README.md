"""
README for integration tests.

Integration tests use real PDF files and actual duplicate detection pipeline.
These tests verify that the system works correctly with real-world data.
"""

# Test Data Location
# ==================
# Integration tests use PDF files from:
# /Users/zijiefeng/Desktop/Guo's lab/My_Research/Dr_Zhong
#
# Primary test file: "STM-Combined Figures.pdf"

# Running Integration Tests
# =========================
# 
# Run all integration tests:
#   pytest tests/integration/ -v
#
# Run specific test file:
#   pytest tests/integration/test_real_pdf.py -v
#
# Run with markers:
#   pytest -m integration -v
#
# Skip integration tests:
#   pytest -m "not integration" -v

# Test Structure
# ==============
# 
# 1. test_real_pdf.py
#    - Tests using actual PDF files
#    - Creates intentional duplicates
#    - Verifies detection pipeline
#
# 2. test_duplicate_creation.py (to be created)
#    - Tests creating various types of duplicates
#    - Verifies detection of each type

# Creating Test Duplicates
# ========================
# 
# The tests create intentional duplicates:
# - Exact duplicates (copy)
# - Rotated duplicates (90, 180, 270 degrees)
# - Noisy duplicates (add random noise)
# - Cropped duplicates (partial match)
#
# These are used to verify the detection pipeline correctly
# identifies duplicates using CLIP, pHash, and SSIM.

# Notes
# =====
# - Integration tests may take longer to run
# - They require actual PDF files
# - They create temporary output directories
# - Some tests may be skipped if required files are missing

