#!/bin/bash
# Quick smoke test - runs in ~2-3 minutes
# Usage: ./quick_test.sh

echo "🔥 QUICK SMOKE TEST"
echo "==================="

# Configuration
PDF_PATH="/Users/zijiefeng/Desktop/Guo's lab/My_Research/Dr_Zhong/PUA-STM-Combined Figures .pdf"
OUTPUT_DIR="./test_output/quick_test"

# Clean previous test
echo "🧹 Cleaning previous test output..."
rm -rf "$OUTPUT_DIR" 2>/dev/null

# Run with permissive settings (should find something)
echo "🚀 Running detection pipeline..."
python ai_pdf_panel_duplicate_check_AUTO.py \
  --pdf "$PDF_PATH" \
  --output "$OUTPUT_DIR" \
  --sim-threshold 0.96 \
  --ssim-threshold 0.9 \
  --phash-max-dist 4 \
  --use-phash-bundles \
  --use-orb \
  --use-tier-gating \
  --enable-cache \
  --no-auto-open

# Check exit code
if [ $? -ne 0 ]; then
    echo "❌ FAILED: Pipeline exited with error"
    exit 1
fi

# Check results
TSV_FILE="$OUTPUT_DIR/final_merged_report.tsv"
METADATA_FILE="$OUTPUT_DIR/RUN_METADATA.json"

echo ""
echo "📊 RESULTS CHECK"
echo "================"

if [ -f "$TSV_FILE" ]; then
    lines=$(wc -l < "$TSV_FILE" | tr -d ' ')
    pairs=$((lines - 1))
    
    if [ "$pairs" -gt 0 ]; then
        echo "✅ SUCCESS: Found $pairs duplicate pairs"
        echo ""
        echo "First 5 results:"
        head -n 6 "$TSV_FILE" | column -t -s $'\t'
    else
        echo "⚠️  WARNING: TSV is empty (0 duplicates found)"
        echo "This may be okay if thresholds are strict"
    fi
else
    echo "❌ FAILED: No output TSV generated"
    exit 1
fi

# Check metadata
if [ -f "$METADATA_FILE" ]; then
    echo ""
    echo "📋 Metadata:"
    echo "  Timestamp: $(jq -r '.timestamp' "$METADATA_FILE")"
    echo "  Runtime: $(jq -r '.runtime_seconds' "$METADATA_FILE")s"
    echo "  Panels: $(jq -r '.counts.total_panels // "N/A"' "$METADATA_FILE")"
    echo "  Pairs: $(jq -r '.counts.total_pairs // "N/A"' "$METADATA_FILE")"
else
    echo "⚠️  WARNING: No metadata file"
fi

echo ""
echo "✅ SMOKE TEST COMPLETE"
echo "Output directory: $OUTPUT_DIR"

