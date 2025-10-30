#!/bin/bash
# Quick Test Script for Tile Detection Fixes
# Usage: ./quick_test.sh /path/to/test.pdf

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
TILE_SIZE=256
TILE_STRIDE=0.70
DPI=150
TIMEOUT=600  # 10 minutes

echo "=================================================="
echo "🧪 Quick Test: Tile Detection Fixes"
echo "=================================================="

# Check arguments
if [ -z "$1" ]; then
    echo -e "${RED}❌ Error: PDF path required${NC}"
    echo "Usage: $0 /path/to/test.pdf"
    exit 1
fi

PDF_PATH="$1"

if [ ! -f "$PDF_PATH" ]; then
    echo -e "${RED}❌ Error: PDF not found: $PDF_PATH${NC}"
    exit 1
fi

echo "📄 PDF: $PDF_PATH"

# Check Python
if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python not found${NC}"
    exit 1
fi

PYTHON_CMD=$(command -v python3 || command -v python)
echo "🐍 Python: $PYTHON_CMD"

# Check dependencies
echo ""
echo "Checking dependencies..."

check_module() {
    $PYTHON_CMD -c "import $1" 2>/dev/null
    return $?
}

MISSING=()
for module in open_clip pandas cv2 PIL imagehash tqdm skimage numpy; do
    if check_module $module; then
        echo "  ✓ $module"
    else
        echo -e "  ${YELLOW}✗ $module${NC}"
        MISSING+=($module)
    fi
done

if [ ${#MISSING[@]} -gt 0 ]; then
    echo -e "${YELLOW}⚠️  Missing dependencies. Install with:${NC}"
    echo "  pip install open-clip-torch pandas opencv-python-headless pillow imagehash tqdm scikit-image numpy"
    exit 1
fi

echo -e "${GREEN}✅ All dependencies satisfied${NC}"

# Check for import errors
echo ""
echo "Checking for import errors..."

if grep -q "open_clip_wrapper" ../src/ai_pdf_panel_duplicate_check_AUTO.py 2>/dev/null; then
    echo -e "${RED}❌ Found open_clip_wrapper import error!${NC}"
    echo "Fix: Edit ai_pdf_panel_duplicate_check_AUTO.py"
    echo "  Replace: from open_clip_wrapper import load_clip_model"
    echo "  With: import open_clip"
    echo ""
    echo "Auto-fix command:"
    echo '  sed -i.bak "s/from open_clip_wrapper import load_clip_model/import open_clip/" ai_pdf_panel_duplicate_check_AUTO.py'
    exit 1
fi

echo -e "${GREEN}✅ No import errors detected${NC}"

# Setup output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="./test_results_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo ""
echo "📁 Output: $OUTPUT_DIR"

# Run pipeline
echo ""
echo "=================================================="
echo "▶️  Running Pipeline (timeout: ${TIMEOUT}s)"
echo "=================================================="
echo "Parameters:"
echo "  - Tile size: ${TILE_SIZE}px"
echo "  - Tile stride: $TILE_STRIDE"
echo "  - DPI: $DPI"
echo ""

LOG_FILE="$OUTPUT_DIR/pipeline.log"

timeout $TIMEOUT $PYTHON_CMD ../src/ai_pdf_panel_duplicate_check_AUTO.py \
    --pdf "$PDF_PATH" \
    --output "$OUTPUT_DIR" \
    --tile-first \
    --tile-size $TILE_SIZE \
    --tile-stride $TILE_STRIDE \
    --dpi $DPI \
    --enable-cache \
    --no-auto-open \
    > "$LOG_FILE" 2>&1

PIPELINE_EXIT_CODE=$?

if [ $PIPELINE_EXIT_CODE -eq 124 ]; then
    echo -e "${RED}❌ Pipeline timed out after ${TIMEOUT}s${NC}"
    exit 1
elif [ $PIPELINE_EXIT_CODE -ne 0 ]; then
    echo -e "${RED}❌ Pipeline failed with exit code $PIPELINE_EXIT_CODE${NC}"
    echo "Last 20 lines of log:"
    tail -n 20 "$LOG_FILE"
    exit 1
fi

echo -e "${GREEN}✅ Pipeline completed successfully${NC}"

# Analyze output
echo ""
echo "=================================================="
echo "📊 Analyzing Results"
echo "=================================================="

TSV_PATH=$(find "$OUTPUT_DIR" -name "final_merged_report.tsv" -type f | head -n 1)

if [ -z "$TSV_PATH" ]; then
    echo -e "${RED}❌ TSV output not found${NC}"
    exit 1
fi

echo "✅ TSV found: $TSV_PATH"

# Analyze with Python
$PYTHON_CMD << EOF
import pandas as pd
import sys

tsv_path = "$TSV_PATH"

try:
    df = pd.read_csv(tsv_path, sep='\t')
    total_pairs = len(df)
    
    print(f"\n📈 Results:")
    print(f"   Total pairs: {total_pairs}")
    
    if 'Tile_Evidence' in df.columns:
        tile_evidence = df['Tile_Evidence'].sum()
        tile_rate = tile_evidence / max(total_pairs, 1)
        
        print(f"   Tile evidence: {tile_evidence} ({tile_rate:.1%})")
        
        if tile_rate < 0.05:
            print("\n⚠️  WARNING: Low tile evidence rate (<5%)")
            print("   Consider: Relax SSIM/NCC thresholds or reduce tile size")
            sys.exit(2)  # Warning exit code
        else:
            print("   ✅ Tile evidence rate is reasonable")
    
    if 'Tile_Evidence_Count' in df.columns:
        avg_tiles = df[df['Tile_Evidence'] == True]['Tile_Evidence_Count'].mean()
        if not pd.isna(avg_tiles):
            print(f"   Avg tiles per match: {avg_tiles:.1f}")
    
    if 'Tier' in df.columns:
        tier_a = len(df[df['Tier'] == 'A'])
        tier_b = len(df[df['Tier'] == 'B'])
        print(f"   Tier A: {tier_a}, Tier B: {tier_b}")
        
        # Check for Multi-Tile-Confirmed
        if 'Tier_Path' in df.columns:
            multi_tile = len(df[df['Tier_Path'].astype(str).str.contains('Multi-Tile-Confirmed', na=False)])
            if multi_tile > 0:
                print(f"   Multi-tile confirmed: {multi_tile}")
    
    print("\n${GREEN}✅ Analysis complete${NC}")
    sys.exit(0)

except Exception as e:
    print(f"\n${RED}❌ Error analyzing TSV: {e}${NC}")
    sys.exit(1)
EOF

ANALYSIS_EXIT=$?

echo ""
echo "=================================================="
if [ $ANALYSIS_EXIT -eq 0 ]; then
    echo -e "${GREEN}🎉 TEST PASSED!${NC}"
    echo "=================================================="
    echo ""
    echo "📁 Results in: $OUTPUT_DIR"
    echo "📊 TSV: $TSV_PATH"
    exit 0
elif [ $ANALYSIS_EXIT -eq 2 ]; then
    echo -e "${YELLOW}⚠️  TEST PASSED WITH WARNINGS${NC}"
    echo "=================================================="
    echo ""
    echo "Consider adjusting thresholds in tile_detection.py"
    exit 0
else
    echo -e "${RED}❌ TEST FAILED${NC}"
    echo "=================================================="
    exit 1
fi
