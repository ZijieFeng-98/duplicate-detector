#!/usr/bin/env python3
"""
Duplicate Detection Studio - Production Ready
All fixes applied for Streamlit Cloud deployment
"""

import streamlit as st
import sys
import subprocess
import json
import os
import re
from pathlib import Path
import time
import pandas as pd
from datetime import datetime
import tempfile
import shutil
from io import BytesIO
import zipfile

# ═══════════════════════════════════════════════════════════════
# ENVIRONMENT SETUP (Prevent warnings on cloud)
# ═══════════════════════════════════════════════════════════════
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# Page config
st.set_page_config(
    page_title="Duplicate Detection Studio",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #3B82F6 0%, #6366F1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .stAlert > div { padding: 1rem; }
    div[data-testid="stMetricValue"] { font-size: 2rem; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION SYSTEM INTEGRATION
# ═══════════════════════════════════════════════════════════════
try:
    from duplicate_detector.models.config import DetectorConfig
    CONFIG_SYSTEM_AVAILABLE = True
except ImportError:
    CONFIG_SYSTEM_AVAILABLE = False
    st.warning("⚠️ Config system not available, using legacy presets")

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════
TEMP_DIR = Path(tempfile.gettempdir()) / "duplicate_detector"
TEMP_DIR.mkdir(exist_ok=True)
MAX_FILE_SIZE_MB = 200
MAX_RUNTIME_SECONDS = 540  # 9 minutes

# ═══════════════════════════════════════════════════════════════
# DEPENDENCY CHECKS
# ═══════════════════════════════════════════════════════════════

@st.cache_resource
def check_dependencies():
    """Check dependencies with fallback suggestions"""
    status = {
        'all_ok': True,
        'missing': [],
        'warnings': [],
        'device': 'CPU',
        'has_poppler': False,
        'has_pymupdf': False,
        'versions': {}
    }
    
    required = {
        'torch': 'PyTorch',
        'open_clip': 'open-clip-torch',
        'PIL': 'Pillow',
        'cv2': 'opencv-python-headless',
        'imagehash': 'imagehash',
        'skimage': 'scikit-image',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'tqdm': 'tqdm'
    }
    
    # Check Python packages
    for module, name in required.items():
        try:
            mod = __import__(module)
            if hasattr(mod, '__version__'):
                status['versions'][module] = mod.__version__
        except ImportError:
            status['missing'].append(name)
            status['all_ok'] = False
    
    # Check PDF converters (pdf2image or PyMuPDF)
    try:
        import pdf2image
        result = subprocess.run(['pdftoppm', '-v'], 
                              capture_output=True, timeout=2)
        if result.returncode == 0:
            status['has_poppler'] = True
    except:
        pass
    
    try:
        import fitz  # PyMuPDF
        status['has_pymupdf'] = True
    except ImportError:
        pass
    
    if not status['has_poppler'] and not status['has_pymupdf']:
        status['warnings'].append('PDF conversion: Install PyMuPDF (pip install pymupdf)')
    
    # Check torch
    try:
        import torch
        if torch.cuda.is_available():
            status['device'] = 'CUDA'
        else:
            status['device'] = 'CPU'
    except:
        pass
    
    return status

# ═══════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════

if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'results' not in st.session_state:
    st.session_state.results = None
if 'pdf_path' not in st.session_state:
    st.session_state.pdf_path = None
if 'config' not in st.session_state:
    st.session_state.config = None

# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def safe_filename(filename: str) -> str:
    """Sanitize filename"""
    import re
    filename = re.sub(r'[^\w\s\-\.]', '', filename)
    return filename.strip()[:100]

def cleanup_old_files():
    """Clean up files older than 1 hour"""
    try:
        current_time = time.time()
        for item in TEMP_DIR.iterdir():
            if item.is_file() or item.is_dir():
                age = current_time - item.stat().st_mtime
                if age > 3600:
                    if item.is_file():
                        item.unlink()
                    else:
                        shutil.rmtree(item)
    except Exception:
        pass

@st.cache_data(show_spinner=False, ttl=300)  # 5-minute TTL for quick re-runs
def load_report(tsv_path: Path):
    """Load TSV report with caching and auto-refresh"""
    try:
        file_bytes = tsv_path.read_bytes()
        if len(file_bytes) == 0:
            raise ValueError("TSV file is empty - detection may have failed or is still running")
        return pd.read_csv(BytesIO(file_bytes), sep="\t", low_memory=False)
    except pd.errors.EmptyDataError:
        raise ValueError("TSV file is empty - detection may have failed or is still running")
    except FileNotFoundError:
        raise ValueError(f"TSV file not found: {tsv_path}")

def parse_results(output_dir: Path) -> dict:
    """Parse results with validation"""
    results = {
        'total_pairs': 0,
        'tier_a': 0,
        'tier_b': 0,
        'other': 0,
        'runtime': 0.0,
        'output_dir': str(output_dir),
        'panels': 0,
        'pages': 0
    }
    
    metadata_path = output_dir / "RUN_METADATA.json"
    if metadata_path.exists():
        try:
            with open(metadata_path) as f:
                meta = json.load(f)
                results['runtime'] = meta.get('runtime_seconds', 0.0)
                res = meta.get('results', {})
                results['panels'] = res.get('panels', 0)
                results['pages'] = res.get('pages', 0)
        except Exception:
            pass
    
    tsv_path = output_dir / "final_merged_report.tsv"
    if tsv_path.exists():
        try:
            df = load_report(tsv_path)
            results['total_pairs'] = len(df)
            
            if 'Tier' in df.columns:
                results['tier_a'] = len(df[df['Tier'] == 'A'])
                results['tier_b'] = len(df[df['Tier'] == 'B'])
                results['other'] = len(df) - results['tier_a'] - results['tier_b']
            else:
                results['other'] = len(df)
        except Exception:
            pass
    
    return results

# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 🔬 Duplicate Detector")
    st.caption("AI-powered image analysis")
    st.markdown("---")
    
    # Initialize current page in session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "📤 Upload"
    
    # Navigation radio button
    pages = ["📤 Upload", "⚙️ Configure", "▶️ Run", "📊 Results", "🧪 Test Lab"]
    current_index = pages.index(st.session_state.current_page) if st.session_state.current_page in pages else 0
    
    selected_page = st.radio(
        "Navigation",
        pages,
        index=current_index,
        label_visibility="collapsed"
    )
    
    # Update session state if user clicked a different page
    if selected_page != st.session_state.current_page:
        st.session_state.current_page = selected_page
        st.rerun()
    
    page = st.session_state.current_page
    
    st.markdown("---")
    
    deps = check_dependencies()
    
    if not deps['all_ok']:
        st.error("❌ Setup incomplete")
        for pkg in deps['missing']:
            st.caption(f"Missing: {pkg}")
    elif deps['warnings']:
        st.warning("⚠️ Limited features")
        for warn in deps['warnings']:
            st.caption(warn)
    else:
        st.success("✅ Ready")
    
    st.caption(f"🖥️ {deps.get('device', 'CPU')}")
    
    with st.expander("ℹ️ About"):
        st.markdown("""
        Upload a PDF and detect:
        - Exact duplicates
        - Rotated copies
        - Cropped/partial matches
        
        **Privacy:** Files deleted after 1 hour.
        """)
    
    with st.expander("⚙️ System Info"):
        if deps['versions']:
            for pkg, ver in list(deps['versions'].items())[:3]:
                st.caption(f"{pkg}: {ver}")

# ═══════════════════════════════════════════════════════════════
# PAGE 1: UPLOAD
# ═══════════════════════════════════════════════════════════════

if page == "📤 Upload":
    st.markdown('<h1 class="main-header">Upload Scientific PDF</h1>', unsafe_allow_html=True)
    st.caption("Drag & drop your PDF file to analyze for duplicate figures")
    
    cleanup_old_files()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose PDF",
            type=['pdf'],
            help=f"Maximum size: {MAX_FILE_SIZE_MB}MB",
            key="pdf_uploader"
        )
        
        if uploaded_file is not None:
            size_mb = uploaded_file.size / 1024 / 1024
            
            if size_mb > MAX_FILE_SIZE_MB:
                st.error(f"❌ File too large: {size_mb:.1f}MB (max {MAX_FILE_SIZE_MB}MB)")
                st.info("💡 Try reducing PDF quality or splitting into smaller files")
                st.stop()
            
            safe_name = safe_filename(uploaded_file.name)
            upload_dir = TEMP_DIR / "uploads"
            upload_dir.mkdir(exist_ok=True)
            pdf_path = upload_dir / safe_name
            
            with st.spinner("Uploading..."):
                pdf_path.write_bytes(uploaded_file.getbuffer())
            
            st.session_state.pdf_path = pdf_path
            
            st.success(f"✅ **{safe_name}** ({size_mb:.1f}MB)")
            
            st.info(f"""
            📄 **Ready to analyze**
            - File: `{safe_name}`
            - Size: {size_mb:.1f}MB
            - Auto-deleted after 1 hour
            """)
    
    with col2:
        st.markdown("### 🚀 Quick Start")
        st.markdown("""
        **3 Simple Steps:**
        
        1. ⬆️ Upload PDF
        2. ⚙️ Configure
        3. ▶️ Run analysis
        
        Takes ~2-5 minutes.
        """)
        
        if uploaded_file is not None:
            st.markdown("---")
            if st.button("Next →", use_container_width=True, type="primary"):
                st.session_state.current_page = "⚙️ Configure"
                st.rerun()
    
    with st.expander("💡 Example Use Cases"):
        st.markdown("""
        - **Before submission:** Check for accidental duplicates
        - **Peer review:** Verify figure integrity
        - **Journal editing:** Quality control
        - **Research integrity:** Detect image manipulation
        """)

# ═══════════════════════════════════════════════════════════════
# PAGE 2: CONFIGURE
# ═══════════════════════════════════════════════════════════════

elif page == "⚙️ Configure":
    st.markdown('<h1 class="main-header">Settings</h1>', unsafe_allow_html=True)
    
    if st.session_state.pdf_path is None:
        st.warning("⚠️ Upload a PDF first")
        if st.button("← Back to Upload"):
            st.session_state.current_page = "📤 Upload"
            st.rerun()
        st.stop()
    
    # ═══════════════════════════════════════════════════════════════
    # PERSIST USER INTENT: Session State + Query Params
    # ═══════════════════════════════════════════════════════════════
    
    # Initialize from query params (shareable URLs)
    if 'tile_mode' not in st.session_state:
        try:
            # Streamlit 1.30+ uses st.query_params (new API)
            query_tile_mode = st.query_params.get('tile_mode', 'Auto (Recommended)')
            st.session_state.tile_mode = query_tile_mode
        except:
            # Fallback for older Streamlit versions
            st.session_state.tile_mode = 'Auto (Recommended)'
    
    if 'tile_size' not in st.session_state:
        st.session_state.tile_size = 384
    
    if 'tile_stride' not in st.session_state:
        st.session_state.tile_stride = 0.65
    
    st.caption("Choose a preset (recommended) or customize")
    
    # Presets
    st.subheader("🎯 Quick Presets")
    st.info("**📊 Test Results (14 known duplicates):** Balanced = 100% detect + 9 FPs | Thorough = 78% detect + 0 FPs | Fast = 100% detect + 25 FPs")
    col1, col2, col3 = st.columns(3)
    
    # Load presets from config system or use legacy
    if CONFIG_SYSTEM_AVAILABLE:
        # Use config system presets
        fast_config = DetectorConfig.from_preset("fast")
        balanced_config = DetectorConfig.from_preset("balanced")
        thorough_config = DetectorConfig.from_preset("thorough")
        
        presets = {
            'fast': {
                'dpi': fast_config.dpi,
                'sim_threshold': fast_config.duplicate_detection.sim_threshold,
                'phash_max_dist': fast_config.duplicate_detection.phash_max_dist,
                'ssim_threshold': fast_config.duplicate_detection.ssim_threshold,
                'use_phash_bundles': fast_config.feature_flags.use_phash_bundles,
                'use_orb': fast_config.feature_flags.use_orb_ransac,
                'use_tier_gating': fast_config.feature_flags.use_tier_gating,
                'batch_size': fast_config.performance.batch_size,
                'name': 'Fast',
                'time': '~2 min',
                'desc': '25+ false positives'
            },
            'balanced': {
                'dpi': balanced_config.dpi,
                'sim_threshold': balanced_config.duplicate_detection.sim_threshold,
                'phash_max_dist': balanced_config.duplicate_detection.phash_max_dist,
                'ssim_threshold': balanced_config.duplicate_detection.ssim_threshold,
                'use_phash_bundles': balanced_config.feature_flags.use_phash_bundles,
                'use_orb': balanced_config.feature_flags.use_orb_ransac,
                'use_tier_gating': balanced_config.feature_flags.use_tier_gating,
                'batch_size': balanced_config.performance.batch_size,
                'name': 'Balanced ⭐',
                'time': '~5 min',
                'desc': '~9 false positives'
            },
            'thorough': {
                'dpi': thorough_config.dpi,
                'sim_threshold': thorough_config.duplicate_detection.sim_threshold,
                'phash_max_dist': thorough_config.duplicate_detection.phash_max_dist,
                'ssim_threshold': thorough_config.duplicate_detection.ssim_threshold,
                'use_phash_bundles': thorough_config.feature_flags.use_phash_bundles,
                'use_orb': thorough_config.feature_flags.use_orb_ransac,
                'use_tier_gating': thorough_config.feature_flags.use_tier_gating,
                'batch_size': thorough_config.performance.batch_size,
                'name': 'Thorough',
                'time': '~8 min',
                'desc': '0 false positives'
            }
        }
    else:
        # Legacy presets (fallback)
        presets = {
            'fast': {
                'dpi': 100,
                'sim_threshold': 0.97,
                'phash_max_dist': 3,
                'ssim_threshold': 0.92,
                'use_phash_bundles': True,
                'use_orb': False,
                'use_tier_gating': True,
                'batch_size': 32,
                'name': 'Fast',
                'time': '~2 min',
                'desc': '25+ false positives'
            },
            'balanced': {
                'dpi': 150,
                'sim_threshold': 0.96,
                'phash_max_dist': 4,
                'ssim_threshold': 0.90,
                'use_phash_bundles': True,
                'use_orb': True,
                'use_tier_gating': True,
                'batch_size': 32,
                'name': 'Balanced ⭐',
                'time': '~5 min',
                'desc': '~9 false positives'
            },
            'thorough': {
                'dpi': 200,
                'sim_threshold': 0.94,
                'phash_max_dist': 5,
                'ssim_threshold': 0.88,
                'use_phash_bundles': True,
                'use_orb': True,
                'use_tier_gating': True,
                'batch_size': 32,
                'name': 'Thorough',
                'time': '~8 min',
                'desc': '0 false positives'
            }
        }
    
    with col1:
        if st.button(f"⚡ Fast\n{presets['fast']['time']}", use_container_width=True):
            st.session_state.preset = presets['fast']
            st.session_state.run_completed = False  # Reset run state when config changes
            st.rerun()
    
    with col2:
        if st.button(f"🎯 Balanced\n{presets['balanced']['time']}", use_container_width=True, type="primary"):
            st.session_state.preset = presets['balanced']
            st.session_state.run_completed = False  # Reset run state when config changes
            st.rerun()
    
    with col3:
        if st.button(f"🔬 Thorough\n{presets['thorough']['time']}", use_container_width=True):
            st.session_state.preset = presets['thorough']
            st.session_state.run_completed = False  # Reset run state when config changes
            st.rerun()
    
    st.markdown("---")
    
    # Load preset
    preset = st.session_state.get('preset', presets['balanced'])
    if 'preset' in st.session_state:
        st.success(f"✅ Using **{preset['name']}** preset")
    
    # Advanced options
    with st.expander("🔧 Advanced Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            dpi = st.slider("DPI", 100, 200, preset['dpi'], 25)
            sim_threshold = st.slider("CLIP", 0.70, 0.99, preset['sim_threshold'], 0.01, 
                                     help="Lower = more sensitive (catches compressed/modified images)")
        
        with col2:
            phash_max_dist = st.slider("pHash", 1, 15, preset['phash_max_dist'], 1,
                                      help="Higher = more tolerant (catches rotated/cropped images)")
            ssim_threshold = st.slider("SSIM", 0.50, 0.95, preset['ssim_threshold'], 0.01,
                                      help="Lower = more sensitive (tolerates structural differences)")
        
        st.markdown("**Features**")
        col1, col2, col3 = st.columns(3)
        with col1:
            use_phash_bundles = st.checkbox("Rotation", preset['use_phash_bundles'])
        with col2:
            use_orb = st.checkbox("Crop Detection", preset['use_orb'])
        with col3:
            use_tier_gating = st.checkbox("Tiers", preset['use_tier_gating'])
        
        st.markdown("---")
        
        # ═══════════════════════════════════════════════════════════════
        # TILE-FIRST MODE (Feature Flag)
        # Owner: Research Team | Review: Q2 2025 | Plan: Merge or Remove
        # ═══════════════════════════════════════════════════════════════
        
        # Check if module is available
        try:
            from tile_first_pipeline import TileFirstConfig, run_tile_first_pipeline
            tile_first_available = True
        except ImportError:
            tile_first_available = False
        
        st.markdown("**🔬 Micro-Tiles Mode** (Experimental)")
        
        if not tile_first_available:
            st.warning("⚠️ Tile-First module not found. Feature disabled.")
            tile_mode = "Disabled"
            tile_size = 384
            tile_stride = 0.65
        else:
            # Three-way radio: Auto / Force ON / Force OFF
            tile_mode = st.radio(
                "Tile-First Strategy:",
                options=["Auto (Recommended)", "Force ON", "Force OFF"],
                index=0 if st.session_state.tile_mode == "Auto (Recommended)" else (1 if st.session_state.tile_mode == "Force ON" else 2),
                help="""
                **Auto:** Use micro-tiles if ≥3 confocal panels detected
                **Force ON:** Always use micro-tiles only (no panel detection)
                **Force OFF:** Never use micro-tiles (standard panel pipeline)
                
                💡 Micro-tiles = More thorough but slower. Use for confocal grids with false positives.
                """
            )
            
            # Show parameters only if not Force OFF
            if tile_mode != "Force OFF":
                st.markdown("**Tile Parameters:**")
                col1, col2 = st.columns(2)
                
                with col1:
                    tile_size = st.slider(
                        "Tile Size (px)",
                        min_value=256,
                        max_value=512,
                        value=st.session_state.tile_size,
                        step=32,
                        help="Size of micro-tiles. Recommended: 384px"
                    )
                
                with col2:
                    tile_stride = st.slider(
                        "Stride Ratio",
                        min_value=0.50,
                        max_value=0.80,
                        value=st.session_state.tile_stride,
                        step=0.05,
                        format="%.2f",
                        help="Overlap between tiles. 0.65 = 35% overlap"
                    )
                    st.caption(f"Overlap: {(1-tile_stride)*100:.0f}% | Smaller tiles & more overlap = thorough but slower")
                
                # Guardrail: Warn if tile size might be too large
                if tile_size >= 448:
                    st.warning(f"""
                    ⚠️ Large tile size ({tile_size}px) may not work well with small panels.
                    
                    **Recommendation:** If panels are <500px, try 256-320px tiles.
                    """)
                
                # Performance note
                if tile_mode == "Force ON" or tile_stride < 0.60:
                    st.info("⚡ More tiles → more candidate pairs → longer runtime (~2-3x)")
            else:
                tile_size = 384
                tile_stride = 0.65
        
        st.markdown("---")
        
        st.markdown("**Detection Strategy**")
        use_modality_specific = st.radio(
            "Choose Detection Method:",
            options=[False, True],
            format_func=lambda x: "🎯 Universal (Recommended)" if not x else "🔬 Modality-Specific (Advanced)",
            index=0,
            help="""
            **📊 Test Results: Both methods perform identically (F1=88.0 on Thorough, F1=75.7 on Balanced)**
            
            **Universal (Recommended):** Single set of rules for all image types. Faster.
            
            **Modality-Specific (Advanced):** Pre-classifies images by type (Western blot, confocal, TEM, etc.) 
            and applies custom thresholds per modality. Same accuracy, more thorough classification.
            
            💡 Recommendation: Use Universal (simpler). Both give identical detection results.
            """
        )
    
    output_dir = TEMP_DIR / "output" / datetime.now().strftime("%Y%m%d_%H%M%S")
    
    st.markdown("---")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if st.button("← Back", use_container_width=True):
            st.session_state.current_page = "📤 Upload"
            st.rerun()
    
    with col2:
        if st.button("Start Analysis →", use_container_width=True, type="primary"):
            # Store query params for shareable URLs
            try:
                st.query_params.update({
                    'tile_mode': tile_mode,
                    'tile_size': str(tile_size),
                    'tile_stride': str(tile_stride)
                })
            except:
                pass  # Graceful fallback for older Streamlit
            
            st.session_state.config = {
                'dpi': dpi,
                'sim_threshold': sim_threshold,
                'phash_max_dist': phash_max_dist,
                'ssim_threshold': ssim_threshold,
                'use_phash_bundles': use_phash_bundles,
                'use_orb': use_orb,
                'use_tier_gating': use_tier_gating,
                'use_modality_specific': use_modality_specific,
                'batch_size': 32,
                'output_dir': str(output_dir),
                'enable_cache': True,
                'highlight_diffs': True,
                'suppress_same_page': False,
                'debug_mode': False,
                'auto_open': False,
                
                # Tile-First parameters
                'tile_mode': tile_mode if tile_first_available else "Disabled",
                'tile_size': tile_size,
                'tile_stride': tile_stride,
            }
            
            # Update session state
            st.session_state.tile_mode = tile_mode
            st.session_state.tile_size = tile_size
            st.session_state.tile_stride = tile_stride
            
            output_dir.mkdir(parents=True, exist_ok=True)
            st.session_state.current_page = "▶️ Run"
            st.rerun()

# ═══════════════════════════════════════════════════════════════
# PAGE 3: RUN ANALYSIS (WITH ALL FIXES)
# ═══════════════════════════════════════════════════════════════

elif page == "▶️ Run":
    st.markdown('<h1 class="main-header">Running Analysis</h1>', unsafe_allow_html=True)
    
    if st.session_state.config is None or st.session_state.pdf_path is None:
        st.error("⚠️ Please configure settings first")
        if st.button("← Back"):
            st.session_state.current_page = "⚙️ Configure"
            st.rerun()
        st.stop()
    
    # Check if we already ran - if so, just show the logs and results
    if 'run_completed' in st.session_state and st.session_state.run_completed:
        st.info("✅ Analysis already completed. Showing previous logs.")
        
        # Show stored logs
        if 'run_logs' in st.session_state and st.session_state.run_logs:
            with st.expander("📋 Detailed Logs", expanded=True):
                st.code("\n".join(st.session_state.run_logs), language="text")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👀 View Results", use_container_width=True, type="primary"):
                st.session_state.current_page = "📊 Results"
                st.rerun()
        with col2:
            if st.button("🔄 Run Again", use_container_width=True):
                st.session_state.run_completed = False
                st.session_state.processing_complete = False
                st.rerun()
        st.stop()
    
    config = st.session_state.config
    
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    
    # Log container with persistent logs
    with st.expander("📋 Detailed Logs", expanded=False):
        log_display = st.empty()
        
        # Show stored logs from previous run if available
        if 'run_logs' in st.session_state and st.session_state.run_logs:
            log_display.code("\n".join(st.session_state.run_logs[-100:]), language="text")
    
    detector_script = Path(__file__).parent / "ai_pdf_panel_duplicate_check_AUTO.py"
    
    if not detector_script.exists():
        st.error("❌ Detector backend not found")
        st.info(f"Make sure `ai_pdf_panel_duplicate_check_AUTO.py` is in: {detector_script.parent}")
        st.stop()
    
    # ═══════════════════════════════════════════════════════════════
    # SINGLE DECISION POINT: Route to Tile-First or Standard Pipeline
    # ═══════════════════════════════════════════════════════════════
    
    tile_mode = config.get('tile_mode', 'Auto (Recommended)')
    tile_first_enabled = False
    mode_reason = ""
    
    # Decision logic
    if tile_mode == "Force ON":
        tile_first_enabled = True
        mode_reason = "user requested Force ON"
    elif tile_mode == "Force OFF":
        tile_first_enabled = False
        mode_reason = "user requested Force OFF"
    elif tile_mode == "Auto (Recommended)":
        # Auto-enable heuristic: ≥3 confocal panels
        # (We don't know panel count yet, so we'll pass --tile-first-auto flag)
        tile_first_enabled = "auto"
        mode_reason = "auto-detect (≥3 confocal panels)"
    else:
        tile_first_enabled = False
        mode_reason = "module disabled"
    
    # Validate PDF file exists
    pdf_path = Path(st.session_state.pdf_path)
    if not pdf_path.exists():
        st.error(f"❌ PDF file not found: {pdf_path}")
        st.error("This may be a file upload or permissions issue on Streamlit Cloud")
        st.info("💡 Try re-uploading your PDF")
        if st.button("← Back to Upload"):
            st.session_state.current_page = "📤 Upload"
            st.rerun()
        st.stop()
    
    # Build command
    cmd = [
        sys.executable, str(detector_script),
        "--pdf", str(pdf_path.absolute()),  # Use absolute path
        "--output", str(Path(config['output_dir']).absolute()),
        "--dpi", str(config['dpi']),
        "--sim-threshold", str(config['sim_threshold']),
        "--phash-max-dist", str(config['phash_max_dist']),
        "--ssim-threshold", str(config['ssim_threshold']),
        "--batch-size", str(config['batch_size']),
        "--no-auto-open",  # Don't open browser in Streamlit
    ]
    
    # Debug: Log file info
    st.info(f"📄 Processing: {pdf_path.name} ({pdf_path.stat().st_size / 1024 / 1024:.1f}MB)")
    
    # Display actual thresholds being used
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("CLIP", f"{config['sim_threshold']:.2f}")
    with col2:
        st.metric("pHash", config['phash_max_dist'])
    with col3:
        st.metric("SSIM", f"{config['ssim_threshold']:.2f}")
    with col4:
        features = []
        if config['use_phash_bundles']: features.append("Rot")
        if config['use_orb']: features.append("ORB")
        if config['use_tier_gating']: features.append("Tier")
        st.metric("Features", "+".join(features) if features else "None")
    
    print(f"DEBUG: PDF path: {pdf_path.absolute()}")
    print(f"DEBUG: PDF exists: {pdf_path.exists()}")
    print(f"DEBUG: PDF readable: {os.access(pdf_path, os.R_OK)}")
    
    if config['use_phash_bundles']:
        cmd.append("--use-phash-bundles")
    if config['use_orb']:
        cmd.append("--use-orb")
    if config['use_tier_gating']:
        cmd.append("--use-tier-gating")
    if config['highlight_diffs']:
        cmd.append("--highlight-diffs")
    if config['enable_cache']:
        cmd.append("--enable-cache")
    if config.get('use_modality_specific', False):
        cmd.append("--use-modality-specific")
        cmd.append("--enable-modality-detection")
    
    # Tile-First routing (single decision point)
    if tile_first_enabled == True:
        # Force ON
        cmd.append("--tile-first")
        cmd.append("--tile-size")
        cmd.append(str(config.get('tile_size', 384)))
        cmd.append("--tile-stride")
        cmd.append(str(config.get('tile_stride', 0.65)))
    elif tile_first_enabled == "auto":
        # Auto mode - let backend decide
        cmd.append("--tile-first-auto")
        cmd.append("--tile-size")
        cmd.append(str(config.get('tile_size', 384)))
        cmd.append("--tile-stride")
        cmd.append(str(config.get('tile_stride', 0.65)))
    # else: Force OFF or Disabled - don't add any flags
    
    # Log mode for debugging
    print(f"🔬 Tile-First Mode: {tile_mode} ({mode_reason})")
    
    # Debug: Print the full command to server logs
    print(f"DEBUG: Running command: {' '.join(cmd)}")
    st.info(f"🔍 Command: `{detector_script.name} --pdf {pdf_path.name}...`")
    
    # Progress patterns
    stage_patterns = [
        (re.compile(r'converting pdf', re.I), "Converting PDF", 0.15),
        (re.compile(r'auto-?detecting.*panel', re.I), "Auto-detecting panels", 0.30),
        (re.compile(r'embed(ding)?|CLIP', re.I), "Computing CLIP embeddings", 0.50),
        (re.compile(r'\bphash\b', re.I), "Running pHash analysis", 0.65),
        (re.compile(r'\borb(-?ransac)?\b', re.I), "ORB-RANSAC detection", 0.78),
        (re.compile(r'\bssim\b', re.I), "SSIM validation", 0.88),
        (re.compile(r'visual|duplicate_comparisons|difference map', re.I), "Generating visual reports", 0.95),
        (re.compile(r'RUN_METADATA|finalizing|saved|complete', re.I), "Finalizing", 0.99),
    ]
    
    try:
        start_time = time.time()
        current_progress = 0.05
        progress_bar.progress(current_progress)
        
        # Show mode in status
        if tile_first_enabled == True:
            mode_badge = f"🔬 Tile-First (tiles: {config.get('tile_size', 384)}px, overlap: {(1-config.get('tile_stride', 0.65))*100:.0f}%)"
        elif tile_first_enabled == "auto":
            mode_badge = f"🔬 Tile-First (Auto-detect)"
        else:
            mode_badge = "📊 Standard Pipeline"
        
        status_text.info(f"⏳ Starting analysis... {mode_badge}")
        
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Store logs in session state so they persist
        st.session_state.run_logs = []
        log_lines = []
        
        # Timeout handling
        while proc.poll() is None:
            elapsed = time.time() - start_time
            
            if elapsed > MAX_RUNTIME_SECONDS:
                proc.kill()
                st.error(f"⏱️ Timed out after {MAX_RUNTIME_SECONDS//60} minutes")
                st.info("💡 Try the **Fast** preset or upload a smaller PDF")
                st.stop()
            
            line = proc.stdout.readline()
            if not line:
                time.sleep(0.05)
                continue
            
            line = line.rstrip()
            log_lines.append(line)
            st.session_state.run_logs.append(line)  # Save to session state
            
            log_display.code("\n".join(log_lines[-40:]), language="text")
            
            # Progress matching
            for pat, label, prog in stage_patterns:
                if pat.search(line):
                    current_progress = max(current_progress, prog)
                    progress_bar.progress(current_progress)
                    status_text.info(f"⏳ {label}...")
                    break
        
        return_code = proc.wait()
        elapsed = time.time() - start_time
        
        # Capture any remaining output (for immediate crashes)
        remaining_output = proc.stdout.read()
        if remaining_output:
            for line in remaining_output.split('\n'):
                if line.strip():
                    log_lines.append(line.rstrip())
                    st.session_state.run_logs.append(line.rstrip())
        
        # Show logs if process failed
        if return_code != 0 and log_lines:
            log_display.code("\n".join(log_lines[-40:]), language="text")
        
        if return_code == 0:
            progress_bar.progress(1.0)
            status_text.success("✅ Complete!")
            
            output_dir = Path(config['output_dir'])
            
            # Validate outputs
            expected_files = [
                output_dir / "final_merged_report.tsv",
                output_dir / "RUN_METADATA.json"
            ]
            missing = [f.name for f in expected_files if not f.exists()]
            
            if missing:
                st.warning(f"⚠️ Run completed but missing files: {', '.join(missing)}")
                with log_container:
                    st.info("Check dependency warnings and logs above")
            
            results = parse_results(output_dir)
            
            if results['total_pairs'] == 0 and not expected_files[0].exists():
                st.warning("⚠️ No pairs found or report missing. Check logs above.")
            
            st.session_state.results = results
            st.session_state.processing_complete = True
            st.session_state.run_completed = True  # Mark as completed
            
            st.balloons()
            st.success(f"✅ Found {results['total_pairs']} pairs in {elapsed:.1f}s")
            
            time.sleep(2)
            st.session_state.current_page = "📊 Results"
            st.rerun()
        else:
            st.session_state.run_completed = True  # Mark as completed even if failed
            st.error(f"❌ Analysis failed (exit code {return_code})")
            st.info("Check logs above for details")
            
            # Show last 50 lines of error output
            if log_lines:
                with st.expander("🔍 Error Output (Last 50 Lines)", expanded=True):
                    st.code("\n".join(log_lines[-50:]))
    
    except Exception as e:
        error_msg = f"❌ Error: {e}"
        st.error(error_msg)
        st.session_state.run_completed = True
        st.session_state.run_logs.append(error_msg)

        import traceback
        tb_text = traceback.format_exc()
        st.code(tb_text)
        st.session_state.run_logs.extend(tb_text.rstrip().splitlines())
        
        # Show command that failed
        st.warning("**Failed Command:**")
        failed_cmd = " ".join(str(c) for c in cmd)
        st.code(failed_cmd)
        st.session_state.run_logs.append(f"CMD: {failed_cmd}")
    finally:
        try:
            proc.kill()  # type: ignore[name-defined]
        except Exception:
            pass

# ═══════════════════════════════════════════════════════════════
# PAGE 4: RESULTS (WITH INLINE PREVIEW)
# ═══════════════════════════════════════════════════════════════

elif page == "📊 Results":
    st.markdown('<h1 class="main-header">Results</h1>', unsafe_allow_html=True)
    
    if not st.session_state.processing_complete:
        st.info("No results yet. Run an analysis first!")
        if st.button("← Start Analysis"):
            st.session_state.current_page = "▶️ Run"
            st.rerun()
        st.stop()
    
    results = st.session_state.results
    output_dir = Path(results['output_dir'])
    
    # ═══════════════════════════════════════════════════════════════
    # RUN MODE BANNER (one-liner)
    # ═══════════════════════════════════════════════════════════════
    if st.session_state.config:
        tile_mode = st.session_state.config.get('tile_mode', 'Standard')
        if tile_mode == "Force ON":
            tile_size = st.session_state.config.get('tile_size', 384)
            tile_stride = st.session_state.config.get('tile_stride', 0.65)
            st.info(f"🔬 **Mode:** Tile-First ({tile_size}px, stride {tile_stride:.2f})")
        elif tile_mode == "Auto (Recommended)":
            st.info(f"🔬 **Mode:** Tile-First (Auto-detect)")
        else:
            st.caption("📊 **Mode:** Standard (auto-tile when needed)")
    
    # Summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total", results['total_pairs'])
    with col2:
        st.metric("Tier A", results['tier_a'], delta="High" if results['tier_a'] > 0 else None)
    with col3:
        st.metric("Tier B", results['tier_b'], delta="Med" if results['tier_b'] > 0 else None)
    with col4:
        st.metric("Time", f"{results['runtime']:.0f}s")
    
    st.markdown("---")
    
    # Downloads
    st.subheader("📥 Download Results")
    
    tsv_path = output_dir / "final_merged_report.tsv"
    index_path = output_dir / "duplicate_comparisons" / "index.html"
    comp_dir = output_dir / "duplicate_comparisons"
    
    col1, col2 = st.columns(2)
    
    with col1:
        if tsv_path.exists():
            st.download_button(
                "📊 Download Report (TSV)",
                data=tsv_path.read_bytes(),
                file_name=f"duplicates_{datetime.now().strftime('%Y%m%d')}.tsv",
                mime="text/tab-separated-values",
                use_container_width=True,
                type="primary"
            )
        else:
            st.warning("Report not found")
    
    with col2:
        if comp_dir.exists():
            # Create zip (memory-safe with temp file)
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(mode='w+b', suffix='.zip', delete=False) as tmp:
                tmp_path = tmp.name
            
            try:
                with zipfile.ZipFile(tmp_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                    # TSV report
                    if tsv_path.exists():
                        zf.write(tsv_path, tsv_path.name)
                    
                    # Metadata
                    meta_json = output_dir / 'RUN_METADATA.json'
                    if meta_json.exists():
                        zf.write(meta_json, meta_json.name)
                    
                    # Visual comparisons (limit to first 50 to avoid huge archives)
                    pair_count = 0
                    for pair_dir in sorted(comp_dir.glob("pair_*_detailed")):
                        if pair_count >= 50:  # Cap at 50 pairs
                            break
                        for file in pair_dir.glob("*"):
                            if file.is_file():
                                zf.write(file, f"{pair_dir.name}/{file.name}")
                        pair_count += 1
                
                # Read temp file for download
                with open(tmp_path, 'rb') as f:
                    zip_bytes = f.read()
            
                st.download_button(
                    "📦 Download All (ZIP)",
                    data=zip_bytes,
                    file_name=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip",
                    use_container_width=True,
                    help="Includes TSV report, metadata, and first 50 pairs (memory-safe)"
                )
            finally:
                # Cleanup
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
    
    st.markdown("---")
    
    # Inline HTML Preview
    with st.expander("📊 Interactive Comparison Preview", expanded=False):
        st.info("Preview the first Tier-A pair's interactive comparison")
        
        # Find first Tier-A detailed folder
        if comp_dir.exists():
            tier_a_dirs = sorted([d for d in comp_dir.glob("pair_*_detailed") 
                                 if (d / "interactive.html").exists()])
            
            if tier_a_dirs:
                first_pair = tier_a_dirs[0]
                html_path = first_pair / "interactive.html"
                
                if html_path.exists():
                    html_content = html_path.read_text(encoding='utf-8')
                    
                    # Embed with iframe
                    st.components.v1.html(html_content, height=800, scrolling=True)
                else:
                    st.warning("Interactive HTML not found")
            else:
                st.info("No Tier-A pairs to preview")
        else:
            st.warning("No comparison directory found")
    
    # Native Streamlit visualization display
    st.subheader("🌐 Interactive Results Viewer")
    
    if tsv_path.exists() and comp_dir.exists():
        df = pd.read_csv(tsv_path, sep='\t')
        
        # Group by tier
        tier_a = df[df['Tier'] == 'A'] if 'Tier' in df.columns else pd.DataFrame()
        tier_b = df[df['Tier'] == 'B'] if 'Tier' in df.columns else pd.DataFrame()
        other = df[~df['Tier'].isin(['A', 'B'])] if 'Tier' in df.columns else df
        
        # Display Tier A pairs (priority)
        if len(tier_a) > 0:
            st.markdown("### 🚨 Tier A - Review Required")
            st.caption(f"{len(tier_a)} high-confidence duplicate{'s' if len(tier_a) > 1 else ''} detected")
            
            for seq_num, (idx, row) in enumerate(tier_a.iterrows(), start=1):
                pair_path_a = row.get('Path_A', row.get('Image_A', ''))
                pair_path_b = row.get('Path_B', row.get('Image_B', ''))
                img_a_name = Path(pair_path_a).name if pair_path_a else f"Image A"
                img_b_name = Path(pair_path_b).name if pair_path_b else f"Image B"
                
                with st.expander(f"**Pair #{seq_num:03d}**: {img_a_name} vs {img_b_name}", expanded=(seq_num == 1)):
                    # Find the pair directory
                    pair_dir = comp_dir / f"pair_{seq_num:03d}_detailed"
                    
                    # Display scores
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        clip_val = row.get('Cosine_Similarity', 'N/A')
                        if pd.notna(clip_val) and clip_val != '':
                            st.metric("🎯 CLIP", f"{float(clip_val):.3f}")
                        else:
                            st.metric("🎯 CLIP", "N/A")
                    with col2:
                        ssim_val = row.get('SSIM', 'N/A')
                        if pd.notna(ssim_val) and ssim_val != '':
                            st.metric("📊 SSIM", f"{float(ssim_val):.3f}")
                        else:
                            st.metric("📊 SSIM", "N/A")
                    with col3:
                        phash_val = row.get('Hamming_Distance', 'N/A')
                        if pd.notna(phash_val) and phash_val != '':
                            st.metric("🔍 pHash", f"{int(phash_val)}")
                        else:
                            st.metric("🔍 pHash", "N/A")
                    with col4:
                        tier_path = row.get('Tier_Path', 'N/A')
                        if pd.notna(tier_path) and tier_path != '':
                            st.metric("🎯 Path", tier_path)
                        else:
                            st.metric("🎯 Path", "N/A")
                    
                    # Display visualizations if they exist
                    if pair_dir.exists():
                        # Image comparison slider
                        img_a_path = pair_dir / "1_raw_A.png"
                        img_b_path = pair_dir / "2_raw_B_aligned.png"
                        
                        if img_a_path.exists() and img_b_path.exists():
                            st.markdown("**📷 Side-by-Side Comparison**")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.image(str(img_a_path), caption="Image A (Original)", use_container_width=True)
                            with col2:
                                st.image(str(img_b_path), caption="Image B (Aligned)", use_container_width=True)
                        
                        # Additional visualizations in tabs
                        overlay_path = pair_dir / "3_overlay_50_50.png"
                        ssim_path = pair_dir / "4_ssim_viridis.png"
                        diff_path = pair_dir / "5_hard_diff_mask.png"
                        checker_path = pair_dir / "6_checkerboard.png"
                        blink_path = pair_dir / "7_blink.gif"
                        
                        available_viz = []
                        if overlay_path.exists(): available_viz.append("Overlay")
                        if ssim_path.exists(): available_viz.append("SSIM Map")
                        if diff_path.exists(): available_viz.append("Diff Mask")
                        if checker_path.exists(): available_viz.append("Checkerboard")
                        if blink_path.exists(): available_viz.append("Blink GIF")
                        
                        if available_viz:
                            st.markdown("**🎨 Advanced Visualizations**")
                            viz_tabs = st.tabs(available_viz)
                            
                            tab_idx = 0
                            if "Overlay" in available_viz:
                                with viz_tabs[tab_idx]:
                                    st.image(str(overlay_path), caption="50/50 Overlay", use_container_width=True)
                                tab_idx += 1
                            if "SSIM Map" in available_viz:
                                with viz_tabs[tab_idx]:
                                    st.image(str(ssim_path), caption="SSIM Dissimilarity Map", use_container_width=True)
                                    st.caption("🟢 Green = Similar | 🔴 Red = Different")
                                tab_idx += 1
                            if "Diff Mask" in available_viz:
                                with viz_tabs[tab_idx]:
                                    st.image(str(diff_path), caption="Hard Difference Mask", use_container_width=True)
                                tab_idx += 1
                            if "Checkerboard" in available_viz:
                                with viz_tabs[tab_idx]:
                                    st.image(str(checker_path), caption="Checkerboard Composite", use_container_width=True)
                                tab_idx += 1
                            if "Blink GIF" in available_viz:
                                with viz_tabs[tab_idx]:
                                    st.image(str(blink_path), caption="Blink Comparator", use_container_width=True)
                        
                        # Download links for HTML versions
                        st.markdown("**📥 Download Interactive HTML**")
                        col1, col2 = st.columns(2)
                        
                        interactive_html = pair_dir / "interactive.html"
                        offline_html = pair_dir / "interactive_offline.html"
                        
                        with col1:
                            if interactive_html.exists():
                                st.download_button(
                                    "📊 Interactive (CDN)",
                                    data=interactive_html.read_bytes(),
                                    file_name=f"pair_{seq_num:03d}_interactive.html",
                                    mime="text/html",
                                    key=f"cdn_{seq_num}",
                use_container_width=True
            )
                        
                        with col2:
                            if offline_html.exists():
                                st.download_button(
                                    "💾 Offline Slider",
                                    data=offline_html.read_bytes(),
                                    file_name=f"pair_{seq_num:03d}_offline.html",
                                    mime="text/html",
                                    key=f"offline_{seq_num}",
                                    use_container_width=True
                                )
                    else:
                        st.info(f"Visualizations not found for pair #{seq_num:03d}")
        
        st.markdown("---")
        
        # Display Tier B pairs (collapsible)
        if len(tier_b) > 0:
            with st.expander(f"⚠️ Tier B - Manual Check ({len(tier_b)} pairs)", expanded=False):
                st.caption("These pairs require manual verification")
                
                for seq_num in range(len(tier_a) + 1, len(tier_a) + len(tier_b) + 1):
                    idx = tier_b.index[seq_num - len(tier_a) - 1]
                    row = tier_b.loc[idx]
                    
                    pair_path_a = row.get('Path_A', row.get('Image_A', ''))
                    pair_path_b = row.get('Path_B', row.get('Image_B', ''))
                    img_a_name = Path(pair_path_a).name if pair_path_a else f"Image A"
                    img_b_name = Path(pair_path_b).name if pair_path_b else f"Image B"
                    
                    st.markdown(f"**Pair #{seq_num:03d}**: {img_a_name} vs {img_b_name}")
                    
                    pair_dir = comp_dir / f"pair_{seq_num:03d}_detailed"
                    if pair_dir.exists():
                        img_a_path = pair_dir / "1_raw_A.png"
                        img_b_path = pair_dir / "2_raw_B_aligned.png"
                        
                        if img_a_path.exists() and img_b_path.exists():
                            col1, col2 = st.columns(2)
                            with col1:
                                st.image(str(img_a_path), use_container_width=True)
                            with col2:
                                st.image(str(img_b_path), use_container_width=True)
                    
                    st.markdown("---")
        
        # Display other pairs (if any)
        if len(other) > 0:
            with st.expander(f"📋 Other Pairs ({len(other)} pairs)", expanded=False):
                st.caption("Lower confidence pairs")
                for seq_num in range(len(tier_a) + len(tier_b) + 1, len(tier_a) + len(tier_b) + len(other) + 1):
                    idx_in_other = seq_num - len(tier_a) - len(tier_b) - 1
                    if idx_in_other < len(other):
                        idx = other.index[idx_in_other]
                        row = other.loc[idx]
                        
                        pair_path_a = row.get('Path_A', row.get('Image_A', ''))
                        pair_path_b = row.get('Path_B', row.get('Image_B', ''))
                        img_a_name = Path(pair_path_a).name if pair_path_a else f"Image A"
                        img_b_name = Path(pair_path_b).name if pair_path_b else f"Image B"
                        
                        st.markdown(f"**Pair #{seq_num:03d}**: {img_a_name} vs {img_b_name}")
    else:
        st.info("No results found. Please download the ZIP file above to view all visualizations offline.")
    
    st.markdown("---")
    
    # Preview table
    st.subheader("🔍 Results Table")
    
    if tsv_path.exists():
        try:
            df = load_report(tsv_path)
        except ValueError as e:
            st.error(f"⚠️ {str(e)}")
            st.info("The detection process may have encountered an error. Please check the logs or try running again.")
            st.stop()
        
        # Normalize Tier column
        if 'Tier' not in df.columns:
            df['Tier'] = 'Other'
        
        tier_filter = st.multiselect(
            "Filter by Tier",
            options=['A', 'B', 'Other'],
            default=['A', 'B']
        )
        
        if tier_filter:
            df = df[df['Tier'].isin(tier_filter)]
        
        st.dataframe(df, use_container_width=True, height=400, hide_index=True)
        st.caption(f"Showing {len(df)} pairs")
    
    # Performance details
    with st.expander("⚡ Performance Details"):
        st.json({
            "Runtime": f"{results['runtime']:.2f}s",
            "Total Pairs": results['total_pairs'],
            "Panels": results.get('panels', 'N/A'),
            "Pages": results.get('pages', 'N/A'),
            "Device": deps.get('device', 'CPU'),
            "Timestamp": datetime.now().isoformat()
        })
    
    # New analysis
    st.markdown("---")
    if st.button("🔄 Analyze New PDF", use_container_width=True, type="primary"):
        try:
            if output_dir.exists():
                shutil.rmtree(output_dir)
            upload_dir = TEMP_DIR / "uploads"
            if upload_dir.exists():
                shutil.rmtree(upload_dir)
        except:
            pass
        
        st.session_state.processing_complete = False
        st.session_state.results = None
        st.session_state.pdf_path = None
        st.session_state.config = None
        if 'preset' in st.session_state:
            del st.session_state.preset
        st.session_state.current_page = "📤 Upload"
        st.rerun()

# ═══════════════════════════════════════════════════════════════
# PAGE 5: TEST LAB (Performance Evaluation & Auto-Tuning)
# ═══════════════════════════════════════════════════════════════

elif page == "🧪 Test Lab":
    st.markdown('<h1 class="main-header">Test Lab - Performance Evaluation</h1>', unsafe_allow_html=True)
    
    # Check if we have results to analyze
    if st.session_state.results is None or st.session_state.results.get('output_dir') is None:
        st.warning("⚠️ No analysis results available")
        st.info("👉 Run an analysis first, then come back to Test Lab to evaluate performance")
        
        if st.button("← Back to Upload"):
            st.session_state.current_page = "📤 Upload"
            st.rerun()
        st.stop()
    
    output_dir = Path(st.session_state.results['output_dir'])
    tsv_file = output_dir / "final_merged_report.tsv"
    
    if not tsv_file.exists():
        st.error(f"❌ Results file not found: {tsv_file}")
        st.stop()
    
    # Get current parameters from config
    config = st.session_state.config or {}
    current_sim = config.get('sim_threshold', 0.96)
    current_ssim = config.get('ssim_threshold', 0.90)
    current_phash = config.get('phash_max_dist', 4)
    
    # Import evaluation function
    sys.path.insert(0, str(Path(__file__).parent / "tools"))
    try:
        from local_eval_policy import evaluate
    except ImportError:
        st.error("❌ Evaluation module not found. Please ensure tools/local_eval_policy.py exists.")
        st.stop()
    
    # Run evaluation
    with st.spinner("🔍 Evaluating performance..."):
        result = evaluate(str(tsv_file), current_sim, current_ssim, current_phash)
    
    if result is None:
        st.error("❌ Evaluation failed")
        st.stop()
    
    # Store evaluation result in session state
    st.session_state.evaluation_result = result
    
    # ═══════════════════════════════════════════════════════════════
    # PERFORMANCE OVERVIEW
    # ═══════════════════════════════════════════════════════════════
    
    st.subheader("📊 Performance Overview")
    
    overall_status = "✅ PASS" if result['overall_pass'] else "❌ FAIL"
    status_color = "green" if result['overall_pass'] else "red"
    
    st.markdown(f"<h2 style='color: {status_color};'>{overall_status}</h2>", unsafe_allow_html=True)
    
    # Metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Pairs",
            result['counts']['total_pairs'],
            help="All duplicate candidates found"
        )
    
    with col2:
        tier_a = result['counts']['tierA']
        tier_a_ratio = result['metrics']['tierA_ratio']
        st.metric(
            "Tier A (Review)",
            tier_a,
            f"{tier_a_ratio:.1%}",
            help="High-confidence duplicates"
        )
    
    with col3:
        tier_b = result['counts']['tierB']
        tier_b_ratio = result['metrics']['tierB_ratio']
        st.metric(
            "Tier B (Check)",
            tier_b,
            f"{tier_b_ratio:.1%}",
            help="Borderline cases for manual review"
        )
    
    with col4:
        fp_proxy = result['counts']['fp_proxy']
        fp_rate = result['metrics']['fp_rate']
        fp_status = "✅" if result['pass_fail']['fp_rate'] else "❌"
        st.metric(
            f"FP Proxy {fp_status}",
            fp_proxy,
            f"{fp_rate:.1%}",
            help="Estimated false positives (high CLIP, low SSIM, weak geometry)",
            delta_color="inverse"
        )
    
    # Detailed metrics
    st.markdown("---")
    st.subheader("📈 Quality Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Policy Checks:**")
        
        # FP Rate
        fp_pass = result['pass_fail']['fp_rate']
        fp_icon = "✅" if fp_pass else "❌"
        st.markdown(f"{fp_icon} **FP Rate:** {result['metrics']['fp_rate']:.1%} (target: ≤{result['policy']['max_fp_rate']:.0%})")
        
        # Cross-page ratio
        cross_pass = result['pass_fail']['cross_page']
        cross_icon = "✅" if cross_pass else "❌"
        st.markdown(f"{cross_icon} **Cross-page Ratio:** {result['metrics']['cross_page_ratio']:.1%} (target: ≥{result['policy']['min_cross_page_ratio']:.0%})")
        
        # Tier A share
        tier_pass = result['pass_fail']['tierA_ratio']
        tier_icon = "✅" if tier_pass else "❌"
        st.markdown(f"{tier_icon} **Tier A Share:** {result['metrics']['tierA_ratio']:.1%} (target: ≥{result['policy']['min_tierA_ratio']:.0%})")
        
        # Anchor precision
        anchor_prec = result['metrics']['anchor_precision']
        if anchor_prec is not None:
            anchor_pass = result['pass_fail']['anchor_precision']
            anchor_icon = "✅" if anchor_pass else "❌"
            st.markdown(f"{anchor_icon} **Anchor Precision:** {anchor_prec:.1%} (target: ≥{result['policy']['min_anchor_precision']:.0%})")
        else:
            st.markdown(f"ℹ️ **Anchor Precision:** N/A (no exact/strong matches found)")
    
    with col2:
        st.markdown("**Current Parameters:**")
        st.code(f"""
sim_threshold:    {result['current_params']['sim_threshold']}
ssim_threshold:   {result['current_params']['ssim_threshold']}
phash_max_dist:   {result['current_params']['phash_max_dist']}
        """)
    
    # ═══════════════════════════════════════════════════════════════
    # AUTO-TUNING SUGGESTIONS
    # ═══════════════════════════════════════════════════════════════
    
    st.markdown("---")
    st.subheader("💡 Auto-Tuning Suggestions")
    
    suggestions = result['suggestions']
    has_suggestions = any(v is not None for v in suggestions.values())
    
    if not result['overall_pass'] and has_suggestions:
        st.info("📊 Based on the 95th percentile of false positive candidates, we suggest:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if suggestions['sim_threshold']:
                delta = suggestions['sim_threshold'] - current_sim
                st.metric(
                    "CLIP Threshold",
                    f"{suggestions['sim_threshold']:.3f}",
                    f"+{delta:.3f}",
                    help="Push CLIP threshold above FP cluster"
                )
            else:
                st.metric("CLIP Threshold", f"{current_sim:.3f}", "No change")
        
        with col2:
            if suggestions['ssim_threshold']:
                delta = suggestions['ssim_threshold'] - current_ssim
                st.metric(
                    "SSIM Threshold",
                    f"{suggestions['ssim_threshold']:.3f}",
                    f"+{delta:.3f}",
                    help="Push SSIM threshold above FP cluster"
                )
            else:
                st.metric("SSIM Threshold", f"{current_ssim:.3f}", "No change")
        
        with col3:
            if suggestions['phash_max_dist']:
                delta = suggestions['phash_max_dist'] - current_phash
                st.metric(
                    "pHash Max Distance",
                    suggestions['phash_max_dist'],
                    f"{delta:+d}" if delta != 0 else "No change",
                    help="Adjust pHash tolerance for anchors"
                )
            else:
                st.metric("pHash Max Distance", current_phash, "No change")
        
        # Apply suggestions button
        st.markdown("###")
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("✨ Apply Suggestions", use_container_width=True, type="primary"):
                # Update config with suggestions
                if suggestions['sim_threshold']:
                    st.session_state.config['sim_threshold'] = suggestions['sim_threshold']
                if suggestions['ssim_threshold']:
                    st.session_state.config['ssim_threshold'] = suggestions['ssim_threshold']
                if suggestions['phash_max_dist']:
                    st.session_state.config['phash_max_dist'] = suggestions['phash_max_dist']
                
                st.success("✅ Suggestions applied to configuration!")
                st.info("👉 Go to Configure page to review, or click 'Run Quick Re-test' to test immediately")
                st.rerun()
        
        with col2:
            if st.button("▶️ Run Quick Re-test", use_container_width=True):
                # Apply suggestions and jump to Run page
                if suggestions['sim_threshold']:
                    st.session_state.config['sim_threshold'] = suggestions['sim_threshold']
                if suggestions['ssim_threshold']:
                    st.session_state.config['ssim_threshold'] = suggestions['ssim_threshold']
                if suggestions['phash_max_dist']:
                    st.session_state.config['phash_max_dist'] = suggestions['phash_max_dist']
                
                st.session_state.current_page = "▶️ Run"
                st.rerun()
        
        with col3:
            if st.button("⚙️ Configure", use_container_width=True):
                st.session_state.current_page = "⚙️ Configure"
                st.rerun()
    
    elif result['overall_pass']:
        st.success("✅ **All quality metrics passed!** Current parameters are performing well.")
        st.info("💡 No tuning needed. Your detection is working optimally.")
    
    else:
        st.warning("⚠️ Some metrics failed, but no automatic suggestions available.")
        st.info("💡 Try manually adjusting parameters in the Configure page.")
    
    # ═══════════════════════════════════════════════════════════════
    # PAGE-FOCUSED ANALYSIS
    # ═══════════════════════════════════════════════════════════════
    
    st.markdown("---")
    st.subheader("📄 Page-Focused Analysis")
    
    # Load results DataFrame
    df = pd.read_csv(tsv_file, sep='\t', low_memory=False)
    
    # Page selector
    col1, col2 = st.columns(2)
    with col1:
        page_num_1 = st.number_input("Page Number 1", min_value=1, value=19, step=1, key="page1")
    with col2:
        page_num_2 = st.number_input("Page Number 2", min_value=1, value=30, step=1, key="page2")
    
    # Filter results for specific pages
    def extract_page_num(path_str):
        import re
        if not isinstance(path_str, str):
            return None
        m = re.search(r'page[_-]?(\d+)', path_str, flags=re.I)
        return int(m.group(1)) if m else None
    
    df['Page_A'] = df['Path_A'].apply(extract_page_num)
    df['Page_B'] = df['Path_B'].apply(extract_page_num)
    
    # Results for page 1
    page1_pairs = df[(df['Page_A'] == page_num_1) | (df['Page_B'] == page_num_1)]
    
    # Results for page 2
    page2_pairs = df[(df['Page_A'] == page_num_2) | (df['Page_B'] == page_num_2)]
    
    tab1, tab2 = st.tabs([f"📄 Page {page_num_1}", f"📄 Page {page_num_2}"])
    
    with tab1:
        st.markdown(f"### Page {page_num_1} Analysis")
        st.caption(f"Found {len(page1_pairs)} pairs involving page {page_num_1}")
        
        if len(page1_pairs) > 0:
            # Show top matches
            display_cols = ['Image_A', 'Image_B', 'Cosine_Similarity', 'SSIM', 'Hamming_Distance', 'Tier', 'Tier_Path', 'Confocal_FP']
            available_cols = [col for col in display_cols if col in page1_pairs.columns]
            
            st.dataframe(
                page1_pairs[available_cols].head(20),
                use_container_width=True,
                hide_index=True
            )
            
            # Statistics
            tier_a_count = len(page1_pairs[page1_pairs.get('Tier') == 'A'])
            tier_b_count = len(page1_pairs[page1_pairs.get('Tier') == 'B'])
            fp_count = len(page1_pairs[page1_pairs.get('Confocal_FP', False) == True])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Tier A", tier_a_count)
            with col2:
                st.metric("Tier B", tier_b_count)
            with col3:
                st.metric("FP Proxy", fp_count)
        else:
            st.info(f"No pairs found involving page {page_num_1}")
    
    with tab2:
        st.markdown(f"### Page {page_num_2} Analysis")
        st.caption(f"Found {len(page2_pairs)} pairs involving page {page_num_2}")
        
        if len(page2_pairs) > 0:
            # Show top matches
            available_cols = [col for col in display_cols if col in page2_pairs.columns]
            
            st.dataframe(
                page2_pairs[available_cols].head(20),
                use_container_width=True,
                hide_index=True
            )
            
            # Statistics
            tier_a_count = len(page2_pairs[page2_pairs.get('Tier') == 'A'])
            tier_b_count = len(page2_pairs[page2_pairs.get('Tier') == 'B'])
            fp_count = len(page2_pairs[page2_pairs.get('Confocal_FP', False) == True])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Tier A", tier_a_count)
            with col2:
                st.metric("Tier B", tier_b_count)
            with col3:
                st.metric("FP Proxy", fp_count)
        else:
            st.info(f"No pairs found involving page {page_num_2}")
    
    # ═══════════════════════════════════════════════════════════════
    # NAVIGATION
    # ═══════════════════════════════════════════════════════════════
    
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("← Back to Results", use_container_width=True):
            st.session_state.current_page = "📊 Results"
            st.rerun()
    
    with col2:
        if st.button("⚙️ Configure", use_container_width=True):
            st.session_state.current_page = "⚙️ Configure"
            st.rerun()
    
    with col3:
        if st.button("▶️ Re-run Analysis", use_container_width=True):
            st.session_state.current_page = "▶️ Run"
        st.rerun()

# Footer
st.markdown("---")
st.caption("🔬 Duplicate Detection Studio v2.5 | Production Ready | Files auto-deleted after 1 hour")

