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

@st.cache_data(show_spinner=False)
def load_report(tsv_path: Path):
    """Load TSV report with caching"""
    return pd.read_csv(BytesIO(tsv_path.read_bytes()), sep="\t", low_memory=False)

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
    
    st.caption("Choose a preset (recommended) or customize")
    
    # Presets
    st.subheader("🎯 Quick Presets")
    st.info("**📊 Test Results (14 known duplicates):** Balanced = 100% detect + 9 FPs | Thorough = 78% detect + 0 FPs | Fast = 100% detect + 25 FPs")
    col1, col2, col3 = st.columns(3)
    
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
            st.rerun()
    
    with col2:
        if st.button(f"🎯 Balanced\n{presets['balanced']['time']}", use_container_width=True, type="primary"):
            st.session_state.preset = presets['balanced']
            st.rerun()
    
    with col3:
        if st.button(f"🔬 Thorough\n{presets['thorough']['time']}", use_container_width=True):
            st.session_state.preset = presets['thorough']
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
            sim_threshold = st.slider("CLIP", 0.90, 0.99, preset['sim_threshold'], 0.01)
        
        with col2:
            phash_max_dist = st.slider("pHash", 1, 8, preset['phash_max_dist'], 1)
            ssim_threshold = st.slider("SSIM", 0.85, 0.95, preset['ssim_threshold'], 0.01)
        
        st.markdown("**Features**")
        col1, col2, col3 = st.columns(3)
        with col1:
            use_phash_bundles = st.checkbox("Rotation", preset['use_phash_bundles'])
        with col2:
            use_orb = st.checkbox("Crop Detection", preset['use_orb'])
        with col3:
            use_tier_gating = st.checkbox("Tiers", preset['use_tier_gating'])
        
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
                'auto_open': False
            }
            
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
    
    config = st.session_state.config
    
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    log_container = st.expander("📋 Detailed Logs", expanded=False)
    
    detector_script = Path(__file__).parent / "ai_pdf_panel_duplicate_check_AUTO.py"
    
    if not detector_script.exists():
        st.error("❌ Detector backend not found")
        st.info(f"Make sure `ai_pdf_panel_duplicate_check_AUTO.py` is in: {detector_script.parent}")
        st.stop()
    
    # Build command - uses the CLI interface we'll add
    cmd = [
        sys.executable, str(detector_script),
        "--pdf", str(st.session_state.pdf_path),
        "--output", str(config['output_dir']),
        "--dpi", str(config['dpi']),
        "--sim-threshold", str(config['sim_threshold']),
        "--phash-max-dist", str(config['phash_max_dist']),
        "--ssim-threshold", str(config['ssim_threshold']),
        "--batch-size", str(config['batch_size']),
        "--no-auto-open",  # Don't open browser in Streamlit
    ]
    
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
        status_text.info("⏳ Starting analysis...")
        
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
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
            
            with log_container:
                st.code("\n".join(log_lines[-40:]), language="text")
            
            # Progress matching
            for pat, label, prog in stage_patterns:
                if pat.search(line):
                    current_progress = max(current_progress, prog)
                    progress_bar.progress(current_progress)
                    status_text.info(f"⏳ {label}...")
                    break
        
        return_code = proc.wait()
        elapsed = time.time() - start_time
        
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
            
            st.balloons()
            st.success(f"✅ Found {results['total_pairs']} pairs in {elapsed:.1f}s")
            
            time.sleep(2)
            st.session_state.current_page = "📊 Results"
            st.rerun()
        else:
            st.error(f"❌ Analysis failed (exit code {return_code})")
            st.info("Check logs above for details")
    
    except Exception as e:
        st.error(f"❌ Error: {e}")
        import traceback
        st.code(traceback.format_exc())
    finally:
        try:
            proc.kill()
        except:
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
            # Create zip
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for file_path in output_dir.rglob('*'):
                    if file_path.is_file():
                        zip_file.write(file_path, file_path.relative_to(output_dir))
            
            st.download_button(
                "📦 Download All (ZIP)",
                data=zip_buffer.getvalue(),
                file_name=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip",
                use_container_width=True
            )
    
    st.markdown("---")
    
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
        df = load_report(tsv_path)
        
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

