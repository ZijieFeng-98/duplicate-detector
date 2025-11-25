"""
Similarity Engine Module

Handles all similarity computation methods:
- CLIP semantic embeddings
- pHash perceptual hashing
- SSIM structural similarity

This module provides the core duplicate detection algorithms.
"""

from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
import json
import hashlib
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import imagehash
import cv2
from PIL import Image, ImageOps
from tqdm import tqdm
from multiprocessing import Pool
from skimage.metrics import structural_similarity as ssim_func

try:
    import open_clip
except ImportError:
    open_clip = None


# ============================================================================
# CACHE HELPERS
# ============================================================================

def get_cache_path(cache_name: str, output_dir: Path, cache_version: str) -> Path:
    """Get cache file path with version"""
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{cache_name}_{cache_version}.npy"


def get_cache_meta_path(cache_name: str, output_dir: Path, cache_version: str) -> Path:
    """Get per-cache metadata path (for file-hash validation)"""
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{cache_name}_{cache_version}_meta.json"


def compute_file_hash(paths: List[Path]) -> str:
    """Compute hash of file list for cache validation (mtime + size)"""
    hasher = hashlib.md5()
    for p in sorted(paths):
        st = p.stat()
        hasher.update(str(p).encode())
        hasher.update(str(st.st_mtime).encode())
        hasher.update(str(st.st_size).encode())
    return hasher.hexdigest()


# ============================================================================
# CLIP EMBEDDINGS
# ============================================================================

@dataclass
class CLIPModel:
    """CLIP model container"""
    model: any
    preprocess: any
    device: str


def load_clip(
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    device: str = "cpu"
) -> CLIPModel:
    """
    Load CLIP model for semantic similarity
    
    Args:
        model_name: CLIP model name
        pretrained: Pretrained weights identifier
        device: Device to load model on
    
    Returns:
        CLIPModel instance
    """
    if open_clip is None:
        raise ImportError("open-clip-torch not installed")
    
    print(f"  Loading CLIP on {device}...")
    
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    model.to(device)
    model.eval()
    print(f"  ✓ CLIP loaded on {device}")
    return CLIPModel(model, preprocess, device)


@torch.no_grad()
def embed_images(
    paths: List[Path],
    clip: CLIPModel,
    batch_size: int = 64
) -> np.ndarray:
    """
    Generate CLIP embeddings with batching
    
    Args:
        paths: List of image paths
        clip: CLIPModel instance
        batch_size: Batch size for processing
    
    Returns:
        Array of normalized embeddings (n_images, 512)
    """
    vecs = []
    
    print(f"  Using batch_size={batch_size} on {clip.device}")
    
    for i in tqdm(range(0, len(paths), batch_size), desc="Embedding"):
        batch = paths[i:i+batch_size]
        imgs = [clip.preprocess(Image.open(p).convert("RGB")) for p in batch]
        if not imgs:
            continue
        
        x = torch.stack(imgs).to(clip.device)
        feats = clip.model.encode_image(x)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        vecs.append(feats.cpu().numpy())
    
    return np.vstack(vecs).astype("float32") if vecs else np.zeros((0, 512), dtype="float32")


def load_or_compute_embeddings(
    panel_paths: List[Path],
    clip: CLIPModel,
    output_dir: Path,
    cache_version: str,
    enable_cache: bool = True,
    batch_size: int = 64,
    use_mmap: bool = True
) -> np.ndarray:
    """
    Load cached embeddings or compute new ones
    
    Args:
        panel_paths: List of panel image paths
        clip: CLIPModel instance
        output_dir: Output directory for cache
        cache_version: Cache version string
        enable_cache: Whether to use cache
        batch_size: Batch size for embedding
        use_mmap: Use memory-mapped loading
    
    Returns:
        Array of embeddings
    """
    cache_path = get_cache_path("clip_embeddings", output_dir, cache_version)
    meta_cache_path = get_cache_meta_path("clip_embeddings", output_dir, cache_version)
    
    # Check if cache exists and is valid
    if enable_cache and cache_path.exists() and meta_cache_path.exists():
        try:
            with open(meta_cache_path, 'r') as f:
                cached_meta = json.load(f)
            
            current_hash = compute_file_hash(panel_paths)
            
            if cached_meta.get("file_hash") == current_hash:
                if use_mmap:
                    print("  ✓ Loading cached embeddings (memory-mapped, zero-copy)...")
                    vecs = np.load(cache_path, mmap_mode='r')
                else:
                    print("  ✓ Loading cached embeddings (in-memory)...")
                    vecs = np.load(cache_path)
                
                if vecs.shape[0] == len(panel_paths):
                    return vecs
        except Exception as e:
            print(f"  ⚠ Cache load failed: {e}, recomputing...")
    
    # Compute new embeddings
    print("  Computing new embeddings...")
    vecs = embed_images(panel_paths, clip, batch_size=batch_size)
    
    # Save cache
    if enable_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, vecs)
        with open(meta_cache_path, 'w') as f:
            json.dump({
                "file_hash": compute_file_hash(panel_paths),
                "num_panels": len(panel_paths),
                "timestamp": datetime.now().isoformat()
            }, f)
        print(f"  ✓ Cached embeddings to {cache_path}")
    
    return vecs


def clip_find_duplicates_threshold(
    panel_paths: List[Path],
    vecs: np.ndarray,
    threshold: float,
    meta_df: pd.DataFrame,
    suppress_same_page: bool = False,
    suppress_adjacent_page: bool = False,
    adjacent_page_max_gap: int = 1,
    max_output_pairs: Optional[int] = None
) -> pd.DataFrame:
    """
    Find duplicate pairs using CLIP similarity threshold
    
    Args:
        panel_paths: List of panel paths
        vecs: CLIP embeddings array
        threshold: Similarity threshold
        meta_df: Panel metadata dataframe
        suppress_same_page: Suppress same-page pairs
        suppress_adjacent_page: Suppress adjacent-page pairs
        adjacent_page_max_gap: Maximum page gap for adjacent
        max_output_pairs: Maximum number of pairs to return
    
    Returns:
        DataFrame with duplicate pairs
    """
    print(f"  Finding similar pairs (threshold≥{threshold})...")
    
    n = len(vecs)
    if n < 2:
        return pd.DataFrame()
    
    # Compute full similarity matrix
    sim_matrix = vecs @ vecs.T
    np.fill_diagonal(sim_matrix, -1.0)
    
    rows = []
    
    # Global scan: Consider ALL pairs above threshold
    iu, ju = np.triu_indices(n, k=1)  # Upper triangle (avoid duplicates)
    mask = sim_matrix[iu, ju] >= threshold
    cand_i = iu[mask]
    cand_j = ju[mask]
    cand_s = sim_matrix[iu, ju][mask]
    
    # Sort by score descending (best pairs first)
    order = np.argsort(-cand_s)
    if max_output_pairs is not None and len(order) > max_output_pairs:
        order = order[:max_output_pairs]
    cand_i, cand_j, cand_s = cand_i[order], cand_j[order], cand_s[order]
    
    # Helper functions for page filtering
    def same_page(pa: str, pb: str) -> bool:
        a = meta_df.loc[meta_df["Panel_Path"] == pa]
        b = meta_df.loc[meta_df["Panel_Path"] == pb]
        if a.empty or b.empty:
            return False
        return a.iloc[0]["Page"] == b.iloc[0]["Page"]
    
    def page_of(panel_path: str) -> Optional[int]:
        row = meta_df.loc[meta_df["Panel_Path"] == panel_path]
        if row.empty:
            return None
        page_val = row.iloc[0]["Page"]
        if isinstance(page_val, str):
            if page_val.startswith('page_'):
                return int(page_val.replace('page_', ''))
            else:
                return int(page_val)
        else:
            return int(page_val)
    
    def adjacent_pages(pa: str, pb: str) -> bool:
        a = page_of(pa)
        b = page_of(pb)
        if a is None or b is None:
            return False
        return abs(a - b) <= adjacent_page_max_gap
    
    for i, j, score in zip(cand_i, cand_j, cand_s):
        pa, pb = str(panel_paths[i]), str(panel_paths[j])
        
        if suppress_same_page and same_page(pa, pb):
            continue
        if suppress_adjacent_page and adjacent_pages(pa, pb):
            continue
        
        rows.append({
            "Image_A": Path(pa).name,
            "Image_B": Path(pb).name,
            "Path_A": pa,
            "Path_B": pb,
            "Cosine_Similarity": float(score)
        })
    
    df = pd.DataFrame(rows)
    print(f"  ✓ {len(df)} pairs (CLIP ≥ {threshold})")
    return df


# ============================================================================
# PHASH (PERCEPTUAL HASHING)
# ============================================================================

def phash_hex(img_path: Path) -> str:
    """Compute pHash for a single image"""
    try:
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = ImageOps.autocontrast(img)
        return str(imagehash.phash(img))
    except Exception:
        return ""


def compute_phash_bundle(img: Image.Image) -> Dict[str, str]:
    """
    Compute pHash for 8 transform variants (rotation + mirror)
    
    Args:
        img: PIL Image
    
    Returns:
        Dictionary with 8 hash values
    """
    bundle = {}
    
    try:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = ImageOps.autocontrast(img)
        
        for angle in [0, 90, 180, 270]:
            rotated = img.rotate(angle, expand=True) if angle > 0 else img
            bundle[f'rot_{angle}'] = str(imagehash.phash(rotated))
        
        mirrored = ImageOps.mirror(img)
        for angle in [0, 90, 180, 270]:
            rotated = mirrored.rotate(angle, expand=True) if angle > 0 else mirrored
            bundle[f'mirror_h_rot_{angle}'] = str(imagehash.phash(rotated))
    except Exception as e:
        warnings.warn(f"pHash bundle computation failed: {e}")
        for key in ['rot_0', 'rot_90', 'rot_180', 'rot_270',
                   'mirror_h_rot_0', 'mirror_h_rot_90', 'mirror_h_rot_180', 'mirror_h_rot_270']:
            bundle[key] = ""
    
    return bundle


def hamming_min_transform(
    bundle_a: Dict[str, str],
    bundle_b: Dict[str, str],
    short_circuit: int = 3
) -> Tuple[int, str]:
    """
    Find minimum Hamming distance across all transform pairs
    
    Args:
        bundle_a: First pHash bundle
        bundle_b: Second pHash bundle
        short_circuit: Early exit if distance <= this value
    
    Returns:
        Tuple of (min_distance, transform_description)
    """
    min_dist = 999
    min_transform = "none"
    
    for key_a in bundle_a:
        hash_a = bundle_a[key_a]
        if not hash_a:
            continue
            
        for key_b in bundle_b:
            hash_b = bundle_b[key_b]
            if not hash_b:
                continue
            
            try:
                dist = imagehash.hex_to_hash(hash_a) - imagehash.hex_to_hash(hash_b)
                
                if dist < min_dist:
                    min_dist = dist
                    min_transform = f"{key_a} vs {key_b}"
                    
                    if dist <= short_circuit:
                        return (min_dist, min_transform)
            except Exception:
                continue
    
    return (min_dist, min_transform)


def load_or_compute_phash_bundles(
    panel_paths: List[Path],
    output_dir: Path,
    cache_version: str,
    enable_cache: bool = True,
    num_workers: int = 4
) -> List[Dict[str, str]]:
    """
    Load cached pHash bundles or compute new ones
    
    Args:
        panel_paths: List of panel paths
        output_dir: Output directory
        cache_version: Cache version
        enable_cache: Enable caching
        num_workers: Number of parallel workers
    
    Returns:
        List of pHash bundles (one per panel)
    """
    cache_path = get_cache_path("phash_bundles", output_dir, cache_version)
    meta_path = get_cache_meta_path("phash_bundles", output_dir, cache_version)
    
    if enable_cache and cache_path.exists() and meta_path.exists():
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            if meta.get("file_hash") == compute_file_hash(panel_paths):
                cached_data = np.load(cache_path, allow_pickle=True)
                print(f"  ✓ Loaded cached pHash bundles")
                return cached_data.tolist()
        except Exception:
            pass
    
    print(f"  Computing pHash bundles (8 transforms)...")
    bundles = []
    
    # Parallel computation
    if num_workers > 1:
        with Pool(num_workers) as pool:
            imgs = [Image.open(p) for p in panel_paths]
            bundles = list(tqdm(
                pool.imap(compute_phash_bundle, imgs),
                total=len(panel_paths),
                desc="pHash bundles"
            ))
    else:
        for path in tqdm(panel_paths, desc="pHash bundles"):
            img = Image.open(path)
            bundle = compute_phash_bundle(img)
            bundles.append(bundle)
    
    if enable_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, np.array(bundles, dtype=object))
        with open(meta_path, 'w') as f:
            json.dump({
                "file_hash": compute_file_hash(panel_paths),
                "num_panels": len(panel_paths),
                "timestamp": datetime.now().isoformat()
            }, f)
    
    return bundles


def phash_find_duplicates_with_bundles(
    panel_paths: List[Path],
    max_dist: int,
    meta_df: pd.DataFrame,
    output_dir: Path,
    cache_version: str,
    enable_cache: bool = True,
    num_workers: int = 4,
    suppress_same_page: bool = False,
    suppress_adjacent_page: bool = False,
    adjacent_page_max_gap: int = 1,
    bundle_short_circuit: int = 3
) -> pd.DataFrame:
    """
    Find duplicates using rotation/mirror-robust pHash bundles
    
    Args:
        panel_paths: List of panel paths
        max_dist: Maximum Hamming distance
        meta_df: Panel metadata dataframe
        output_dir: Output directory for cache
        cache_version: Cache version
        enable_cache: Enable caching
        num_workers: Number of workers
        suppress_same_page: Suppress same-page pairs
        suppress_adjacent_page: Suppress adjacent-page pairs
        adjacent_page_max_gap: Maximum page gap
        bundle_short_circuit: Early exit threshold
    
    Returns:
        DataFrame with duplicate pairs
    """
    print(f"\n[Stage 3] pHash-RT with rotation/mirror bundles (≤{max_dist})...")
    
    bundles = load_or_compute_phash_bundles(
        panel_paths, output_dir, cache_version, enable_cache, num_workers
    )
    
    rows = []
    # Bucket prefilter (use all 8 transforms for better recall)
    buckets = {}
    for i, bundle in enumerate(bundles):
        prefixes = []
        for k in ['rot_0','rot_90','rot_180','rot_270',
                  'mirror_h_rot_0','mirror_h_rot_90','mirror_h_rot_180','mirror_h_rot_270']:
            h = bundle.get(k, '')
            if h:
                prefixes.append(h[:4])
        for pref in set(prefixes):
            if pref:
                buckets.setdefault(pref, []).append(i)
    
    checked = set()
    for idxs in buckets.values():
        for a in range(len(idxs)-1):
            for b in range(a+1, len(idxs)):
                i, j = idxs[a], idxs[b]
                if (i, j) in checked:
                    continue
                checked.add((i, j))
                
                d, transform = hamming_min_transform(
                    bundles[i], bundles[j], bundle_short_circuit
                )
                
                if d <= max_dist:
                    pa, pb = str(panel_paths[i]), str(panel_paths[j])
                    
                    # Page filtering helpers
                    def same_page(pa: str, pb: str) -> bool:
                        a = meta_df.loc[meta_df["Panel_Path"] == pa]
                        b = meta_df.loc[meta_df["Panel_Path"] == pb]
                        if a.empty or b.empty:
                            return False
                        return a.iloc[0]["Page"] == b.iloc[0]["Page"]
                    
                    def page_of(panel_path: str) -> Optional[int]:
                        row = meta_df.loc[meta_df["Panel_Path"] == panel_path]
                        if row.empty:
                            return None
                        page_val = row.iloc[0]["Page"]
                        if isinstance(page_val, str):
                            if page_val.startswith('page_'):
                                return int(page_val.replace('page_', ''))
                            else:
                                return int(page_val)
                        else:
                            return int(page_val)
                    
                    def adjacent_pages(pa: str, pb: str) -> bool:
                        a = page_of(pa)
                        b = page_of(pb)
                        if a is None or b is None:
                            return False
                        return abs(a - b) <= adjacent_page_max_gap
                    
                    if suppress_same_page and same_page(pa, pb):
                        continue
                    if suppress_adjacent_page and adjacent_pages(pa, pb):
                        continue
                    
                    rows.append({
                        "Image_A": Path(pa).name,
                        "Image_B": Path(pb).name,
                        "Path_A": pa,
                        "Path_B": pb,
                        "Hamming_Distance": int(d),
                        "Transform_Matched": transform
                    })
    
    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values("Hamming_Distance")
    
    print(f"  ✓ {len(df)} pairs (pHash ≤ {max_dist})")
    return df


# ============================================================================
# SSIM (STRUCTURAL SIMILARITY)
# ============================================================================

def apply_clahe(img_gray: np.ndarray, clip_limit: float = 2.5, tile_size: int = 8) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization"""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(img_gray)


def normalize_photometric(
    img: np.ndarray,
    apply_clahe_flag: bool = True,
    clip_limit: float = 2.5,
    tile_size: int = 8
) -> Tuple[np.ndarray, Dict]:
    """
    Apply deterministic photometric normalization
    
    Args:
        img: Input image (BGR or grayscale)
        apply_clahe_flag: Apply CLAHE
        clip_limit: CLAHE clip limit
        tile_size: CLAHE tile size
    
    Returns:
        Tuple of (normalized_image, parameters_dict)
    """
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img.copy()
    
    params = {'clahe_applied': apply_clahe_flag, 'clip_limit': clip_limit, 'tile_size': tile_size}
    
    if apply_clahe_flag:
        img_gray = apply_clahe(img_gray, clip_limit, tile_size)
    
    mean = np.mean(img_gray)
    std = np.std(img_gray)
    
    if std > 1e-6:
        img_normalized = (img_gray - mean) / std
    else:
        img_normalized = img_gray - mean
    
    params['mean'] = float(mean)
    params['std'] = float(std)
    
    img_normalized = np.clip(img_normalized * 50 + 128, 0, 255).astype(np.uint8)
    
    return img_normalized, params


def compute_ssim_normalized(
    path_a: str,
    path_b: str,
    target_h: int = 512,
    apply_norm: bool = True,
    use_patchwise: bool = True,
    grid_h: int = 3,
    grid_w: int = 3,
    topk_patches: int = 4,
    mix_weight: float = 0.6,
    enable_text_masking: bool = False,
    detect_text_regions_heuristic: Optional[callable] = None
) -> Tuple[float, Dict]:
    """
    Compute SSIM with photometric normalization + patch-wise refinement
    
    Args:
        path_a: Path to first image
        path_b: Path to second image
        target_h: Target height for resizing
        apply_norm: Apply photometric normalization
        use_patchwise: Enable patch-wise SSIM
        grid_h: Grid rows
        grid_w: Grid columns
        topk_patches: Number of top patches to average
        mix_weight: Weight for patch vs global SSIM
        enable_text_masking: Enable text masking
        detect_text_regions_heuristic: Function to detect text regions
    
    Returns:
        Tuple of (mixed_score, metadata_dict)
    """
    try:
        img_a = cv2.imread(path_a)
        img_b = cv2.imread(path_b)
        
        if img_a is None or img_b is None:
            return np.nan, {}
        
        # Optional text masking
        if enable_text_masking and detect_text_regions_heuristic:
            try:
                mask_a = detect_text_regions_heuristic(cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB))
                mask_b = detect_text_regions_heuristic(cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB))
                img_a = cv2.inpaint(img_a, mask_a, 3, cv2.INPAINT_TELEA)
                img_b = cv2.inpaint(img_b, mask_b, 3, cv2.INPAINT_TELEA)
            except Exception:
                pass
        
        # Photometric normalization
        if apply_norm:
            img_a_gray, params_a = normalize_photometric(img_a)
            img_b_gray, params_b = normalize_photometric(img_b)
        else:
            img_a_gray = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
            img_b_gray = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
            params_a = params_b = {}
        
        # Resize by height and pad to same width
        def resize_to_height(img, h):
            scale = h / img.shape[0]
            new_w = max(1, int(round(img.shape[1] * scale)))
            return cv2.resize(img, (new_w, h), interpolation=cv2.INTER_AREA)
        
        img_a_gray = resize_to_height(img_a_gray, target_h)
        img_b_gray = resize_to_height(img_b_gray, target_h)
        
        max_w = max(img_a_gray.shape[1], img_b_gray.shape[1])
        if img_a_gray.shape[1] < max_w:
            img_a_gray = cv2.copyMakeBorder(img_a_gray, 0, 0, 0, max_w - img_a_gray.shape[1],
                                           cv2.BORDER_CONSTANT, value=255)
        if img_b_gray.shape[1] < max_w:
            img_b_gray = cv2.copyMakeBorder(img_b_gray, 0, 0, 0, max_w - img_b_gray.shape[1],
                                           cv2.BORDER_CONSTANT, value=255)
        
        # Compute global SSIM
        ssim_global, _ = ssim_func(img_a_gray, img_b_gray, full=True, data_range=255)
        
        # Compute patch-wise SSIM
        patch_scores = []
        patch_min = np.nan
        patch_topk_mean = np.nan
        
        if use_patchwise:
            gh = int(max(1, grid_h))
            gw = int(max(1, grid_w))
            
            ha, wa = img_a_gray.shape
            hb, wb = img_b_gray.shape
            
            # Use the smaller common grid-aligned area
            H = min(ha - (ha % gh), hb - (hb % gh))
            W = min(wa - (wa % gw), wb - (wb % gw))
            
            if H > 0 and W > 0:
                # Crop to aligned grid
                ca = img_a_gray[:H, :W]
                cb = img_b_gray[:H, :W]
                
                # Patch dimensions
                ph = H // gh
                pw = W // gw
                
                # Compute SSIM for each patch
                for r in range(gh):
                    for c in range(gw):
                        ya, yb = r * ph, (r + 1) * ph
                        xa, xb = c * pw, (c + 1) * pw
                        
                        patch_a = ca[ya:yb, xa:xb]
                        patch_b = cb[ya:yb, xa:xb]
                        
                        # Skip tiny patches
                        if patch_a.size < 64 or patch_b.size < 64:
                            continue
                        
                        try:
                            ps, _ = ssim_func(patch_a, patch_b, full=True, data_range=255)
                            if not np.isnan(ps):
                                patch_scores.append(float(ps))
                        except Exception:
                            continue
                
                # Aggregate patch statistics
                if patch_scores:
                    patch_scores_arr = np.array(patch_scores, dtype=float)
                    patch_min = float(np.min(patch_scores_arr))
                    
                    # Top-K mean
                    k = int(max(1, min(len(patch_scores_arr), topk_patches)))
                    topk_patches_arr = np.sort(patch_scores_arr)[-k:]
                    patch_topk_mean = float(np.mean(topk_patches_arr))
        
        # Compute mixed SSIM score
        if use_patchwise and not np.isnan(patch_topk_mean):
            w = float(np.clip(mix_weight, 0.0, 1.0))
            score_mixed = float((1.0 - w) * ssim_global + w * patch_topk_mean)
        else:
            score_mixed = float(ssim_global)
        
        # Package metadata
        metadata = {
            'normalization_applied': apply_norm,
            'params_a': params_a,
            'params_b': params_b,
            'ssim_global': float(ssim_global),
            'patch_min': float(patch_min) if not np.isnan(patch_min) else None,
            'patch_topk_mean': float(patch_topk_mean) if not np.isnan(patch_topk_mean) else None,
            'num_patches': len(patch_scores) if patch_scores else 0,
            'grid': [int(grid_h), int(grid_w)],
            'mix_weight': float(mix_weight) if use_patchwise else 0.0
        }
        
        return score_mixed, metadata
        
    except Exception as e:
        warnings.warn(f"SSIM computation failed: {e}")
        return np.nan, {}


def add_ssim_validation(
    df: pd.DataFrame,
    use_patchwise: bool = True,
    ssim_threshold: float = 0.85,
    patch_min_gate: float = 0.85,
    enable_text_masking: bool = False,
    detect_text_regions_heuristic: Optional[callable] = None
) -> pd.DataFrame:
    """
    Add SSIM validation to duplicate pairs DataFrame
    
    Args:
        df: DataFrame with duplicate pairs
        use_patchwise: Enable patch-wise SSIM
        ssim_threshold: SSIM threshold (for reference)
        patch_min_gate: Minimum patch SSIM gate
        enable_text_masking: Enable text masking
        detect_text_regions_heuristic: Text detection function
    
    Returns:
        DataFrame with SSIM columns added
    """
    if df.empty:
        return df
    
    print(f"\n[4b/7] SSIM annotation (patch-wise={use_patchwise}, threshold≥{ssim_threshold} for reference)...")
    
    ssim_scores = []
    patch_mins = []
    patch_topks = []
    global_ssims = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="SSIM"):
        try:
            score, metadata = compute_ssim_normalized(
                row['Path_A'],
                row['Path_B'],
                apply_norm=True,
                use_patchwise=use_patchwise,
                enable_text_masking=enable_text_masking,
                detect_text_regions_heuristic=detect_text_regions_heuristic
            )
            ssim_scores.append(score)
            patch_mins.append(metadata.get('patch_min', np.nan))
            patch_topks.append(metadata.get('patch_topk_mean', np.nan))
            global_ssims.append(metadata.get('ssim_global', score))
        except Exception as e:
            print(f"  ⚠️ SSIM failed for pair: {e}")
            ssim_scores.append(np.nan)
            patch_mins.append(np.nan)
            patch_topks.append(np.nan)
            global_ssims.append(np.nan)
    
    # Attach all metrics (NO FILTERING - let Tier gating decide)
    df = df.copy()
    df["SSIM"] = ssim_scores
    df["Patch_SSIM_Min"] = patch_mins
    df["Patch_SSIM_TopK"] = patch_topks
    df["Global_SSIM"] = global_ssims
    
    # Show discrimination effect
    if use_patchwise:
        valid_patch_mins = pd.to_numeric(df['Patch_SSIM_Min'], errors='coerce').dropna()
        if not valid_patch_mins.empty:
            high_patch = len(df[pd.to_numeric(df['Patch_SSIM_Min'], errors='coerce') >= patch_min_gate])
            print(f"    → {high_patch}/{len(df)} pairs pass patch gate (min≥{patch_min_gate})")
    
    print(f"  ✓ Computed SSIM for {len(df)} pairs (no filtering, Tier gate will decide)")
    
    return df

