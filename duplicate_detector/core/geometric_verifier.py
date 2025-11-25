"""
Geometric Verifier Module

Handles ORB-RANSAC geometric verification for partial duplicates.
Detects rotated, cropped, or partially overlapping images.
"""

from pathlib import Path
from typing import List, Tuple, Optional, Dict, Set
import json
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

from duplicate_detector.core.similarity_engine import (
    normalize_photometric,
    get_cache_path,
    get_cache_meta_path,
    compute_file_hash
)


def extract_orb_features(
    img_path: str,
    max_keypoints: int = 1000,
    retry_scales: List[float] = None,
    enable_text_masking: bool = False,
    detect_text_regions_heuristic: Optional[callable] = None
) -> Tuple[Optional[List], Optional[np.ndarray]]:
    """
    Extract ORB keypoints with multi-scale robustness
    
    Args:
        img_path: Path to image
        max_keypoints: Maximum number of keypoints
        retry_scales: List of scales to try (default: [1.0, 2.0, 0.5])
        enable_text_masking: Enable text masking
        detect_text_regions_heuristic: Function to detect text regions
    
    Returns:
        Tuple of (keypoints_list, descriptors_array)
    """
    if retry_scales is None:
        retry_scales = [1.0, 2.0, 0.5]
    
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None, None
        
        # Optional text masking for ORB
        if enable_text_masking and detect_text_regions_heuristic:
            try:
                mask = detect_text_regions_heuristic(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
            except Exception:
                pass
        
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray, _ = normalize_photometric(img_gray)
        
        orb = cv2.ORB_create(nfeatures=max_keypoints)
        kp, desc = [], None
        
        for scale in retry_scales:
            if scale != 1.0:
                h, w = img_gray.shape
                new_h, new_w = int(h * scale), int(w * scale)
                img_scaled = cv2.resize(img_gray, (new_w, new_h))
            else:
                img_scaled = img_gray
            
            kpi, desci = orb.detectAndCompute(img_scaled, None)
            
            if desci is not None and len(kpi) >= 50:
                if scale != 1.0:
                    for k in kpi:
                        k.pt = (k.pt[0] / scale, k.pt[1] / scale)
                return kpi, desci
        
        return (kp if len(kp) else None, desc)
        
    except Exception as e:
        warnings.warn(f"ORB extraction failed for {img_path}: {e}")
        return None, None


def match_orb_features(
    desc_a: np.ndarray,
    desc_b: np.ndarray,
    ratio_threshold: float = 0.75
) -> List[cv2.DMatch]:
    """
    Match ORB features using BFMatcher with Lowe's ratio test
    
    Args:
        desc_a: Descriptors from image A
        desc_b: Descriptors from image B
        ratio_threshold: Ratio threshold for Lowe's test
    
    Returns:
        List of good matches
    """
    if desc_a is None or desc_b is None:
        return []
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    try:
        matches = bf.knnMatch(desc_a, desc_b, k=2)
    except Exception:
        return []
    
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)
    
    return good_matches


def estimate_homography_ransac(
    kp_a: List,
    kp_b: List,
    matches: List[cv2.DMatch],
    min_inliers: int = 30,
    max_reproj_error: float = 4.0
) -> Dict:
    """
    Estimate homography using RANSAC with degenerate detection
    
    Args:
        kp_a: Keypoints from image A
        kp_b: Keypoints from image B
        matches: List of matches
        min_inliers: Minimum number of inliers
        max_reproj_error: Maximum reprojection error
    
    Returns:
        Dictionary with homography results
    """
    result = {
        'H': None, 'homography_type': None, 'inliers': 0, 'inlier_ratio': 0.0,
        'reproj_error': 999.0, 'crop_coverage': 0.0, 'is_partial_dupe': False, 'is_degenerate': False
    }
    
    if len(matches) < min_inliers:
        return result
    
    pts_a = np.float32([kp_a[m.queryIdx].pt for m in matches])
    pts_b = np.float32([kp_b[m.trainIdx].pt for m in matches])
    
    try:
        H, mask = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, ransacReprojThreshold=5.0)
        
        if H is None:
            return result
        
        det_H = np.linalg.det(H[:2, :2])
        if abs(det_H) < 0.1 or abs(det_H) > 10.0:
            result['is_degenerate'] = True
            return result
        
        inliers = np.sum(mask)
        inlier_ratio = inliers / len(matches)
        
        if inliers < min_inliers or inlier_ratio < 0.30:
            return result
        
        inlier_pts_a = pts_a[mask.ravel() == 1]
        inlier_pts_b = pts_b[mask.ravel() == 1]
        
        inlier_pts_a_h = np.hstack([inlier_pts_a, np.ones((len(inlier_pts_a), 1))])
        projected = (H @ inlier_pts_a_h.T).T
        projected = projected[:, :2] / projected[:, 2:3]
        
        errors = np.linalg.norm(projected - inlier_pts_b, axis=1)
        median_error = np.median(errors)
        
        if median_error > max_reproj_error:
            return result
        
        is_affine = np.allclose(H[2, :2], 0) and np.isclose(H[2, 2], 1)
        
        result['H'] = H
        result['homography_type'] = 'affine' if is_affine else 'projective'
        result['inliers'] = int(inliers)
        result['inlier_ratio'] = float(inlier_ratio)
        result['reproj_error'] = float(median_error)
        
        return result
        
    except Exception as e:
        warnings.warn(f"Homography estimation failed: {e}")
        return result


def compute_crop_coverage(
    H: np.ndarray,
    img_a_shape: Tuple[int, int],
    img_b_shape: Tuple[int, int]
) -> float:
    """
    Compute coverage of image A when projected into image B's coordinate space
    
    Args:
        H: Homography matrix
        img_a_shape: Shape of image A (height, width)
        img_b_shape: Shape of image B (height, width)
    
    Returns:
        Coverage ratio (0.0 to 1.0)
    """
    if H is None:
        return 0.0
    
    h_a, w_a = img_a_shape[:2]
    h_b, w_b = img_b_shape[:2]
    
    corners_a = np.float32([[0, 0], [w_a, 0], [w_a, h_a], [0, h_a]])
    
    try:
        corners_a_h = np.hstack([corners_a, np.ones((4, 1))])
        projected = (H @ corners_a_h.T).T
        projected = projected[:, :2] / projected[:, 2:3]
        
        inside_count = 0
        for pt in projected:
            if 0 <= pt[0] <= w_b and 0 <= pt[1] <= h_b:
                inside_count += 1
        
        quad_area = cv2.contourArea(projected.astype(np.float32))
        b_area = w_b * h_b
        
        coverage = min(inside_count / 4.0, quad_area / b_area)
        return float(coverage)
        
    except Exception:
        return 0.0


def load_or_compute_orb_features(
    panel_paths: List[Path],
    output_dir: Path,
    cache_version: str,
    enable_cache: bool = True,
    max_keypoints: int = 1000,
    retry_scales: List[float] = None,
    enable_text_masking: bool = False,
    detect_text_regions_heuristic: Optional[callable] = None
) -> Dict[str, Dict]:
    """
    Load cached ORB features or compute new ones
    
    Args:
        panel_paths: List of panel paths
        output_dir: Output directory
        cache_version: Cache version string
        enable_cache: Enable caching
        max_keypoints: Maximum keypoints per image
        retry_scales: Scales to try
        enable_text_masking: Enable text masking
        detect_text_regions_heuristic: Text detection function
    
    Returns:
        Dictionary mapping path -> {keypoints, descriptors}
    """
    if retry_scales is None:
        retry_scales = [1.0, 2.0, 0.5]
    
    cache_path = output_dir / "cache" / f"orb_features_{cache_version}.pkl"
    meta_path = get_cache_meta_path("orb_features", output_dir, cache_version)
    
    if enable_cache and cache_path.exists() and meta_path.exists():
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            if meta.get("file_hash") == compute_file_hash(panel_paths):
                with open(cache_path, 'rb') as f:
                    cached = pickle.load(f)
                    # Convert back from tuples to KeyPoint objects
                    orb_data = {}
                    for path, data in cached.items():
                        kp_list = [cv2.KeyPoint(x=kp[0], y=kp[1], size=kp[2], angle=kp[3], 
                                                response=kp[4], octave=kp[5], class_id=kp[6]) 
                                   for kp in data['keypoints_tuples']]
                        orb_data[path] = {'keypoints': kp_list, 'descriptors': data['descriptors']}
                    print(f"  ✓ Loaded cached ORB features")
                    return orb_data
        except Exception:
            pass
    
    print(f"  Computing ORB features...")
    orb_data = {}
    for path in tqdm(panel_paths, desc="ORB"):
        kp, desc = extract_orb_features(
            str(path), max_keypoints, retry_scales,
            enable_text_masking, detect_text_regions_heuristic
        )
        orb_data[str(path)] = {'keypoints': kp, 'descriptors': desc}
    
    if enable_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # Convert KeyPoint objects to tuples for pickling
        cacheable = {}
        for path, data in orb_data.items():
            kp_tuples = [(kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, 
                         kp.octave, kp.class_id) for kp in data['keypoints']] if data['keypoints'] else []
            cacheable[path] = {'keypoints_tuples': kp_tuples, 'descriptors': data['descriptors']}
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cacheable, f)
        with open(meta_path, 'w') as f:
            json.dump({
                "file_hash": compute_file_hash(panel_paths),
                "num_panels": len(panel_paths),
                "timestamp": datetime.now().isoformat()
            }, f)
    
    return orb_data


def get_orb_features_for_subset(
    request_paths: List[Path],
    output_dir: Path,
    cache_version: str
) -> Dict[str, Dict]:
    """
    Read global ORB cache without recomputing, filter to requested paths
    
    Args:
        request_paths: List of paths to retrieve
        output_dir: Output directory
        cache_version: Cache version
    
    Returns:
        Dictionary mapping path -> {keypoints, descriptors}
    """
    cache_path = output_dir / "cache" / f"orb_features_{cache_version}.pkl"
    
    if not cache_path.exists():
        return {}
    
    try:
        with open(cache_path, 'rb') as f:
            full_cache = pickle.load(f)
        
        subset = {}
        for p in request_paths:
            key = str(p)
            if key in full_cache:
                # Convert tuples back to KeyPoints
                data = full_cache[key]
                kps = [cv2.KeyPoint(x=t[0], y=t[1], size=t[2], angle=t[3],
                                    response=t[4], octave=t[5], class_id=t[6])
                       for t in data['keypoints_tuples']]
                subset[key] = {'keypoints': kps, 'descriptors': data['descriptors']}
        
        return subset
    except Exception:
        return {}


def orb_find_partial_duplicates(
    panel_paths: List[Path],
    clip_df: pd.DataFrame,
    phash_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    output_dir: Path,
    cache_version: str,
    orb_trigger_clip_threshold: float = 0.95,
    orb_trigger_phash_threshold: int = 5,
    orb_ratio_threshold: float = 0.75,
    min_inliers: int = 30,
    min_inlier_ratio: float = 0.30,
    max_reproj_error: float = 4.0,
    min_coverage: float = 0.50,
    suppress_same_page: bool = False,
    suppress_adjacent_page: bool = False,
    adjacent_page_max_gap: int = 1,
    enable_cache: bool = True,
    max_keypoints: int = 1000,
    retry_scales: List[float] = None
) -> pd.DataFrame:
    """
    Find partial duplicates using ORB-RANSAC geometric verification
    
    Args:
        panel_paths: List of all panel paths
        clip_df: CLIP duplicate pairs DataFrame
        phash_df: pHash duplicate pairs DataFrame
        meta_df: Panel metadata DataFrame
        output_dir: Output directory
        cache_version: Cache version
        orb_trigger_clip_threshold: CLIP threshold to trigger ORB
        orb_trigger_phash_threshold: pHash threshold to trigger ORB
        orb_ratio_threshold: ORB match ratio threshold
        min_inliers: Minimum inliers for homography
        min_inlier_ratio: Minimum inlier ratio
        max_reproj_error: Maximum reprojection error
        min_coverage: Minimum crop coverage
        suppress_same_page: Suppress same-page pairs
        suppress_adjacent_page: Suppress adjacent-page pairs
        adjacent_page_max_gap: Maximum page gap
        enable_cache: Enable caching
        max_keypoints: Maximum keypoints
        retry_scales: Scales to try
    
    Returns:
        DataFrame with ORB-RANSAC results
    """
    if retry_scales is None:
        retry_scales = [1.0, 2.0, 0.5]
    
    print(f"\n[Stage 4] ORB-RANSAC partial duplicate detection...")
    
    # Build trigger set
    trigger_pairs = set()
    
    # Trigger 1: High CLIP similarity
    for _, row in clip_df.iterrows():
        if pd.to_numeric(row.get('Cosine_Similarity', 0), errors='coerce') >= orb_trigger_clip_threshold:
            trigger_pairs.add(tuple(sorted([row['Path_A'], row['Path_B']])))
    
    # Trigger 2: Low pHash distance
    for _, row in phash_df.iterrows():
        if pd.to_numeric(row.get('Hamming_Distance', 999), errors='coerce') <= orb_trigger_phash_threshold:
            trigger_pairs.add(tuple(sorted([row['Path_A'], row['Path_B']])))
    
    if not trigger_pairs:
        print("  ✓ No pairs triggered ORB-RANSAC")
        return pd.DataFrame()
    
    print(f"  Triggered on {len(trigger_pairs)} pairs")
    
    # Collect only the paths we need (subset optimization)
    triggered_paths = set()
    for path_a, path_b in trigger_pairs:
        triggered_paths.add(path_a)
        triggered_paths.add(path_b)
    
    triggered_paths_list = [Path(p) for p in triggered_paths]
    print(f"  Loading ORB for {len(triggered_paths_list)} unique panels (subset cache)...")
    
    # Try subset read from cache first
    orb_data = get_orb_features_for_subset(triggered_paths_list, output_dir, cache_version)
    
    # If cache is empty (first run), compute only triggered subset
    if not orb_data or len(orb_data) < len(triggered_paths_list):
        print(f"  Computing ORB for triggered subset only...")
        orb_data = load_or_compute_orb_features(
            triggered_paths_list, output_dir, cache_version, enable_cache,
            max_keypoints, retry_scales
        )
    
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
    
    # Run RANSAC
    rows = []
    for path_a, path_b in tqdm(trigger_pairs, desc="ORB-RANSAC"):
        # Skip same-page and adjacent pages
        if suppress_same_page and same_page(path_a, path_b):
            continue
        if suppress_adjacent_page and adjacent_pages(path_a, path_b):
            continue
        
        data_a = orb_data.get(path_a)
        data_b = orb_data.get(path_b)
        
        if not data_a or not data_b:
            continue
        
        if data_a['descriptors'] is None or data_b['descriptors'] is None:
            continue
        
        # Match
        matches = match_orb_features(data_a['descriptors'], data_b['descriptors'], orb_ratio_threshold)
        
        if len(matches) < min_inliers:
            continue
        
        # RANSAC
        result = estimate_homography_ransac(
            data_a['keypoints'], data_b['keypoints'], matches,
            min_inliers=min_inliers, max_reproj_error=max_reproj_error
        )
        
        if result['H'] is None or result['is_degenerate']:
            continue
        
        # Compute coverage
        img_a = cv2.imread(path_a)
        img_b = cv2.imread(path_b)
        if img_a is None or img_b is None:
            continue
            
        coverage = compute_crop_coverage(result['H'], img_a.shape, img_b.shape)
        
        is_partial_dupe = (
            result['inliers'] >= min_inliers and
            result['inlier_ratio'] >= min_inlier_ratio and
            result['reproj_error'] <= max_reproj_error and
            coverage >= min_coverage
        )
        
        rows.append({
            'Image_A': Path(path_a).name,
            'Image_B': Path(path_b).name,
            'Path_A': path_a,
            'Path_B': path_b,
            'ORB_Inliers': result['inliers'],
            'Inlier_Ratio': result['inlier_ratio'],
            'Reproj_Error': result['reproj_error'],
            'Crop_Coverage': coverage,
            'Homography_Type': result['homography_type'],
            'Is_Partial_Dupe': is_partial_dupe
        })
    
    df = pd.DataFrame(rows)
    partial_count = len(df[df['Is_Partial_Dupe']]) if not df.empty else 0
    print(f"  ✓ Found {partial_count} partial duplicates")
    return df

