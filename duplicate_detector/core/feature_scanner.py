
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
from duplicate_detector.utils.logger import StageLogger

def whole_page_feature_scan(
    pages: List[Path], 
    out_dir: Path,
    min_inliers: int = 20,
    max_pages_to_scan: int = 50
) -> pd.DataFrame:
    """
    Robust feature matching (ORB+RANSAC) on whole pages to detect small or complex duplicates.
    """
    print("\n" + "="*70)
    print("🔎 WHOLE-PAGE FEATURE SCAN (Hybrid Mode)")
    print("="*70)
    
    # Pre-compute ORB features for all pages
    orb = cv2.ORB_create(nfeatures=10000, scoreType=cv2.ORB_FAST_SCORE)
    page_features = {}
    
    for p_path in tqdm(pages, desc="Extracting Page Features"):
        try:
            img = cv2.imread(str(p_path), cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            kp, des = orb.detectAndCompute(img, None)
            if des is not None and len(kp) > 100:
                page_features[p_path.name] = (kp, des, img.shape)
        except Exception as e:
            print(f"Error processing {p_path}: {e}")

    page_names = sorted(list(page_features.keys()))
    matches = []
    
    # Brute-force match all page pairs (O(N^2))
    # Optimized by only checking pages with enough features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    pair_count = 0
    for i, p1_name in enumerate(tqdm(page_names, desc="Scanning Page Pairs")):
        for j, p2_name in enumerate(page_names):
            if i >= j: continue # Skip self and duplicate checks
            
            kp1, des1, shape1 = page_features[p1_name]
            kp2, des2, shape2 = page_features[p2_name]
            
            # Initial rough match
            raw_matches = bf.match(des1, des2)
            raw_matches = sorted(raw_matches, key=lambda x: x.distance)
            
            # Filter: Keep top 15%
            keep_count = int(len(raw_matches) * 0.15)
            good_matches = raw_matches[:keep_count]
            
            if len(good_matches) < min_inliers:
                continue
                
            # Geometric Verification (RANSAC)
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if mask is not None:
                inliers = int(sum(mask.ravel()))
                if inliers >= min_inliers:
                    matches.append({
                        'Path_A': str(out_dir / "pages" / p1_name),
                        'Path_B': str(out_dir / "pages" / p2_name),
                        'Image_A': p1_name,
                        'Image_B': p2_name,
                        'Cosine_Similarity': 0.95, # Synthetic score for sorting
                        'SSIM': 0.95, # Synthetic score
                        'ORB_Inliers': inliers,
                        'Tier': 'A',
                        'Tier_Path': 'WholePage-FeatureMatch',
                        'Source': 'FeatureScan'
                    })
                    
    print(f"  ✓ Found {len(matches)} whole-page duplicate regions")
    return pd.DataFrame(matches)

