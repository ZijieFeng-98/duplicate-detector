"""
Panel Detection Module

Extracts panels from PDF pages using computer vision techniques.
Handles PDF to pages conversion and panel detection with NMS.
"""

from pathlib import Path
from typing import List, Tuple, Dict, Set
import sys
import fitz  # PyMuPDF
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw
from tqdm import tqdm


def ensure_dir(p: Path):
    """Ensure directory exists"""
    p.mkdir(parents=True, exist_ok=True)


def page_stem(i: int) -> str:
    """Generate page stem name"""
    return f"page_{i+1}"


def compute_iou(box_a: Dict, box_b: Dict) -> float:
    """Compute Intersection over Union (IoU) for two bounding boxes"""
    x1_a, y1_a = box_a['x'], box_a['y']
    x2_a, y2_a = x1_a + box_a['w'], y1_a + box_a['h']
    
    x1_b, y1_b = box_b['x'], box_b['y']
    x2_b, y2_b = x1_b + box_b['w'], y1_b + box_b['h']
    
    # Intersection
    x1_i = max(x1_a, x1_b)
    y1_i = max(y1_a, y1_b)
    x2_i = min(x2_a, x2_b)
    y2_i = min(y2_a, y2_b)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area_a = box_a['w'] * box_a['h']
    area_b = box_b['w'] * box_b['h']
    union = area_a + area_b - intersection
    
    return intersection / union if union > 0 else 0.0


def pdf_to_pages(
    pdf_path: Path,
    out_dir: Path,
    dpi: int,
    caption_pages: Set[int] = None,
    debug_mode: bool = False
) -> List[Path]:
    """
    Convert PDF to PNG pages using PyMuPDF
    
    Args:
        pdf_path: Path to input PDF file
        out_dir: Output directory for pages
        dpi: DPI for rendering
        caption_pages: Set of page numbers (1-indexed) to exclude
        debug_mode: Enable debug output
    
    Returns:
        List of paths to extracted page images
    """
    if caption_pages is None:
        caption_pages = set()
    
    pages_dir = out_dir / "pages"
    ensure_dir(pages_dir)
    
    if not pdf_path.exists():
        print(f"❌ PDF not found: {pdf_path}")
        sys.exit(1)
    
    print(f"\n[1/7] Converting PDF to PNGs at {dpi} DPI...")
    
    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        print(f"❌ PDF conversion failed: {e}")
        print("\n💡 Make sure the PDF is valid and not corrupted")
        sys.exit(1)
    
    saved = []
    excluded = 0
    
    # Calculate zoom factor from DPI (72 is default PDF DPI)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    
    for i in tqdm(range(len(doc)), desc="Converting pages"):
        if (i + 1) in caption_pages:
            excluded += 1
            continue
        
        page = doc[i]
        pix = page.get_pixmap(matrix=mat)
        
        fp = pages_dir / f"{page_stem(i)}.png"
        pix.save(str(fp))
        saved.append(fp)
    
    doc.close()
    print(f"  ✓ Saved {len(saved)} pages (excluded {excluded} caption pages)")
    return saved


def detect_panels_cv(
    page_png: Path,
    out_dir: Path,
    min_panel_area: int = 80000,
    max_panel_area: int = 10000000,
    min_aspect_ratio: float = 0.2,
    max_aspect_ratio: float = 5.0,
    edge_threshold1: int = 40,
    edge_threshold2: int = 120,
    contour_approx_epsilon: float = 0.02,
    debug_mode: bool = False
) -> List[Tuple[Path, Dict]]:
    """
    Detect panels in a page image using computer vision
    
    Args:
        page_png: Path to page image
        out_dir: Output directory for panels
        min_panel_area: Minimum panel area in pixels
        max_panel_area: Maximum panel area in pixels
        min_aspect_ratio: Minimum aspect ratio (width/height)
        max_aspect_ratio: Maximum aspect ratio (width/height)
        edge_threshold1: Canny edge detection threshold 1
        edge_threshold2: Canny edge detection threshold 2
        contour_approx_epsilon: Contour approximation epsilon
        debug_mode: Enable debug visualization
    
    Returns:
        List of tuples (panel_path, metadata_dict)
    """
    ensure_dir(out_dir)
    
    img_pil = Image.open(page_png).convert("RGB")
    img_np = np.array(img_pil)
    
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, edge_threshold1, edge_threshold2)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_panels = []
    for contour in contours:
        epsilon = contour_approx_epsilon * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        x, y, w, h = cv2.boundingRect(approx)
        area = w * h
        aspect_ratio = w / h if h > 0 else 0
        
        if (min_panel_area <= area <= max_panel_area and
            min_aspect_ratio <= aspect_ratio <= max_aspect_ratio):
            valid_panels.append({
                'x': x, 'y': y, 'w': w, 'h': h,
                'area': area, 'aspect_ratio': aspect_ratio
            })
    
    # Non-Maximum Suppression (NMS)
    valid_panels.sort(key=lambda p: p['area'], reverse=True)
    nms_panels = []
    for panel in valid_panels:
        if all(compute_iou(panel, kept) < 0.5 for kept in nms_panels):
            nms_panels.append(panel)
    
    nms_panels.sort(key=lambda p: (p['y'], p['x']))
    
    # Save panels
    saved = []
    base = page_png.stem
    
    for idx, panel in enumerate(nms_panels):
        x, y, w, h = panel['x'], panel['y'], panel['w'], panel['h']
        crop = img_pil.crop((x, y, x+w, y+h))
        crop = ImageOps.autocontrast(crop)
        
        out = out_dir / f"{base}_panel{idx+1:02d}.png"
        crop.save(out)
        
        metadata = {
            "page": base,
            "panel_num": idx + 1,
            "x": x, "y": y,
            "width": w, "height": h,
            "area": panel['area']
        }
        saved.append((out, metadata))
    
    # Debug visualization
    if debug_mode and len(nms_panels) > 0:
        debug_img = img_pil.copy()
        draw = ImageDraw.Draw(debug_img)
        for panel in nms_panels:
            x, y, w, h = panel['x'], panel['y'], panel['w'], panel['h']
            draw.rectangle([x, y, x+w, y+h], outline='red', width=5)
        debug_path = out_dir / f"{base}_debug.png"
        debug_img.save(debug_path)
    
    return saved


def pages_to_panels_auto(
    pages: List[Path],
    out_dir: Path,
    min_panel_area: int = 80000,
    max_panel_area: int = 10000000,
    min_aspect_ratio: float = 0.2,
    max_aspect_ratio: float = 5.0,
    edge_threshold1: int = 40,
    edge_threshold2: int = 120,
    contour_approx_epsilon: float = 0.02,
    debug_mode: bool = False
) -> Tuple[List[Path], pd.DataFrame]:
    """
    Extract panels from all pages
    
    Args:
        pages: List of page image paths
        out_dir: Output directory
        min_panel_area: Minimum panel area
        max_panel_area: Maximum panel area
        min_aspect_ratio: Minimum aspect ratio
        max_aspect_ratio: Maximum aspect ratio
        edge_threshold1: Canny threshold 1
        edge_threshold2: Canny threshold 2
        contour_approx_epsilon: Contour epsilon
        debug_mode: Enable debug mode
    
    Returns:
        Tuple of (panel_paths, metadata_dataframe)
    """
    panels_dir = out_dir / "panels"
    ensure_dir(panels_dir)
    
    print(f"\n[2/7] Auto-detecting panels (MIN_AREA={min_panel_area:,}, NMS enabled)...")
    
    all_items: List[Tuple[Path, Dict]] = []
    page_stats = []
    
    for p in tqdm(pages, desc="Detecting panels"):
        stem = p.stem
        subdir = panels_dir / stem
        items = detect_panels_cv(
            p, subdir,
            min_panel_area=min_panel_area,
            max_panel_area=max_panel_area,
            min_aspect_ratio=min_aspect_ratio,
            max_aspect_ratio=max_aspect_ratio,
            edge_threshold1=edge_threshold1,
            edge_threshold2=edge_threshold2,
            contour_approx_epsilon=contour_approx_epsilon,
            debug_mode=debug_mode
        )
        all_items.extend(items)
        page_stats.append((stem, len(items)))
    
    print(f"\n  ✓ Extracted {len(all_items)} panels total")
    
    if not all_items:
        return [], pd.DataFrame()
    
    panel_paths = [it[0] for it in all_items]
    meta_df = pd.DataFrame([{
        "Panel_Path": str(path),
        "Panel_Name": path.name,
        "Page": md["page"],
        "Panel_Num": md["panel_num"],
        "X": md["x"], "Y": md["y"],
        "Width": md["width"], "Height": md["height"],
        "Area": md["area"]
    } for path, md in all_items])
    
    return panel_paths, meta_df

