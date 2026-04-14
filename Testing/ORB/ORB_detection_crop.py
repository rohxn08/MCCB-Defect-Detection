import sys
import os

# Add the project root to sys.path to allow imports from the 'Testing' package
# The script is located in D:\MCCB-Defect-Detection\Testing\ORB\
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import cv2
import numpy as np
import re
import time
from datetime import datetime
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import matplotlib
matplotlib.use('Qt5Agg')


# Local import from the Testing folder
from Testing.layout_detector import detect_layout_regions

# --- 1. Image Alignment Engine ---

from collections import defaultdict


def spatially_balanced_matches(matches, kp_query, img_shape, grid=4, per_cell=12):
    """Distribute matches evenly across spatial grid to prevent drift."""
    h, w = img_shape[:2]
    cell_h, cell_w = h / grid, w / grid
    cells = defaultdict(list)
    for m in matches:
        pt   = kp_query[m.queryIdx].pt
        cell = (min(int(pt[1] / cell_h), grid - 1), min(int(pt[0] / cell_w), grid - 1))
        cells[cell].append(m)
    selected = []
    for cell_matches in cells.values():
        selected.extend(sorted(cell_matches, key=lambda x: x.distance)[:per_cell])
    return selected


def verify_alignment(aligned, master, threshold=0.5):
    """NCC-based alignment quality check."""
    def norm(img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        g -= g.mean()
        g /= (g.std() + 1e-6)
        return g
    h, w  = master.shape[:2]
    score = float(np.sum(norm(aligned) * norm(master)) / (h * w))
    status = "Good" if score > threshold else "Poor"
    print(f"  Alignment NCC: {score:.3f}  ({status})")
    return score > threshold, score


def align_images(master, img_raw):
    """
    Aligns img_raw to match the perspective and size of master using
    spatially-balanced ORB feature matching (mirrors master.py logic).
    """
    img      = img_raw  # caller is responsible for any pre-rotation
    h_m, w_m = master.shape[:2]

    orb = cv2.ORB_create(10000)
    bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    kp_m, des_m = orb.detectAndCompute(master, None)
    kp_t, des_t = orb.detectAndCompute(img, None)

    if des_m is None or des_t is None or len(kp_t) < 10:
        print("  Not enough keypoints — returning resized")
        return cv2.resize(img, (w_m, h_m))

    all_matches = sorted(bf.match(des_m, des_t), key=lambda x: x.distance)
    matches     = spatially_balanced_matches(all_matches, kp_m, master.shape, grid=4, per_cell=12)

    if len(matches) < 10:
        print("  Not enough balanced matches — returning resized")
        return cv2.resize(img, (w_m, h_m))

    src_pts = np.float32([kp_m[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_t[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if M is None:
        print("  Homography failed — returning resized")
        return cv2.resize(img, (w_m, h_m))

    inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    print(f"    Inliers: {inliers}/{len(matches)} ({inliers/len(matches):.1%})")

    return cv2.warpPerspective(img, M, (w_m, h_m), flags=cv2.WARP_INVERSE_MAP)

# --- 2. Defect Detection Engine ---

def compare_images(reference_image_path, input_image_path, min_contour_area=100, threshold_value=200):
    """
    Compares images using a non-destructive edge mask to avoid warp artifacts.
    """
    reference_image = cv2.imread(reference_image_path)
    input_image = cv2.imread(input_image_path)

    if reference_image is None or input_image is None:
        print("Error: Could not load images.")
        return

    print("Aligning input image to reference image...")
    try:
        input_image = align_images(reference_image, input_image)
        is_good, ncc_score = verify_alignment(input_image, reference_image)
        if not is_good:
            print(f"  Warning: Poor alignment (NCC={ncc_score:.3f}) — result may be unreliable")
    except Exception as e:
        print(f"Alignment failed: {e}")

    reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    input_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # --- IMPROVEMENT: INTENSITY NORMALIZATION ---
    print("Normalizing local contrast (preserving part identity)...")
    # CLAHE fixes lighting/shadows without changing the fundamental color (White vs black)
    # We use a gentle clipLimit=2.0 to avoid over-amplifying sensor noise
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    reference_gray = clahe.apply(reference_gray)
    input_gray = clahe.apply(input_gray)

    # Get image dimensions — used to scale all parameters to resolution
    h_blur, w_blur = reference_gray.shape[:2]
    shorter_dim = min(h_blur, w_blur)

    # --- RESOLUTION-AWARE GAUSSIAN BLUR ---
    # The label/text panel of the MCCB has embossed serial numbers (~2-5px wide at 4K).
    # A 31px blur (1.5% of shorter dim) suppresses those text differences while keeping
    # structural features like screw holes (~200-300px diameter at 4K) fully visible.
    # 1.5% rule:  4K (2068px short) → 31px  |  HD → 15px  |  small → 11px (min)
    blur_k = max(11, int(shorter_dim * 0.015))
    if blur_k % 2 == 0: blur_k += 1   # must be odd
    print(f"Using resolution-aware blur kernel: ({blur_k}, {blur_k})  [image: {w_blur}x{h_blur}]")
    reference_blurred = cv2.GaussianBlur(reference_gray, (blur_k, blur_k), 0)
    input_blurred     = cv2.GaussianBlur(input_gray,     (blur_k, blur_k), 0)

    # 1. Calculate SSIM — win_size=21 matches the 31px blur scale
    (score, diff) = ssim(reference_blurred, input_blurred, full=True, win_size=21)

    # --- EDGE MASKING (suppress warp artefacts at borders) ---
    h, w = diff.shape
    border = int(shorter_dim * 0.03)
    edge_mask = np.zeros((h, w), dtype=np.uint8)
    edge_mask[border:h-border, border:w-border] = 255

    # Global SSIM score for reporting (masked valid region only)
    score = np.mean((diff[edge_mask == 255] + 1) / 2)
    print(f"Realistic Global SSIM Score: {score:.4f}")

    # --- LOCAL ANOMALY SCORING ---
    # Absolute SSIM thresholding fails when master and test have different
    # overall brightness/exposure: the ENTIRE image looks 'different' and
    # everything gets flagged.
    #
    # This approach instead asks: is THIS pixel significantly WORSE than its
    # LOCAL NEIGHBOURHOOD?  Uniform lighting shifts produce anomaly ≈ 0
    # everywhere (flagged=nothing). A missing screw is locally much worse
    # than the surrounding plastic surface → anomaly is HIGH → flagged.
    diff_f = diff.astype(np.float32)

    # Fill borders with the in-region mean so the Gaussian blur isn’t
    # biased towards 0 at the edges.
    border_fill = float(np.mean(diff_f[edge_mask == 255]))
    diff_filled = diff_f.copy()
    diff_filled[edge_mask == 0] = border_fill

    # Neighbourhood window: ~8% of shorter dim
    # 4K (shorter=2068) → 165px  |  HD → 82px  |  minimum 51px
    local_win = max(51, int(shorter_dim * 0.08))
    if local_win % 2 == 0: local_win += 1
    print(f"Local anomaly window: {local_win}px")
    local_mean_diff = cv2.GaussianBlur(diff_filled, (local_win, local_win), 0)

    # anomaly = how much lower (worse) is local SSIM vs neighbourhood?
    # 0 = no anomaly, 1 = maximally anomalous
    anomaly   = np.clip(local_mean_diff - diff_f, 0.0, 1.0)
    anomaly_img = (anomaly * 255).astype("uint8")
    anomaly_img[edge_mask == 0] = 0   # zero borders

    # threshold_value is now an anomaly score (0-255).
    # 80  → flag if local SSIM is >0.31 below neighbourhood mean.
    # Missing screw: anomaly typically 0.5-0.9 → anomaly_img 127-229 → well above 80.
    # Lighting shift: anomaly ≈ 0 everywhere → nothing flagged.
    _, thresh = cv2.threshold(anomaly_img, threshold_value, 255, cv2.THRESH_BINARY)

    # --- RESOLUTION-AWARE MORPHOLOGICAL CLEANING ---
    # ~0.3% of shorter dimension: 4K→~6px, HD→~3px. Capped at 13 to never over-erase.
    # CRITICAL: only 2 iterations — 5 iterations was erasing real blobs at 4K.
    clean_k = max(3, min(int(shorter_dim * 0.003), 13))
    if clean_k % 2 == 0: clean_k += 1
    print(f"Morphological kernel: {clean_k}px  (2 iterations)")
    kernel = np.ones((clean_k, clean_k), np.uint8)
    thresh_cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(thresh_cleaned.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Dynamic min area: 0.1% of image area scales correctly with resolution.
    # 4K (3840x2160) → ~8,294 px²  ≈ a ~90px diameter circle — about right for a screw.
    # User-supplied min_contour_area is still honoured if larger.
    dynamic_min_area = (h_blur * w_blur) * 0.001
    actual_min_area  = max(min_contour_area, dynamic_min_area)
    print(f"Min contour area: {actual_min_area:.0f} px²")

    # Max area: 1.5% of image captures screws (~250px diam) and small parts,
    # but rejects the large handle/switch shine blobs (100K-160K px² at 4K).
    max_defect_area = (h_blur * w_blur) * 0.015
    # Aspect ratio cap: screws are compact (1.0-1.5), elongated strips > 2.0
    max_aspect_ratio = 2.0
    # Fill ratio: solid defects fill >=30% of their bounding box.
    # Scattered noise merged by morphology fills only 5-15% of its bbox.
    min_fill_ratio   = 0.30
    print(f"Area: [{actual_min_area:.0f} — {max_defect_area:.0f}] px²  aspect<{max_aspect_ratio}  fill>{min_fill_ratio}")

    output_image = input_image.copy()
    defects_found = 0
    for c in contours:
        area = cv2.contourArea(c)
        x, y, w_box, h_box = cv2.boundingRect(c)
        aspect_ratio = max(w_box, h_box) / (min(w_box, h_box) + 1e-6)
        fill_ratio   = area / (w_box * h_box + 1e-6)

        if area < actual_min_area:
            continue
        if area > max_defect_area:
            print(f"  [skip] area={area:.0f}  too large (handle/reflective surface)")
            continue
        if aspect_ratio >= max_aspect_ratio:
            print(f"  [skip] aspect={aspect_ratio:.1f}  too elongated")
            continue
        if fill_ratio < min_fill_ratio:
            print(f"  [skip] fill={fill_ratio:.2f}  sparse noise cluster (bbox {w_box}×{h_box})")
            continue

        print(f"  [DEFECT] x={x}  y={y}  w={w_box}  h={h_box}  area={area:.0f}  aspect={aspect_ratio:.2f}  fill={fill_ratio:.2f}")
        cv2.rectangle(output_image, (x, y), (x + w_box, y + h_box), (0, 0, 255), 4)
        cv2.putText(output_image, "Defect", (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        defects_found += 1

    status = f"OK(similarity {score:.4f})" if score >= 0.95 else f"Defects_Found(similarity {score:.4f})"
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_plot_path = f"Result_{status}_{timestamp}.png"
    
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 4, 1); plt.imshow(cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)); plt.title("Master Reference"); plt.axis('off')
    plt.subplot(1, 4, 2); plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)); plt.title("Aligned Input Image"); plt.axis('off')
    plt.subplot(1, 4, 3); plt.imshow(thresh_cleaned, cmap='gray'); plt.title("Difference Mask"); plt.axis('off')
    plt.subplot(1, 4, 4); plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)); plt.title(f"Result: {status}"); plt.axis('off')
    plt.tight_layout()
    plt.savefig(result_plot_path)
    print(f"Saved optimized plot to {result_plot_path}")
    plt.show()

# --- 3. Main Orchestrator ---

def process_test_image(input_image_path, reference_image_path):
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n--- Step 1: Smart Cropping Test Image: {os.path.basename(input_image_path)} ---")
    img_raw = cv2.imread(input_image_path)
    cropped_master = cv2.imread(reference_image_path)
    
    if img_raw is None or cropped_master is None:
        print("Error: Could not load images.")
        return

    # Flip to CCW rotation
    img_rotated = cv2.rotate(img_raw, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    # --- ALIGNMENT CROPPING LOGIC (mirrors master.py align_to_master) ---
    print("Aligning test image to master using spatially-balanced ORB features...")
    aligned_test = align_images(cropped_master, img_rotated)
    is_good, ncc_score = verify_alignment(aligned_test, cropped_master)
    if not is_good:
        print(f"  Warning: Poor alignment (NCC={ncc_score:.3f}) — result may be unreliable")

    cropped_test_path = os.path.join(output_dir, "visual_cropped_" + os.path.basename(input_image_path))
    cv2.imwrite(cropped_test_path, aligned_test)
    print(f"Aligned test image saved to {cropped_test_path}")
    input_for_analysis = cropped_test_path

    print(f"\n--- Step 2: Comparing against Master: {os.path.basename(reference_image_path)} ---")
    compare_images(reference_image_path, input_for_analysis, min_contour_area=0, threshold_value=80)

if __name__ == "__main__":
    # Update these paths as needed for your tests
    
    master_reference = r"cropped_master_imaeges\cropped_masterXT13P_mccb.png"
    latest_test_image =r"Testing_images\CG36355343067392.png" 
    
    if os.path.exists(latest_test_image):
        process_test_image(latest_test_image, master_reference)
    else:
        print(f"Test image not found at {latest_test_image}")
