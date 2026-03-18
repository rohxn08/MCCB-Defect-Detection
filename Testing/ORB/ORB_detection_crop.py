import cv2
import numpy as np
import os
import re
import time
from datetime import datetime
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import matplotlib
matplotlib.use('Qt5Agg')


# Local import from the same folder
from layout_detector import detect_layout_regions

# --- 1. Image Alignment Engine ---

def align_images(im1, im2, max_features=5000, keep_percent=0.2):
    """
    Aligns im2 to match the perspective and position of im1 using ORB feature matching.
    """
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(max_features)
    keypoints1, descriptors1 = orb.detectAndCompute(im1_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2_gray, None)

    if descriptors1 is None or descriptors2 is None:
        return im2

    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors2, descriptors1, None)

    matches = sorted(matches, key=lambda x: x.distance)
    keep = int(len(matches) * keep_percent)
    matches = matches[:keep]

    if len(matches) < 4:
        return im2

    pts1 = np.zeros((len(matches), 2), dtype=np.float32)
    pts2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        pts2[i, :] = keypoints2[match.queryIdx].pt
        pts1[i, :] = keypoints1[match.trainIdx].pt

    h, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC)
    
    if h is None:
        return im2

    height, width, channels = im1.shape
    im2_aligned = cv2.warpPerspective(im2, h, (width, height))

    return im2_aligned

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

    # Get dimensions for dynamic cleaning further down
    h_blur, w_blur = reference_gray.shape[:2]

    # Blur to ignore glare and minor textures
    # --- USING FIXED KERNEL AS REQUESTED ---
    blur_kernel = (69, 69)
    print(f"Using fixed blur kernel: {blur_kernel}")
    
    reference_blurred = cv2.GaussianBlur(reference_gray, blur_kernel, 0)
    input_blurred = cv2.GaussianBlur(input_gray, blur_kernel, 0)

    # 1. Calculate Global SSIM
    (score, diff) = ssim(reference_blurred, input_blurred, full=True, win_size=25)
    
    # Scale diff from [-1, 1] to [0, 255] safely
    diff_img = (diff * 127.5 + 127.5).astype("uint8")

    # --- IMPROVEMENT: EDGE MASKING ---
    h, w = diff_img.shape
    border = int(min(h, w) * 0.03)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[border:h-border, border:w-border] = 255
    
    # Recalculate score only within the valid mask region
    score = np.mean((diff[mask == 255] + 1) / 2)
    print(f"Realistic Global SSIM Score: {score:.4f}")

    # Use a tighter threshold to catch color mismatches (White vs Black is a huge change)
    diff_masked = diff_img.copy()
    diff_masked[mask == 0] = 255 
    
    # We use a standard threshold to catch any drop in similarity
    _, thresh = cv2.threshold(diff_masked, threshold_value, 255, cv2.THRESH_BINARY_INV)

    # --- IMPROVEMENT: DYNAMIC CLEANING KERNEL ---
    # We scale the 'cleaning' noise filter based on resolution too
    # Reference width is used to keep sensitivity consistent
    clean_k_size = int(w_blur * 0.003) 
    if clean_k_size < 3: clean_k_size = 3
    if clean_k_size % 2 == 0: clean_k_size += 1
    
    kernel = np.ones((clean_k_size, clean_k_size), np.uint8)
    thresh_cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)

    contours, _ = cv2.findContours(thresh_cleaned.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate a dynamic min area based on total pixel count (approx 0.01% of area)
    dynamic_min_area = (h_blur * w_blur) * 0.0001
    # Use whichever is larger: the user's requirement or our noise floor
    actual_min_area = max(min_contour_area, dynamic_min_area)

    output_image = input_image.copy()
    defects_found = 0
    for c in contours:
        if cv2.contourArea(c) > actual_min_area:
            x, y, w_box, h_box = cv2.boundingRect(c)
            cv2.rectangle(output_image, (x, y), (x + w_box, y + h_box), (0, 0, 255), 4)
            cv2.putText(output_image, "Defect", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
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
    
    # --- VISUAL REFERENCE CROPPING LOGIC ---
    print("Finding master template region using ORB features...")
    orb_finder = cv2.ORB_create(10000)
    kp_m, des_m = orb_finder.detectAndCompute(cropped_master, None)
    kp_t, des_t = orb_finder.detectAndCompute(img_rotated, None)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_m, des_t)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) < 10:
        print("Error: Not enough features to find the master region. Falling back to full image.")
        input_for_analysis = input_image_path
    else:
        src_pts = np.float32([kp_m[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_t[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if M is not None:
            h_m, w_m = cropped_master.shape[:2]
            corners = np.float32([[0, 0], [0, h_m-1], [w_m-1, h_m-1], [w_m-1, 0]]).reshape(-1, 1, 2)
            transformed_corners = cv2.perspectiveTransform(corners, M)
            
            x_min, y_min = np.int32(transformed_corners.min(axis=0).ravel())
            x_max, y_max = np.int32(transformed_corners.max(axis=0).ravel())
            
            h_t, w_t = img_rotated.shape[:2]
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(w_t, x_max), min(h_t, y_max)
            
            cropped_test = img_rotated[y_min:y_max, x_min:x_max]
            cropped_test_path = os.path.join(output_dir, "visual_cropped_" + os.path.basename(input_image_path))
            cv2.imwrite(cropped_test_path, cropped_test)
            print(f"Successfully found and cropped test region. Saved to {cropped_test_path}")
            input_for_analysis = cropped_test_path
        else:
            print("Warning: Could not find master region. Using uncropped image.")
            input_for_analysis = input_image_path

    print(f"\n--- Step 2: Comparing against Master: {os.path.basename(reference_image_path)} ---")
    compare_images(reference_image_path, input_for_analysis, min_contour_area=1000, threshold_value=50)

if __name__ == "__main__":
    # Update these paths as needed for your tests
    
    master_reference = r"D:\MCCB-Defect-Detection\cropped_master_imaeges\cropped_masterP14P_mccb.png"
    latest_test_image =r"Testing_images\image.png" 
    
    if os.path.exists(latest_test_image):
        process_test_image(latest_test_image, master_reference)
    else:
        print(f"Test image not found at {latest_test_image}")
