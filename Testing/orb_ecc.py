# robust_ocr_pipeline.py
# Integrated MCCB Pipeline:
#   QR Scan (Serial) → Camera Capture → OCR → Defect Detection
# 
# Changes from original:
#   - Removed double CLAHE (was over-amplifying text/chars causing noisy mask)
#   - Added two-stage alignment: ORB (coarse) → ECC (fine) to fix right-side artifacts
#   - Integrated camera capture, serial QR scanner, and OCR pipeline from reference code
#   - align_images now uses crop_margin_pct to ignore edge backgrounds during ORB matching

# ── Standard Library ──────────────────────────────────────────────────────────
import os
import re
import sys
import time
import json
from datetime import datetime

# ── Third Party ───────────────────────────────────────────────────────────────
import cv2
import numpy as np
import matplotlib.pyplot as plt
#import serial
from skimage.metrics import structural_similarity as ssim
from paddleocr import PaddleOCR

# ── Hikrobot Camera SDK ───────────────────────────────────────────────────────
#from MvCameraControl_class import *

# ── Local Imports ─────────────────────────────────────────────────────────────
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "OCR_Extraction"))
from json_formatter import format_image_result
from layout_detector import detect_layout_regions


# ══════════════════════════════════════════════════════════════════════════════
# 1. CAMERA CAPTURE
# ══════════════════════════════════════════════════════════════════════════════

def capture_image(save_path="snapshot.jpg"):
    """
    Captures a single frame from the connected Hikrobot camera.
    Supports both Trigger Mode and Continuous Mode automatically.

    Args:
        save_path (str): File path where the captured image will be saved.

    Returns:
        bool: True if capture was successful, False otherwise.
    """
    cam = MvCamera()

    # Enumerate devices
    device_list = MV_CC_DEVICE_INFO_LIST()
    ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, device_list)
    if ret != 0 or device_list.nDeviceNum == 0:
        print("❌ No Hikrobot camera detected.")
        return False

    stDeviceInfo = cast(device_list.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents

    # Create & open handle
    ret = cam.MV_CC_CreateHandle(stDeviceInfo)
    if ret != 0:
        print(f"❌ CreateHandle failed: {ret}")
        return False

    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0:
        print(f"❌ OpenDevice failed: {ret}")
        cam.MV_CC_DestroyHandle()
        return False

    # Detect trigger mode
    trigger_mode = c_uint()
    ret = cam.MV_CC_GetEnumValue("TriggerMode", trigger_mode)
    is_trigger_mode = (ret == 0 and trigger_mode.value == 1)

    if is_trigger_mode:
        print("🎯 Camera is in Trigger Mode → using software trigger.")
        cam.MV_CC_SetEnumValue("TriggerSource", 7)  # Software trigger
    else:
        print("📷 Camera is in Continuous Mode.")

    # Start grabbing
    cam.MV_CC_StartGrabbing()
    time.sleep(0.3)

    if is_trigger_mode:
        cam.MV_CC_SetCommandValue("TriggerSoftware")

    # Dynamic resolution + buffer
    width_val  = MVCC_INTVALUE()
    height_val = MVCC_INTVALUE()
    cam.MV_CC_GetIntValue("Width",  width_val)
    cam.MV_CC_GetIntValue("Height", height_val)
    width, height = width_val.nCurValue, height_val.nCurValue
    buf_size = width * height * 3
    data_buf = (c_ubyte * buf_size)()

    frame_info = MV_FRAME_OUT_INFO_EX()
    ret = cam.MV_CC_GetOneFrameTimeout(byref(data_buf), buf_size, frame_info, 3000)
    print("GetOneFrameTimeout ret:", ret)

    success = False
    if ret == 0:
        print(f"✅ Frame captured: {frame_info.nWidth}x{frame_info.nHeight}")

        frame = np.frombuffer(data_buf, dtype=np.uint8,
                              count=frame_info.nWidth * frame_info.nHeight)
        frame = frame.reshape(frame_info.nHeight, frame_info.nWidth)

        # Pixel format conversion
        if frame_info.enPixelType == 17301505:  # Mono8
            image = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            image = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2BGR)

        cv2.imwrite(save_path, image)
        print(f"📸 Image saved: {save_path}")
        success = True
    else:
        print(f"⚠️ Failed to get frame. Return code: {ret}")

    # Cleanup
    cam.MV_CC_StopGrabbing()
    cam.MV_CC_CloseDevice()
    cam.MV_CC_DestroyHandle()
    return success


# ══════════════════════════════════════════════════════════════════════════════
# 2. IMAGE ALIGNMENT ENGINE (Two-Stage: ORB → ECC)
# ══════════════════════════════════════════════════════════════════════════════

def align_images_orb(im1, im2, max_features=5000, keep_percent=0.2, crop_margin_pct=0.15):
    """
    Stage 1 — Coarse alignment using ORB feature matching + Homography.

    crop_margin_pct ignores left/right edge regions during feature detection,
    preventing background from narrower 3-pole breakers polluting the matches.

    Args:
        im1: Reference image (target).
        im2: Input image to be aligned.
        max_features (int): Max ORB features to detect.
        keep_percent (float): Top % of matches to keep.
        crop_margin_pct (float): % of width to ignore on each side during matching.

    Returns:
        np.ndarray: Roughly aligned version of im2.
    """
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Central strip mask — ignore distracting edges
    h1, w1 = im1_gray.shape
    mask1 = np.zeros_like(im1_gray)
    mask1[:, int(w1 * crop_margin_pct):int(w1 * (1.0 - crop_margin_pct))] = 255

    h2, w2 = im2_gray.shape
    mask2 = np.zeros_like(im2_gray)
    mask2[:, int(w2 * crop_margin_pct):int(w2 * (1.0 - crop_margin_pct))] = 255

    orb = cv2.ORB_create(max_features)
    keypoints1, descriptors1 = orb.detectAndCompute(im1_gray, mask=mask1)
    keypoints2, descriptors2 = orb.detectAndCompute(im2_gray, mask=mask2)

    if descriptors1 is None or descriptors2 is None:
        print("⚠️ ORB: No descriptors found. Skipping coarse alignment.")
        return im2

    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors2, descriptors1, None)
    matches = sorted(matches, key=lambda x: x.distance)

    keep = int(len(matches) * keep_percent)
    matches = matches[:keep]

    if len(matches) < 4:
        print("⚠️ ORB: Not enough matches. Skipping coarse alignment.")
        return im2

    pts1 = np.zeros((len(matches), 2), dtype=np.float32)
    pts2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        pts2[i, :] = keypoints2[match.queryIdx].pt
        pts1[i, :] = keypoints1[match.trainIdx].pt

    H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC)
    if H is None:
        print("⚠️ ORB: Homography failed. Skipping coarse alignment.")
        return im2

    height, width = im1.shape[:2]
    return cv2.warpPerspective(im2, H, (width, height))


def align_images_ecc(im1, im2):
    """
    Stage 2 — Fine alignment using ECC (Enhanced Correlation Coefficient).

    Uses affine transform (translation + rotation + scale) instead of homography,
    which prevents the wild right-side warping artifacts seen with ORB alone.
    Sub-pixel accurate — fixes residual misalignment after ORB coarse pass.

    Args:
        im1: Reference image.
        im2: Roughly aligned image from ORB stage.

    Returns:
        np.ndarray: Finely aligned version of im2.
    """
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        1000,   # max iterations
        1e-7    # precision threshold
    )

    try:
        _, warp_matrix = cv2.findTransformECC(
            im1_gray,
            im2_gray,
            warp_matrix,
            cv2.MOTION_AFFINE,
            criteria
        )
        h, w = im1.shape[:2]
        aligned = cv2.warpAffine(
            im2,
            warp_matrix,
            (w, h),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
        )
        return aligned
    except cv2.error as e:
        print(f"⚠️ ECC fine alignment failed: {e}. Keeping ORB result.")
        return im2


def align_images(im1, im2):
    """
    Full two-stage alignment pipeline:
        Stage 1: ORB  → coarse homography alignment
        Stage 2: ECC  → fine affine sub-pixel correction

    This combination fixes:
        - General perspective differences (ORB)
        - Right-side warp artifacts from bad homography (ECC)

    Args:
        im1: Reference image.
        im2: Input image to align.

    Returns:
        np.ndarray: Precisely aligned version of im2.
    """
    print("  Stage 1: ORB coarse alignment...")
    roughly_aligned = align_images_orb(im1, im2)

    print("  Stage 2: ECC fine alignment...")
    precisely_aligned = align_images_ecc(im1, roughly_aligned)

    print("  ✅ Two-stage alignment complete!")
    return precisely_aligned


# ══════════════════════════════════════════════════════════════════════════════
# 3. DEFECT DETECTION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def compare_images(reference_image_path, input_image_path,
                   min_contour_area=100, threshold_value=50):
    """
    Compares input image against master reference to detect physical defects.

    Pipeline:
        1. Load images
        2. Two-stage alignment (ORB → ECC)
        3. Grayscale conversion (NO CLAHE — removing it fixed noisy mask on 3-pole)
        4. Gaussian blur to suppress text/chars and minor textures
        5. SSIM comparison with edge masking
        6. Threshold + morphological cleaning
        7. Contour detection + bounding boxes
        8. Save & display result plot

    Args:
        reference_image_path (str): Path to the master reference image.
        input_image_path (str): Path to the test/input image.
        min_contour_area (int): Minimum contour area to be flagged as a defect.
        threshold_value (int): SSIM difference threshold for binary mask.
    """
    reference_image = cv2.imread(reference_image_path)
    input_image     = cv2.imread(input_image_path)

    if reference_image is None or input_image is None:
        print("❌ Error: Could not load one or both images.")
        return

    # Step 1: Align
    print("\nAligning input image to reference...")
    try:
        input_image = align_images(reference_image, input_image)
    except Exception as e:
        print(f"⚠️ Alignment failed: {e}. Proceeding without alignment.")

    # Step 2: Grayscale
    # NOTE: CLAHE intentionally removed — it was over-amplifying text/chars,
    # causing them to survive the blur and appear as false defects in the mask.
    reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    input_gray     = cv2.cvtColor(input_image,     cv2.COLOR_BGR2GRAY)

    h_img, w_img = reference_gray.shape[:2]

    # Step 3: Gaussian blur — suppresses text, minor glare, sub-pixel noise
    blur_kernel = (69, 69)
    print(f"  Applying Gaussian blur with kernel {blur_kernel}...")
    reference_blurred = cv2.GaussianBlur(reference_gray, blur_kernel, 0)
    input_blurred     = cv2.GaussianBlur(input_gray,     blur_kernel, 0)

    # Step 4: SSIM
    (score, diff) = ssim(reference_blurred, input_blurred,
                         full=True, win_size=23)
    diff_img = (diff * 255).astype("uint8")

    # Step 5: Edge mask — ignore border pixels (warp artifacts live here)
    border = int(min(h_img, w_img) * 0.03)
    edge_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    edge_mask[border:h_img - border, border:w_img - border] = 255

    # Recalculate score only within valid (non-border) region
    score = np.mean((diff[edge_mask == 255] + 1) / 2)
    print(f"  SSIM Score (edge-masked): {score:.4f}")

    # Step 6: Threshold
    diff_masked = diff_img.copy()
    diff_masked[edge_mask == 0] = 255   # Ignore borders in threshold
    _, thresh = cv2.threshold(diff_masked, threshold_value, 255, cv2.THRESH_BINARY_INV)

    # Step 7: Morphological cleaning — removes thin noise, preserves real defects
    clean_k = max(3, int(w_img * 0.003))
    if clean_k % 2 == 0:
        clean_k += 1
    kernel        = np.ones((clean_k, clean_k), np.uint8)
    thresh_cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Step 8: Contours
    contours, _ = cv2.findContours(thresh_cleaned.copy(),
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    # Dynamic minimum area floor (~0.01% of image area)
    dynamic_min_area = (h_img * w_img) * 0.0001
    actual_min_area  = max(min_contour_area, dynamic_min_area)

    output_image  = input_image.copy()
    defects_found = 0

    for c in contours:
        if cv2.contourArea(c) > actual_min_area:
            x, y, w_box, h_box = cv2.boundingRect(c)
            cv2.rectangle(output_image,
                          (x, y), (x + w_box, y + h_box),
                          (0, 0, 255), 4)
            cv2.putText(output_image, "Defect",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        (0, 0, 255), 3)
            defects_found += 1

    print(f"  Defect regions found: {defects_found}")

    # Step 9: Status & save
    status    = (f"OK(similarity {score:.4f})"
                 if score >= 0.965
                 else f"Defects_Found(similarity {score:.4f})")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plot_path = f"Result_{status}_{timestamp}.png"

    plt.figure(figsize=(20, 8))
    plt.subplot(1, 4, 1)
    plt.imshow(cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB))
    plt.title("Master Reference"); plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
    plt.title("Aligned Input Image"); plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(thresh_cleaned, cmap='gray')
    plt.title("Difference Mask"); plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Result: {status}"); plt.axis('off')

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"  ✅ Saved result plot: {plot_path}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 4. SMART CROP + DEFECT ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def process_test_image(input_image_path, reference_image_path):
    """
    Orchestrates the defect detection pipeline:
        1. Rotate input image 90° CCW
        2. Use ORB to locate and crop the MCCB region from the full frame
        3. Run compare_images on the cropped region vs master reference

    Args:
        input_image_path (str): Path to the raw captured test image.
        reference_image_path (str): Path to the cropped master reference image.
    """
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  DEFECT DETECTION")
    print(f"{'='*60}")
    print(f"  Test image : {os.path.basename(input_image_path)}")
    print(f"  Master ref : {os.path.basename(reference_image_path)}")

    img_raw        = cv2.imread(input_image_path)
    cropped_master = cv2.imread(reference_image_path)

    if img_raw is None or cropped_master is None:
        print("❌ Error: Could not load images for defect detection.")
        return

    # Rotate to match master orientation
    img_rotated = cv2.rotate(img_raw, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # ── Find MCCB region in full frame using ORB ──────────────────────────────
    print("\nStep 1: Locating MCCB region in full frame...")
    orb_finder = cv2.ORB_create(10000)
    kp_m, des_m = orb_finder.detectAndCompute(cropped_master, None)
    kp_t, des_t = orb_finder.detectAndCompute(img_rotated,    None)

    bf      = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des_m, des_t), key=lambda x: x.distance)

    input_for_analysis = input_image_path  # fallback

    if len(matches) < 10:
        print("⚠️ Not enough features to locate MCCB region. Using full image.")
    else:
        src_pts = np.float32([kp_m[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_t[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        M, _    = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if M is not None:
            h_m, w_m  = cropped_master.shape[:2]
            corners   = np.float32([[0, 0], [0, h_m-1],
                                    [w_m-1, h_m-1], [w_m-1, 0]]).reshape(-1, 1, 2)
            t_corners = cv2.perspectiveTransform(corners, M)

            x_min, y_min = np.int32(t_corners.min(axis=0).ravel())
            x_max, y_max = np.int32(t_corners.max(axis=0).ravel())

            h_t, w_t = img_rotated.shape[:2]
            x_min = max(0, x_min); y_min = max(0, y_min)
            x_max = min(w_t, x_max); y_max = min(h_t, y_max)

            cropped_test      = img_rotated[y_min:y_max, x_min:x_max]
            cropped_test_path = os.path.join(output_dir,
                                             "cropped_" + os.path.basename(input_image_path))
            cv2.imwrite(cropped_test_path, cropped_test)
            print(f"  ✅ MCCB region found and cropped → {cropped_test_path}")
            input_for_analysis = cropped_test_path
        else:
            print("⚠️ Homography failed. Using full image.")

    # ── Compare against master ────────────────────────────────────────────────
    print("\nStep 2: Comparing against master reference...")
    compare_images(reference_image_path, input_for_analysis,
                   min_contour_area=10000, threshold_value=50)


# ══════════════════════════════════════════════════════════════════════════════
# 5. OCR PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

class RobustMCCBPipeline:
    """
    Main orchestrator for MCCB OCR data extraction.

    Initializes PaddleOCR once, then processes images by:
        1. Detecting layout regions (via layout_detector)
        2. Extracting top lines, table data, and rating from those regions
        3. Returning structured JSON-ready data
    """

    def __init__(self):
        print("Initializing PaddleOCR engine...")
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        print("✅ PaddleOCR ready.")

    def extract_top_lines(self, ocr_results, y_max):
        """Extracts text from the top section of the MCCB (above the spec table)."""
        top_items = []
        for line in ocr_results:
            box  = line[0]
            text = line[1][0]
            cy   = sum(p[1] for p in box) / 4
            if cy < y_max:
                top_items.append({"text": text, "cy": cy})
        top_items.sort(key=lambda k: k['cy'])
        return [item['text'] for item in top_items]

    def extract_table(self, ocr_results, y_min, y_max):
        """Filters OCR text within the table ROI and restructures into a 2D grid."""
        table_boxes = []
        for line in ocr_results:
            box  = line[0]
            text = line[1][0]
            cy   = sum(p[1] for p in box) / 4
            cx   = sum(p[0] for p in box) / 4
            h    = abs(box[2][1] - box[0][1])
            if y_min < cy < y_max:
                table_boxes.append({"text": text, "cy": cy, "cx": cx, "h": h})

        if not table_boxes:
            return []

        table_boxes.sort(key=lambda k: k['cy'])

        rows        = []
        current_row = []
        last_y      = table_boxes[0]['cy']
        avg_height  = np.mean([b['h'] for b in table_boxes])
        y_tolerance = avg_height * 0.6

        for item in table_boxes:
            if abs(item['cy'] - last_y) <= y_tolerance:
                current_row.append(item)
            else:
                current_row.sort(key=lambda k: k['cx'])
                rows.append([x['text'] for x in current_row])
                current_row = [item]
                last_y = item['cy']

        if current_row:
            current_row.sort(key=lambda k: k['cx'])
            rows.append([x['text'] for x in current_row])

        return rows

    def extract_rating_from_roi(self, ocr_results, y_min, y_max):
        """Finds the 'In=...A' rating string within the rating ROI."""
        rating_pattern = re.compile(r'(?:I|l|1)n\s*=\s*(\d+\s*A)', re.IGNORECASE)
        for line in ocr_results:
            box  = line[0]
            text = line[1][0]
            cy   = sum(p[1] for p in box) / 4
            if y_min < cy < y_max:
                match = rating_pattern.search(text)
                if match:
                    return match.group(0)
        return "Not Found"

    def process_image(self, image_path):
        """
        Full OCR pipeline for a single MCCB image.

        Returns:
            dict: Structured data with top_lines, rating, table_data.
                  None if image is unreadable or OCR produces no output.
        """
        filename = os.path.basename(image_path)
        print(f"\nOCR Processing: {filename}")

        img = cv2.imread(image_path)
        if img is None:
            print(f"  ❌ Could not read image: {image_path}")
            return None

        # Rotate to match layout_detector's expected orientation
        img_rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        ocr_result  = self.ocr.ocr(img_rotated, cls=True)

        if not ocr_result or ocr_result[0] is None:
            print("  ⚠️ No text found by OCR.")
            return None

        all_lines = ocr_result[0]

        # Detect layout regions
        debug_dir = os.path.join(os.path.dirname(image_path), "debug_pipeline")
        regions   = detect_layout_regions(image_path, debug_dir, ocr_lines=all_lines)

        h, w = img_rotated.shape[:2]

        if regions:
            table_y_min,  table_y_max  = regions["table"]
            rating_y_min, rating_y_max = regions["rating"]
        else:
            print("  ⚠️ Layout detection failed. Searching full image.")
            table_y_min  = rating_y_min = 0
            table_y_max  = rating_y_max = h

        top_lines  = self.extract_top_lines(all_lines, table_y_min)
        rating_val = self.extract_rating_from_roi(all_lines, rating_y_min, rating_y_max)
        table_rows = self.extract_table(all_lines, table_y_min, table_y_max)

        return {
            "filename"  : filename,
            "top_lines" : top_lines,
            "rating"    : rating_val,
            "table_data": table_rows,
        }


# ══════════════════════════════════════════════════════════════════════════════
# 6. MAIN ORCHESTRATOR
#    Flow: QR Scan → Camera Capture → OCR → Defect Detection
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Paths — update as needed ───────────────────────────────────────────────
    MASTER_REFERENCE = r"cropped_master_imaeges\cropped_masterXT13P_mccb.png"  # Update for each MCCB model
    input_image_path=r"Testing_images\CG36350266067415.png"

    # ── Output directory ───────────────────────────────────────────────────────
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)

    # ── Step 1: QR Scan via Serial ─────────────────────────────────────────────
    print("\n" + "="*60)
    print("  MCCB INSPECTION PIPELINE")
    print("="*60)
    print("\nStep 1: Waiting for QR scan...")

    # ser      = serial.Serial('COM4', 9600, timeout=1)
    # qr_data  = None

    # while True:
    #     if ser.in_waiting > 0:
    #         qr_data = ser.readline().decode("utf-8").strip()
    #         if qr_data:
    #             print(f"  ✅ Scanned: {qr_data}")
    #             break

    # # ── Step 2: Camera Capture ─────────────────────────────────────────────────
    # print("\nStep 2: Capturing image from camera...")
    # input_image_path = f"{qr_data}.png"
    # time.sleep(10)  # Wait for MCCB to be positioned

    # success = capture_image(input_image_path)
    # if not success:
    #     print("❌ Camera capture failed. Exiting.")
    #     ser.close()
    #     exit(1)

    # print(f"  ✅ Image captured: {input_image_path}")

    # ── Step 3: OCR ────────────────────────────────────────────────────────────
    print("\nStep 3: Running OCR pipeline...")
    print("="*60)
    print("  ROBUST MCCB OCR PIPELINE")
    print("="*60)

    pipeline = RobustMCCBPipeline()
    raw_data = pipeline.process_image(input_image_path)

    if raw_data is None:
        print("  ⚠️ OCR produced no data. Skipping JSON output.")
    else:
        formatted_data = format_image_result(raw_data)
        serial_number  = formatted_data.get("serial_number")

        # Build output filename from serial number if available
        if serial_number:
            safe_sn  = "".join(c for c in serial_number
                               if c.isalnum() or c in ('-', '_')).strip()
            out_name = f"{safe_sn}.json" if safe_sn else f"{os.path.splitext(qr_data)[0]}.json"
        else:
            out_name = f"{os.path.splitext(qr_data)[0]}.json"

        out_path = os.path.join(output_dir, out_name)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(formatted_data, fh, indent=2, ensure_ascii=False)

        print(f"\n  ✅ OCR Result:")
        print(json.dumps(formatted_data, indent=2, ensure_ascii=False))
        print(f"  ✅ Saved JSON: {out_path}")

    # ── Step 4: Defect Detection ───────────────────────────────────────────────
    print("\nStep 4: Running defect detection...")
    if os.path.exists(MASTER_REFERENCE):
        process_test_image(input_image_path, MASTER_REFERENCE)
    else:
        print(f"❌ Master reference not found: {MASTER_REFERENCE}")
        print("   Update MASTER_REFERENCE path at the top of __main__.")

    # ser.close()
    print("\n✅ Pipeline complete.")