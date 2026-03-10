#robst_ocr_pipeline.py



#below for OCR engine
from paddleocr import PaddleOCR

#below for image processing
import cv2

#for OCR formating
import re

#to obtain path
import os

#for image processing
import numpy as np

#for output
import json
from json_formatter import format_image_result

#to detect roi
from layout_detector import detect_layout_regions

#to capture image
from MvCameraControl_class import *

#to detect defect
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

import time
from datetime import datetime

# for QR Code
import serial

def capture_image(save_path="snapshot.jpg"):

    cam = MvCamera()

    # Enumerate devices
    device_list = MV_CC_DEVICE_INFO_LIST()
    ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, device_list)
    if ret != 0 or device_list.nDeviceNum == 0:
        print("❌ No Hikrobot camera detected.")
        return None

    stDeviceInfo = cast(device_list.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents

    # Create & open handle
    ret = cam.MV_CC_CreateHandle(stDeviceInfo)
    if ret != 0:
        print(f"❌ CreateHandle failed: {ret}")
        return None

    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0:
        print(f"❌ OpenDevice failed: {ret}")
        cam.MV_CC_DestroyHandle()
        return None

    # Get current trigger mode
    trigger_mode = c_uint()
    ret = cam.MV_CC_GetEnumValue("TriggerMode", trigger_mode)
    is_trigger_mode = False
    if ret == 0 and trigger_mode.value == 1:
        is_trigger_mode = True
        print("🎯 Camera is in Trigger Mode → using software trigger.")
    else:
        print("📷 Camera is in Continuous Mode.")

    # If trigger mode is ON, set trigger source to software
    if is_trigger_mode:
        cam.MV_CC_SetEnumValue("TriggerSource", 7)  # Software trigger

    # Start grabbing
    cam.MV_CC_StartGrabbing()
    time.sleep(0.3)

    # Fire trigger if needed
    if is_trigger_mode:
        cam.MV_CC_SetCommandValue("TriggerSoftware")

    # Prepare buffer
    # === Dynamic resolution and buffer ===
    width_val = MVCC_INTVALUE()
    height_val = MVCC_INTVALUE()
    cam.MV_CC_GetIntValue("Width", width_val)
    cam.MV_CC_GetIntValue("Height", height_val)
    width, height = width_val.nCurValue, height_val.nCurValue
    buf_size = width * height * 3
    data_buf = (c_ubyte * buf_size)()

    frame_info = MV_FRAME_OUT_INFO_EX()
    ret = cam.MV_CC_GetOneFrameTimeout(byref(data_buf), buf_size, frame_info, 3000)
    print("GetOneFrameTimeout ret:", ret)
    if ret == 0:
        print(f"✅ Frame captured: {frame_info.nWidth}x{frame_info.nHeight}")

        frame = np.frombuffer(data_buf, dtype=np.uint8,
                              count=frame_info.nWidth * frame_info.nHeight)
        frame = frame.reshape(frame_info.nHeight, frame_info.nWidth)

        # Pixel type conversion
        if frame_info.enPixelType == 17301505:  # Mono8
            image = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            image = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2BGR)


        # timestamp = datetime.now().strftime(result_data['timestamp'])

        save_path = save_path
        cv2.imwrite(save_path, image)
        print(f"📸 Image saved: {save_path}")
    else:
        print(f"⚠️ Failed to get frame. Return code: {ret}")

    # Cleanup
    cam.MV_CC_StopGrabbing()
    cam.MV_CC_CloseDevice()
    cam.MV_CC_DestroyHandle()
    return ret == 0

def align_images(im1, im2, max_features=5000, keep_percent=0.2):
    """
    Aligns im2 to match the perspective and position of im1 using ORB feature matching.
    """
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(max_features)
    keypoints1, descriptors1 = orb.detectAndCompute(im1_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2_gray, None)

    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors2, descriptors1, None)

    matches = sorted(matches, key=lambda x: x.distance)
    keep = int(len(matches) * keep_percent)
    matches = matches[:keep]

    pts1 = np.zeros((len(matches), 2), dtype=np.float32)
    pts2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        pts2[i, :] = keypoints2[match.queryIdx].pt
        pts1[i, :] = keypoints1[match.trainIdx].pt

    h, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC)
    
    height, width, channels = im1.shape
    im2_aligned = cv2.warpPerspective(im2, h, (width, height))

    return im2_aligned

# --- 2. The Core Image Comparison Function ---
def compare_images(reference_image_path, input_image_path, min_contour_area=100, threshold_value=30):
    """
    Compares an input image with a reference image to find and label defects,
    and displays the results using matplotlib.
    """
    reference_image = cv2.imread(reference_image_path)
    input_image = cv2.imread(input_image_path)

    print("Aligning input image to reference image using ORB...")
    try:
        input_image = align_images(reference_image, input_image)
    except Exception as e:
        print(f"Alignment failed: {e}. Proceeding without alignment.")

    reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    input_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    
    #Blur the image
    reference_blurred = cv2.GaussianBlur(reference_gray, (69, 69), 0)
    input_blurred = cv2.GaussianBlur(input_gray, (69, 69), 0)
    # reference_blurred = cv2.cvtColor(reference_image, cv2.COLOR_BGR2HSV)
    # input_blurred = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
    # cv2.imwrite('input_blurred.jpg', input_blurred )
    # cv2.imwrite("reference_blurred.jpg", reference_image)
    (score, diff) = ssim(reference_blurred, input_blurred, full=True)
    diff = (diff * 255).astype("uint8")
    print(f"Image Similarity Score: {score:.4f}")


    _, thresh = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # **FIX APPLIED HERE**: Directly unpack the two return values from cv2.findContours.
    # This is the standard, robust way to do it for OpenCV versions 3 and 4.
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output_image = input_image.copy()
    defects_found = 0
    # Now, the 'contours' variable is guaranteed to be the list of contour arrays.
    for c in contours:
        if cv2.contourArea(c) > min_contour_area:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 0, 255), 4)
            cv2.putText(output_image, "Defect", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            defects_found += 1

    print(f"Found {defects_found} potential defects.")
    status = f"Defects_Found(similarity {score:.4f})" if score <= 0.98 else f"OK(similarity {score:.4f})"

    cv2.imwrite("output_with_defects.png", output_image)
    print("Saved the output image as 'output_with_defects.png'")

    plt.figure(figsize=(20, 8))
    plt.subplot(1, 4, 1)
    plt.imshow(cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB))
    plt.title("Master Image")
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
    plt.title("Input Image")
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(thresh, cmap='gray')
    plt.title("Difference Mask (Defects)")
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Result: {status}")
    plt.axis('off')

    plt.tight_layout()


    now = datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    # **FIX APPLIED HERE**: Save the complete figure to a file before displaying it.

    plt.savefig(rf"testing\\030326\\P1_PRD_{status}_comparison_plot_{timestamp_str}.png")
    print("Saved the entire comparison plot as 'comparison_plot.png'")

    plt.show()

class RobustMCCBPipeline:
    """
        Main orchestrator for the MCCB data extraction process.

        This class manages the lifecycle of the OCR process. It initializes a single
        instance of the PaddleOCR engine (heavy operation), processes images by coordinating
        with the layout_detector to find physical Regions of Interest (ROIs), and then
        filters the OCR text into specialized buckets (Top Lines, Table, Rating) based
        on those spatial boundaries.
        """
    def __init__(self):
        """
                Initializes the pipeline and pre-loads the PaddleOCR engine models into memory.

                Note: PaddleOCR is initialized only once per class instantiation to prevent
                costly model reloading times when processing directories of multiple images.
                Uses angle classification (use_angle_cls=True) to handle rotated text.
                """
        # Initialize PaddleOCR once
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

    def extract_top_lines(self, ocr_results, y_max):
        """
                Collects OCR text from the uppermost section of the MCCB label.

                This targets text physically located above the specification table
                (e.g., brand logos, product names, 'S/N' serial numbers). It calculates
                the center Y-coordinate (cy) of every OCR bounding box and filters for
                items positioned higher than the y_max threshold.

                Args:
                    ocr_results (list): Raw OCR outputs directly from PaddleOCR.
                    y_max (int): The vertical cutoff (Y-coordinate). Usually the top edge
                                 of the Table ROI calculated by the layout_detector.

                Returns:
                    list: Strings extracted from the top section, sorted physically top-to-bottom.
        """
        top_items = []
        for line in ocr_results:
            box = line[0] # The 4 corners of the text box
            text = line[1][0] # The actual readable string (e.g., "SACE FORMULA")
            cy = sum([p[1] for p in box]) / 4 # Calculating the center Y-coordinate


            if cy < y_max:  # If it's physically higher than the table
                top_items.append({"text": text, "cy": cy})

        # Sort top-to-bottom and return just the text strings
        top_items.sort(key=lambda k: k['cy'])
        return [item['text'] for item in top_items]

    def extract_table(self, ocr_results, y_min, y_max):
        """
                Filters and restructures OCR text bounded within the Table ROI into a 2D grid.

                Architecture:
                1. Filtering: Extracts only text boxes whose center Y-coordinate falls between
                   y_min (top of table) and y_max (bottom of table, usually the switch top).
                2. Virtual Grid Construction: Sorts the extracted text by height (Y). Because
                   text on the same row may not align perfectly physically, it calculates an
                   average line height and defines a `y_tolerance` (60% of average height).
                   It then "snaps" all horizontal text sharing a vertical window into grouped rows.

                Args:
                    ocr_results (list): Raw OCR outputs directly from PaddleOCR.
                    y_min (int): Top boundary of the Table ROI.
                    y_max (int): Bottom boundary of the Table ROI.

                Returns:
                    list of lists: A structured 2D array where each inner list acts as a
                                   physical row of specifications (e.g., [['Ue (V)', '415']]).
                                   Returns an empty list if no text falls inside the ROI.
                """
        table_boxes = []
        for line in ocr_results:
            box = line[0]  # The 4 corners of the text box
            text = line[1][0]  # The actual readable string
            cy = sum([p[1] for p in box]) / 4  # Center Y-coordinate
            cx = sum([p[0] for p in box]) / 4  # Center X-coordinate
            h = abs(box[2][1] - box[0][1])  # Height of the text box
            
            if y_min < cy < y_max:   # If text falls within the table boundaries
                table_boxes.append({"text": text, "cy": cy, "cx": cx, "h": h})
        
        if not table_boxes:
            return []

        # Virtual Grid Construction
        # 1. Sort by Y to find rows top to bottom
        table_boxes.sort(key=lambda k: k['cy'])
        
        rows = []  # Final list of rows
        current_row = []   # Temporary cluster for the current horizontal row
        last_y = table_boxes[0]['cy']   # Tracker for the vertical baseline
        avg_height = np.mean([b['h'] for b in table_boxes])  # Average text height
        y_tolerance = avg_height * 0.6  # Allowable vertical drift for items in the same row
        
        for item in table_boxes:
            if abs(item['cy'] - last_y) <= y_tolerance:
                current_row.append(item)  # If text belongs to the current vertical row
            else:
                # Finish row
                current_row.sort(key=lambda k: k['cx'])
                rows.append([x['text'] for x in current_row])
                # Start new
                current_row = [item]
                last_y = item['cy']
        # Catching remaining items
        if current_row:
            current_row.sort(key=lambda k: k['cx'])
            rows.append([x['text'] for x in current_row])
            
        return rows

    def extract_rating_from_roi(self, ocr_results, y_min, y_max):
        """
                Scans specifically within the physical Rating ROI for the breaker's amperage.

                Filters OCR text to the bounding box surrounding the switch and searches
                for the 'In=...' electrical rating structure using Regular Expressions.

                Args:
                    ocr_results (list): Raw OCR outputs directly from PaddleOCR.
                    y_min (int): Top boundary of the Rating ROI (usually the top of the switch).
                    y_max (int): Bottom boundary of the Rating ROI (usually 90% of image height).

                Returns:
                    str: The fully extracted rating string (e.g., "In=160A").
                         Returns "Not Found" if the pattern isn't matched within the region.
                """
        rating_pattern = re.compile(r'(?:I|l|1)n\s*=\s*(\d+\s*A)', re.IGNORECASE)
        
        for line in ocr_results:
            box = line[0]
            text = line[1][0]
            cy = sum([p[1] for p in box]) / 4
            
            if y_min < cy < y_max:
                match = rating_pattern.search(text)
                if match:
                    return match.group(0) # Return entire string like "In=63A"
        return "Not Found"

    def process_image(self, image_path):
        """
                Coordinates the full pipeline: loading the image, detecting ROIs, running OCR,
                and extracting structured data.

                Args:
                    image_path (str): The file path to the original MCCB image.

                Returns:
                    dict: Structured data containing product name, rating, table, etc.
                          Returns None if image is unreadable or OCR fails.
                """
        filename = os.path.basename(image_path)
        print(f"Processing {filename}...")
        
        img = cv2.imread(image_path)
        if img is None:
            return None

        # 1. Align orientations
        # Rotate the image the same way layout_detector does
        # so OCR coordinates align perfectly with ROI coordinates
        img_rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        ocr_result = self.ocr.ocr(img_rotated, cls=True)
        if not ocr_result or ocr_result[0] is None:
            print("  No text found.")
            return None
        all_lines = ocr_result[0]

        # 2. Detect Layout Regions (via layout_detector)
        debug_dir = os.path.join(os.path.dirname(image_path), "debug_pipeline")
        regions = detect_layout_regions(image_path, debug_dir, ocr_lines=all_lines)
        
        h, w = img_rotated.shape[:2]
        
        if regions:
            # Use layout detector's ROI boundaries directly
            table_y_min, table_y_max = regions["table"]
            rating_y_min, rating_y_max = regions["rating"]
        else:
            # Fallback
            # h = img.shape[0]
            print("  Warning: Switch not detected. Using full image search.")
            table_y_min, table_y_max = 0, h
            rating_y_min, rating_y_max = 0, h

        # 3. Extract Data
        # Top-section lines (product name, serial number, etc.)
        top_lines = self.extract_top_lines(all_lines, table_y_min)

        # Rating
        rating_val = self.extract_rating_from_roi(all_lines, rating_y_min, rating_y_max)
        
        # Table
        table_rows = self.extract_table(all_lines, table_y_min, table_y_max)

        return {
            "filename": filename,
            "top_lines": top_lines,
            "rating": rating_val,
            "table_data": table_rows        }

if __name__ == "__main__":

    ser = serial.Serial('COM4', 9600, timeout=1)
    print("Scanner ready.")
    while True:
        # qr_data = "check"
        success = None
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     print("Exiting...")
        #     break

        if ser.in_waiting > 0:
            qr_data = ser.readline().decode("utf-8").strip()
            if qr_data:
                print(f"Scanned data: {qr_data}")
                time.sleep(10)
                success = capture_image(qr_data+".png")
                if success:
                    print("✅ Capture successful.")
                else:
                    print("❌ Capture failed.")
                break
        # time.sleep(1)

    input_image = rf"{qr_data}.png"
    capture_image(input_image)
    pipeline = RobustMCCBPipeline()
    
    print("\n" + "="*50)
    print("  ROBUST MCCB OCR PIPELINE")
    print("="*50)

    # choice = input("Do you want to process a single image (1) or a directory (2)? [1/2]: ").strip()

    # Adjust path
    project_root = os.getcwd()
    if "OCR_Extraction" in project_root:
        project_root = os.path.dirname(project_root)
        
    # Output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    images_to_process = []

    img_path = input_image

    images_to_process.append(img_path)

    # if choice == '1':
    #     # img_path = input("Enter the full path to the image: ").strip()
    #     # Remove surrounding quotes if dragged-and-dropped in terminal
    #     img_path = img_path.strip('\'"')
    #     if os.path.isfile(img_path):
    #         images_to_process.append(img_path)
    #     else:
    #         print(f"Invalid image path: {img_path}")
    # elif choice == '2':
    #     default_dir = os.path.join(project_root, "images", "master_mccb")
    #     input_dir = input(f"Enter directory path [Press Enter for default: {default_dir}]: ").strip()
    #     if not input_dir:
    #         input_dir = default_dir
    #     else:
    #         input_dir = input_dir.strip('\'"')
    #
    #     if os.path.isdir(input_dir):
    #         files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    #         images_to_process.extend([os.path.join(input_dir, f) for f in files])
    #     else:
    #         print(f"Directory not found: {input_dir}")
    # else:
    #     print("Invalid choice. Please enter 1 or 2.")
        
    if images_to_process:
        print(f"\nFound {len(images_to_process)} image(s) to process.")
        
        for img_path in images_to_process:
            filename = os.path.basename(img_path)
            raw_data = pipeline.process_image(img_path)
            
            if raw_data is None:
                print(f"  ⚠ Skipped {filename} (no data extracted)")
                continue
            
            # Format JSON
            formatted_data = format_image_result(raw_data)
            
            # Determine filename
            serial_number = formatted_data.get("serial_number")
            
            # Clean serial number for filesystem if it exists
            if serial_number:
                # Remove any characters that might be invalid in filenames
                safe_sn = "".join(c for c in serial_number if c.isalnum() or c in ('-', '_')).strip()
                if safe_sn:
                    out_name = f"{safe_sn}.json"
                else:
                    out_name = f"{os.path.splitext(filename)[0]}.json"
            else:
                out_name = f"{os.path.splitext(filename)[0]}.json"
                
            out_path = os.path.join(output_dir, out_name)
            
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(formatted_data, fh, indent=2, ensure_ascii=False)
                
            print(f"\n  ✓ Result JSON:")
            print(json.dumps(formatted_data, indent=2, ensure_ascii=False))
            print(f"  ✓ Saved {filename} results to: {out_path}\n")


    # detect defects
    ref_image_file = r"masterImages/masterXT14P_mccb.png"

    compare_images(ref_image_file, input_image, min_contour_area=5000, threshold_value=50)



