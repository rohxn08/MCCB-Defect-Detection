import cv2
import numpy as np
import re
import os
"""
If the lighting conditions change, these are the values to be modified.

- Threshold value: Current Value = 40
-----------------> If contrast is high (brighter lighting), increase the threshold value.
-----------------> If contrast is low (darker lighting), decrease the threshold value.

- Table ROI: Current Value = (sy - int(h * 0.25), sy)
-----------------> If the top of the table is cut off, increase the multiplier (e.g., from 0.25 to 0.30) to make the box taller.
-----------------> If the box is grabbing empty space or logos above the table, decrease the multiplier (e.g., from 0.25 to 0.20).

- Rating ROI: Current Value = (sy, int(h * 0.90))
-----------------> If the bottom ratings are cut off, increase the multiplier (e.g., from 0.90 to 0.95) to reach further down.
-----------------> If the box is grabbing bottom screws/plastic instead of just text, decrease the multiplier (e.g., from 0.90 to 0.85).


"""

def detect_layout_regions(image_path, output_dir, ocr_lines=None):
    """    
    Detects the switch component in an MCCB image to caluculate Regions of Intrest (ROIs)
    for the tabular and the rating sections.

    Args:
        image_path(str): Path to the input image file
        output_dir(str): Directory where the debug visualizations are stored
        ocr_lines(list): List of OCR lines from the image   

    Returns:
        dict: Dictionary containing the ROIs for the table and rating sections
        None: If the switch is not detected

"""
    filename = os.path.basename(image_path)
    img = cv2.imread(image_path)
    img = cv2.rotate(img,cv2.ROTATE_90_COUNTERCLOCKWISE)
    if img is None:
        print(f"Failed to load: {image_path}")
        return None
    # Creating a copy of the image for visualization
    vis_img = img.copy()
    h, w = img.shape[:2]

    
    
    # Preprocessing to find the Switch (dark object in the middle)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Thresholding to isolate the Switch 
    # We apply an inverted binary threshold because the switch is distinctively dark 
    # against the white label. Inverting it makes the dark switch 'white' (easier to detect as a contour).
    
    # The threshold value (currently 40) is highly sensitive to the image's exposure.
    # If testing images from different lighting environments, follow these rules to tune:
    # - Brighter lighting / Glare: INCREASE threshold (e.g., 60-80) to catch the grayed-out switch.
    # - Darker lighting / Shadows: DECREASE threshold (e.g., 20-30) so shadows aren't mistaken for the switch.
    # - For the current lighting 40 is the optimal value
    _, thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV)


    # 2. Find Contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 3. Filter for the Switch
    # Heuristics:
    # - Must be somewhat central (x in 20% to 80% range)
    # - Must be somewhat large (area > 5000)
    # - Must be in the vertical middle-ish (y in 30% to 80% range)
    switch_contour = None
    max_area = 0
    
    for c in contours:
        area = cv2.contourArea(c)
        x, y, cw, ch = cv2.boundingRect(c)
        cx, cy = x + cw/2, y + ch/2
        
        # Constraints check (Somewhat between 20% to 80% of the image width)
        is_centered_x = (0.2 * w < cx < 0.8 * w)
        # Constraints check (30% to 80% of the image height)
        is_centered_y = (0.3 * h < cy < 0.8 * h) 
        # Constraints check (Adjust based on resolution)
        is_large_enough = area > 5000  
        
        if is_centered_x and is_centered_y and is_large_enough:
            if area > max_area:
                max_area = area
                switch_contour = c

    if switch_contour is not None:
        # Get Bounding Box of Switch
        sx, sy, sw, sh = cv2.boundingRect(switch_contour)
        
        # Define Regions
        # Table: Estimate table start relative to the switch position.
        # The table occupies roughly the area between the top mounting hardware
        # and the switch. Table typically starts ~20% of image height above the switch.
        # NOTE: OCR-based refinement is disabled because OCR runs on the original
        # (non-rotated) image, so its coordinates don't match the rotated layout.
        table_roi_y1 = max(0, sy - int(h * 0.25))
        table_roi_y2 = sy 
        
        # Rating: from switch top to bottom 90%
        # Covers the switch body area (In= rating) and the bottom table (MIN/MED/MAX)
        rating_roi_y1 = sy # Starts at sy
        rating_roi_y2 = int(h * 0.90) # ends at 90% of the image height
        
        # VISUALIZATION

        # Draw Switch Box (Red)
        cv2.rectangle(vis_img, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 10)
        cv2.putText(vis_img, "SWITCH", (sx, sy-20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
        
        # Draw Table Region (Green)
        cv2.rectangle(vis_img, (0, table_roi_y1), (w, table_roi_y2), (0, 255, 0), 10)
        cv2.putText(vis_img, "TABLE ROI", (50, table_roi_y1 + 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)

        # Draw Rating Region (Blue)
        cv2.rectangle(vis_img, (0, rating_roi_y1), (w, rating_roi_y2), (255, 0, 0), 10)
        cv2.putText(vis_img, "RATING ROI", (50, rating_roi_y1 + 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 5)
        
        print(f"[{filename}] SUCCESS - Switch found at Y={sy}")
        
        # Return ROI data for pipeline use
        regions = {
            "switch": (sy, sy + sh),
            "table":  (table_roi_y1, table_roi_y2),
            "rating": (rating_roi_y1, rating_roi_y2),
        }
        
    else:
        print(f"[{filename}] FAILED - No switch detected")
        regions = None
        
    # Save debug image
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, f"layout_{filename}")
    cv2.imwrite(output_path, vis_img)
    
    return regions

if __name__ == "__main__":
    # Adjust paths relative to where the script is run (project root or folder)
    # Assuming running from project root
    project_root = os.getcwd()
    if "OCR_Extraction" in project_root: # If run from inside the folder
        project_root = os.path.dirname(project_root)
        
    input_dir = os.path.join(project_root, "images", "master_mccb")
    output_dir = os.path.join(project_root, "images", "debug_layout")
    
    if os.path.exists(input_dir):
        files = [f for f in os.listdir(input_dir) if f.startswith("master") and f.lower().endswith(('.png', '.jpg'))]
        for f in files:
            detect_layout_regions(os.path.join(input_dir, f), output_dir)
        print(f"\nDebug images saved to {output_dir}")
    else:
        print(f"Directory not found: {input_dir}")
