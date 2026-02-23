
import cv2
import numpy as np
import os

def detect_layout_regions(image_path, output_dir):
    filename = os.path.basename(image_path)
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load: {image_path}")
        return None
    
    vis_img = img.copy()
    h, w = img.shape[:2]
    
    # Preprocessing to find the Switch (dark object in the middle)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Thresholding to find dark regions (The switch is usually distinctively dark)
    # Using simple thresholding often works best for the switch vs white label
    # Inverted because we want the dark switch to be white in the binary image
    _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
    
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
        
        # Constraints check
        is_centered_x = (0.2 * w < cx < 0.8 * w)
        # Switch is usually in the middle or slightly lower middle
        is_centered_y = (0.3 * h < cy < 0.8 * h) 
        is_large_enough = area > 5000  # Adjust based on resolution
        
        if is_centered_x and is_centered_y and is_large_enough:
            if area > max_area:
                max_area = area
                switch_contour = c

    if switch_contour is not None:
        # Get Bounding Box of Switch
        sx, sy, sw, sh = cv2.boundingRect(switch_contour)
        
        # Define Regions
        # Table counts as roughly Top 10% to Switch Top
        # We assume the logo is at the very top, so we start below it (e.g., 10% or detected logo bottom)
        table_roi_y1 = int(h * 0.1) 
        table_roi_y2 = sy 
        
        # Rating counts as Switch Bottom to Bottom 10% (Barcode area)
        rating_roi_y1 = sy + sh
        rating_roi_y2 = int(h * 0.9)
        
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
