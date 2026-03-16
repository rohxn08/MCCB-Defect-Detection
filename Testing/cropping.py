import cv2
import os
import sys
import numpy as np

# Import layout detector
# Assuming it's in the same Testing folder now
try:
    from layout_detector import detect_layout_regions
except ImportError:
    # Fallback if run from different context
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from layout_detector import detect_layout_regions

def crop_for_oem_inspection(image_path, output_dir="cropped_inspection"):
    """
    Crops a test image strictly around the label area using the switch as an anchor.
    This helps in manual inspection and verifies the cropping logic for the alignment engine.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    filename = os.path.basename(image_path)
    print(f"\nProcessing: {filename}")
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load {image_path}")
        return
        
    # Rotate the image 90 CCW to match the layout detector's perspective
    img_rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    h, w = img_rotated.shape[:2]
    
    # Run the layout detector to find the switch bounding box
    # We pass the image_path so it can reload it internally if needed
    regions = detect_layout_regions(image_path, output_dir)
    
    if regions and "switch" in regions:
        sy, sy_bottom = regions["switch"]
        
        # --- CALIBRATED PERCENTAGES ---
        top_percentage = 0.33     # Decrease to remove top jig (Original Right side)
        bottom_percentage = 0.50  # Increase to capture bottom screws/TMD (Original Left side)
        
        crop_top = max(0, int(sy - (h * top_percentage)))
        crop_bottom = min(h, int(sy_bottom + (h * bottom_percentage)))
        
        # Extract the label area
        cropped_img = img_rotated[crop_top:crop_bottom, :]
        
        output_path = os.path.join(output_dir, f"cropped_{filename}")
        cv2.imwrite(output_path, cropped_img)
        
        print(f"SUCCESS: Cropped region extracted.")
        print(f"Switch height: {sy_bottom - sy}px")
        print(f"Total crop height: {crop_bottom - crop_top}px")
        print(f"Saved to: {output_path}")
    else:
        print(f"FAILED: Switch not detected in {filename}. Visualization saved to {output_dir}")

if __name__ == "__main__":
    # Test images to process
    test_files = [
        r"D:\MCCB-Defect-Detection\Test_images\P1_3P.png",
        r"D:\MCCB-Defect-Detection\Test_images\XT1_3P.png",
        r"D:\MCCB-Defect-Detection\Test_images\P1_4P_Testing.png"
        # Add more paths here as needed
    ]
    
    print("--- MCCB Physical Cropping Utility ---")
    for file in test_files:
        if os.path.exists(file):
            crop_for_oem_inspection(file)
        else:
            print(f"File not found: {file}")
