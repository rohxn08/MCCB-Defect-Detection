import cv2
import numpy as np
import os

def visual_reference_crop(full_test_path, cropped_master_path, output_dir="visual_crop_inspection"):
    """
    Uses a cropped master image as a 'visual template' to find and crop 
    the same region from a raw, uncropped test image.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Load images
    full_test = cv2.imread(full_test_path)
    cropped_master = cv2.imread(cropped_master_path)
    
    if full_test is None or cropped_master is None:
        if full_test is None: print(f"Error loading full_test: {full_test_path}")
        if cropped_master is None: print(f"Error loading master: {cropped_master_path}")
        return

    # In our pipeline, the test image is usually rotated 90 CCW 
    # Let's rotate it to match the perspective of the master crop
    test_rotated = cv2.rotate(full_test, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    print(f"\n--- Processing {os.path.basename(full_test_path)} ---")
    print(f"Master Reference: {os.path.basename(cropped_master_path)}")
    
    # 1. ORB Feature Detection
    orb = cv2.ORB_create(10000)
    kp_master, des_master = orb.detectAndCompute(cropped_master, None)
    kp_test, des_test = orb.detectAndCompute(test_rotated, None)
    
    # 2. Match Features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_master, des_test)
    matches = sorted(matches, key=lambda x: x.distance)
    
    if len(matches) < 10:
        print("Error: Not enough feature matches found.")
        return

    # 3. Calculate Homography
    src_pts = np.float32([kp_master[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_test[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    if M is None:
        print("Error: Could not calculate transformation (Homography failed).")
        return

    # 4. Define corners and project
    h_m, w_m = cropped_master.shape[:2]
    master_corners = np.float32([[0, 0], [0, h_m-1], [w_m-1, h_m-1], [w_m-1, 0]]).reshape(-1, 1, 2)
    test_corners = cv2.perspectiveTransform(master_corners, M)
    
    # 5. Get bounding box
    x_min, y_min = np.int32(test_corners.min(axis=0).ravel())
    x_max, y_max = np.int32(test_corners.max(axis=0).ravel())
    
    # Clamp
    h_t, w_t = test_rotated.shape[:2]
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(w_t, x_max), min(h_t, y_max)
    
    # 6. Crop and Save
    cropped_result = test_rotated[y_min:y_max, x_min:x_max]
    
    if cropped_result.size == 0:
        print("Error: Crop resulted in empty image.")
        return

    out_name = "visual_crop_" + os.path.basename(full_test_path)
    out_path = os.path.join(output_dir, out_name)
    cv2.imwrite(out_path, cropped_result)
    
    print(f"SUCCESS: Crop region found at [{x_min}:{x_max}, {y_min}:{y_max}]")
    print(f"Saved result: {out_path}")

if __name__ == "__main__":
    base_dir = r"D:\MCCB-Defect-Detection"
    master_dir = os.path.join(base_dir, "cropped_master_imaeges")
    test_dir = os.path.join(base_dir, "Test_images")
    
    tests = [
        (os.path.join(test_dir, "P1_3P.png"), os.path.join(master_dir, "cropped_masterP13P_mccb.png")),
        (os.path.join(test_dir, "P1_4P_Testing.png"), os.path.join(master_dir, "cropped_masterP14P_mccb.png")),
        (os.path.join(test_dir, "XT1_3P.png"), os.path.join(master_dir, "cropped_masterXT13P_mccb.png"))
    ]
    
    for t_path, m_path in tests:
        visual_reference_crop(t_path, m_path)
