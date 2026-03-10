import cv2
import numpy as np

def align_images(im1, im2, max_features=5000, keep_percent=0.2):
    """
    Aligns im2 to match the perspective and position of im1 using ORB feature matching.
    
    Args:
        im1 (numpy.ndarray): The reference image (Master Image).
        im2 (numpy.ndarray): The input image to be aligned (Defective Image).
        max_features (int): Maximum number of features to extract.
        keep_percent (float): Fraction of best matches to keep for calculating homography.
        
    Returns:
        numpy.ndarray: The aligned version of im2, perfectly matching im1.
    """
    # 1. Convert images to grayscale
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # 2. Detect ORB features and compute descriptors
    orb = cv2.ORB_create(max_features)
    keypoints1, descriptors1 = orb.detectAndCompute(im1_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2_gray, None)

    # 3. Match features
    # Use Hamming distance for ORB descriptors
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors2, descriptors1, None)

    # 4. Sort matches by score (distance)
    matches = sorted(matches, key=lambda x: x.distance)

    # 5. Keep only the top matches to define the transformation
    # E.g. keeping top 20% of features to remove noise and bad matches
    keep = int(len(matches) * keep_percent)
    matches = matches[:keep]

    # 6. Extract the physical (x, y) coordinates of those matched features
    pts1 = np.zeros((len(matches), 2), dtype=np.float32)
    pts2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        pts2[i, :] = keypoints2[match.queryIdx].pt
        pts1[i, :] = keypoints1[match.trainIdx].pt

    # 7. Compute the Homography matrix (the mathematical transformation map)
    # RANSAC isolates the actual physical movement and throws out crazy statistical outliers
    h, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC)

    # 8. Use the Matrix to warp the Defective Image exactly over the Master Image
    height, width, channels = im1.shape
    im2_aligned = cv2.warpPerspective(im2, h, (width, height))

    return im2_aligned

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import os

    # We use paths similar to what you had in your Defect_OCR.py for testing
    master_path = r"C:\ALL PROJECTS\MCCB-Defect-Detection\Testing\ORB_Testing_images\CG36350207067415.png"
    test_path = r"C:\ALL PROJECTS\MCCB-Defect-Detection\Testing\ORB_Testing_images\CG36350161067415.png"

    # Fallback to local paths if absolute paths fail
    if not os.path.exists(master_path):
        master_path = r"masterImages/masterP14P_mccb.png"
    if not os.path.exists(test_path):
        test_path = r"masterImages/masterP13P_mccb.png"

    print(f"Loading Master: {master_path}")
    print(f"Loading Test: {test_path}")

    master_img = cv2.imread(master_path)
    test_img = cv2.imread(test_path)

    if master_img is not None and test_img is not None:
        print("Calculating ORB alignment... (This might take a second)")
        # Align the test image to perfectly match the master image
        aligned_img = align_images(master_img, test_img)
        
        # Save the aligned image to disk
        aligned_filename = f"aligned_{os.path.basename(test_path)}"
        aligned_filepath = os.path.join(os.path.dirname(test_path), aligned_filename)
        cv2.imwrite(aligned_filepath, aligned_img)
        print(f"Alignment complete! Saved aligned image to: {aligned_filepath}")
        
        # Create a beautiful 1x3 comparison plot
        plt.figure(figsize=(18, 6))
        
        # Plot 1: The perfect Master Reference
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(master_img, cv2.COLOR_BGR2RGB))
        plt.title("1. Master Image (Reference)", fontsize=14, fontweight='bold')
        plt.axis('off')
        
        # Plot 2: The raw Test Image (Notice how it might be slightly shifted/rotated)
        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
        plt.title("2. Raw Test Image (Misaligned)", fontsize=14, fontweight='bold')
        plt.axis('off')
        
        # Plot 3: The ORB Aligned Image (Snaps identically to the Master)
        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB))
        plt.title("3. ORB Aligned Image (Ready for SSIM)", fontsize=14, fontweight='bold', color='green')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    else:
        print("❌ Could not load one or both images. Please check the paths.")
