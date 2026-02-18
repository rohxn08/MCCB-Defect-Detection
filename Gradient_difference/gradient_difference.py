import cv2
import matplotlib.pyplot as plt

def gradient_magnitude(image):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sobelx, sobely)
    magnitude = cv2.convertScaleAbs(magnitude)
    return magnitude

def gradient_difference(master_img_path, test_img_path, image_id):
    # Read the master and test images
    master_img = cv2.imread(master_img_path)
    test_img = cv2.imread(test_img_path)

    # Basic error handling
    if master_img is None:
        print(f"Error: Could not read master image at {master_img_path}")
        return
    if test_img is None:
        print(f"Error: Could not read test image at {test_img_path}")
        return

    # Warn if images are not the same size
    master_h, master_w = master_img.shape[:2]
    test_h, test_w = test_img.shape[:2]
    if (master_h, master_w) != (test_h, test_w):
        print(f"WARNING: Image size mismatch! Master: ({master_w}x{master_h}), Test: ({test_w}x{test_h})")
        print("Images must be the same size for gradient difference. Aborting.")
        return

    # Convert the same to grayscale
    master_gray = cv2.cvtColor(master_img, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    # Applying Gaussian Blur to remove noise
    master_blur = cv2.GaussianBlur(master_gray, (5, 5), 0)
    test_blur = cv2.GaussianBlur(test_gray, (5, 5), 0)

    # Applying CLAHE onto the master and test images
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    master_clahe = clahe.apply(master_blur)
    test_clahe = clahe.apply(test_blur)

    # Gradient magnitude
    master_grad = gradient_magnitude(master_clahe)
    test_grad = gradient_magnitude(test_clahe)

    # Differences of the gradients between the master and the test images
    diff_grad = cv2.subtract(master_grad, test_grad)

    # Thresholding the difference image
    _, thresh = cv2.threshold(diff_grad, 30, 255, cv2.THRESH_BINARY)
    thresh = cv2.medianBlur(thresh, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    output = test_img.copy() 
    
    for i in contours:
        area = cv2.contourArea(i)
        x, y, w, h = cv2.boundingRect(i)
        aspect_ratio = w / float(h)
        if area > 200:
            if 0.2 < aspect_ratio < 5.0:
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # VISUALIZATION FIXES
    # Convert BGR to RGB for correct execution in Matplotlib
    master_rgb = cv2.cvtColor(master_img, cv2.COLOR_BGR2RGB)
    test_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 8))
    
    ax1.imshow(master_rgb)
    ax1.set_title("Master Image") 
    ax1.axis("off")

    ax2.imshow(test_rgb)
    ax2.set_title("Test Image")
    ax2.axis("off")

    ax3.imshow(diff_grad, cmap='gray') 
    ax3.set_title("Difference Image")
    ax3.axis("off")

    ax4.imshow(output_rgb)
    ax4.set_title("Output Image")
    ax4.axis("off")

    plt.tight_layout()
    
    # Save to output folder
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    save_name = f"{image_id}.png"
    save_path = os.path.join(output_dir, save_name)
    plt.savefig(save_path)
    print(f"Result saved to: {save_path}")
    
    plt.show()

def main():
    master_img=input("Enter the master image path: ")
    test_img=input("Enter the test image path: ")
    test_id=input("Enter the output image ID/Name (without extension): ")
    gradient_difference(master_img, test_img, test_id)

if __name__ == "__main__":
    main()

