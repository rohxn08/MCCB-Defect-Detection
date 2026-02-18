"""
Threshold Tuner for Gradient Difference Pipeline
=================================================
Uses Matplotlib sliders (works with opencv-python-headless).

Adjust the sliders and observe the results in real-time.
Close the window — the final slider values will be printed
so you can plug them into gradient_difference.py.

Usage:
    python threshold_tuner.py
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


# ─── HELPER ───────────────────────────────────────────────
def gradient_magnitude(image):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sobelx, sobely)
    magnitude = cv2.convertScaleAbs(magnitude)
    return magnitude


def preprocess(img):
    """Grayscale → Blur → CLAHE → Gradient"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blur)
    grad = gradient_magnitude(enhanced)
    return grad


def process_and_draw(test_img, diff_grad, bin_thresh, med_blur_k, morph_k, min_area, ar_min, ar_max):
    """Apply thresholds and return output image, threshold mask, and defect count."""
    # Thresholding
    _, thresh = cv2.threshold(diff_grad, bin_thresh, 255, cv2.THRESH_BINARY)
    
    # Median blur (must be odd and >= 3)
    med_k = int(med_blur_k)
    if med_k % 2 == 0:
        med_k += 1
    med_k = max(med_k, 3)
    thresh = cv2.medianBlur(thresh, med_k)
    
    # Morphological closing
    mk = max(int(morph_k), 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (mk, mk))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find & filter contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = test_img.copy()
    defect_count = 0

    for c in contours:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = w / float(h) if h > 0 else 0
        if area > min_area and ar_min < aspect_ratio < ar_max:
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
            defect_count += 1

    # Convert BGR → RGB for matplotlib
    output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    return output_rgb, thresh, defect_count


# ─── MAIN ─────────────────────────────────────────────────
def main():
    master_path = input("Enter the master image path: ").strip()
    test_path   = input("Enter the test image path: ").strip()

    master_img = cv2.imread(master_path)
    test_img   = cv2.imread(test_path)

    if master_img is None:
        print(f"Error: Could not read master image at {master_path}")
        return
    if test_img is None:
        print(f"Error: Could not read test image at {test_path}")
        return

    if master_img.shape[:2] != test_img.shape[:2]:
        print("WARNING: Image sizes don't match!")
        return

    # Preprocess once
    master_grad = preprocess(master_img)
    test_grad   = preprocess(test_img)
    diff_grad   = cv2.absdiff(master_grad, test_grad)

    # Initial values
    init_thresh   = 30
    init_med_blur = 5
    init_morph    = 15
    init_area     = 200
    init_ar_min   = 0.2
    init_ar_max   = 5.0

    # Initial processing
    output_rgb, thresh_mask, defect_count = process_and_draw(
        test_img, diff_grad, init_thresh, init_med_blur, init_morph,
        init_area, init_ar_min, init_ar_max
    )

    # ─── Setup figure ────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    plt.subplots_adjust(bottom=0.40)  # Make room for sliders

    img_plot   = ax1.imshow(output_rgb)
    ax1.set_title(f"Detections (Defects: {defect_count})")
    ax1.axis("off")

    thresh_plot = ax2.imshow(thresh_mask, cmap='gray')
    ax2.set_title("Threshold Mask")
    ax2.axis("off")

    # ─── Create sliders ──────────────────────────────────
    slider_color = '#4CAF50'

    ax_thresh   = plt.axes([0.15, 0.28, 0.70, 0.03])
    ax_med_blur = plt.axes([0.15, 0.23, 0.70, 0.03])
    ax_morph    = plt.axes([0.15, 0.18, 0.70, 0.03])
    ax_area     = plt.axes([0.15, 0.13, 0.70, 0.03])
    ax_ar_min   = plt.axes([0.15, 0.08, 0.70, 0.03])
    ax_ar_max   = plt.axes([0.15, 0.03, 0.70, 0.03])

    s_thresh   = Slider(ax_thresh,   'Bin Threshold', 0, 255, valinit=init_thresh, valstep=1, color=slider_color)
    s_med_blur = Slider(ax_med_blur, 'Med Blur K',    3, 25,  valinit=init_med_blur, valstep=2, color=slider_color)
    s_morph    = Slider(ax_morph,    'Morph Kernel',   1, 50,  valinit=init_morph, valstep=1, color=slider_color)
    s_area     = Slider(ax_area,     'Min Area',       0, 5000, valinit=init_area, valstep=10, color=slider_color)
    s_ar_min   = Slider(ax_ar_min,   'AR Min',         0.1, 5.0, valinit=init_ar_min, valstep=0.1, color=slider_color)
    s_ar_max   = Slider(ax_ar_max,   'AR Max',         0.5, 10.0, valinit=init_ar_max, valstep=0.1, color=slider_color)

    # ─── Update function ─────────────────────────────────
    def update(val):
        output_rgb, thresh_mask, defect_count = process_and_draw(
            test_img, diff_grad,
            int(s_thresh.val),
            int(s_med_blur.val),
            int(s_morph.val),
            int(s_area.val),
            s_ar_min.val,
            s_ar_max.val
        )
        img_plot.set_data(output_rgb)
        thresh_plot.set_data(thresh_mask)
        ax1.set_title(f"Detections (Defects: {defect_count})")
        fig.canvas.draw_idle()

    # Connect sliders to update function
    s_thresh.on_changed(update)
    s_med_blur.on_changed(update)
    s_morph.on_changed(update)
    s_area.on_changed(update)
    s_ar_min.on_changed(update)
    s_ar_max.on_changed(update)

    print("\n" + "="*50)
    print("  THRESHOLD TUNER — Adjust sliders in the window")
    print("  Close the window when done to see final values")
    print("="*50 + "\n")

    plt.show()

    # ─── Print final values after window is closed ───────
    final_thresh   = int(s_thresh.val)
    final_med_blur = int(s_med_blur.val)
    final_morph    = int(s_morph.val)
    final_area     = int(s_area.val)
    final_ar_min   = round(s_ar_min.val, 1)
    final_ar_max   = round(s_ar_max.val, 1)

    print("\n" + "="*50)
    print("  FINAL THRESHOLD VALUES")
    print("="*50)
    print(f"  Binary Threshold : {final_thresh}")
    print(f"  Median Blur K    : {final_med_blur}")
    print(f"  Morph Kernel     : ({final_morph}, {final_morph})")
    print(f"  Min Contour Area : {final_area}")
    print(f"  Aspect Ratio     : {final_ar_min} – {final_ar_max}")
    print("="*50)
    print("\nCopy these values into gradient_difference.py!")


if __name__ == "__main__":
    main()
