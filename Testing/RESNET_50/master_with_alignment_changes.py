# inspect.py — POC MCCB Defect Inspection using PatchCore
# ─────────────────────────────────────────────────────────
# Usage: python inspect.py
# Requires: memory bank built with build_bank.py first.
# ─────────────────────────────────────────────────────────

import os
import cv2
import pickle
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from datetime import datetime

# ── CONFIG ────────────────────────────────────────────────
MASTER_PATH  = r"cropped_master_imaeges\cropped_masterXT13P_mccb.png"
BANK_PATH    = r"banks\XT1_3P.pkl"
TEST_IMAGE   = r"Testing_images\CG36355365067392.png"
OUTPUT_DIR   = r"output"

THRESHOLD    = 0.20  # ← Tune this. Lower = more sensitive.
                      #   Good images typically score 0.10–0.20
                      #   Defective images score 0.25+
# MIN_DEFECT_AREA = 80  # Minimum contour area to count as defect (pixels²)
# ─────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════
# 1. BACKBONE
# ══════════════════════════════════════════════════════════

def get_backbone(device):
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    backbone = nn.Sequential(
        resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
        resnet.layer1, resnet.layer2, resnet.layer3,
    ).eval().requires_grad_(False).to(device)
    return backbone


def get_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


# ══════════════════════════════════════════════════════════
# 2. CLAHE
# ══════════════════════════════════════════════════════════

def apply_clahe(img_bgr):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    return cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)


# ══════════════════════════════════════════════════════════
# 3. ALIGNMENT
# ══════════════════════════════════════════════════════════

def align_to_master(img_raw, master):
    """ORB-based alignment + fine registration to master size."""
    img = cv2.rotate(img_raw, cv2.ROTATE_90_COUNTERCLOCKWISE)

    orb = cv2.ORB_create(10000)
    bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    h_m, w_m = master.shape[:2]

    kp_m, des_m = orb.detectAndCompute(master, None)
    kp_t, des_t = orb.detectAndCompute(img, None)

    if des_m is None or des_t is None or len(des_t) < 10:
        print("  ⚠️  Alignment fallback: insufficient keypoints")
        return cv2.resize(img, (w_m, h_m))

    matches = sorted(bf.match(des_m, des_t), key=lambda x: x.distance)[:200]
    if len(matches) < 10:
        print("  ⚠️  Alignment fallback: insufficient matches")
        return cv2.resize(img, (w_m, h_m))

    src_pts = np.float32([kp_m[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp_t[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    M, _    = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if M is None:
        print("  ⚠️  Alignment fallback: homography failed")
        return cv2.resize(img, (w_m, h_m))

    corners   = np.float32([[0,0],[0,h_m-1],[w_m-1,h_m-1],[w_m-1,0]]).reshape(-1,1,2)
    t_corners = cv2.perspectiveTransform(corners, M)
    h_t, w_t  = img.shape[:2]
    x1 = max(0, int(t_corners[:,0,0].min()))
    y1 = max(0, int(t_corners[:,0,1].min()))
    x2 = min(w_t, int(t_corners[:,0,0].max()))
    y2 = min(h_t, int(t_corners[:,0,1].max()))

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return cv2.resize(img, (w_m, h_m))

    # Fine registration
    kp_c, des_c = orb.detectAndCompute(crop, None)
    if des_c is not None and len(des_c) >= 10:
        matches_fine = sorted(bf.match(des_c, des_m), key=lambda x: x.distance)[:200]
        if len(matches_fine) >= 10:
            pts_c = np.float32([kp_c[m.queryIdx].pt for m in matches_fine]).reshape(-1,1,2)
            pts_m = np.float32([kp_m[m.trainIdx].pt for m in matches_fine]).reshape(-1,1,2)
            H_fine, _ = cv2.findHomography(pts_c, pts_m, cv2.RANSAC, 5.0)
            if H_fine is not None:
                return cv2.warpPerspective(crop, H_fine, (w_m, h_m))

    return cv2.resize(crop, (w_m, h_m))


# ══════════════════════════════════════════════════════════
# 4. PATCHCORE DETECTION  (updated — patch-level border masking)
# ══════════════════════════════════════════════════════════

@torch.no_grad()
def detect(img_bgr, memory_bank, patch_grid, device, backbone, transform):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor  = transform(img_rgb).unsqueeze(0).to(device)
    feat    = backbone(tensor).squeeze(0).permute(1, 2, 0)
    h_f, w_f, c_f = feat.shape
    H, W = img_bgr.shape[:2]

    MIN_DEFECT_AREA = int((H * W) * 0.001)
    MAX_DEFECT_AREA = int((H * W) * 0.08)

    test_vecs = feat.reshape(-1, c_f)
    test_vecs = test_vecs / (torch.norm(test_vecs, dim=1, keepdim=True) + 1e-6)

    sims   = test_vecs @ memory_bank.T
    scores = (1.0 - sims.max(dim=1)[0]).cpu().numpy()

    anomaly_grid = scores.reshape(h_f, w_f)

    # ── PATCH-LEVEL BORDER MASKING ─────────────────────────
    # Each patch ≈ 16px at 224 scale → 2 patches ≈ 32px
    # Covers the 2-3mm mechanical jig shift causing edge false positives
    # Increase BORDER_PATCHES to 3 if edges still fire
    BORDER_PATCHES = 2
    anomaly_grid[:BORDER_PATCHES, :]   = 0   # top
    anomaly_grid[-BORDER_PATCHES:, :]  = 0   # bottom
    anomaly_grid[:, :BORDER_PATCHES]   = 0   # left
    anomaly_grid[:, -BORDER_PATCHES:]  = 0   # right
    # ───────────────────────────────────────────────────────

    smooth       = cv2.GaussianBlur(anomaly_grid.astype(np.float32), (3, 3), 0)
    heatmap_full = cv2.resize(smooth, (W, H), interpolation=cv2.INTER_LINEAR)

    # Color heatmap
    heatmap_vis = cv2.applyColorMap(
        cv2.normalize(heatmap_full, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        cv2.COLORMAP_JET
    )

    # ── Extract DARK RED regions from heatmap ──────────────
    heatmap_hsv = cv2.cvtColor(heatmap_vis, cv2.COLOR_BGR2HSV)

    lower_dark_red1 = np.array([0,   200, 180])
    upper_dark_red1 = np.array([8,   255, 255])
    lower_dark_red2 = np.array([172, 200, 180])
    upper_dark_red2 = np.array([180, 255, 255])

    mask_r1  = cv2.inRange(heatmap_hsv, lower_dark_red1, upper_dark_red1)
    mask_r2  = cv2.inRange(heatmap_hsv, lower_dark_red2, upper_dark_red2)
    red_mask = cv2.bitwise_or(mask_r1, mask_r2)

    # Pixel-level edge exclusion (secondary safety net)
    border = int(min(H, W) * 0.03)  # increased from 0.01 to 0.03
    red_mask[:border, :]   = 0
    red_mask[H-border:, :] = 0
    red_mask[:, :border]   = 0
    red_mask[:, W-border:] = 0

    # ── Draw bounding boxes on output image ────────────────
    out_img  = img_bgr.copy()
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    defect_count = 0

    print(f"  Total red contours: {len(contours)}")
    for c in contours:
        area = cv2.contourArea(c)
        x, y, w_b, h_b = cv2.boundingRect(c)
        aspect_ratio = max(w_b, h_b) / (min(w_b, h_b) + 1e-6)
        print(f"    Contour → x={x:4d}  y={y:4d}  w={w_b:4d}  h={h_b:4d}  area={area:.0f}")
        if area > MIN_DEFECT_AREA and area < MAX_DEFECT_AREA and aspect_ratio < 3.0:
            region_score = heatmap_full[y:y+h_b, x:x+w_b].max()
            cv2.rectangle(out_img, (x, y), (x+w_b, y+h_b), (0, 255, 0), 6)
            cv2.putText(out_img, f"DEFECT {region_score:.2f}",
                        (x, max(y-12, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            defect_count += 1

    return {
        "score":   float(np.max(scores)),  # uses pre-mask scores for overall score
        "heatmap": heatmap_vis,
        "output":  out_img,
        "defects": defect_count,
    }
# ══════════════════════════════════════════════════════════
# 5. MAIN
# ══════════════════════════════════════════════════════════

def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*55}")
    print(f"  MCCB INSPECTION  |  {device}")
    print(f"{'='*55}")

    # Load master
    master = cv2.imread(MASTER_PATH)
    if master is None:
        raise FileNotFoundError(f"❌ Master not found: {MASTER_PATH}")

    # Load memory bank
    if not os.path.exists(BANK_PATH):
        raise FileNotFoundError(f"❌ Bank not found: {BANK_PATH}\n   Run build_bank.py first!")

    with open(BANK_PATH, "rb") as f:
        data = pickle.load(f)
    memory_bank = torch.from_numpy(data["memory_bank_normalized"]).float().to(device)
    patch_grid  = data.get("patch_grid", (14, 14))
    print(f"  ✅ Memory bank loaded  ({len(memory_bank)} patches)")

    # Load test image
    test_img = cv2.imread(TEST_IMAGE)
    if test_img is None:
        raise FileNotFoundError(f"❌ Test image not found: {TEST_IMAGE}")

    # Pipeline
    print(f"\n  1. Aligning to master...")
    aligned = align_to_master(test_img, master)

    print(f"  2. Applying CLAHE...")
    aligned_clahe = apply_clahe(aligned)
    master_clahe  = apply_clahe(master)

    print(f"  3. Running PatchCore detection...")
    backbone  = get_backbone(device)
    transform = get_transform()
    result    = detect(aligned_clahe, memory_bank, patch_grid, device, backbone, transform)

    # Result
    status = "PASS" if result["defects"] == 0 else "FAIL"
    color  = "green" if status == "PASS" else "red"
    print(f"\n  Result   : {status}")
    print(f"  Score    : {result['score']:.4f}  (threshold: {THRESHOLD})")
    print(f"  Defects  : {result['defects']}")

    # Dashboard
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUTPUT_DIR, f"result_{status}_{ts}.png")

    fig, axes = plt.subplots(2, 2, figsize=(20, 10))
    axes[0,0].imshow(cv2.cvtColor(master_clahe,       cv2.COLOR_BGR2RGB)); axes[0,0].set_title("Master Reference");         axes[0,0].axis("off")
    axes[0,1].imshow(cv2.cvtColor(aligned_clahe,      cv2.COLOR_BGR2RGB)); axes[0,1].set_title("Aligned Input");             axes[0,1].axis("off")
    axes[1,0].imshow(cv2.cvtColor(result["heatmap"],  cv2.COLOR_BGR2RGB)); axes[1,0].set_title(f"Anomaly Heatmap  (score: {result['score']:.3f})"); axes[1,0].axis("off")
    axes[1,1].imshow(cv2.cvtColor(result["output"],   cv2.COLOR_BGR2RGB)); axes[1,1].set_title(f"Detection  ({result['defects']} defects)");         axes[1,1].axis("off")

    fig.suptitle(
        f"INSPECTION: {status}  |  SCORE: {result['score']:.3f}  |  THRESHOLD: {THRESHOLD}",
        fontsize=20, fontweight="bold", color=color
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\n  ✅ Dashboard saved → {out_path}")
    plt.show()
    print(f"{'='*55}\n")


if __name__ == "__main__":
    run()