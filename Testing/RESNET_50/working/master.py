# inspect.py — POC MCCB Defect Inspection using PatchCore
# ─────────────────────────────────────────────────────────
# Usage: python inspect.py
# Requires: memory bank built with build_bank.py first.
# ─────────────────────────────────────────────────────────

import os
import sys
import cv2
import pickle
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from datetime import datetime
import faiss

# Fix Windows cp1252 terminal choking on emoji characters
sys.stdout.reconfigure(encoding='utf-8')

# ── CONFIG ────────────────────────────────────────────────
MASTER_PATH  = r"cropped_master_imaeges\cropped_masterXT13P_mccb.png"
BANK_PATH    = r"banks\XT13P.pkl"           # used for metadata (patch_grid) only
FAISS_PATH   = r"faiss\XT13P.index"         # FAISS coreset index for fast search
TEST_IMAGE   = r"Testing_images\CG36355374067392.png"
OUTPUT_DIR   = r"070426\xt13p_r"

THRESHOLD    = 0.20  # ← Tune this. Lower = more sensitive.
                      #   Good images typically score 0.10–0.20
                      #   Defective images score 0.25+
# MIN_DEFECT_AREA = 80  # Minimum contour area to count as defect (pixels²)
# ─────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════
# 1. BACKBONE
# ══════════════════════════════════════════════════════════


#   Resnet 50 model 
# def get_backbone(device):
#     resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
#     backbone = nn.Sequential(
#         resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
#         resnet.layer1, resnet.layer2, resnet.layer3,
#     ).eval().requires_grad_(False).to(device)
#     return backbone


#   Wide resnet50 model
def get_backbone(device):
    wide_resnet = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
    backbone = nn.Sequential(
        wide_resnet.conv1, wide_resnet.bn1, wide_resnet.relu, wide_resnet.maxpool,
        wide_resnet.layer1, wide_resnet.layer2, wide_resnet.layer3,
    ).eval().requires_grad_(False).to(device)
    return backbone


def get_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((320, 320)),   # must match build_memory_bank.py
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

def spatially_balanced_matches(matches, kp_query, img_shape, grid=4, per_cell=12):
    """Distribute matches evenly across spatial grid to prevent drift."""
    h, w = img_shape[:2]
    cell_h, cell_w = h / grid, w / grid
    from collections import defaultdict
    cells = defaultdict(list)
    for m in matches:
        pt   = kp_query[m.queryIdx].pt
        cell = (min(int(pt[1] / cell_h), grid-1), min(int(pt[0] / cell_w), grid-1))
        cells[cell].append(m)
    selected = []
    for cell_matches in cells.values():
        selected.extend(sorted(cell_matches, key=lambda x: x.distance)[:per_cell])
    return selected


def verify_alignment(aligned, master, threshold=0.5):
    """NCC-based alignment quality check."""
    def norm(img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        g -= g.mean(); g /= (g.std() + 1e-6)
        return g
    h, w  = master.shape[:2]
    score = float(np.sum(norm(aligned) * norm(master)) / (h * w))
    status = "✅ Good" if score > threshold else "⚠️  Poor"
    print(f"  Alignment NCC: {score:.3f}  ({status})")
    return score > threshold, score


def align_to_master(img_raw, master):
    img      = cv2.rotate(img_raw, cv2.ROTATE_90_COUNTERCLOCKWISE)  # match inspect.py
    h_m, w_m = master.shape[:2]

    orb = cv2.ORB_create(10000)
    bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    kp_m, des_m = orb.detectAndCompute(master, None)
    kp_t, des_t = orb.detectAndCompute(img, None)

    if des_m is None or des_t is None or len(kp_t) < 10:
        print("  ⚠️  Not enough keypoints — returning resized")
        return cv2.resize(img, (w_m, h_m))

    all_matches = sorted(bf.match(des_m, des_t), key=lambda x: x.distance)
    matches     = spatially_balanced_matches(all_matches, kp_m, master.shape, grid=4, per_cell=12)

    if len(matches) < 10:
        print("  ⚠️  Not enough balanced matches — returning resized")
        return cv2.resize(img, (w_m, h_m))

    src_pts = np.float32([kp_m[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_t[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if M is None:
        print("  ⚠️  Homography failed — returning resized")
        return cv2.resize(img, (w_m, h_m))

    inliers = inlier_mask.sum() if inlier_mask is not None else 0
    print(f"    Inliers: {inliers}/{len(matches)} ({inliers/len(matches):.1%})")

    return cv2.warpPerspective(img, M, (w_m, h_m), flags=cv2.WARP_INVERSE_MAP)
# ══════════════════════════════════════════════════════════
# 4. PATCHCORE DETECTION
# ══════════════════════════════════════════════════════════

@torch.no_grad()
def detect(img_bgr, faiss_index, patch_grid, device, backbone, transform):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor  = transform(img_rgb).unsqueeze(0).to(device)
    feat    = backbone(tensor).squeeze(0).permute(1, 2, 0)
    h_f, w_f, c_f = feat.shape
    H, W = img_bgr.shape[:2]

    MIN_DEFECT_AREA = int((H * W) * 0.001)
    MAX_DEFECT_AREA = int((H * W) * 0.08)

    test_vecs = feat.reshape(-1, c_f)
    test_vecs = test_vecs / (torch.norm(test_vecs, dim=1, keepdim=True) + 1e-6)
    test_np   = test_vecs.cpu().numpy()              # FAISS requires numpy float32

    # FAISS k=1 nearest-neighbour search (inner product = cosine for normalised vecs)
    sims, _  = faiss_index.search(test_np, k=1)      # sims: (N_patches, 1)
    scores   = (1.0 - sims[:, 0])                     # anomaly score: 0=normal, 1=defect

    anomaly_grid = scores.reshape(h_f, w_f)
    smooth       = cv2.GaussianBlur(anomaly_grid.astype(np.float32), (3, 3), 0)
    heatmap_full = cv2.resize(smooth, (W, H), interpolation=cv2.INTER_LINEAR)

    # ── Visualization heatmap (for dashboard only — NOT used for detection) ──
    heatmap_norm = cv2.normalize(heatmap_full, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_vis  = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)

    # ── Direct score threshold ─────────────────────────────────────
    thresh_map = (heatmap_full > THRESHOLD).astype(np.uint8) * 255

    # ── Exclusion zones — variable text regions that are never defects ──
    # These areas contain unit-specific ratings/serial numbers that differ
    # across MCCBs and will always produce high anomaly scores.
    # Zones are expressed as fractions of (H, W) so they scale with any image size.
    exclusion_zones = [
        # (y_start_frac, y_end_frac, x_start_frac, x_end_frac)
        (0.10, 0.75, 0.00, 0.12),   # Left ratings panel  (MIN/MED/MAX, In=, I3)
        (0.80, 1.00, 0.00, 0.15),   # TMD label            (bottom-left text)
        (0.00, 0.92, 0.82, 1.00),   # Right specs panel    (S/N, voltage table)
    ]
    for (ys, ye, xs, xe) in exclusion_zones:
        thresh_map[int(ys*H):int(ye*H), int(xs*W):int(xe*W)] = 0

    # Edge exclusion
    border = int(min(H, W) * 0.01)
    thresh_map[:border, :]   = 0
    thresh_map[H-border:, :] = 0
    thresh_map[:, :border]   = 0
    thresh_map[:, W-border:] = 0

    # Morphological cleanup — 5×5 so thin low-severity blobs survive
    kernel     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh_map = cv2.morphologyEx(thresh_map, cv2.MORPH_CLOSE, kernel)
    thresh_map = cv2.morphologyEx(thresh_map, cv2.MORPH_OPEN,  kernel)

    # ── Draw bounding boxes ────────────────────────────────────────
    out_img  = img_bgr.copy()

    # Draw exclusion zones in blue on output for visibility
    for (ys, ye, xs, xe) in exclusion_zones:
        cv2.rectangle(out_img,
                      (int(xs*W), int(ys*H)), (int(xe*W), int(ye*H)),
                      (255, 140, 0), 3)

    contours, _ = cv2.findContours(thresh_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    defect_count = 0

    print(f"  Total anomaly contours: {len(contours)}")
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
        "score":   float(np.max(scores)),
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

    # Load FAISS index (coreset bank)
    if not os.path.exists(FAISS_PATH):
        raise FileNotFoundError(f"❌ FAISS index not found: {FAISS_PATH}\n   Run build_memory_bank.py first!")
    faiss_index = faiss.read_index(FAISS_PATH)
    print(f"  ✅ FAISS index loaded  ({faiss_index.ntotal} coreset patches)")

    # Load pkl for metadata only
    with open(BANK_PATH, "rb") as f:
        data = pickle.load(f)
    patch_grid = data.get("patch_grid", (14, 14))

    # Load test image
    test_img = cv2.imread(TEST_IMAGE)
    if test_img is None:
        raise FileNotFoundError(f"❌ Test image not found: {TEST_IMAGE}")

    # Pipeline
    print(f"\n  1. Aligning to master...")
    aligned = align_to_master(test_img, master)
    is_good, ncc_score = verify_alignment(aligned, master)
    if not is_good:
        print(f"  ⚠️  Poor alignment (NCC={ncc_score:.3f}) — result may be unreliable")

    print(f"  2. Applying CLAHE...")
    aligned_clahe = apply_clahe(aligned)
    master_clahe  = apply_clahe(master)

    print(f"  3. Running PatchCore detection...")
    backbone  = get_backbone(device)
    transform = get_transform()
    result    = detect(aligned_clahe, faiss_index, patch_grid, device, backbone, transform)

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