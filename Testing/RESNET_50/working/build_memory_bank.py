# build_bank.py — POC Memory Bank Builder for MCCB Defect Detection
# ─────────────────────────────────────────────────────────────────
# Run this ONCE per MCCB model using good/normal reference images.
# Output: a .pkl memory bank used at inspection time.
# ─────────────────────────────────────────────────────────────────

import os
import cv2
import pickle
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

# ── CONFIG ────────────────────────────────────────────────────────
REFERENCE_DIR   = r"reference_images\xt13p"          # Folder of good MCCB images
MASTER_PATH     = r"cropped_master_imaeges\cropped_masterXT13P_mccb.png" # Cropped master reference
OUTPUT_BANK     = r"banks\XT13P.pkl"                 # Output memory bank path
MODEL_ID        = "XT13P"
# ─────────────────────────────────────────────────────────────────


def get_backbone(device):
    """ResNet50 up to Layer3 → (1024, 14, 14) feature maps."""
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


def apply_clahe(img_bgr):
    """Normalize local contrast using CLAHE on L-channel."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    lab = cv2.merge((clahe.apply(l), a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


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
        pt    = kp_query[m.queryIdx].pt
        cell  = (min(int(pt[1] / cell_h), grid-1), min(int(pt[0] / cell_w), grid-1))
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
    h, w   = master.shape[:2]
    score  = float(np.sum(norm(aligned) * norm(master)) / (h * w))
    status = "✅ Good" if score > threshold else "⚠️  Poor"
    print(f"  Alignment NCC: {score:.3f}  ({status})")
    return score > threshold, score


def align_to_master(img_raw, master):
    """Spatially-balanced ORB alignment with direct warpPerspective."""
    img      = cv2.rotate(img_raw, cv2.ROTATE_90_COUNTERCLOCKWISE)  # remove if not needed
    h_m, w_m = master.shape[:2]

    orb = cv2.ORB_create(10000)
    bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    kp_m, des_m = orb.detectAndCompute(master, None)
    kp_t, des_t = orb.detectAndCompute(img, None)

    if des_m is None or des_t is None or len(kp_t) < 10:
        print("  ⚠️  Fallback: insufficient keypoints")
        return cv2.resize(img, (w_m, h_m))

    all_matches = sorted(bf.match(des_m, des_t), key=lambda x: x.distance)

    # ── SPATIAL BALANCING (fixes left-drift) ──────────────
    matches = spatially_balanced_matches(all_matches, kp_m, master.shape, grid=4, per_cell=12)
    # ───────────────────────────────────────────────────────

    if len(matches) < 10:
        print("  ⚠️  Fallback: insufficient balanced matches")
        return cv2.resize(img, (w_m, h_m))

    src_pts = np.float32([kp_m[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_t[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if M is None:
        print("  ⚠️  Fallback: homography failed")
        return cv2.resize(img, (w_m, h_m))

    inliers = inlier_mask.sum() if inlier_mask is not None else 0
    print(f"  Homography inliers: {inliers}/{len(matches)} ({inliers/len(matches):.1%})")

    # ── DIRECT WARP — no crop/resize amplification ────────
    aligned = cv2.warpPerspective(img, M, (w_m, h_m), flags=cv2.WARP_INVERSE_MAP)
    # ───────────────────────────────────────────────────────

    return aligned



def extract_features(img_bgr, backbone, transform, device):
    """Extract 196 patch vectors of 1024-dim from a single image."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor  = transform(img_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = backbone(tensor).squeeze(0).permute(1, 2, 0)  # (14,14,1024)
    h, w, c = feat.shape
    return feat.reshape(-1, c).cpu().numpy(), (h, w)


def build_bank():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*55}")
    print(f"  MEMORY BANK BUILDER  |  {MODEL_ID}  |  {device}")
    print(f"{'='*55}")

    backbone  = get_backbone(device)
    transform = get_transform()

    master = cv2.imread(MASTER_PATH)
    if master is None:
        raise FileNotFoundError(f"❌ Master not found: {MASTER_PATH}")

    files = [f for f in os.listdir(REFERENCE_DIR)
             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not files:
        raise FileNotFoundError(f"❌ No images in: {REFERENCE_DIR}")

    print(f"\n  Reference images : {len(files)}")
    print(f"  Master           : {MASTER_PATH}\n")

    all_features = []
    patch_grid   = None

    for i, fname in enumerate(files):
        path    = os.path.join(REFERENCE_DIR, fname)
        img_raw = cv2.imread(path)
        if img_raw is None:
            print(f"  ⚠️  [{i+1}/{len(files)}] Skipped (unreadable): {fname}")
            continue

        aligned = align_to_master(img_raw, master)

        # ── NCC QUALITY GATE — skip bad alignments ────────
        is_good, ncc = verify_alignment(aligned, master, threshold=0.4)
        if not is_good:
            print(f"  ⚠️  [{i+1}/{len(files)}] SKIPPED poor alignment (NCC={ncc:.3f}): {fname}")
            continue
        # ─────────────────────────────────────────────────

        aligned_clahe     = apply_clahe(aligned)
        feats, patch_grid = extract_features(aligned_clahe, backbone, transform, device)
        all_features.append(feats)
        print(f"  ✅  [{i+1}/{len(files)}] {fname}  →  {feats.shape[0]} patches  NCC={ncc:.3f}")

    if not all_features:
        raise RuntimeError("❌ No features extracted. Check your images.")

    # Stack and normalize for cosine similarity at inference
    bank      = np.vstack(all_features)                          # (N*196, 1024)
    norms     = np.linalg.norm(bank, axis=1, keepdims=True)
    norms[norms == 0] = 1
    bank_norm = bank / norms

    os.makedirs(os.path.dirname(OUTPUT_BANK) if os.path.dirname(OUTPUT_BANK) else ".", exist_ok=True)

    with open(OUTPUT_BANK, "wb") as f:
        pickle.dump({
            "memory_bank_normalized": bank_norm,
            "patch_grid": patch_grid,
            "model_id":   MODEL_ID,
            "n_images":   len(all_features),
        }, f)

    size_mb = os.path.getsize(OUTPUT_BANK) / (1024 * 1024)
    print(f"\n  Memory bank : {bank_norm.shape}  |  {size_mb:.1f} MB")
    print(f"  Saved to    : {OUTPUT_BANK}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    build_bank()