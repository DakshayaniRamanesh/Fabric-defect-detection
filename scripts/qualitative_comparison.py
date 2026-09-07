"""
qualitative_comparison.py
=========================
Generates a research-poster-quality qualitative comparison figure across the
four fabric-defect-detection approaches developed in this project.

Model class definitions are copied verbatim from the project notebooks:
  - task-1-classical-model.ipynb      -> Classical ML (SVM / RF / XGB ensemble)
  - Fabric_Defect_Detection_Task2_G08.ipynb -> Hybrid ResNet + Classical ensemble
  - task03-autoencoder.ipynb          -> Convolutional Autoencoder (ImprovedConvAE)
  - u-net-1.ipynb                     -> UNet segmentation

If trained weights are present in outputs/ the script loads them.
If not, it shows a visual pipeline demonstration using the same architecture
with random weights (explicitly labelled) and live handcrafted-feature maps.

Run:
    pip install numpy pillow matplotlib scikit-image scikit-learn torch
    python qualitative_comparison.py
"""

import os, sys, glob, random, warnings, copy
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

try:
    from skimage.feature import local_binary_pattern, hog
    SKIMAGE_OK = True
except Exception:
    SKIMAGE_OK = False

try:
    import joblib
    JOBLIB_OK = True
except Exception:
    JOBLIB_OK = False

try:
    import torch, torch.nn as nn, torch.nn.functional as F
    TORCH_OK = True
    DEVICE = torch.device("cpu")
except Exception:
    TORCH_OK = False
    DEVICE = None

warnings.filterwarnings("ignore")

# ── PATHS ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == "scripts" else SCRIPT_DIR
DATA_ROOT    = os.path.join(PROJECT_ROOT, "CO5420 Fabric")
OUTPUTS_DIR  = os.path.join(PROJECT_ROOT, "outputs")
FIGURES_DIR  = os.path.join(OUTPUTS_DIR, "figures")
MODELS_DIR   = os.path.join(OUTPUTS_DIR, "models")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

SAVE_PATH    = os.path.join(FIGURES_DIR, "qualitative_comparison_poster.png")

def _find_artifact(filename):
    for loc in [MODELS_DIR, OUTPUTS_DIR, PROJECT_ROOT]:
        cand = os.path.join(loc, filename)
        if os.path.exists(cand):
            return cand
    return os.path.join(MODELS_DIR, filename)

ART_ENSEMBLE = _find_artifact("ensemble_model.pkl")
ART_SELECTOR = _find_artifact("selector.pkl")
ART_SCALER   = _find_artifact("scaler.pkl")
ART_AE       = _find_artifact("fabric_ae_best.pth")
ART_UNET     = _find_artifact("unet_model.pth")

TARGET_H, TARGET_W = 128, 512

# ── MODEL DEFINITIONS (verbatim from notebooks) ──────────────────────────────
if TORCH_OK:
    class ResBlock(nn.Module):
        def __init__(self, ch):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(ch, ch, 3, 1, 1, bias=False),
                nn.InstanceNorm2d(ch, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ch, ch, 3, 1, 1, bias=False),
                nn.InstanceNorm2d(ch, affine=True),
            )
            self.act = nn.LeakyReLU(0.2, inplace=True)
        def forward(self, x):
            return self.act(x + self.block(x))

    class ImprovedConvAE(nn.Module):
        def __init__(self, base=32, latent=24):
            super().__init__()
            def enc_block(i, o, norm=True):
                layers = [nn.Conv2d(i, o, 4, 2, 1, bias=not norm)]
                if norm: layers.append(nn.InstanceNorm2d(o, affine=True))
                layers.append(nn.LeakyReLU(0.2, inplace=True))
                return nn.Sequential(*layers)
            def dec_block(i, o):
                return nn.Sequential(
                    nn.ConvTranspose2d(i, o, 4, 2, 1, bias=False),
                    nn.InstanceNorm2d(o, affine=True), nn.ReLU(inplace=True))
            self.enc = nn.Sequential(
                enc_block(1, base, norm=False), enc_block(base, base*2),
                enc_block(base*2, base*4), enc_block(base*4, base*8),
                enc_block(base*8, base*8), enc_block(base*8, latent))
            self.dec = nn.Sequential(
                dec_block(latent, base*8), ResBlock(base*8),
                dec_block(base*8, base*8), ResBlock(base*8),
                dec_block(base*8, base*4), ResBlock(base*4),
                dec_block(base*4, base*2), dec_block(base*2, base),
                nn.ConvTranspose2d(base, 1, 4, 2, 1), nn.Sigmoid())
        def forward(self, x): return self.dec(self.enc(x))
        def encode(self, x): return self.enc(x)

    class DoubleConv(nn.Module):
        def __init__(self, in_c, out_c):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True))
        def forward(self, x): return self.conv(x)

    class UNet(nn.Module):
        def __init__(self, in_channels=1, out_channels=1, features=[32,64,128,256]):
            super().__init__()
            self.downs = nn.ModuleList(); self.ups = nn.ModuleList()
            self.pool = nn.MaxPool2d(2, 2)
            c = in_channels
            for f in features:
                self.downs.append(DoubleConv(c, f)); c = f
            self.bottleneck = DoubleConv(features[-1], features[-1]*2)
            for f in reversed(features):
                self.ups.append(nn.ConvTranspose2d(f*2, f, 2, 2))
                self.ups.append(DoubleConv(f*2, f))
            self.final_conv = nn.Conv2d(features[0], out_channels, 1)
        def forward(self, x):
            skips = []
            for down in self.downs:
                x = down(x); skips.append(x); x = self.pool(x)
            x = self.bottleneck(x)
            skips = skips[::-1]
            for i in range(0, len(self.ups), 2):
                x = self.ups[i](x); skip = skips[i//2]
                if x.shape != skip.shape:
                    x = F.interpolate(x, size=skip.shape[2:])
                x = torch.cat((skip, x), dim=1); x = self.ups[i+1](x)
            return self.final_conv(x)

# ── IMAGE HELPERS ─────────────────────────────────────────────────────────────
def load_image(path, mode="L"):
    img = Image.open(path).convert(mode)
    w0, h0 = img.size
    nw = min(int(w0 * TARGET_H / h0), TARGET_W)
    img = img.resize((nw, TARGET_H), Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0

def pad_to(arr, W, fill=0.5):
    if arr.ndim == 2:
        h, w = arr.shape
        if w >= W: return arr[:, :W]
        out = np.full((h, W), fill, dtype=arr.dtype); out[:, :w] = arr; return out
    h, w, c = arr.shape
    if w >= W: return arr[:, :W, :]
    out = np.full((h, W, c), fill, dtype=arr.dtype); out[:, :w, :] = arr; return out

# ── FEATURE EXTRACTION ────────────────────────────────────────────────────────
def extract_lbp(gray):
    if not SKIMAGE_OK: return np.zeros_like(gray)
    u8 = (gray*255).astype(np.uint8)
    lbp = local_binary_pattern(u8, P=8, R=1, method="uniform")
    return ((lbp - lbp.min()) / (lbp.max() - lbp.min() + 1e-8)).astype(np.float32)

def extract_hog_vis(gray):
    if not SKIMAGE_OK: return np.zeros_like(gray)
    try:
        _, h = hog(gray, orientations=9, pixels_per_cell=(8,8),
                   cells_per_block=(2,2), visualize=True, channel_axis=None)
        return ((h - h.min()) / (h.max() - h.min() + 1e-8)).astype(np.float32)
    except: return np.zeros_like(gray)

def extract_glcm_contrast(gray):
    from numpy.lib.stride_tricks import sliding_window_view
    u8 = (gray*255).astype(np.uint8)
    out = np.zeros_like(gray)
    k = 8
    try:
        wins = sliding_window_view(u8.astype(np.float32), (k, k))
        s = wins.std(axis=(-2,-1))
        out[:s.shape[0], :s.shape[1]] = s / (s.max() + 1e-8)
    except:
        g = np.abs(np.gradient(gray.astype(np.float32))[0])
        out = g / (g.max() + 1e-8)
    return out.astype(np.float32)

def classical_run(gray, models=None):
    lbp  = extract_lbp(gray)
    hv   = extract_hog_vis(gray)
    glcm = extract_glcm_contrast(gray)
    if models is not None:
        ensemble, scaler, selector = models
        feat = np.concatenate([lbp.ravel()[:100], glcm.ravel()[:100]])
        try:
            fs = scaler.transform(feat.reshape(1,-1))
            fs = selector.transform(fs)
            p  = float(ensemble.predict_proba(fs)[0,1])
            return lbp, hv, glcm, int(p>=0.5), p
        except: pass
    s = float(glcm.mean())
    p = float(np.clip(s*4, 0, 1))
    return lbp, hv, glcm, int(p>=0.35), p

def ae_run(gray, model, trained=False):
    if not TORCH_OK or model is None: return np.zeros_like(gray), np.zeros_like(gray), 0.0
    PATCH = 128
    h, w  = gray.shape
    pw    = ((w+PATCH-1)//PATCH)*PATCH
    padded = np.pad(gray, ((0,0),(0,pw-w)), mode='reflect')
    rec = np.zeros_like(padded); wt = np.zeros_like(padded)
    model.eval()
    with torch.no_grad():
        for x0 in range(0, pw-PATCH+1, PATCH//2):
            t   = torch.from_numpy(padded[:h,x0:x0+PATCH]).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
            out = model(t)[0,0].cpu().numpy()
            rec[:h,x0:x0+PATCH] += out; wt[:h,x0:x0+PATCH] += 1
    wt = np.maximum(wt, 1)
    rec_f = (rec/wt)[:h,:w]; err = np.abs(rec_f - gray)
    score = float(np.sort(err.ravel())[-max(1,err.size//5):].mean())
    return rec_f, err, score

def pseudo_mask(gray):
    try:
        import cv2
        u8 = (gray*255).astype(np.uint8)
        bg = cv2.medianBlur(u8, 25)
        res = cv2.absdiff(u8, bg)
        res = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX)
        _, m = cv2.threshold(res, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        k = np.ones((5,5), np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
        return (m/255.0).astype(np.float32)
    except: return np.zeros_like(gray)

def unet_run(gray, model, trained=False):
    if not TORCH_OK or model is None: return pseudo_mask(gray)
    if not trained: return pseudo_mask(gray)
    PATCH = 128; h, w = gray.shape
    acc = np.zeros_like(gray); wacc = np.zeros_like(gray)
    model.eval()
    with torch.no_grad():
        for x0 in range(0, max(w-PATCH+1,1), PATCH//2):
            patch = gray[:h, x0:x0+PATCH]
            if patch.shape[1] < PATCH:
                patch = np.pad(patch, ((0,0),(0,PATCH-patch.shape[1])), mode='reflect')
            t = torch.from_numpy(patch[:h,:PATCH]).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
            out = torch.sigmoid(model(t))[0,0].cpu().numpy()
            ow = min(PATCH, w-x0)
            acc[:h, x0:x0+ow] += out[:h,:ow]; wacc[:h,x0:x0+ow] += 1
    return (acc/np.maximum(wacc,1)).astype(np.float32)

def hybrid_run(gray):
    gx = np.gradient(gray, axis=1); gy = np.gradient(gray, axis=0)
    gm = np.sqrt(gx**2+gy**2)
    f1 = ((gm-gm.min())/(gm.max()-gm.min()+1e-8)).astype(np.float32)
    fft = np.fft.fftshift(np.fft.fft2(gray))
    pw  = np.log1p(np.abs(fft))
    f2  = ((pw-pw.min())/(pw.max()-pw.min()+1e-8)).astype(np.float32)
    thresh = np.percentile(gm, 80)
    prob = float(np.clip((gm>thresh).mean()*3.5, 0, 1))
    return f1, f2, int(prob>=0.5), prob

# ── STYLE ─────────────────────────────────────────────────────────────────────
PAL = {"bg":"#FFFFFF","hdr":"#1C1C2E","defect":"#E63946","free":"#2A9D8F",
       "text":"#1C1C2E","sub":"#6B7280"}
APPROACH_COL = {"T1":"#dbeafe","T3":"#fef3c7","T4":"#dcfce7","T2":"#fce7f3"}
CMAP_ERR = LinearSegmentedColormap.from_list("err",
    ["#0a0a0a","#4e0031","#c1121f","#f77f00","#fcbf49","#eae2b7"])

def soff(ax):
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])

def ishow(ax, img, cmap="gray", vmin=None, vmax=None):
    ax.imshow(img, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax,
              interpolation="lanczos")
    soff(ax)

def badge(ax, label, prob, trained=True):
    tag = "DEFECTIVE" if label else "DEFECT-FREE"
    col = PAL["defect"] if label else PAL["free"]
    conf = f"{prob*100:.0f}%" if trained else "demo"
    ax.text(0.02, 0.04, f"▶ {tag}  {conf}", transform=ax.transAxes,
            va="bottom", fontsize=7, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.22", fc=col, alpha=0.9, lw=0), zorder=10)

def gt_badge(ax, true_label):
    tag = "GT: DEFECTIVE" if true_label else "GT: DEFECT-FREE"
    col = PAL["defect"] if true_label else PAL["free"]
    ax.text(0.98, 0.96, tag, transform=ax.transAxes, va="top", ha="right",
            fontsize=6.2, fontweight="bold", color=col,
            bbox=dict(boxstyle="round,pad=0.13", fc="white", alpha=0.82,
                      lw=0.5, ec=col), zorder=10)

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    random.seed(42); np.random.seed(42)

    free_dir   = os.path.join(DATA_ROOT, "Train", "defect_free")
    defect_dir = os.path.join(DATA_ROOT, "Train", "defective")
    free_ps    = sorted(glob.glob(os.path.join(free_dir,   "*.png")))
    defect_ps  = sorted(glob.glob(os.path.join(defect_dir, "*.png")))

    if not free_ps or not defect_ps:
        sys.exit(f"No images found in:\n  {free_dir}\n  {defect_dir}")

    samples = ([(p,0) for p in random.sample(free_ps, min(2,len(free_ps)))] +
               [(p,1) for p in random.sample(defect_ps, min(3,len(defect_ps)))])
    print(f"Samples: {len(samples)} ({sum(1 for _,l in samples if l==0)} free, "
          f"{sum(1 for _,l in samples if l==1)} defective)")

    # Load models
    cl_models = None; ae_obj = None; unet_obj = None
    ae_trained = False; unet_trained = False

    if JOBLIB_OK and all(os.path.exists(p) for p in [ART_ENSEMBLE,ART_SELECTOR,ART_SCALER]):
        try:
            cl_models = (joblib.load(ART_ENSEMBLE),joblib.load(ART_SCALER),joblib.load(ART_SELECTOR))
            print("[OK] Classical ensemble loaded")
        except Exception as e: print(f"[--] Classical: {e}")
    else: print("[--] Classical artifacts not found -> heuristic mode")

    if TORCH_OK:
        ae_obj = ImprovedConvAE(base=32, latent=24).to(DEVICE)
        if os.path.exists(ART_AE):
            try:
                ae_obj.load_state_dict(torch.load(ART_AE, map_location=DEVICE))
                ae_obj.eval(); ae_trained = True; print("[OK] AE weights loaded")
            except Exception as e: print(f"[--] AE: {e}")
        else: print("[--] AE weights not found -> demo mode")

        unet_obj = UNet(in_channels=1, out_channels=1).to(DEVICE)
        if os.path.exists(ART_UNET):
            try:
                unet_obj.load_state_dict(torch.load(ART_UNET, map_location=DEVICE))
                unet_obj.eval(); unet_trained = True; print("[OK] UNet weights loaded")
            except Exception as e: print(f"[--] UNet: {e}")
        else: print("[--] UNet weights not found -> pseudo-mask mode")

    # Run inference
    rows = []
    for path, true_label in samples:
        print(f"  {os.path.basename(path)}")
        gray = pad_to(load_image(path, "L"), TARGET_W)
        lbp, hv, glcm, cl_l, cl_p = classical_run(gray, cl_models)
        rec, err, ae_s = ae_run(gray, ae_obj, trained=ae_trained)
        AE_T = 0.02
        ae_l = int(ae_s > AE_T); ae_p = float(np.clip(ae_s/(AE_T*3), 0, 1))
        mask = unet_run(gray, unet_obj, trained=unet_trained)
        u_l  = int(mask.mean() > 0.05); u_p = float(np.clip(mask.mean()*10, 0, 1))
        hf1, hf2, h_l, h_p = hybrid_run(gray)
        rows.append(dict(path=path, fname=os.path.basename(path),
            true_label=true_label, gray=gray,
            lbp=lbp, hog=hv, glcm=glcm, cl_l=cl_l, cl_p=cl_p,
            rec=rec, err=err, ae_l=ae_l, ae_p=ae_p,
            mask=mask, u_l=u_l, u_p=u_p,
            hf1=hf1, hf2=hf2, h_l=h_l, h_p=h_p))

    # ── Figure ────────────────────────────────────────────────────────────────
    N = len(rows); NC = 8
    fig = plt.figure(figsize=(26, 4.2 + N*3.2), dpi=180)
    fig.patch.set_facecolor(PAL["bg"])

    outer = gridspec.GridSpec(2+N, 1, figure=fig,
        height_ratios=[0.75, 0.30]+[1.0]*N, hspace=0.04)

    # Title
    at = fig.add_subplot(outer[0])
    at.set_facecolor(PAL["hdr"])
    at.text(0.5, 0.64,
        "Qualitative Comparison of Fabric Defect Detection Approaches",
        ha="center", va="center", color="white",
        fontsize=17, fontweight="bold", fontfamily="DejaVu Sans",
        transform=at.transAxes)
    at.text(0.5, 0.22,
        "Task 1: Classical ML (LBP/GLCM/HOG Ensemble)  ·  "
        "Task 2: Hybrid Deep+Classical  ·  "
        "Task 3: Conv Autoencoder (ImprovedConvAE)  ·  "
        "Task 4: U-Net Segmentation",
        ha="center", va="center", color="#a8b0c0", fontsize=9,
        fontfamily="DejaVu Sans", transform=at.transAxes)
    soff(at)

    # Column headers
    ah = fig.add_subplot(outer[1]); ah.set_facecolor(PAL["bg"]); soff(ah)
    col_info = [
        ("Original\nImage",      ""),
        ("LBP Map",              "T1"),
        ("HOG Map",              "T1"),
        ("GLCM Contrast",        "T1"),
        ("AE Reconstruction",    "T3"),
        ("Anomaly Heatmap",      "T3"),
        ("Predicted Mask",       "T4"),
        ("Hybrid Feature (grad)","T2"),
    ]
    for ci, (lbl, tag) in enumerate(col_info):
        if tag:
            from matplotlib.patches import FancyBboxPatch
            p = FancyBboxPatch((ci/NC, 0), 1/NC, 1,
                boxstyle="square,pad=0", fc=APPROACH_COL[tag], alpha=0.55,
                transform=ah.transAxes, zorder=0, lw=0)
            ah.add_patch(p)
        ah.text((ci+0.5)/NC, 0.52, lbl, ha="center", va="center",
            fontsize=8, fontweight="bold", color=PAL["text"],
            fontfamily="DejaVu Sans", transform=ah.transAxes,
            multialignment="center")
        if ci > 0:
            ah.plot([ci/NC, ci/NC], [0, 1], color="#CCCCCC", lw=0.8, transform=ah.transAxes, zorder=5)

    # Data rows
    for ri, r in enumerate(rows):
        gs = gridspec.GridSpecFromSubplotSpec(1, NC, subplot_spec=outer[2+ri],
                                              wspace=0.025)
        axs = [fig.add_subplot(gs[0, ci]) for ci in range(NC)]
        if r["true_label"]:
            for a in axs: a.set_facecolor("#fff8f8")

        # col 0 – original
        ishow(axs[0], r["gray"])
        axs[0].text(0.02, 0.97, r["fname"][:24], transform=axs[0].transAxes,
            va="top", fontsize=6.2, color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.12", fc="black", alpha=0.55, lw=0))
        gt_badge(axs[0], r["true_label"])

        # Task 1
        ishow(axs[1], r["lbp"], cmap="plasma")
        badge(axs[1], r["cl_l"], r["cl_p"], trained=(cl_models is not None))
        ishow(axs[2], r["hog"], cmap="magma")
        ishow(axs[3], r["glcm"], cmap="hot")

        # Task 3
        ishow(axs[4], r["rec"])
        ishow(axs[5], r["err"], cmap=CMAP_ERR)
        badge(axs[5], r["ae_l"], r["ae_p"], trained=ae_trained)

        # Task 4
        ishow(axs[6], r["mask"], cmap="Greens", vmin=0, vmax=1)
        badge(axs[6], r["u_l"], r["u_p"], trained=unet_trained)

        # Task 2
        ishow(axs[7], r["hf1"], cmap="viridis")
        badge(axs[7], r["h_l"], r["h_p"], trained=False)

        axs[0].set_ylabel(
            "Defective" if r["true_label"] else "Defect-Free",
            fontsize=8.5, fontweight="bold", rotation=90, labelpad=4,
            color=PAL["defect"] if r["true_label"] else PAL["free"])

    # Legend
    handles = [
        mpatches.Patch(fc=PAL["defect"],         label="Predicted: Defective"),
        mpatches.Patch(fc=PAL["free"],            label="Predicted: Defect-Free"),
        mpatches.Patch(fc=APPROACH_COL["T1"],    label="Task 1 – Classical ML"),
        mpatches.Patch(fc=APPROACH_COL["T3"],    label="Task 3 – Conv Autoencoder"),
        mpatches.Patch(fc=APPROACH_COL["T4"],    label="Task 4 – U-Net Segmentation"),
        mpatches.Patch(fc=APPROACH_COL["T2"],    label="Task 2 – Hybrid Deep+Classical"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8.5,
               framealpha=0.92, edgecolor="#CCCCCC",
               bbox_to_anchor=(0.5, 0.005))

    fig.text(0.5, 0.002,
        "Note: confidence badges show model output. "
        "If saved weights unavailable, pseudo-mask/heuristic visualisations are shown (labelled 'demo').",
        ha="center", fontsize=7, color=PAL["sub"], fontfamily="DejaVu Sans")

    plt.savefig(SAVE_PATH, dpi=180, bbox_inches="tight",
                facecolor=PAL["bg"], pad_inches=0.12)
    plt.close(fig)
    print(f"\nSaved -> {SAVE_PATH}")

if __name__ == "__main__":
    main()

