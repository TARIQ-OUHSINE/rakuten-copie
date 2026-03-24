"""
Phase 3 — Advanced (Optimized): CLIP ViT-B/32 + MLP (text + image multimodal)

Optimizations over clip_mlp.py:
  1. Batched CLIP inference via CLIPDataset + DataLoader (batch_size=256, num_workers=4)
  2. pin_memory=True for faster CPU→GPU transfer
  3. Mixed precision (torch.cuda.amp) in both embedding extraction and MLP training
  4. GradScaler for stable float16 gradient updates
  5. Parallel ablation using ThreadPoolExecutor (3 configs train concurrently)

Input : data/train_cleaned.pkl, data/test_cleaned.pkl
        models/label_encoder.pkl   (from Phase 1)
Output: docs/figures/mod_fig7_clip_training.png
        docs/figures/mod_fig8_clip_ablation.png
        models/clip_mlp_head.pth
        data/test_predictions.csv

CLIP embeddings are cached to disk on first run:
  data/clip_img_emb_train.npy  data/clip_img_emb_val.npy
  data/clip_txt_emb_train.npy  data/clip_txt_emb_val.npy
  data/clip_img_emb_test.npy   data/clip_txt_emb_test.npy
"""

import os
import pickle
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import open_clip
from PIL import Image
from tqdm import tqdm

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
IMG_DIR = os.path.join(ROOT, "images", "image_train")
IMG_TEST_DIR = os.path.join(ROOT, "images", "image_test")
FIG_DIR = os.path.join(ROOT, "docs", "figures")
MODEL_DIR = os.path.join(ROOT, "models")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"          # AMP only meaningful on GPU
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"
EMBED_DIM = 512
COMBINED_DIM = 1024
NUM_CLASSES = 27
BATCH_SIZE = 256
NUM_EPOCHS = 20
LR = 1e-3
NUM_WORKERS = min(4, os.cpu_count() or 1)


# ── Helpers ──────────────────────────────────────────────────────────────────
def weighted_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="weighted")


def img_path(row, split="train"):
    folder = IMG_DIR if split == "train" else IMG_TEST_DIR
    return os.path.join(
        folder,
        f"image_{row['imageid']}_product_{row['productid']}.jpg",
    )


# ── Data ─────────────────────────────────────────────────────────────────────
def load_data():
    train = pd.read_pickle(os.path.join(DATA_DIR, "train_cleaned.pkl"))
    test = pd.read_pickle(os.path.join(DATA_DIR, "test_cleaned.pkl"))

    with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)

    train["label"] = le.transform(train["prdtypecode"])

    X_train, X_val, y_train, y_val = train_test_split(
        train, train["label"],
        test_size=0.20, random_state=42, stratify=train["label"],
    )
    print(f"Train: {len(X_train):,}  Val: {len(X_val):,}  Classes: {le.classes_.shape[0]}")
    return X_train, X_val, y_train, y_val, le, test


# ── CLIP model ───────────────────────────────────────────────────────────────
def load_clip_model():
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED,
    )
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
    model = model.to(DEVICE).eval()
    return model, preprocess, tokenizer


# ── Batched CLIP Dataset ──────────────────────────────────────────────────────
class CLIPDataset(Dataset):
    """Returns (preprocessed_img_tensor [3,224,224], token_ids [77]) per sample."""

    def __init__(self, df, preprocess, tokenizer, split="train"):
        self.df = df.reset_index(drop=True)
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.split = split

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]

        # Image — zero tensor fallback for missing/corrupt files
        try:
            img = Image.open(img_path(row, self.split)).convert("RGB")
            img_tensor = self.preprocess(img)           # [3, 224, 224]
        except Exception:
            img_tensor = torch.zeros(3, 224, 224)

        # Text — tokenizer returns [1, 77]; squeeze to [77]
        text = str(row.get("text_clean", ""))[:77]
        tok = self.tokenizer([text])[0]                 # [77]

        return img_tensor, tok


# ── Batched embedding extraction ──────────────────────────────────────────────
def extract_embeddings(df, clip_model, preprocess, tokenizer, split="train"):
    """Extract and return (img_emb, txt_emb) as float32 numpy arrays.

    Uses a DataLoader with num_workers for parallel preprocessing and
    batch CLIP inference (batch_size=BATCH_SIZE) instead of one sample at a time.
    """
    img_cache = os.path.join(DATA_DIR, f"clip_img_emb_{split}.npy")
    txt_cache = os.path.join(DATA_DIR, f"clip_txt_emb_{split}.npy")

    if os.path.exists(img_cache) and os.path.exists(txt_cache):
        print(f"  Loading cached CLIP embeddings for {split}...")
        return np.load(img_cache), np.load(txt_cache)

    print(f"  Extracting CLIP embeddings for {split} ({len(df):,} samples)  "
          f"batch={BATCH_SIZE}  workers={NUM_WORKERS}  AMP={USE_AMP}")

    dataset = CLIPDataset(df, preprocess, tokenizer, split)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
    )

    img_embs, txt_embs = [], []

    with torch.no_grad():
        for imgs, toks in tqdm(loader, desc=f"CLIP {split}"):
            imgs = imgs.to(DEVICE, non_blocking=True)
            toks = toks.to(DEVICE, non_blocking=True)

            if USE_AMP:
                with torch.cuda.amp.autocast():
                    img_feats = clip_model.encode_image(imgs)
                    txt_feats = clip_model.encode_text(toks)
            else:
                img_feats = clip_model.encode_image(imgs)
                txt_feats = clip_model.encode_text(toks)

            # L2-normalize
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)

            img_embs.append(img_feats.cpu().float().numpy())
            txt_embs.append(txt_feats.cpu().float().numpy())

    img_arr = np.vstack(img_embs)
    txt_arr = np.vstack(txt_embs)
    np.save(img_cache, img_arr)
    np.save(txt_cache, txt_arr)
    print(f"  Cached → {img_cache}")
    return img_arr, txt_arr


# ── Dataset & MLP ─────────────────────────────────────────────────────────────
class EmbedDataset(Dataset):
    def __init__(self, img_emb, txt_emb, labels=None):
        self.img = torch.tensor(img_emb, dtype=torch.float32)
        self.txt = torch.tensor(txt_emb, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None

    def __len__(self):
        return len(self.img)

    def __getitem__(self, i):
        x = torch.cat([self.img[i], self.txt[i]])
        if self.labels is not None:
            return x, self.labels[i]
        return x


class MLPHead(nn.Module):
    def __init__(self, in_dim=COMBINED_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def make_class_weights(y_train, num_classes):
    counts = np.bincount(y_train, minlength=num_classes).astype(float)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)


# ── Training (AMP + parallel-ready) ──────────────────────────────────────────
def train_mlp(img_tr, txt_tr, y_train, img_va, txt_va, y_val, in_dim=COMBINED_DIM):
    ds_tr = EmbedDataset(img_tr, txt_tr, np.array(y_train))
    ds_va = EmbedDataset(img_va, txt_va, np.array(y_val))
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True,
                       num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"))
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"))

    model = MLPHead(in_dim=in_dim, num_classes=NUM_CLASSES).to(DEVICE)
    weights = make_class_weights(np.array(y_train), NUM_CLASSES)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    history = {"train_loss": [], "val_f1": []}
    best_f1 = 0.0
    best_state = None

    for epoch in range(1, NUM_EPOCHS + 1):
        # Train
        model.train()
        total_loss = 0.0
        for x_batch, y_batch in dl_tr:
            x_batch = x_batch.to(DEVICE, non_blocking=True)
            y_batch = y_batch.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                loss = criterion(model(x_batch), y_batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(y_batch)
        scheduler.step()
        avg_loss = total_loss / len(ds_tr)

        # Validate
        model.eval()
        all_preds = []
        with torch.no_grad():
            for x_batch, _ in dl_va:
                with torch.cuda.amp.autocast(enabled=USE_AMP):
                    preds = model(x_batch.to(DEVICE, non_blocking=True)).argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
        val_f1 = weighted_f1(np.array(y_val), np.array(all_preds))

        history["train_loss"].append(avg_loss)
        history["val_f1"].append(val_f1)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:02d}/{NUM_EPOCHS}  loss={avg_loss:.4f}  val_F1={val_f1:.4f}")

    model.load_state_dict(best_state)
    print(f"\n  Best val F1: {best_f1:.4f}")
    return model, history, best_f1


# ── Inference ─────────────────────────────────────────────────────────────────
def predict(model, img_emb, txt_emb):
    ds = EmbedDataset(img_emb, txt_emb)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"))
    model.eval()
    preds = []
    with torch.no_grad():
        for x_batch in dl:
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                preds.extend(model(x_batch.to(DEVICE, non_blocking=True)).argmax(dim=1).cpu().numpy())
    return np.array(preds)


# ── Parallel ablation ────────────────────────────────────────────────────────
def _ablation_worker(config, img_tr, txt_tr, y_train, img_va, txt_va, y_val):
    """Train one ablation config and return (label, f1)."""
    label, in_dim, _img_tr, _txt_tr, _img_va, _txt_va = config
    print(f"\nAblation — {label} (in_dim={in_dim})")
    _, _, f1 = train_mlp(_img_tr, _txt_tr, y_train, _img_va, _txt_va, y_val, in_dim=in_dim)
    return {"modality": label, "f1": round(f1, 4)}


def run_ablation(img_tr, txt_tr, y_train, img_va, txt_va, y_val):
    """Train text-only, image-only, and combined MLPs concurrently."""
    zero_img_tr = np.zeros_like(img_tr)
    zero_txt_tr = np.zeros_like(txt_tr)
    zero_img_va = np.zeros_like(img_va)
    zero_txt_va = np.zeros_like(txt_va)

    configs = [
        ("Text only (512-d)",   EMBED_DIM,    zero_img_tr, txt_tr, zero_img_va, txt_va),
        ("Image only (512-d)",  EMBED_DIM,    img_tr, zero_txt_tr, img_va, zero_txt_va),
        ("Combined (1024-d)",   COMBINED_DIM, img_tr, txt_tr,      img_va, txt_va),
    ]

    results = []
    # ThreadPoolExecutor: PyTorch releases GIL during GPU ops → true parallelism,
    # and threads share tensor memory (no serialization overhead).
    with ThreadPoolExecutor(max_workers=len(configs)) as pool:
        futures = {
            pool.submit(_ablation_worker, cfg, img_tr, txt_tr, y_train, img_va, txt_va, y_val): cfg[0]
            for cfg in configs
        }
        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda r: r["f1"], reverse=True)
    return results


# ── Figures ───────────────────────────────────────────────────────────────────
def plot_training_history(history):
    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs, y=history["train_loss"],
        mode="lines+markers", name="Train Loss", yaxis="y1",
    ))
    fig.add_trace(go.Scatter(
        x=epochs, y=history["val_f1"],
        mode="lines+markers", name="Val Weighted F1", yaxis="y2",
    ))
    fig.update_layout(
        title="Phase 3 (Optimized) — CLIP MLP Training History",
        xaxis_title="Epoch",
        yaxis=dict(title="Cross-Entropy Loss"),
        yaxis2=dict(title="Weighted F1", overlaying="y", side="right"),
        legend=dict(x=0.7, y=0.5),
        height=420,
    )
    return fig


def plot_ablation(results):
    df = pd.DataFrame(results).sort_values("f1", ascending=False)
    fig = px.bar(
        df, x="modality", y="f1",
        title="Phase 3 (Optimized) — CLIP Modality Ablation (Weighted F1)",
        labels={"f1": "Weighted F1", "modality": "Modality"},
        color="f1", color_continuous_scale="RdYlGn", range_y=[0, 1],
        text="f1",
    )
    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig.update_layout(coloraxis_showscale=False, height=400)
    return fig


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Phase 3 (Optimized) — CLIP ViT-B/32 + MLP")
    print(f"Device: {DEVICE}  AMP: {USE_AMP}  Workers: {NUM_WORKERS}")
    print("=" * 60)

    X_train, X_val, y_train, y_val, le, test = load_data()

    print(f"\nLoading CLIP model ({CLIP_MODEL_NAME}, {CLIP_PRETRAINED})...")
    clip_model, preprocess, tokenizer = load_clip_model()

    # Batched embedding extraction
    print("\nExtracting embeddings (batched)...")
    img_tr, txt_tr = extract_embeddings(X_train, clip_model, preprocess, tokenizer, "train")
    img_va, txt_va = extract_embeddings(X_val, clip_model, preprocess, tokenizer, "val")

    # Train combined model
    print("\nTraining combined MLP (text + image)...")
    best_model, history, best_f1 = train_mlp(img_tr, txt_tr, y_train, img_va, txt_va, y_val)

    y_pred = predict(best_model, img_va, txt_va)
    print(f"\nCLIP MLP (combined) — weighted F1: {best_f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_val, y_pred, target_names=[str(c) for c in le.classes_]))

    # Save model
    torch.save(best_model.state_dict(), os.path.join(MODEL_DIR, "clip_mlp_head.pth"))
    print("Model saved to models/clip_mlp_head.pth")

    # Training history figure
    fig_hist = plot_training_history(history)
    try:
        fig_hist.write_image(os.path.join(FIG_DIR, "mod_fig7_clip_training.png"), scale=2)
        print("Figure saved: mod_fig7_clip_training.png")
    except Exception as e:
        print(f"Figure export skipped: {e}")

    # Parallel ablation
    print("\nRunning modality ablation (3 configs in parallel)...")
    ablation_results = run_ablation(img_tr, txt_tr, y_train, img_va, txt_va, y_val)
    print("\nAblation results:")
    for r in ablation_results:
        print(f"  {r['modality']:25s}  F1={r['f1']:.4f}")

    fig_abl = plot_ablation(ablation_results)
    try:
        fig_abl.write_image(os.path.join(FIG_DIR, "mod_fig8_clip_ablation.png"), scale=2)
        print("Figure saved: mod_fig8_clip_ablation.png")
    except Exception as e:
        print(f"Figure export skipped: {e}")

    # Test predictions
    print("\nGenerating test predictions...")
    img_te, txt_te = extract_embeddings(test, clip_model, preprocess, tokenizer, "test")
    y_test_pred = predict(best_model, img_te, txt_te)
    test_labels = le.inverse_transform(y_test_pred)
    pd.DataFrame({"prdtypecode": test_labels}).to_csv(
        os.path.join(DATA_DIR, "test_predictions.csv"), index=False,
    )
    print("Test predictions saved to data/test_predictions.csv")

    return best_f1


if __name__ == "__main__":
    main()
