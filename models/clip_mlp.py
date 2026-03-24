"""
Phase 3 — Advanced: CLIP ViT-B/32 + MLP (text + image multimodal)

Input : data/train_cleaned.pkl, data/test_cleaned.pkl
        models/label_encoder.pkl   (from Phase 1)
Output: docs/figures/mod_fig7_clip_training.png
        docs/figures/mod_fig8_clip_ablation.png
        models/clip_mlp_head.pth   (saved MLP head)
        data/test_predictions.csv  (final test predictions)

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
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"
EMBED_DIM = 512
COMBINED_DIM = 1024
NUM_CLASSES = 27
BATCH_SIZE = 256
NUM_EPOCHS = 20
LR = 1e-3


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


# ── CLIP embeddings ───────────────────────────────────────────────────────────
def load_clip_model():
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED,
    )
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
    model = model.to(DEVICE).eval()
    return model, preprocess, tokenizer


def extract_embeddings(df, clip_model, preprocess, tokenizer, split="train"):
    """Extract and return (img_emb, txt_emb) as float32 numpy arrays."""
    img_cache = os.path.join(DATA_DIR, f"clip_img_emb_{split}.npy")
    txt_cache = os.path.join(DATA_DIR, f"clip_txt_emb_{split}.npy")

    if os.path.exists(img_cache) and os.path.exists(txt_cache):
        print(f"  Loading cached CLIP embeddings for {split}...")
        return np.load(img_cache), np.load(txt_cache)

    print(f"  Extracting CLIP embeddings for {split} ({len(df):,} samples)...")
    img_embs, txt_embs = [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"CLIP {split}"):
        # Image
        try:
            img = Image.open(img_path(row, split)).convert("RGB")
            img_tensor = preprocess(img).unsqueeze(0).to(DEVICE)
        except Exception:
            img_tensor = torch.zeros(1, 3, 224, 224).to(DEVICE)

        # Text
        text = str(row.get("text_clean", ""))
        # Truncate to ~400 chars (safe estimate for 77 tokens, ~5 chars per token)
        text = text[:400] if len(text) > 400 else text
        tok = tokenizer([text]).to(DEVICE)

        with torch.no_grad():
            img_feat = clip_model.encode_image(img_tensor)
            txt_feat = clip_model.encode_text(tok)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

        img_embs.append(img_feat.cpu().numpy().astype(np.float32))
        txt_embs.append(txt_feat.cpu().numpy().astype(np.float32))

    img_arr = np.vstack(img_embs)
    txt_arr = np.vstack(txt_embs)
    np.save(img_cache, img_arr)
    np.save(txt_cache, txt_arr)
    print(f"  Cached to {img_cache}, {txt_cache}")
    return img_arr, txt_arr


# ── Dataset & MLP ─────────────────────────────────────────────────────────────
class EmbedDataset(Dataset):
    def __init__(self, img_emb, txt_emb, labels=None, use_single_modality=None):
        """
        Args:
            img_emb: Image embeddings
            txt_emb: Text embeddings
            labels: Optional labels
            use_single_modality: None (concatenate both), 'img' (image only), or 'txt' (text only)
        """
        self.img = torch.tensor(img_emb, dtype=torch.float32)
        self.txt = torch.tensor(txt_emb, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None
        self.use_single_modality = use_single_modality

    def __len__(self):
        return len(self.img)

    def __getitem__(self, i):
        if self.use_single_modality == 'img':
            x = self.img[i]
        elif self.use_single_modality == 'txt':
            x = self.txt[i]
        else:
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


# ── Training ─────────────────────────────────────────────────────────────────
def train_mlp(img_tr, txt_tr, y_train, img_va, txt_va, y_val, in_dim=COMBINED_DIM, use_single_modality=None):
    ds_tr = EmbedDataset(img_tr, txt_tr, np.array(y_train), use_single_modality=use_single_modality)
    ds_va = EmbedDataset(img_va, txt_va, np.array(y_val), use_single_modality=use_single_modality)
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = MLPHead(in_dim=in_dim, num_classes=NUM_CLASSES).to(DEVICE)
    weights = make_class_weights(np.array(y_train), NUM_CLASSES)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    history = {"train_loss": [], "val_f1": []}
    best_f1 = 0.0
    best_state = None

    for epoch in range(1, NUM_EPOCHS + 1):
        # Train
        model.train()
        total_loss = 0.0
        for x_batch, y_batch in dl_tr:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y_batch)
        scheduler.step()
        avg_loss = total_loss / len(ds_tr)

        # Validate
        model.eval()
        all_preds = []
        with torch.no_grad():
            for x_batch, _ in dl_va:
                preds = model(x_batch.to(DEVICE)).argmax(dim=1).cpu().numpy()
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
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    model.eval()
    preds = []
    with torch.no_grad():
        for x_batch in dl:
            preds.extend(model(x_batch.to(DEVICE)).argmax(dim=1).cpu().numpy())
    return np.array(preds)


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
        title="Phase 3 — CLIP MLP Training History",
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
        title="Phase 3 — CLIP Modality Ablation (Weighted F1)",
        labels={"f1": "Weighted F1", "modality": "Modality"},
        color="f1", color_continuous_scale="RdYlGn", range_y=[0, 1],
        text="f1",
    )
    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig.update_layout(coloraxis_showscale=False, height=400)
    return fig


# ── Ablation ─────────────────────────────────────────────────────────────────
def run_ablation(img_tr, txt_tr, y_train, img_va, txt_va, y_val):
    """Train three MLPs: text-only, image-only, combined."""
    results = []
    configs = [
        ("Text only (512-d)", EMBED_DIM, "txt"),
        ("Image only (512-d)", EMBED_DIM, "img"),
        ("Combined (1024-d)", COMBINED_DIM, None),
    ]
    for label, in_dim, modality in configs:
        print(f"\nAblation — {label} (in_dim={in_dim})")
        _, _, f1 = train_mlp(
            img_tr, txt_tr, y_train, img_va, txt_va, y_val,
            in_dim=in_dim, use_single_modality=modality
        )
        results.append({"modality": label, "f1": round(f1, 4)})

    return results


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Phase 3 — Advanced: CLIP ViT-B/32 + MLP")
    print("=" * 60)

    X_train, X_val, y_train, y_val, le, test = load_data()

    print(f"\nLoading CLIP model ({CLIP_MODEL_NAME}, {CLIP_PRETRAINED})...")
    clip_model, preprocess, tokenizer = load_clip_model()
    print(f"Device: {DEVICE}")

    # Extract / load cached embeddings
    print("\nExtracting embeddings...")
    img_tr, txt_tr = extract_embeddings(X_train.reset_index(drop=True), clip_model, preprocess, tokenizer, "train")
    img_va, txt_va = extract_embeddings(X_val.reset_index(drop=True), clip_model, preprocess, tokenizer, "val")

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

    # Ablation
    print("\nRunning modality ablation (3 × 20 epochs)...")
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
    img_te, txt_te = extract_embeddings(test.reset_index(drop=True), clip_model, preprocess, tokenizer, "test")
    y_test_pred = predict(best_model, img_te, txt_te)
    test_labels = le.inverse_transform(y_test_pred)
    pd.DataFrame({"prdtypecode": test_labels}).to_csv(
        os.path.join(DATA_DIR, "test_predictions.csv"), index=False,
    )
    print("Test predictions saved to data/test_predictions.csv")

    return best_f1


if __name__ == "__main__":
    main()
