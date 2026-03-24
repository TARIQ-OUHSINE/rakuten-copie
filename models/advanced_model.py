"""
Phase 2 — Optimization: TF-IDF + OpenCV features + LightGBM (text + image)

Requires phase1 artifacts: models/tfidf_vectorizer.pkl, models/label_encoder.pkl

Input : data/train_cleaned.pkl, data/test_cleaned.pkl, images/image_train/
Output: data/img_feats_train.npy  (cached, ~5 min to compute)
        data/img_feats_val.npy
        docs/figures/mod_fig3_opencv_features.png
        docs/figures/mod_fig4_ablation_opencv.png
        docs/figures/mod_fig5_feature_importance.png
        docs/figures/mod_fig6_shap.png
        models/lgb_combined_model.pkl

Image feature vector: 8222-d
  - color histogram (HSV, 3×32 bins) :   96-d
  - HOG descriptor (128×128 px)       : 8100-d
  - LBP texture (24 pts, r=3)         :   26-d
"""

import os
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import cv2
from skimage.feature import local_binary_pattern
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ThreadPoolExecutor


from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
import lightgbm as lgb

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
IMG_DIR = os.path.join(ROOT, "images", "image_train")
FIG_DIR = os.path.join(ROOT, "docs", "figures")
MODEL_DIR = os.path.join(ROOT, "models")
os.makedirs(FIG_DIR, exist_ok=True)

# Feature vector layout (must match extraction functions)
N_COLOR = 96    # 3 channels × 32 bins
N_HOG = 8100    # 15×15 blocks × 4 cells × 9 bins (128×128 target)
N_LBP = 26      # 24 points + 2 (uniform LBP)
N_TOTAL = N_COLOR + N_HOG + N_LBP  # 8222


# ── Helpers ──────────────────────────────────────────────────────────────────
def weighted_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="weighted")


def _make_tqdm_callback(desc="Training LightGBM", position=0):
    """Return a LightGBM callback that displays a tqdm progress bar."""
    pbar = None

    def callback(env):
        nonlocal pbar
        if pbar is None:
            pbar = tqdm(
                total=env.end_iteration - env.begin_iteration,
                desc=desc,
                unit="iter",
                position=position,
                leave=True,
            )
        pbar.update(1)
        if env.iteration + 1 >= env.end_iteration:
            pbar.close()

    callback.order = 10
    return callback


def get_image_path(productid, imageid):
    return os.path.join(IMG_DIR, f"image_{int(imageid)}_product_{int(productid)}.jpg")


# ── Image feature extraction ─────────────────────────────────────────────────

# HOG descriptor created once at module level and reused for every image
_HOG_DESCRIPTOR = cv2.HOGDescriptor(
    _winSize=(128, 128),
    _blockSize=(16, 16),
    _blockStride=(8, 8),
    _cellSize=(8, 8),
    _nbins=9,
)


def extract_color_histogram(img_bgr, bins=32):
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([img_hsv], [0], None, [bins], [0, 180]).flatten()
    s = cv2.calcHist([img_hsv], [1], None, [bins], [0, 256]).flatten()
    v = cv2.calcHist([img_hsv], [2], None, [bins], [0, 256]).flatten()
    feat = np.concatenate([h, s, v])
    return feat / (feat.sum() + 1e-7)


def extract_lbp_from_gray(img_gray, num_points=24, radius=3):
    """Accept a pre-resized grayscale image directly."""
    lbp = local_binary_pattern(img_gray, num_points, radius, method="uniform")
    n_bins = num_points + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist


def extract_image_features(productid, imageid):
    path = get_image_path(productid, imageid)
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        return np.zeros(N_TOTAL)
    color = extract_color_histogram(img_bgr)
    # Compute grayscale once — reused for HOG and LBP
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.resize(img_gray, (128, 128))
    hog = _HOG_DESCRIPTOR.compute(img_gray).flatten()
    lbp = extract_lbp_from_gray(img_gray)
    return np.concatenate([color, hog, lbp])


def _extract_worker(args):
    """Picklable top-level worker for multiprocessing.Pool."""
    return extract_image_features(*args)


def batch_extract(df, desc="Extracting"):
    args = df[["productid", "imageid"]].values.tolist()
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        results = list(tqdm(
            pool.imap(_extract_worker, args, chunksize=64),
            total=len(args), desc=desc,
        ))
    return np.vstack(results)


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
    return X_train, X_val, y_train, y_val, le, test


def load_tfidf_features(X_train, X_val):
    with open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
        tfidf = pickle.load(f)
    return tfidf, tfidf.transform(X_train["text_clean"]), tfidf.transform(X_val["text_clean"])


def load_or_compute_image_features(X_train, X_val):
    train_npy = os.path.join(DATA_DIR, "img_feats_train.npy")
    val_npy = os.path.join(DATA_DIR, "img_feats_val.npy")
    if os.path.exists(train_npy) and os.path.exists(val_npy):
        print("Loading cached image features...")
        return np.load(train_npy), np.load(val_npy)
    print("Extracting image features (~5 min)...")
    feats_tr = batch_extract(X_train, desc="Train")
    feats_va = batch_extract(X_val,   desc="Val")
    np.save(train_npy, feats_tr)
    np.save(val_npy,   feats_va)
    return feats_tr, feats_va


# ── Visualization ─────────────────────────────────────────────────────────────
def plot_feature_viz(X_train):
    sample_row = X_train.iloc[0]
    img_bgr = cv2.imread(get_image_path(sample_row["productid"], sample_row["imageid"]))
    color_feat = extract_color_histogram(img_bgr)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.resize(img_gray, (128, 128))
    hog_feat = _HOG_DESCRIPTOR.compute(img_gray).flatten()
    lbp_feat = extract_lbp_from_gray(img_gray)

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[
            f"Color histogram (HSV, {len(color_feat)}-d)",
            f"HOG descriptor ({len(hog_feat)}-d — first 200)",
            f"LBP texture ({len(lbp_feat)}-d)",
        ],
    )
    fig.add_trace(go.Bar(y=color_feat, showlegend=False, marker_color="steelblue"), row=1, col=1)
    fig.add_trace(go.Bar(y=hog_feat[:200], showlegend=False, marker_color="orange"), row=1, col=2)
    fig.add_trace(go.Bar(y=lbp_feat, showlegend=False, marker_color="green"), row=1, col=3)
    fig.update_layout(title="OpenCV Image Features — Sample Product", height=400)
    return fig


def plot_ablation(ablation_df):
    fig = px.bar(
        ablation_df.sort_values("Weighted F1", ascending=False),
        x="Features", y="Weighted F1",
        title="Phase 2 Ablation — Contribution of Each OpenCV Feature Group",
        text="Weighted F1", range_y=[0, 1],
        color="Weighted F1", color_continuous_scale="Greens",
    )
    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig.update_layout(coloraxis_showscale=False, height=420)
    return fig


def plot_feature_importance(model, all_feat_names):
    importance_df = pd.DataFrame({
        "feature": all_feat_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).head(50)

    importance_df["type"] = importance_df["feature"].apply(
        lambda f: "TF-IDF" if not f.startswith(("color_", "hog_", "lbp_")) else f.split("_")[0].upper()
    )
    fig = px.bar(
        importance_df, x="importance", y="feature",
        orientation="h", color="type",
        title="LightGBM — Top 50 Feature Importances (text + image)",
        labels={"importance": "Importance", "feature": "Feature"},
    )
    fig.update_layout(height=900, yaxis={"categoryorder": "total ascending"})
    return fig


def plot_shap(model, X_va_combined, all_feat_names):
    import shap
    explainer = shap.TreeExplainer(model)
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(X_va_combined.shape[0], size=100, replace=False)
    # Convert to dense numpy array (works for both sparse matrices and DataFrames)
    if isinstance(X_va_combined, pd.DataFrame):
        X_sample = X_va_combined.iloc[sample_idx].sparse.to_dense().values
    else:
        X_sample = X_va_combined[sample_idx].toarray()  # SHAP TreeExplainer requires dense input
    shap_values = explainer.shap_values(X_sample, check_additivity=False)

    # shap_values: list of (n_samples, n_features) arrays, one per class
    mean_abs = np.mean([np.abs(sv) for sv in shap_values], axis=0).mean(axis=0)
    top_idx = mean_abs.argsort()[::-1][:30]
    shap_df = pd.DataFrame({
        "feature": [all_feat_names[i] for i in top_idx],
        "mean_abs_shap": mean_abs[top_idx],
        "type": [
            "TF-IDF" if not all_feat_names[i].startswith(("color_", "hog_", "lbp_"))
            else all_feat_names[i].split("_")[0].upper()
            for i in top_idx
        ],
    })
    fig = px.bar(
        shap_df, x="mean_abs_shap", y="feature",
        orientation="h", color="type",
        title="SHAP — Top 30 Features by Mean |SHAP| (all classes)",
        labels={"mean_abs_shap": "Mean |SHAP|", "feature": "Feature"},
    )
    fig.update_layout(height=700, yaxis={"categoryorder": "total ascending"})
    return fig


# ── Ablation study ────────────────────────────────────────────────────────────
def _train_single_ablation(X_tr, X_va, y_train, y_val, n_jobs):
    """Train one ablation LightGBM model; returns weighted F1 on validation set."""
    m = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.1, num_leaves=31,
        feature_fraction=0.3, bagging_fraction=0.8, bagging_freq=5,
        max_bin=127, class_weight="balanced", force_col_wise=True,
        n_jobs=n_jobs, random_state=42, verbose=-1,
    )
    m.fit(
        X_tr, y_train,
        eval_set=[(X_va, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )
    return weighted_f1(y_val, m.predict(X_va))


def run_ablation(X_tr_tfidf, X_va_tfidf, img_tr, img_va, y_train, y_val):
    groups = [
        ("TF-IDF only",         None,                                None),
        ("TF-IDF + color hist", csr_matrix(img_tr[:, :N_COLOR]),    csr_matrix(img_va[:, :N_COLOR])),
        ("TF-IDF + HOG",        csr_matrix(img_tr[:, N_COLOR:N_COLOR+N_HOG]),
                                csr_matrix(img_va[:, N_COLOR:N_COLOR+N_HOG])),
        ("TF-IDF + LBP",        csr_matrix(img_tr[:, N_COLOR+N_HOG:]),
                                csr_matrix(img_va[:, N_COLOR+N_HOG:])),
        ("TF-IDF + all OpenCV", csr_matrix(img_tr), csr_matrix(img_va)),
    ]

    # Distribute CPU cores evenly: each worker gets cpu_count // n_workers threads
    n_workers = max(1, min(len(groups), multiprocessing.cpu_count() // 2))
    n_jobs_per = max(1, multiprocessing.cpu_count() // n_workers)

    tasks = [
        (name,
         X_tr_tfidf if tr_extra is None else hstack([X_tr_tfidf, tr_extra]),
         X_va_tfidf if va_extra is None else hstack([X_va_tfidf, va_extra]))
        for name, tr_extra, va_extra in groups
    ]

    pbar = tqdm(total=len(tasks), desc="Ablation groups", unit="group")

    def run_one(name, X_tr, X_va):
        f1 = _train_single_ablation(X_tr, X_va, y_train, y_val, n_jobs_per)
        pbar.update(1)
        pbar.write(f"  {name:<30} F1={f1:.4f}")
        return f1

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(run_one, name, X_tr, X_va) for name, X_tr, X_va in tasks]
    pbar.close()

    # Reconstruct DataFrame in original group order
    return pd.DataFrame([
        {"Features": name, "Weighted F1": fut.result()}
        for (name, _, _), fut in zip(tasks, futures)
    ])


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Phase 2 — Optimization: TF-IDF + OpenCV + LightGBM")
    print("=" * 60)

    X_train, X_val, y_train, y_val, le, _ = load_data()
    tfidf, X_tr_tfidf, X_va_tfidf = load_tfidf_features(X_train, X_val)
    img_feats_tr, img_feats_va = load_or_compute_image_features(X_train, X_val)

    print(f"\nImage feature matrix: train={img_feats_tr.shape}, val={img_feats_va.shape}")

    # Feature visualization
    fig_feat = plot_feature_viz(X_train)
    try:
        fig_feat.write_image(os.path.join(FIG_DIR, "mod_fig3_opencv_features.png"), scale=2)
        print("Saved mod_fig3_opencv_features.png")
    except Exception as e:
        print(f"Figure export skipped: {e}")

    # Combined feature matrix
    img_tr_sparse = csr_matrix(img_feats_tr)
    img_va_sparse = csr_matrix(img_feats_va)
    X_tr_combined_sparse = hstack([X_tr_tfidf, img_tr_sparse])
    X_va_combined_sparse = hstack([X_va_tfidf, img_va_sparse])
    
    # Feature names (built here so they can be passed to fit)
    all_feat_names = (
        list(tfidf.get_feature_names_out()) +
        [f"color_{i}" for i in range(N_COLOR)] +
        [f"hog_{i}"   for i in range(N_HOG)] +
        [f"lbp_{i}"   for i in range(N_LBP)]
    )
    
    # Convert to pandas DataFrame with feature names (sparse format to save memory)
    X_tr_combined = pd.DataFrame.sparse.from_spmatrix(X_tr_combined_sparse, columns=all_feat_names)
    X_va_combined = pd.DataFrame.sparse.from_spmatrix(X_va_combined_sparse, columns=all_feat_names)
    print(f"Combined feature matrix: {X_tr_combined.shape}")

    # Train combined LightGBM
    print("\nTraining LightGBM on combined features...")
    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.1,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=20,
        feature_fraction=0.3,   # evaluate 30% of features per tree (~17K vs 58K)
        bagging_fraction=0.8,   # subsample 80% of rows per tree
        bagging_freq=5,
        max_bin=127,            # reduce from default 255 for faster histograms
        class_weight="balanced",
        force_col_wise=True,    # correct for wide datasets (58K features)
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )
    model.fit(
        X_tr_combined, y_train,
        eval_set=[(X_va_combined, y_val)],
        feature_name=all_feat_names,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100),
            _make_tqdm_callback("Training LightGBM (TF-IDF + OpenCV)"),
        ],
    )
    y_pred = model.predict(X_va_combined)
    f1_combined = weighted_f1(y_val, y_pred)
    print(f"\nLightGBM (TF-IDF + OpenCV) — weighted F1: {f1_combined:.4f}")
    print(classification_report(y_val, y_pred, target_names=[str(c) for c in le.classes_]))

    # Ablation
    print("\nRunning ablation study...")
    ablation_df = run_ablation(X_tr_tfidf, X_va_tfidf, img_feats_tr, img_feats_va, y_train, y_val)
    fig_abl = plot_ablation(ablation_df)
    try:
        fig_abl.write_image(os.path.join(FIG_DIR, "mod_fig4_ablation_opencv.png"), scale=2)
        print("Saved mod_fig4_ablation_opencv.png")
    except Exception as e:
        print(f"Figure export skipped: {e}")

    # Feature importance
    fig_imp = plot_feature_importance(model, all_feat_names)
    try:
        fig_imp.write_image(os.path.join(FIG_DIR, "mod_fig5_feature_importance.png"), scale=2)
        print("Saved mod_fig5_feature_importance.png")
    except Exception as e:
        print(f"Figure export skipped: {e}")

    # SHAP
    print("\nComputing SHAP values (100 samples)...")
    fig_shap = plot_shap(model, X_va_combined, all_feat_names)
    try:
        fig_shap.write_image(os.path.join(FIG_DIR, "mod_fig6_shap.png"), scale=2)
        print("Saved mod_fig6_shap.png")
    except Exception as e:
        print(f"Figure export skipped: {e}")

    # Save model
    with open(os.path.join(MODEL_DIR, "lgb_combined_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    print("Model saved to models/lgb_combined_model.pkl")

    return f1_combined


if __name__ == "__main__":
    main()
