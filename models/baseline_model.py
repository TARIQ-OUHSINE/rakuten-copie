"""
Phase 1 — Baseline: TF-IDF + LightGBM (text-only)

Input : data/train_cleaned.pkl, data/test_cleaned.pkl
Output: docs/figures/mod_fig1a_lgb_text_confusion.png
        docs/figures/mod_fig1b_lgb_text_per_class_f1.png
        models/lgb_text_model.pkl   (saved model)
        models/tfidf_vectorizer.pkl (saved vectorizer)
        models/label_encoder.pkl    (saved label encoder)
"""

import os
import pickle
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import nltk
from nltk.corpus import stopwords
nltk.download("stopwords", quiet=True)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import lightgbm as lgb

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
FIG_DIR = os.path.join(ROOT, "docs", "figures")
MODEL_DIR = os.path.join(ROOT, "models")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ── Helpers ──────────────────────────────────────────────────────────────────
def weighted_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="weighted")


def plot_confusion_matrix_fig(y_true, y_pred, classes, title):
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig = px.imshow(
        cm,
        x=[str(c) for c in classes],
        y=[str(c) for c in classes],
        color_continuous_scale="Blues",
        title=title,
        labels=dict(x="Predicted", y="True", color="Rate"),
    )
    fig.update_layout(height=650)
    return fig


def plot_per_class_f1(y_true, y_pred, classes, title):
    report = classification_report(
        y_true, y_pred,
        target_names=[str(c) for c in classes],
        output_dict=True,
    )
    rows = [
        {"class": k, "f1": v["f1-score"]}
        for k, v in report.items()
        if k not in ("accuracy", "macro avg", "weighted avg")
    ]
    df = pd.DataFrame(rows).sort_values("f1", ascending=False)
    fig = px.bar(
        df, x="class", y="f1",
        title=title,
        labels={"f1": "F1-score", "class": "Category"},
        color="f1", color_continuous_scale="RdYlGn", range_y=[0, 1],
    )
    fig.add_hline(
        y=df["f1"].mean(), line_dash="dash", line_color="navy",
        annotation_text=f"Mean F1={df['f1'].mean():.3f}",
    )
    fig.update_layout(coloraxis_showscale=False, height=420)
    return fig


# ── Data ─────────────────────────────────────────────────────────────────────
def load_data():
    train = pd.read_pickle(os.path.join(DATA_DIR, "train_cleaned.pkl"))
    test = pd.read_pickle(os.path.join(DATA_DIR, "test_cleaned.pkl"))

    le = LabelEncoder()
    train["label"] = le.fit_transform(train["prdtypecode"])

    X_train, X_val, y_train, y_val = train_test_split(
        train, train["label"],
        test_size=0.20, random_state=42, stratify=train["label"],
    )
    print(f"Train: {len(X_train):,}  Val: {len(X_val):,}  Classes: {le.classes_.shape[0]}")
    return X_train, X_val, y_train, y_val, le, test


# ── TF-IDF ───────────────────────────────────────────────────────────────────
def build_tfidf(X_train, X_val):
    stop_words = list(set(
        stopwords.words("french") +
        stopwords.words("english") +
        stopwords.words("german")
    ))
    tfidf = TfidfVectorizer(
        max_features=20_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        stop_words=stop_words,
        min_df=2,
    )
    X_tr = tfidf.fit_transform(X_train["text_clean"])
    X_va = tfidf.transform(X_val["text_clean"])
    print(f"TF-IDF matrix: {X_tr.shape}")
    return tfidf, X_tr, X_va


# ── LightGBM ─────────────────────────────────────────────────────────────────
def train_lgbm(X_tr, X_va, y_train, y_val):
    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=127,
        max_depth=-1,
        min_child_samples=20,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )
    model.fit(
        X_tr, y_train,
        eval_set=[(X_va, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )
    return model


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Phase 1 — Baseline: TF-IDF + LightGBM")
    print("=" * 60)

    X_train, X_val, y_train, y_val, le, test = load_data()
    tfidf, X_tr_tfidf, X_va_tfidf = build_tfidf(X_train, X_val)

    print("\nTraining LightGBM...")
    model = train_lgbm(X_tr_tfidf, X_va_tfidf, y_train, y_val)

    y_pred = model.predict(X_va_tfidf)
    f1 = weighted_f1(y_val, y_pred)
    print(f"\nLightGBM (TF-IDF) — weighted F1: {f1:.4f}")
    print(f"Best iteration: {model.best_iteration_}")
    print("\nClassification report:")
    print(classification_report(y_val, y_pred, target_names=[str(c) for c in le.classes_]))

    # Figures
    fig_cm = plot_confusion_matrix_fig(
        y_val, y_pred, le.classes_,
        f"Baseline — LightGBM (text-only) Confusion Matrix  [F1={f1:.4f}]",
    )
    fig_f1 = plot_per_class_f1(
        y_val, y_pred, le.classes_,
        "Baseline — LightGBM Per-Class F1 (text-only)",
    )
    try:
        fig_cm.write_image(os.path.join(FIG_DIR, "mod_fig1a_lgb_text_confusion.png"), scale=2)
        fig_f1.write_image(os.path.join(FIG_DIR, "mod_fig1b_lgb_text_per_class_f1.png"), scale=2)
        print("Figures saved to docs/figures/")
    except Exception as e:
        print(f"Figure export skipped: {e}")

    # Save artifacts
    with open(os.path.join(MODEL_DIR, "lgb_text_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(tfidf, f)
    with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)
    print("Models saved to models/")

    return f1


if __name__ == "__main__":
    main()
