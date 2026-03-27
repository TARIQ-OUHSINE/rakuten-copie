"""
TF-IDF with language-aware preprocessing.

Pipeline:
  1. Detect language per row (langdetect)
  2. Translate non-French text → French (deep-translator / Google)
  3. Apply TF-IDF on unified French text

Input : data/train_cleaned.pkl, data/test_cleaned.pkl
Output: data/train_translated.pkl  (adds 'lang' + 'text_fr' columns)
        data/test_translated.pkl
        models/tfidf_translated_vectorizer.pkl
        models/lgb_translated_model.pkl
        models/label_encoder.pkl
        docs/figures/mod_fig_lang_distribution.png
        docs/figures/mod_fig_translated_confusion.png
        docs/figures/mod_fig_translated_per_class_f1.png

Requirements:
  pip install langdetect deep-translator
"""

import os
import re
import pickle
import time
import pandas as pd
import numpy as np
import plotly.express as px

from langdetect import detect, LangDetectException
from deep_translator import GoogleTranslator
from tqdm import tqdm

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


# ── 1. Language Detection ────────────────────────────────────────────────────
def detect_language(text):
    """Detect language of a text string. Returns ISO 639-1 code."""
    if not text or len(text.strip()) < 3:
        return "unknown"
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def add_language_column(df):
    """Add a 'lang' column with detected language for each row."""
    tqdm.pandas(desc="Detecting languages")
    df["lang"] = df["text_clean"].progress_apply(detect_language)
    return df


# ── 2. Translation ───────────────────────────────────────────────────────────
# Google Translate free tier has rate limits — we batch and retry.
SUPPORTED_SOURCES = {"en", "de", "es", "it", "pt", "nl", "ja", "ko", "zh-cn"}


def translate_batch(texts, source_lang, target_lang="fr", batch_size=50):
    """Translate a list of texts from source_lang to target_lang."""
    translator = GoogleTranslator(source=source_lang, target=target_lang)
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        for text in batch:
            try:
                # deep-translator handles one string at a time
                translated = translator.translate(text[:4500])  # API char limit
                results.append(translated if translated else text)
            except Exception:
                results.append(text)  # keep original on failure
            time.sleep(0.05)  # rate-limit politeness
    return results


def translate_to_french(df):
    """Translate non-French text_clean → French. Adds 'text_fr' column."""
    df["text_fr"] = df["text_clean"].copy()

    for lang in SUPPORTED_SOURCES:
        mask = df["lang"] == lang
        n = mask.sum()
        if n == 0:
            continue
        print(f"Translating {n:,} rows from '{lang}' → 'fr' ...")
        texts = df.loc[mask, "text_clean"].tolist()
        translated = translate_batch(texts, source_lang=lang, target_lang="fr")
        df.loc[mask, "text_fr"] = translated

    # For unsupported / unknown languages, keep original text
    return df


# ── 3. Text normalization (post-translation) ────────────────────────────────
def normalize_text(text):
    """Lowercase, strip punctuation/digits, collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── 4. TF-IDF on unified French text ────────────────────────────────────────
def build_tfidf_french(X_train, X_val):
    """Build TF-IDF using French stopwords only (all text is now French)."""
    french_stops = list(set(stopwords.words("french")))

    tfidf = TfidfVectorizer(
        max_features=10_000,
        ngram_range=(1, 1),
        sublinear_tf=True,
        stop_words=french_stops,
        min_df=2,
    )
    X_tr = tfidf.fit_transform(X_train["text_fr_clean"])
    X_va = tfidf.transform(X_val["text_fr_clean"])
    print(f"TF-IDF matrix: {X_tr.shape}")
    return tfidf, X_tr, X_va


# ── 5. LightGBM ─────────────────────────────────────────────────────────────
class TqdmCallback:
    """tqdm progress bar callback for LightGBM."""
    def __init__(self, total):
        self.pbar = tqdm(total=total, desc="LightGBM training", unit="iter")

    def __call__(self, env):
        self.pbar.update(1)
        if env.evaluation_result_list:
            score = env.evaluation_result_list[0][2]
            self.pbar.set_postfix(val_loss=f"{score:.4f}", iter=env.iteration)
        if env.iteration == env.end_iteration - 1:
            self.pbar.close()


def train_lgbm(X_tr, X_va, y_train, y_val):
    n_estimators = 500
    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=31,
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
            TqdmCallback(total=n_estimators),
        ],
    )
    return model


# ── Helpers ──────────────────────────────────────────────────────────────────
def weighted_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="weighted")


def plot_lang_distribution(df, title="Language Distribution in Dataset"):
    lang_counts = df["lang"].value_counts().reset_index()
    lang_counts.columns = ["language", "count"]
    fig = px.bar(
        lang_counts, x="language", y="count",
        title=title,
        labels={"language": "Detected Language", "count": "Number of Products"},
        color="count", color_continuous_scale="Viridis",
    )
    fig.update_layout(coloraxis_showscale=False, height=420)
    return fig


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


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("TF-IDF with Language Detection + Translation")
    print("=" * 60)

    # Load data
    train = pd.read_pickle(os.path.join(DATA_DIR, "train_cleaned.pkl"))
    test = pd.read_pickle(os.path.join(DATA_DIR, "test_cleaned.pkl"))

    # Check for cached translated data
    train_trans_path = os.path.join(DATA_DIR, "train_translated.pkl")
    test_trans_path = os.path.join(DATA_DIR, "test_translated.pkl")

    if os.path.exists(train_trans_path) and os.path.exists(test_trans_path):
        print("Loading cached translated data...")
        train = pd.read_pickle(train_trans_path)
        test = pd.read_pickle(test_trans_path)
    else:
        # Step 1: Detect languages
        print("\n── Step 1: Language Detection ──")
        train = add_language_column(train)
        test = add_language_column(test)

        print(f"\nTrain language distribution:\n{train['lang'].value_counts()}")
        print(f"\nTest language distribution:\n{test['lang'].value_counts()}")

        # Plot language distribution
        fig_lang = plot_lang_distribution(train, "Language Distribution (Train)")
        try:
            fig_lang.write_image(os.path.join(FIG_DIR, "mod_fig_lang_distribution.png"), scale=2)
            print("Saved mod_fig_lang_distribution.png")
        except Exception as e:
            print(f"Figure export skipped: {e}")

        # Step 2: Translate non-French text → French
        print("\n── Step 2: Translation to French ──")
        train = translate_to_french(train)
        test = translate_to_french(test)

        # Step 3: Normalize translated text
        print("\n── Step 3: Normalizing translated text ──")
        train["text_fr_clean"] = train["text_fr"].apply(normalize_text)
        test["text_fr_clean"] = test["text_fr"].apply(normalize_text)

        # Cache translated data
        train.to_pickle(train_trans_path)
        test.to_pickle(test_trans_path)
        print(f"Cached translated data to {train_trans_path}")

    # Encode labels
    le = LabelEncoder()
    train["label"] = le.fit_transform(train["prdtypecode"])

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        train, train["label"],
        test_size=0.20, random_state=42, stratify=train["label"],
    )
    print(f"\nTrain: {len(X_train):,}  Val: {len(X_val):,}  Classes: {le.classes_.shape[0]}")

    # Step 4: TF-IDF on unified French text
    print("\n── Step 4: TF-IDF (French-only stopwords) ──")
    tfidf, X_tr_tfidf, X_va_tfidf = build_tfidf_french(X_train, X_val)

    # Step 5: Train LightGBM
    print("\n── Step 5: Training LightGBM ──")
    model = train_lgbm(X_tr_tfidf, X_va_tfidf, y_train, y_val)

    y_pred = model.predict(X_va_tfidf)
    f1 = weighted_f1(y_val, y_pred)
    print(f"\nLightGBM (translated TF-IDF) — weighted F1: {f1:.4f}")
    print(f"Best iteration: {model.best_iteration_}")
    print("\nClassification report:")
    print(classification_report(y_val, y_pred, target_names=[str(c) for c in le.classes_]))

    # Figures
    fig_cm = plot_confusion_matrix_fig(
        y_val, y_pred, le.classes_,
        f"Translated TF-IDF + LightGBM Confusion Matrix  [F1={f1:.4f}]",
    )
    fig_f1 = plot_per_class_f1(
        y_val, y_pred, le.classes_,
        "Translated TF-IDF — Per-Class F1",
    )
    try:
        fig_cm.write_image(os.path.join(FIG_DIR, "mod_fig_translated_confusion.png"), scale=2)
        fig_f1.write_image(os.path.join(FIG_DIR, "mod_fig_translated_per_class_f1.png"), scale=2)
        print("Figures saved to docs/figures/")
    except Exception as e:
        print(f"Figure export skipped: {e}")

    # Save artifacts
    with open(os.path.join(MODEL_DIR, "tfidf_translated_vectorizer.pkl"), "wb") as f:
        pickle.dump(tfidf, f)
    with open(os.path.join(MODEL_DIR, "lgb_translated_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)
    print("Models saved to models/")

    return f1


if __name__ == "__main__":
    main()
