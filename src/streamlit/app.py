"""
Rakuten France — Classification Multimodale
Streamlit demo app (inference only — no retraining).

Lancement :
    streamlit run src/streamlit/app.py
"""

from __future__ import annotations

import pickle
import re
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image
from scipy.sparse import csr_matrix, hstack

# ── Project paths ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
FIG_DIR = ROOT / "docs" / "figures"
REPORT_FIG_DIR = ROOT / "reports" / "figures"

sys.path.insert(0, str(ROOT))

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Rakuten — Classification Multimodale",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    :root {
        --primary: #BF0000;
        --primary-dark: #8B0000;
        --primary-light: #ff4444;
        --bg-card: #f8f9fa;
        --border: #e0e0e0;
        --text-muted: #6c757d;
    }
    .stApp { background: #ffffff; }

    /* Hero header */
    .hero {
        background: linear-gradient(135deg, #BF0000 0%, #8B0000 100%);
        color: white;
        padding: 2.2rem 3rem;
        border-radius: 14px;
        margin-bottom: 1.8rem;
        box-shadow: 0 6px 24px rgba(191,0,0,0.25);
    }
    .hero h1 { font-size: 2.3rem; font-weight: 800; margin: 0; }
    .hero p { font-size: 1.05rem; opacity: 0.92; margin: 0.55rem 0 0; }

    /* KPI cards */
    .kpi {
        background: white;
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.2rem 1rem;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border-top: 4px solid var(--primary);
        height: 100%;
    }
    .kpi-value { font-size: 1.9rem; font-weight: 800; color: var(--primary); line-height: 1.1; }
    .kpi-label { font-size: 0.78rem; color: var(--text-muted); margin-top: 0.4rem;
                 letter-spacing: 0.06em; text-transform: uppercase; }

    /* Section title */
    .section-title {
        font-size: 1.25rem; font-weight: 700; color: #2c3e50;
        border-left: 4px solid var(--primary); padding-left: 0.8rem;
        margin: 1.2rem 0 0.9rem;
    }

    /* Prediction cards */
    .pred-card {
        background: white;
        border: 2px solid var(--primary);
        border-radius: 12px;
        padding: 1.3rem;
        box-shadow: 0 4px 14px rgba(191,0,0,0.12);
    }
    .pred-card-secondary {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.3rem;
    }
    .pred-model { font-size: 0.78rem; color: var(--text-muted);
                  text-transform: uppercase; letter-spacing: 0.06em; }
    .pred-class { font-size: 1.5rem; font-weight: 800; color: var(--primary);
                  margin: 0.3rem 0; }
    .pred-class-secondary { color: #2c3e50; }
    .pred-conf { font-size: 0.95rem; color: var(--text-muted); }

    /* Comment box */
    .comment {
        background: #fff7e6;
        border-left: 4px solid #f39c12;
        padding: 0.85rem 1rem;
        border-radius: 6px;
        margin: 0.6rem 0 1.2rem;
        font-size: 0.93rem;
        color: #5d4e2a;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        background: #f1f3f5; border-radius: 8px 8px 0 0;
        padding: 0.6rem 1.4rem; font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: var(--primary) !important; color: white !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ── Class names (Rakuten France 27 categories) ───────────────────────────────
CLASS_NAMES = {
    10:   "Livre occasion",
    40:   "Jeux vidéo neufs",
    50:   "Accessoires gaming",
    60:   "Consoles de jeux",
    1140: "Figurines",
    1160: "Cartes à collectionner",
    1180: "Jeux de rôle / Figurines",
    1280: "Jouets enfants",
    1281: "Jeux de société",
    1300: "Modélisme",
    1301: "Articles bébé",
    1302: "Jouets de plein air",
    1320: "Puériculture / Peluches",
    1560: "Mobilier",
    1920: "Linge de maison",
    1940: "Épicerie",
    2060: "Décoration",
    2220: "Animalerie",
    2280: "Magazines / Journaux",
    2403: "Livres et BD anciens",
    2462: "Jeux occasion",
    2522: "Papeterie",
    2582: "Mobilier de jardin",
    2583: "Piscine / Spa",
    2585: "Outillage de jardin",
    2705: "Livres neufs",
    2905: "Jeu PC dématérialisé",
}


# ── Helpers ──────────────────────────────────────────────────────────────────
def kpi(value: str, label: str) -> str:
    return f'<div class="kpi"><div class="kpi-value">{value}</div><div class="kpi-label">{label}</div></div>'


def comment(text: str) -> None:
    st.markdown(f'<div class="comment">💡 {text}</div>', unsafe_allow_html=True)


def clean_text(text: str) -> str:
    """Light cleaning consistent with baseline_model preprocessing."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.lower()
    text = re.sub(r"[^\w\sàâäéèêëïîôöùûüç]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def find_figure(name: str) -> Path | None:
    for d in (FIG_DIR, REPORT_FIG_DIR):
        p = d / name
        if p.exists():
            return p
    return None


# ── Cached loaders ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Chargement des modèles…")
def load_models():
    artifacts = {}
    with open(MODEL_DIR / "tfidf_vectorizer.pkl", "rb") as f:
        artifacts["tfidf"] = pickle.load(f)
    with open(MODEL_DIR / "tfidf_translated_vectorizer.pkl", "rb") as f:
        artifacts["tfidf_tr"] = pickle.load(f)
    with open(MODEL_DIR / "label_encoder.pkl", "rb") as f:
        artifacts["le"] = pickle.load(f)
    with open(MODEL_DIR / "lgb_text_model.pkl", "rb") as f:
        artifacts["lgb_text"] = pickle.load(f)
    with open(MODEL_DIR / "lgb_translated_model.pkl", "rb") as f:
        artifacts["lgb_tr"] = pickle.load(f)
    with open(MODEL_DIR / "lgb_combined_model.pkl", "rb") as f:
        artifacts["lgb_combined"] = pickle.load(f)
    return artifacts


@st.cache_data(show_spinner=False)
def load_class_stats() -> pd.DataFrame:
    """Compute per-class stats from train_cleaned.pkl (cached)."""
    train = pd.read_pickle(DATA_DIR / "train_cleaned.pkl")
    rows = []
    for code, sub in train.groupby("prdtypecode"):
        rows.append({
            "code": int(code),
            "classe": CLASS_NAMES.get(int(code), f"Classe {code}"),
            "n_produits": len(sub),
            "pct_total": 100 * len(sub) / len(train),
            "pct_missing_desc": 100 * (1 - sub["has_description"].mean()),
            "mots_moyens": sub["word_count"].mean(),
        })
    return pd.DataFrame(rows).sort_values("n_produits", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_lang_stats() -> pd.DataFrame | None:
    p = DATA_DIR / "train_translated.pkl"
    if not p.exists():
        return None
    df = pd.read_pickle(p)
    if "lang" not in df.columns:
        return None
    counts = df["lang"].value_counts().reset_index()
    counts.columns = ["langue", "n_produits"]
    counts["pct"] = 100 * counts["n_produits"] / counts["n_produits"].sum()
    return counts


@st.cache_data(show_spinner=False)
def load_sample_designations(n: int = 6) -> list[dict]:
    """A few example products for the demo tab."""
    train = pd.read_pickle(DATA_DIR / "train_cleaned.pkl")
    rng = np.random.default_rng(42)
    idx = rng.choice(len(train), size=n, replace=False)
    samples = []
    for i in idx:
        row = train.iloc[i]
        samples.append({
            "designation": str(row["designation"])[:200],
            "description": str(row["description"])[:400] if pd.notna(row["description"]) else "",
            "true_code": int(row["prdtypecode"]),
            "true_name": CLASS_NAMES.get(int(row["prdtypecode"]), "?"),
        })
    return samples


# ── OpenCV feature extraction (for combined model) ───────────────────────────
def extract_image_features(pil_img: Image.Image) -> np.ndarray:
    """Color hist (HSV) + HOG + LBP — same layout as advanced_model.py (8222-d)."""
    import cv2
    from skimage.feature import local_binary_pattern

    img_rgb = np.array(pil_img.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # Color histogram (HSV, 3×32 bins = 96-d)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([img_hsv], [0], None, [32], [0, 180]).flatten()
    s = cv2.calcHist([img_hsv], [1], None, [32], [0, 256]).flatten()
    v = cv2.calcHist([img_hsv], [2], None, [32], [0, 256]).flatten()
    color = np.concatenate([h, s, v])
    color = color / (color.sum() + 1e-7)

    # HOG (128×128, 8100-d)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128))
    hog_desc = cv2.HOGDescriptor(
        _winSize=(128, 128), _blockSize=(16, 16),
        _blockStride=(8, 8), _cellSize=(8, 8), _nbins=9,
    )
    hog = hog_desc.compute(gray).flatten()

    # LBP (24 points, r=3 → 26-d)
    lbp = local_binary_pattern(gray, 24, 3, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=26, range=(0, 26), density=True)

    return np.concatenate([color, hog, lbp_hist])


# ── Prediction helpers ───────────────────────────────────────────────────────
def predict_top3(model, tfidf_vec, designation: str, description: str,
                 image_feats: np.ndarray | None = None) -> list[tuple[int, str, float]]:
    text = clean_text(f"{designation} {description}")
    if not text:
        return []
    X_text = tfidf_vec.transform([text])
    if image_feats is not None:
        X = hstack([X_text, csr_matrix(image_feats.reshape(1, -1))])
    else:
        X = X_text
    try:
        proba = model.predict_proba(X)[0]
    except Exception as exc:
        st.error(f"Erreur de prédiction : {exc}")
        return []
    le = ARTIFACTS["le"]
    top_idx = np.argsort(proba)[::-1][:3]
    out = []
    for i in top_idx:
        code = int(le.classes_[i])
        out.append((code, CLASS_NAMES.get(code, f"Classe {code}"), float(proba[i])))
    return out


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="hero">
    <h1>🛒 Rakuten France — Classification Multimodale</h1>
    <p>84 916 produits · 27 catégories · Texte + Image · Challenge SIGIR eCommerce 2020</p>
</div>
""",
    unsafe_allow_html=True,
)

# Load artifacts once (used across tabs)
ARTIFACTS = load_models()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Présentation",
    "📊 Exploration des données",
    "🤖 Modélisation",
    "🔮 Démo Inférence",
    "📚 Méthodologie",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — PRÉSENTATION
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">Contexte du projet</div>', unsafe_allow_html=True)

    col_text, col_img = st.columns([2, 1])
    with col_text:
        st.markdown("""
**Rakuten France** est l'une des plus grandes marketplaces e-commerce en Europe. Chaque jour,
des milliers de vendeurs ajoutent de nouveaux produits sur la plateforme. La **classification
automatique du type de produit** (`prdtypecode`) est essentielle pour :

- 🔎 Le **moteur de recherche** : trouver rapidement les produits pertinents
- 📂 L'**organisation du catalogue** : navigation par catégorie
- 🎯 Les **recommandations** : suggérer des produits similaires
- 💼 La **gestion commerciale** : analyses de marché par segment

**Le défi** : les listings contiennent du texte (souvent court et bruité, multilingue) et
une image produit. Ces deux modalités doivent être combinées intelligemment pour atteindre
une classification fiable sur **27 catégories** très hétérogènes.

> *Source : Challenge SIGIR eCommerce 2020, organisé conjointement par Rakuten France
> et l'ACM SIGIR. La baseline officielle (TF-IDF + classifieur linéaire) atteint
> ~0.87 de F1 ; la meilleure soumission ~0.92.*
        """)
    with col_img:
        st.markdown(
            """
<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
            height:100%;min-height:280px;padding:2rem 1rem;
            background:linear-gradient(135deg,#BF0000,#8B0000);
            border-radius:14px;color:white;
            box-shadow:0 6px 20px rgba(191,0,0,0.25);">
    <div style="font-size:5rem;font-weight:900;line-height:1;">R</div>
    <div style="font-size:1.4rem;font-weight:700;margin-top:0.4rem;letter-spacing:0.05em;">Rakuten</div>
    <div style="font-size:0.85rem;opacity:0.85;margin-top:0.3rem;">SIGIR eCommerce 2020</div>
    <div style="font-size:0.75rem;opacity:0.7;margin-top:1rem;">🇫🇷 France marketplace</div>
</div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">Vue d\'ensemble du dataset</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.markdown(kpi("84 916", "Produits"), unsafe_allow_html=True)
    with c2: st.markdown(kpi("27", "Catégories"), unsafe_allow_html=True)
    with c3: st.markdown(kpi("35%", "Desc. manquantes"), unsafe_allow_html=True)
    with c4: st.markdown(kpi("79%", "Produits en FR"), unsafe_allow_html=True)
    with c5: st.markdown(kpi("13×", "Ratio déséquilibre"), unsafe_allow_html=True)

    st.markdown('<div class="section-title">Approche en 4 phases</div>', unsafe_allow_html=True)
    phases = pd.DataFrame([
        {"Phase": "1 — Baseline", "Modèle": "TF-IDF + LightGBM", "Modalités": "Texte", "Statut": "✅ Livré"},
        {"Phase": "2 — Optimisation", "Modèle": "TF-IDF + OpenCV + LightGBM", "Modalités": "Texte + Image", "Statut": "✅ Livré"},
        {"Phase": "2 bis — Translation", "Modèle": "TF-IDF (FR) + LightGBM", "Modalités": "Texte traduit FR", "Statut": "✅ Livré"},
        {"Phase": "3 — Avancé", "Modèle": "CLIP ViT-B/32 + MLP", "Modalités": "Embeddings multimodaux", "Statut": "🧪 Expérimental"},
        {"Phase": "4 — Fine-tuning", "Modèle": "GPT-4o Vision", "Modalités": "Multimodal natif", "Statut": "🧪 Expérimental"},
    ])
    st.dataframe(phases, width="stretch", hide_index=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — EXPLORATION
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    stats = load_class_stats()

    # ── Class distribution ──
    st.markdown('<div class="section-title">Répartition des 27 catégories</div>', unsafe_allow_html=True)
    fig_class = px.bar(
        stats.sort_values("n_produits", ascending=True),
        x="n_produits", y="classe", orientation="h",
        color="n_produits", color_continuous_scale="Reds",
        template="plotly_white", height=560,
        labels={"n_produits": "Nb produits", "classe": ""},
    )
    fig_class.update_coloraxes(showscale=False)
    fig_class.update_layout(margin=dict(l=10, r=20, t=10, b=20))
    st.plotly_chart(fig_class, width="stretch")
    comment(f"Dataset très **déséquilibré** : la classe la plus représentée "
            f"({stats.iloc[0]['classe']}, {stats.iloc[0]['n_produits']:,} produits) "
            f"est ~{stats.iloc[0]['n_produits'] / stats.iloc[-1]['n_produits']:.0f}× "
            f"plus fréquente que la moins représentée. Justifie l'usage de `class_weight=balanced` et du F1 pondéré.")

    # ── Missing description ──
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-title">Descriptions manquantes par classe</div>', unsafe_allow_html=True)
        miss = stats.sort_values("pct_missing_desc", ascending=True)
        fig_miss = px.bar(
            miss, x="pct_missing_desc", y="classe", orientation="h",
            color="pct_missing_desc",
            color_continuous_scale=["#2ecc71", "#f39c12", "#e74c3c"],
            template="plotly_white", height=520,
            labels={"pct_missing_desc": "% manquant", "classe": ""},
        )
        fig_miss.update_coloraxes(showscale=False)
        fig_miss.add_vline(x=35, line_dash="dash", line_color="navy",
                           annotation_text="Moyenne 35%", annotation_position="top right")
        fig_miss.update_layout(margin=dict(l=10, r=20, t=10, b=20))
        st.plotly_chart(fig_miss, width="stretch")

    with col_b:
        st.markdown('<div class="section-title">Longueur moyenne du texte (mots)</div>', unsafe_allow_html=True)
        wc = stats.sort_values("mots_moyens", ascending=True)
        fig_wc = px.bar(
            wc, x="mots_moyens", y="classe", orientation="h",
            color="mots_moyens", color_continuous_scale="Blues",
            template="plotly_white", height=520,
            labels={"mots_moyens": "Mots / produit", "classe": ""},
        )
        fig_wc.update_coloraxes(showscale=False)
        fig_wc.update_layout(margin=dict(l=10, r=20, t=10, b=20))
        st.plotly_chart(fig_wc, width="stretch")

    comment("Les produits avec **peu de texte** (jouets, accessoires) souffrent davantage "
            "d'un classifieur purement textuel. C'est pour eux que la branche **image** "
            "(Phase 2) apporte le plus de gain.")

    # ── Language distribution ──
    lang_df = load_lang_stats()
    if lang_df is not None:
        st.markdown('<div class="section-title">Distribution des langues détectées</div>', unsafe_allow_html=True)
        top_langs = lang_df.head(12)
        fig_lang = px.bar(
            top_langs, x="langue", y="n_produits",
            color="n_produits", color_continuous_scale="Viridis",
            template="plotly_white", height=380,
            labels={"langue": "Langue (ISO 639-1)", "n_produits": "Nb produits"},
            text="n_produits",
        )
        fig_lang.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig_lang.update_coloraxes(showscale=False)
        st.plotly_chart(fig_lang, width="stretch")
        fr_pct = lang_df[lang_df["langue"] == "fr"]["pct"].iloc[0] if "fr" in lang_df["langue"].values else 0
        comment(f"Le français domine ({fr_pct:.0f}%), mais l'anglais représente ~12% "
                "du corpus. La Phase 2bis traduit le texte non-FR en FR pour unifier le vocabulaire TF-IDF.")

    # ── Pre-rendered figures ──
    st.markdown('<div class="section-title">Visualisations complémentaires</div>', unsafe_allow_html=True)
    figs = [
        ("fig5_wordclouds.png", "Nuages de mots par classe"),
        ("fig4_image_grid.png", "Exemples d'images produits"),
        ("fig3_text_length.png", "Distribution de la longueur du texte"),
        ("fig6_image_sizes.png", "Tailles des images produits"),
    ]
    cols = st.columns(2)
    for i, (fname, caption) in enumerate(figs):
        p = find_figure(fname)
        if p is not None:
            with cols[i % 2]:
                st.image(str(p), caption=caption, width="stretch")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODÉLISATION
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Comparaison des modèles</div>', unsafe_allow_html=True)

    bench = pd.DataFrame([
        {"Modèle": "SIGIR 2020 — 1ère place",   "F1 pondéré": 0.9194, "Type": "Référence", "Modalités": "Multi"},
        {"Modèle": "SIGIR 2020 — Baseline",     "F1 pondéré": 0.8705, "Type": "Référence", "Modalités": "Texte"},
        {"Modèle": "Phase 2 — TF-IDF + OpenCV", "F1 pondéré": 0.8200, "Type": "Notre modèle", "Modalités": "Texte + Image"},
        {"Modèle": "Phase 2bis — TF-IDF traduit","F1 pondéré": 0.8100, "Type": "Notre modèle", "Modalités": "Texte (FR)"},
        {"Modèle": "Phase 1 — TF-IDF baseline", "F1 pondéré": 0.8000, "Type": "Notre modèle", "Modalités": "Texte"},
    ])
    fig_bench = px.bar(
        bench.sort_values("F1 pondéré", ascending=True),
        x="F1 pondéré", y="Modèle", orientation="h",
        color="Type", color_discrete_map={"Référence": "#f39c12", "Notre modèle": "#BF0000"},
        text="F1 pondéré", template="plotly_white", height=420,
        range_x=[0.75, 0.95],
    )
    fig_bench.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig_bench.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            margin=dict(l=10, r=40, t=30, b=20))
    st.plotly_chart(fig_bench, width="stretch")
    comment("Ajouter la branche image (OpenCV) apporte **+2 points** par rapport au texte seul. "
            "La traduction FR unifie le vocabulaire pour un gain modeste mais robuste.")

    # ── Phase 1 ──
    st.markdown('<div class="section-title">Phase 1 — Baseline texte (TF-IDF + LightGBM)</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        p = find_figure("mod_fig1a_lgb_text_confusion.png")
        if p: st.image(str(p), caption="Matrice de confusion normalisée", width="stretch")
    with col2:
        p = find_figure("mod_fig1b_lgb_text_per_class_f1.png")
        if p: st.image(str(p), caption="F1-score par classe", width="stretch")
    comment("Les classes textuellement proches (1280 / 1281 jeux enfants, 2280 / 2403 / 2705 livres) "
            "sont systématiquement confondues. Sans image, le modèle n'a pas assez de signal pour discriminer.")

    # ── Phase 2bis ──
    st.markdown('<div class="section-title">Phase 2bis — Traduction FR + TF-IDF</div>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        p = find_figure("mod_fig_translated_confusion.png")
        if p: st.image(str(p), caption="Confusion après unification FR", width="stretch")
    with col4:
        p = find_figure("mod_fig_translated_per_class_f1.png")
        if p: st.image(str(p), caption="F1 par classe — texte traduit", width="stretch")

    # ── Phase 2 ──
    st.markdown('<div class="section-title">Phase 2 — Ajout des features OpenCV</div>', unsafe_allow_html=True)
    col5, col6 = st.columns(2)
    with col5:
        p = find_figure("mod_fig3_opencv_features.png")
        if p: st.image(str(p), caption="Features OpenCV (HSV / HOG / LBP) — exemple", width="stretch")
    with col6:
        p = find_figure("mod_fig4_ablation_opencv.png")
        if p: st.image(str(p), caption="Étude d'ablation : contribution de chaque groupe", width="stretch")
    comment("**HOG** (descripteur de forme) apporte le plus de gain, suivi de l'histogramme HSV. "
            "LBP (texture) seul n'est pas suffisant.")

    # ── Interpretability ──
    st.markdown('<div class="section-title">Interprétabilité (Phase 2)</div>', unsafe_allow_html=True)
    col7, col8 = st.columns(2)
    with col7:
        p = find_figure("mod_fig5_feature_importance.png")
        if p: st.image(str(p), caption="Top 50 features (LightGBM)", width="stretch")
    with col8:
        p = find_figure("mod_fig6_shap.png")
        if p: st.image(str(p), caption="SHAP — Top 30 features (toutes classes)", width="stretch")
    comment("Les features dominantes sont des **mots-clés TF-IDF** (livre, jeu, jouet, neuf…) "
            "complétés par quelques HOG très discriminants pour les images stéréotypées "
            "(consoles, couvertures de livres). C'est exactement ce qu'on attend d'un modèle multimodal sain.")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — DÉMO INFÉRENCE
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">Tester un produit en direct</div>', unsafe_allow_html=True)
    st.caption("Saisissez un titre + description (optionnellement une image), comparez les 3 modèles.")

    samples = load_sample_designations(n=6)
    sample_labels = ["✏️ Saisie manuelle"] + [
        f"#{i+1} — {s['true_name']} ({str(s['designation'])[:60]}…)"
        for i, s in enumerate(samples)
    ]
    choice = st.selectbox("Exemple :", sample_labels, index=0)

    if choice != sample_labels[0]:
        idx = sample_labels.index(choice) - 1
        default_des = samples[idx]["designation"]
        default_desc = samples[idx]["description"]
        true_label = f"{samples[idx]['true_code']} — {samples[idx]['true_name']}"
    else:
        default_des = ""
        default_desc = ""
        true_label = None

    col_form, col_img = st.columns([2, 1])
    with col_form:
        designation = st.text_input("Désignation (titre)", value=default_des, key="des_input")
        description = st.text_area("Description", value=default_desc, height=140, key="desc_input")
    with col_img:
        st.markdown("**Image produit (optionnelle)**")
        uploaded = st.file_uploader("Charger une image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        pil_image = None
        if uploaded is not None:
            pil_image = Image.open(BytesIO(uploaded.read()))
            st.image(pil_image, width="stretch")

    predict = st.button("🔮 Prédire", type="primary", width="stretch")

    if predict:
        if not designation and not description:
            st.warning("Saisissez au moins une désignation ou une description.")
        else:
            if true_label:
                st.info(f"**Vraie classe :** {true_label}")

            with st.spinner("Inférence en cours…"):
                pred_text = predict_top3(
                    ARTIFACTS["lgb_text"], ARTIFACTS["tfidf"], designation, description,
                )
                pred_tr = predict_top3(
                    ARTIFACTS["lgb_tr"], ARTIFACTS["tfidf_tr"], designation, description,
                )
                if pil_image is not None:
                    img_feats = extract_image_features(pil_image)
                    pred_combined = predict_top3(
                        ARTIFACTS["lgb_combined"], ARTIFACTS["tfidf"], designation, description,
                        image_feats=img_feats,
                    )
                else:
                    pred_combined = None

            c1, c2, c3 = st.columns(3)

            def render_pred(col, title, preds, primary=False):
                with col:
                    cls = "pred-card" if primary else "pred-card-secondary"
                    cls2 = "" if primary else "pred-class-secondary"
                    if not preds:
                        st.markdown(f'<div class="{cls}">'
                                    f'<div class="pred-model">{title}</div>'
                                    f'<div class="pred-class {cls2}">—</div>'
                                    f'<div class="pred-conf">Pas de texte exploitable</div>'
                                    f'</div>', unsafe_allow_html=True)
                        return
                    top1 = preds[0]
                    rest = "<br>".join(
                        f'<span style="color:#6c757d;">{n}: <b>{p:.1%}</b></span>'
                        for _, n, p in preds[1:]
                    )
                    st.markdown(
                        f'<div class="{cls}">'
                        f'<div class="pred-model">{title}</div>'
                        f'<div class="pred-class {cls2}">{top1[1]}</div>'
                        f'<div class="pred-conf">Code <b>{top1[0]}</b> · Confiance <b>{top1[2]:.1%}</b></div>'
                        f'<hr style="margin:0.6rem 0;border:0;border-top:1px solid #e0e0e0;">'
                        f'<div style="font-size:0.85rem;">{rest}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            render_pred(c1, "Phase 1 — Texte (TF-IDF)", pred_text)
            render_pred(c2, "Phase 2bis — Texte traduit", pred_tr)
            if pred_combined is not None:
                render_pred(c3, "Phase 2 — Texte + Image", pred_combined, primary=True)
            else:
                with c3:
                    st.markdown('<div class="pred-card-secondary">'
                                '<div class="pred-model">Phase 2 — Texte + Image</div>'
                                '<div class="pred-class pred-class-secondary">—</div>'
                                '<div class="pred-conf">📷 Charger une image pour activer ce modèle</div>'
                                '</div>', unsafe_allow_html=True)

            # Top-3 probabilities (best available model)
            best_preds = pred_combined or pred_text or pred_tr
            if best_preds:
                df_preds = pd.DataFrame(best_preds, columns=["code", "Classe", "Probabilité"])
                fig_p = px.bar(
                    df_preds, x="Probabilité", y="Classe", orientation="h",
                    color="Probabilité", color_continuous_scale="Reds",
                    template="plotly_white", height=240,
                    range_x=[0, 1], text="Probabilité",
                )
                fig_p.update_traces(texttemplate="%{text:.1%}", textposition="outside")
                fig_p.update_coloraxes(showscale=False)
                fig_p.update_layout(yaxis={"categoryorder": "total ascending"},
                                    margin=dict(l=10, r=50, t=20, b=20))
                st.plotly_chart(fig_p, width="stretch")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — MÉTHODOLOGIE
# ═════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-title">Pipeline complet</div>', unsafe_allow_html=True)
    st.code(
        """
                ┌──────────────────┐
designation ───►│   Cleaning       │
description ───►│  (HTML, lower,   │────┐
                │   stopwords)     │    │
                └──────────────────┘    │
                                        ▼
          ┌──────────────────┐    ┌──────────────┐
          │ langdetect       │───►│ Translation  │
          │ (fr/en/de/it/…)  │    │     → FR     │
          └──────────────────┘    └──────┬───────┘
                                         │
                                         ▼
                                ┌─────────────────┐
                                │   TF-IDF 10K    │
                                │   (uni-grams)   │
                                └────────┬────────┘
                                         │
image ───► OpenCV (HSV 96d + HOG 8100d ──┤
                  + LBP 26d = 8222d)     │
                                         ▼
                                ┌─────────────────┐
                                │    LightGBM     │
                                │  class_weight = │
                                │     balanced    │
                                └────────┬────────┘
                                         ▼
                            27 prdtypecode probabilities
        """,
        language=None,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">Choix techniques</div>', unsafe_allow_html=True)
        st.markdown("""
- **TF-IDF (10 000 features)** : compromis vitesse / qualité, sublinear TF, min_df = 2
- **LightGBM** : sparse natif, scaling propre sur 28 K features
- **OpenCV** : HOG (forme), HSV (couleur), LBP (texture) — pas de deep learning lourd
- **Translation langdetect → Google** : cache disque, supporte 9 langues
- **`class_weight=\"balanced\"`** : indispensable vu le déséquilibre 13×
        """)
    with col2:
        st.markdown('<div class="section-title">Limites identifiées</div>', unsafe_allow_html=True)
        st.markdown("""
- **TF-IDF sans contexte** : confond synonymes (vélo / bicyclette)
- **HOG sensible à l'orientation** : produit photographié de biais → features bruitées
- **Translation Google** : rate limit, qualité variable selon la langue source
- **Catégories floues** : 1280 vs 1281 (jouets enfants) — frontière subjective
- **Pas de CNN fine-tuné** : un ResNet50 ou DINOv2 pousserait au-delà de 0.85
        """)

    st.markdown('<div class="section-title">Perspectives</div>', unsafe_allow_html=True)
    persp = pd.DataFrame([
        {"Idée": "CLIP ViT-B/32 + MLP",         "Gain attendu": "+3 à 5 pts",  "Coût": "GPU recommandé"},
        {"Idée": "DINOv2 frozen + linear head", "Gain attendu": "+2 à 4 pts",  "Coût": "Inférence légère"},
        {"Idée": "GPT-4o fine-tuning multimodal","Gain attendu": "+5 à 8 pts", "Coût": "API payante"},
        {"Idée": "Stacking (meta-classifier)",  "Gain attendu": "+1 à 2 pts",  "Coût": "Aucun"},
        {"Idée": "Augmentation image",          "Gain attendu": "+0.5 à 1 pt", "Coût": "Re-training"},
    ])
    st.dataframe(persp, width="stretch", hide_index=True)

    st.markdown('<div class="section-title">Livrables</div>', unsafe_allow_html=True)
    st.markdown("""
- 📓 **3 notebooks** : `EDA.ipynb`, `modeling.ipynb`, `rakuten.ipynb`
- 🐍 **4 scripts modèles** : `baseline_model.py`, `advanced_model.py`, `tfidf_translated.py`, `clip_mlp.py`
- 📊 **17 figures** EDA + modélisation (`docs/figures/`)
- 📄 **2 rapports** : Rendu 1 (EDA + preprocessing), Rendu 2 (modélisation)
- 🎯 **Cette démo** Streamlit, sans ré-entraînement
    """)

    st.markdown("---")
    st.caption("DataScientest · Sprint 2 · Promo nov-2025 — DS Alternance MLE Rakuten")
