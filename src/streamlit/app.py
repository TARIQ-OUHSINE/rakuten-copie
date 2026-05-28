"""
Rakuten France — Classification Multimodale
Streamlit demo app (inference only — no retraining).

Architecture v2 :
    Texte  : CamemBERT fine-tuné → 27 probabilités
    Image  : CLIP ViT-B/32 (laion2b) + MLP → 27 probabilités
    Meta   : MetaMLP stacker (meta_best.pkl) sur 61 features
             = 27 probs texte + 27 probs image + 6 meta-scalaires + 1 accord
    Fallback : meta_E1 (texte seul) ou meta_E2 (image seule) si une modalité manque

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
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


# ─── MetaMLP — défini AVANT pickle.load(meta_best.pkl) ──────────────────────
# meta_best.pkl référence __main__.MetaMLP et __main__.MetaMLPClassifier
# (sérialisés depuis NB05). Ces classes doivent être visibles à l'unpickling.

class MetaMLP(nn.Module):
    """FC → BN → ReLU → Dropout (identique à NB05)."""

    def __init__(self, in_dim: int, hidden, n_classes: int):
        super().__init__()
        layers: list = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.3)]
            prev = h
        layers += [nn.Linear(prev, n_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MetaMLPClassifier:
    """Wrapper sklearn-like sérialisé dans meta_best.pkl (cfg + model)."""

    def __init__(self, in_dim: int = 61, hidden=(256, 128, 64), n_classes: int = 27,
                 lr: float = 1e-3, epochs: int = 100, patience: int = 20, batch: int = 256):
        self.cfg = {
            'in_dim': in_dim, 'hidden': hidden, 'n_classes': n_classes,
            'lr': lr, 'epochs': epochs, 'patience': patience, 'batch': batch,
        }
        self.model = MetaMLP(in_dim, hidden, n_classes)

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            Xt = torch.tensor(np.asarray(X), dtype=torch.float32)
            return F.softmax(self.model(Xt), dim=1).numpy()

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


# Rendre les classes visibles sous __main__ pour pickle
sys.modules['__main__'].MetaMLP = MetaMLP
sys.modules['__main__'].MetaMLPClassifier = MetaMLPClassifier

# ── Project paths ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_V2_DIR = MODEL_DIR / "v2"
FIG_DIR = ROOT / "docs" / "figures"
REPORT_FIG_DIR = ROOT / "reports" / "figures"

sys.path.insert(0, str(ROOT))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

    /* Cap figure size so they don't dominate the viewport */
    [data-testid="stImage"] {
        display: flex;
        justify-content: center;
    }
    [data-testid="stImage"] img {
        max-width: 880px;
        max-height: 520px;
        width: auto;
        height: auto;
        object-fit: contain;
        border-radius: 8px;
    }
    /* Image caption styling */
    [data-testid="stImageCaption"], [data-testid="stImage"] figcaption {
        text-align: center;
        font-size: 0.85rem;
        color: var(--text-muted);
        font-style: italic;
        max-width: 880px;
        margin: 0.4rem auto 0;
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

CAMEMBERT_REPO = "camembert-base"
CLIP_REPO = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"


# ── Helpers ──────────────────────────────────────────────────────────────────
def kpi(value: str, label: str) -> str:
    return f'<div class="kpi"><div class="kpi-value">{value}</div><div class="kpi-label">{label}</div></div>'


def comment(text: str) -> None:
    st.markdown(f'<div class="comment">💡 {text}</div>', unsafe_allow_html=True)


def clean_text(text: str) -> str:
    """Light cleaning consistent with the training-time preprocessing."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def find_figure(name: str) -> Path | None:
    for d in (FIG_DIR, REPORT_FIG_DIR):
        p = d / name
        if p.exists():
            return p
    return None


# ── Image classifier head (matches clip_mlp_best.pth) ────────────────────────
class CLIPHead(nn.Module):
    """MLP head trained on top of L2-normalized CLIP image embeddings (512-d)."""

    def __init__(self, in_dim: int = 512, n_classes: int = 27, dropout: float = 0.3):
        super().__init__()
        # Sequential indices must match the saved state_dict
        # (net.0 Linear, net.1 BN, net.4 Linear, net.5 BN, net.8 Linear)
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),     # 0
            nn.BatchNorm1d(512),        # 1
            nn.GELU(),                  # 2
            nn.Dropout(dropout),        # 3
            nn.Linear(512, 256),        # 4
            nn.BatchNorm1d(256),        # 5
            nn.GELU(),                  # 6
            nn.Dropout(dropout),        # 7
            nn.Linear(256, n_classes),  # 8
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Cached loaders ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Chargement du tokenizer + CamemBERT fine-tuné…")
def load_text_model():
    from transformers import AutoTokenizer, CamembertForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(CAMEMBERT_REPO)
    model = CamembertForSequenceClassification.from_pretrained(
        CAMEMBERT_REPO, num_labels=27
    )
    state = torch.load(MODEL_V2_DIR / "camembert_ft_best.pth", map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval().to(DEVICE)
    return tokenizer, model


@st.cache_resource(show_spinner="Chargement de CLIP + MLP image…")
def load_image_model():
    from transformers import CLIPModel, CLIPProcessor

    clip = CLIPModel.from_pretrained(CLIP_REPO)
    processor = CLIPProcessor.from_pretrained(CLIP_REPO)
    clip.eval().to(DEVICE)

    head = CLIPHead(in_dim=512, n_classes=27)
    state = torch.load(MODEL_V2_DIR / "clip_mlp_best.pth", map_location=DEVICE, weights_only=True)
    head.load_state_dict(state)
    head.eval().to(DEVICE)
    return clip, processor, head


class _CPUUnpickler(pickle.Unpickler):
    """Unpickler qui remappe les tenseurs CUDA → CPU à la volée.

    Le pickle de meta_best.pkl (NB05) sérialise un nn.Module entraîné sur GPU.
    Sur une machine CPU-only, le `pickle.load` standard échoue. Ce custom
    Unpickler intercepte `torch.storage._load_from_bytes` pour forcer
    `map_location='cpu'` (équivalent CPU de `torch.load(..., map_location='cpu')`).
    """

    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(BytesIO(b), map_location="cpu", weights_only=False)
        return super().find_class(module, name)


def _load_pickle_cpu(path: Path):
    """Charge un pickle contenant potentiellement des tenseurs CUDA, en CPU."""
    with open(path, "rb") as f:
        return _CPUUnpickler(f).load()


@st.cache_resource(show_spinner="Chargement des stackers…")
def load_stackers():
    le = _load_pickle_cpu(MODEL_V2_DIR / "label_encoder.pkl")
    meta_text = _load_pickle_cpu(MODEL_V2_DIR / "meta_E1.pkl")
    meta_image = _load_pickle_cpu(MODEL_V2_DIR / "meta_E2.pkl")
    # MetaMLP final (61 features = 27 + 27 + 3 + 3 + 1) — meilleur modèle NB05
    meta_best_path = MODEL_V2_DIR / "meta_best.pkl"
    meta_best = None
    if meta_best_path.exists():
        try:
            meta_best = _load_pickle_cpu(meta_best_path)
            # S'assurer que le sous-module torch est bien sur CPU
            if hasattr(meta_best, "model") and isinstance(meta_best.model, nn.Module):
                meta_best.model.to("cpu").eval()
        except Exception as exc:
            st.warning(f"meta_best.pkl introuvable ou corrompu ({exc}) — fallback moyenne E1/E2.")
    return le, meta_text, meta_image, meta_best


def _scalar_meta(P: np.ndarray) -> np.ndarray:
    """entropy, margin, max_conf pour un vecteur de probas (1, n_classes) → (1, 3)."""
    eps = 1e-12
    entropy = -(P * np.log(P + eps)).sum(axis=1, keepdims=True)
    sorted_p = np.sort(P, axis=1)[:, ::-1]
    margin = (sorted_p[:, 0] - sorted_p[:, 1]).reshape(-1, 1)
    max_conf = sorted_p[:, 0].reshape(-1, 1)
    return np.hstack([entropy, margin, max_conf]).astype(np.float32)


def build_meta_features(P_text: np.ndarray, P_image: np.ndarray) -> np.ndarray:
    """Assemble les 61 meta-features pour meta_best (cf. NB05).

    Entrées : P_text, P_image de shape (27,) ou (1, 27).
    Sortie  : (1, 61) = [P_text | P_image | txt_meta(3) | img_meta(3) | accord(1)].
    """
    Pt = np.asarray(P_text, dtype=np.float32).reshape(1, -1)
    Pi = np.asarray(P_image, dtype=np.float32).reshape(1, -1)
    agreement = (Pt.argmax(1) == Pi.argmax(1)).astype(np.float32).reshape(-1, 1)
    return np.hstack([Pt, Pi, _scalar_meta(Pt), _scalar_meta(Pi), agreement]).astype(np.float32)


@st.cache_data(show_spinner=False)
def load_class_stats() -> pd.DataFrame:
    """Per-class stats from train_cleaned.pkl (cached)."""
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


# ── Inference helpers ────────────────────────────────────────────────────────
@torch.inference_mode()
def predict_text_probs(designation: str, description: str) -> np.ndarray | None:
    """Run CamemBERT on cleaned designation+description, return 27-d softmax."""
    text = clean_text(f"{designation} {description}")
    if not text:
        return None
    tokenizer, model = load_text_model()
    enc = tokenizer(
        text, truncation=True, max_length=256, padding=True, return_tensors="pt"
    ).to(DEVICE)
    logits = model(**enc).logits[0]
    return F.softmax(logits, dim=-1).cpu().numpy()


@torch.inference_mode()
def predict_image_probs(pil_img: Image.Image) -> np.ndarray:
    """Encode image with CLIP, classify with the MLP head, return 27-d softmax."""
    clip, processor, head = load_image_model()
    inputs = processor(images=pil_img.convert("RGB"), return_tensors="pt").to(DEVICE)
    feats = clip.get_image_features(**inputs)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    # BatchNorm needs >1 sample → repeat then take the first row
    feats_bn = feats.repeat(2, 1) if feats.shape[0] == 1 else feats
    head.eval()  # disable BN updates; running stats are used in eval mode
    logits = head(feats_bn)[0]
    return F.softmax(logits, dim=-1).cpu().numpy()


def stack_top3(probs: np.ndarray, le) -> list[tuple[int, str, float]]:
    """Convert a 27-d probability vector into a sorted list of (code, name, p)."""
    order = np.argsort(probs)[::-1][:3]
    out = []
    for idx in order:
        code = int(le.classes_[idx])
        out.append((code, CLASS_NAMES.get(code, f"Classe {code}"), float(probs[idx])))
    return out


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="hero">
    <h1>🛒 Rakuten France — Classification Multimodale</h1>
    <p>84 916 produits · 27 catégories · CamemBERT + CLIP+MLP · Challenge SIGIR eCommerce 2020</p>
</div>
""",
    unsafe_allow_html=True,
)

# Light artifacts (stackers + label encoder) load instantly
LE, META_TEXT, META_IMAGE, META_BEST = load_stackers()

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
        {"Phase": "1 — Baseline",      "Modèle": "TF-IDF + LightGBM",              "Modalités": "Texte",            "Statut": "✅ Livré"},
        {"Phase": "2 — Optimisation",  "Modèle": "TF-IDF + OpenCV + LightGBM",     "Modalités": "Texte + Image",    "Statut": "✅ Livré"},
        {"Phase": "2 bis — Translation","Modèle": "TF-IDF (FR) + LightGBM",        "Modalités": "Texte traduit FR", "Statut": "✅ Livré"},
        {"Phase": "3 — Deep texte",    "Modèle": "CamemBERT fine-tuné",            "Modalités": "Texte FR",         "Statut": "🚀 Déployé"},
        {"Phase": "3 — Deep image",    "Modèle": "CLIP ViT-B/32 + MLP",            "Modalités": "Image",            "Statut": "🚀 Déployé"},
        {"Phase": "4 — Stacking",      "Modèle": "Meta-LR (E1 texte + E2 image)",  "Modalités": "Multimodal",       "Statut": "🚀 Déployé"},
    ])
    st.dataframe(phases, width="stretch", hide_index=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — EXPLORATION
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.caption("📄 Figures issues de **Etape1_rendu.docx** — source de vérité du rendu d'exploration.")

    def show_fig(fname: str, caption: str) -> None:
        p = find_figure(fname)
        if p is not None:
            st.image(str(p), caption=caption, width="stretch")
        else:
            st.warning(f"Figure introuvable : `{fname}`")

    # ── §1 Distribution des classes ──────────────────────────────────────
    st.markdown('<div class="section-title">1 · Distribution des classes</div>', unsafe_allow_html=True)
    show_fig("etape1_fig1_class_distribution.png", "Répartition des 27 catégories produits Rakuten")
    comment("**Déséquilibre des classes** : ratio **13:1** entre la classe majoritaire (2583 — Piscine/Spa) "
            "et la classe minoritaire (1180). Validé par un **test du Chi² (p < 0.001)**. "
            "Conséquence directe sur la modélisation : `class_weight='balanced'` et **F1 pondéré** comme métrique.")

    # ── §2 Descriptions manquantes ───────────────────────────────────────
    st.markdown('<div class="section-title">2 · Descriptions manquantes par classe</div>', unsafe_allow_html=True)
    show_fig("etape1_fig2_missing_desc.png", "% de descriptions absentes et produits avec / sans description par classe")
    comment("**35 % de descriptions absentes**, taux qui **varie fortement par classe** "
            "(Chi² p < 0.001 → non aléatoire). Stratégie : feature binaire `has_desc` + concaténation "
            "`designation + description` pour éviter les NaN, et compensation par la **modalité image**.")

    # ── §3 Distribution des langues par classe ───────────────────────────
    st.markdown('<div class="section-title">3 · Distribution des langues par classe</div>', unsafe_allow_html=True)
    show_fig("etape1_fig3_languages.png", "Répartition globale des langues + composition par classe (top 15)")
    show_fig("etape1_fig3b_lang_heatmap.png", "Heatmap langue × classe (%)")
    comment("~**84 % français**, multilinguisme **fr / en / de / es**. La dépendance langue × classe "
            "est significative (**Chi² p < 0.001**) : certaines classes (accessoires gaming) sont quasi "
            "exclusivement en anglais. → Encodage one-hot de la feature `language` (5 colonnes).")

    # ── §4 Longueur du texte par classe ──────────────────────────────────
    st.markdown('<div class="section-title">4 · Longueur du texte par classe</div>', unsafe_allow_html=True)
    show_fig("etape1_fig4_text_length.png", "Histogrammes + boxplots de la longueur (désignation & description)")
    comment("Désignations courtes (**médiane ~8 mots**), descriptions très variables (0 → 1000+ mots). "
            "La longueur est **discriminante par classe** : **Kruskal-Wallis p < 0.001**. "
            "→ Features `len_desig`, `len_desc`, `len_combined` standardisées et injectées dans le modèle.")

    # ── §5 Top mots par classe ───────────────────────────────────────────
    st.markdown('<div class="section-title">5 · Top mots par classe (signal lexical)</div>', unsafe_allow_html=True)
    show_fig("etape1_fig5_top_words.png", "Top 12 mots — exemples sur 3 classes très différenciées")
    comment("Chaque classe possède un **vocabulaire distinctif** : *piscine / kit / bâche* pour Piscines, "
            "*cuisine / matelas / chaise* pour Mobilier, *drone / dji / quadcopter* pour Modélisme. "
            "→ Le signal textuel est **très informatif** pour la classification, justifie TF-IDF et CamemBERT.")

    # ── §6 % images mal cadrées par classe ──────────────────────────────
    st.markdown('<div class="section-title">6 · % images mal cadrées par classe</div>', unsafe_allow_html=True)
    show_fig("etape1_fig6_bad_crop.png", "% d'images mal cadrées (zoom_ratio < 0.7) — relatif + absolu")
    comment("Analyse **originale** sur la qualité du cadrage : ~**48 %** d'images mal cadrées en moyenne, "
            "fortement dépendant de la classe (**Chi² = 8 985, p < 0.001**). "
            "→ Solution : `auto_crop()` qui détecte la bounding box des pixels non-blancs et recadre.")

    # ── Bloc Preprocessing ───────────────────────────────────────────────
    st.markdown('<div class="section-title">Preprocessing — avant / après</div>', unsafe_allow_html=True)
    st.markdown(
        "Quatre visualisations valident l'effet du pipeline de nettoyage (texte + image) "
        "présenté dans la section Méthodologie."
    )
    cA, cB = st.columns(2)
    with cA:
        show_fig("etape1_preproc_graphA_text.png", "Graph A — texte BRUT vs NETTOYÉ (HTML, stopwords, ponctuation)")
        show_fig("etape1_preproc_graphC_vocab.png", "Graph C — −21 % de mots uniques après preprocessing")
    with cB:
        show_fig("etape1_preproc_graphB_length.png", "Graph B — longueur de texte avant / après nettoyage")
        show_fig("etape1_preproc_white_ratio.png", "Distribution du ratio de pixels blancs (seuil 70 %)")
    show_fig("etape1_preproc_bad_examples.png", "Exemples d'images avec > 70 % de pixels blancs (mal cadrées)")
    show_fig("etape1_preproc_autocrop.png", "Auto-crop — résultats avant / après recadrage")
    comment("Le preprocessing **réduit le bruit textuel** (vocab −21 %, longueur médiane 48 → 33 mots) "
            "et **standardise les images** (recadrage automatique sur fond blanc). "
            "Ce sont les leviers les plus déterminants pour la performance finale.")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODÉLISATION
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.caption("📄 Figures issues de **Rendu 2 — Modélisation.docx** + `reports/figures/`.")

    # ── Benchmark global ─────────────────────────────────────────────────
    st.markdown('<div class="section-title">Comparaison des modèles (weighted F1 — test)</div>',
                unsafe_allow_html=True)
    bench = pd.DataFrame([
        {"Modèle": "SIGIR 2020 — 1ère place",                "F1 pondéré": 0.9144, "Type": "Référence"},
        {"Modèle": "Stacking final (E11) — Meta-MLP",        "F1 pondéré": 0.8960, "Type": "Notre modèle"},
        {"Modèle": "CamemBERT fine-tuné (texte)",            "F1 pondéré": 0.8760, "Type": "Notre modèle"},
        {"Modèle": "SIGIR 2020 — Baseline",                  "F1 pondéré": 0.8705, "Type": "Référence"},
        {"Modèle": "TF-IDF + LightGBM + c-TF-IDF (EXP_T_KW)","F1 pondéré": 0.8480, "Type": "Notre modèle"},
        {"Modèle": "TF-IDF + LightGBM baseline (multimodal)","F1 pondéré": 0.8321, "Type": "Notre modèle"},
        {"Modèle": "CLIP ViT-B/32 fine-tuné (image)",        "F1 pondéré": 0.6774, "Type": "Notre modèle"},
    ])
    fig_bench = px.bar(
        bench.sort_values("F1 pondéré", ascending=True),
        x="F1 pondéré", y="Modèle", orientation="h",
        color="Type", color_discrete_map={"Référence": "#f39c12", "Notre modèle": "#BF0000"},
        text="F1 pondéré", template="plotly_white", height=460,
        range_x=[0.60, 0.95],
    )
    fig_bench.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig_bench.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            margin=dict(l=10, r=40, t=30, b=20))
    st.plotly_chart(fig_bench, width="stretch")
    comment("Le **stacking final (E11)** atteint **F1 = 0.896**, à **−1.8 pt** de la 1ʳᵉ place SIGIR 2020 "
            "(0.9144) et **+6.4 pts** au-dessus de la baseline TF-IDF + LightGBM (0.8321).")

    # ── §1 Baseline TF-IDF + LightGBM ────────────────────────────────────
    st.markdown('<div class="section-title">§1 · Baseline multimodale — TF-IDF + LightGBM</div>',
                unsafe_allow_html=True)
    st.markdown(
        "Représentation **TF-IDF** (50 000 features, bigrammes, `sublinear_tf`) + features image légères "
        "(histogramme RGB 8 bins) + meta-features Etape 1 → **LightGBM** (`class_weight=balanced`). "
        "Résultat : **F1 = 0.8262 (val) / 0.8321 (test)**."
    )
    show_fig("baseline_confusion_matrix.png",
             "Figure 1 — Matrice de confusion baseline (test, 12 738 produits)")
    show_fig("baseline_f1_per_class.png",
             "Figure 2 — F1-score par classe : 5 classes < 0.75 (1281, 10, 1180, 2403, 1280)")
    show_fig("baseline_tfidf_importance.png",
             "Figure 3 — Top features TF-IDF (gain LightGBM) — tokens métier intelligibles")
    comment("Trois facteurs structurels expliquent l'erreur résiduelle : **frontière sémantique floue** "
            "entre classes voisines (jeux/figurines, livres/magazines), **35,1 %** de descriptions manquantes "
            "(χ² = 2 847, p < 0.001), et stemming FR pénalisant les **15,8 %** de produits non-FR.")

    # ── §2 Branche texte deep ────────────────────────────────────────────
    st.markdown('<div class="section-title">§2 · Approfondissement de la branche texte</div>',
                unsafe_allow_html=True)
    st.markdown(
        "Benchmark de **4 configurations** texte (TF-IDF brut, TF-IDF + c-TF-IDF, **multilingual-e5-large**, "
        "**CamemBERT fine-tuné**) en protocole **OOF (Out-Of-Fold)** stratifié 5-folds."
    )
    show_fig("texte_benchmark.png",
             "Figure 5 — Benchmark des 4 configurations texte (weighted F1)")
    show_fig("ctfidf_similarity_heatmap.png",
             "Figure 4 — Heatmap c-TF-IDF : similarités cosinus entre 27 centroïdes de classe")
    show_fig("texte_logreg_coefficients.png",
             "Figure 6 — Top coefficients LogReg surrogate sur embeddings e5-large")
    comment("**CamemBERT** (F1 = 0.876) bat e5-large (0.863), lui-même supérieur à TF-IDF + c-TF-IDF (0.848). "
            "Spécialisation FR + token-level contextualisation > généralité multilingue sur un corpus à 84 % français.")

    # ── §3 Branche image ─────────────────────────────────────────────────
    st.markdown('<div class="section-title">§3 · Approfondissement de la branche image</div>',
                unsafe_allow_html=True)
    st.markdown(
        "Trois familles d'architectures comparées : **DINOv2-base** (frozen feature extractor), "
        "**CLIP ViT-B/32** (frozen & fine-tuned), **ResNet50** (fine-tuning progressif 3 phases)."
    )
    col_img1, col_img2, col_img3 = st.columns(3)
    with col_img1: show_fig("viz_resnet50.png",   "ResNet50 — fine-tuning progressif")
    with col_img2: show_fig("viz_clip_ft.png",    "CLIP ViT-B/32 — fine-tuned")
    with col_img3: show_fig("viz_dino_ft.png",    "DINOv2-base — frozen + linear probe")
    comment("CLIP fine-tuné atteint **F1 = 0.677** seul. La modalité image plafonne sur les classes "
            "**visuellement isomorphes** (livres / magazines / BD) et apporte ~12 % de gain en stacking.")

    # ── §4 Stacking multimodal final ─────────────────────────────────────
    st.markdown('<div class="section-title">§4 · Stacking multimodal — meta-classifieur</div>',
                unsafe_allow_html=True)
    st.markdown(
        "**Meta-MLP** agrège les probabilités OOF de CamemBERT (texte) + CLIP-FT (image) + **6 meta-scalaires** "
        "(entropie, marge, max_conf par modalité) + features de preprocessing Etape 1. "
        "Progression sur 12 variantes **E0 → E11**, **F1 final = 0.896 (test)**."
    )
    show_fig("meta_features_distribution.png",
             "Figure 9 — Distribution des 6 meta-scalaires (entropie / marge / max_conf, texte vs image)")
    col_s1, col_s2 = st.columns(2)
    with col_s1: show_fig("stacking_confusion_matrix.png", "Matrice de confusion — stacking final E11")
    with col_s2: show_fig("stacking_f1_per_class.png",     "F1 par classe — stacking final")
    show_fig("stacking_confidence_analysis.png",
             "Analyse de la confiance : entropie vs justesse (calibration du meta-classifieur)")
    show_fig("benchmark_final_vs_sigir2020.png",
             "Benchmark : notre stacking vs top 5 SIGIR 2020 (1ʳᵉ place 0.9144)")
    comment("Le **stacking E11** corrige **+2 pts** sur les classes textuellement ambiguës "
            "(jeux 1180/1281, livres 10/2403) grâce à la complémentarité du signal image.")

    # ── §5 Interprétabilité ──────────────────────────────────────────────
    st.markdown('<div class="section-title">§5 · Interprétabilité</div>', unsafe_allow_html=True)
    st.markdown("**Branche texte** : permutation importance + top mots discriminants. "
                "**Branche image** : Grad-CAM sur les CNN fine-tunés.")
    show_fig("graph05b_top_words_per_class.png",
             "Figure 14 — Top mots discriminants par classe (régularités lexicales légitimes)")
    col_g1, col_g2 = st.columns(2)
    with col_g1: show_fig("gradcam_best_model.png", "Grad-CAM — meilleur modèle image")
    with col_g2: show_fig("gradcam_resnet50.png",   "Grad-CAM — ResNet50 (couche conv finale)")
    comment("Grad-CAM confirme que le réseau **attend les bons endroits** : packaging produit pour les jeux, "
            "couverture pour les livres, silhouette produit pour le mobilier. Pas de focus parasite sur le fond.")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — DÉMO INFÉRENCE
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">Tester un produit en direct</div>', unsafe_allow_html=True)
    st.caption("CamemBERT fine-tuné (texte) · CLIP+MLP (image) · "
               "MetaMLP `meta_best.pkl` (61 features) en fusion finale · "
               "Fallback Meta-LR E1/E2 si une seule modalité.")

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
        if not designation and not description and pil_image is None:
            st.warning("Saisissez au moins une désignation, une description, ou chargez une image.")
        else:
            if true_label:
                st.info(f"**Vraie classe :** {true_label}")

            with st.spinner("Inférence CamemBERT + CLIP en cours…"):
                # Text branch ─────────────────────────────────────────────
                text_probs = predict_text_probs(designation, description)
                if text_probs is not None:
                    text_stacked = META_TEXT.predict_proba(text_probs.reshape(1, -1))[0]
                    pred_text = stack_top3(text_stacked, LE)
                else:
                    pred_text = []

                # Image branch ────────────────────────────────────────────
                if pil_image is not None:
                    image_probs = predict_image_probs(pil_image)
                    image_stacked = META_IMAGE.predict_proba(image_probs.reshape(1, -1))[0]
                    pred_image = stack_top3(image_stacked, LE)
                else:
                    image_probs = None
                    image_stacked = None
                    pred_image = []

                # Fusion ──────────────────────────────────────────────────
                # Si les deux modalités sont disponibles ET meta_best est chargé,
                # utiliser le MetaMLP 61-features (cf. NB05). Sinon : moyenne E1/E2.
                fusion_label = "Fusion — Moyenne E1 + E2"
                if text_probs is not None and image_probs is not None:
                    if META_BEST is not None:
                        meta_X = build_meta_features(text_probs, image_probs)
                        combined = META_BEST.predict_proba(meta_X)[0]
                        fusion_label = "Fusion — MetaMLP (meta_best, 61 features)"
                    else:
                        combined = (text_stacked + image_stacked) / 2.0
                    pred_combined = stack_top3(combined, LE)
                else:
                    combined = None
                    pred_combined = []

            c1, c2, c3 = st.columns(3)

            def render_pred(col, title, preds, primary=False, hint: str = ""):
                with col:
                    cls = "pred-card" if primary else "pred-card-secondary"
                    cls2 = "" if primary else "pred-class-secondary"
                    if not preds:
                        st.markdown(f'<div class="{cls}">'
                                    f'<div class="pred-model">{title}</div>'
                                    f'<div class="pred-class {cls2}">—</div>'
                                    f'<div class="pred-conf">{hint or "Pas d\'entrée exploitable"}</div>'
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

            render_pred(c1, "Texte — CamemBERT + Meta-E1", pred_text,
                        hint="📝 Saisissez un texte pour activer ce modèle")
            render_pred(c2, "Image — CLIP+MLP + Meta-E2", pred_image,
                        hint="📷 Chargez une image pour activer ce modèle")
            render_pred(c3, fusion_label, pred_combined, primary=True,
                        hint="🔗 Texte ET image requis pour la fusion")

            # ── Top-3 probabilities (best available model) ──────────────
            best_preds = pred_combined or pred_text or pred_image
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
    st.markdown('<div class="section-title">Pipeline v2 (architecture déployée)</div>', unsafe_allow_html=True)
    st.code(
        """
                ┌─────────────────────────┐
designation ───►│   Light cleaning        │
description ───►│   (HTML, lower)         │
                └────────────┬────────────┘
                             ▼
                ┌────────────────────────┐
                │  CamemBERT tokenizer   │
                │  (SentencePiece, 32 K) │
                └────────────┬───────────┘
                             ▼
                ┌────────────────────────┐
                │ CamemBERT fine-tuné    │   P_text (27)
                │ (12 layers, 768 dim)   │────────┬────────┐
                └────────────────────────┘        │        │
                                                  │        ▼ (fallback texte seul)
                                                  │   ┌──────────────────┐
                                                  │   │  Meta-LR  E1     │
                                                  │   │  (27 → 27)       │
                                                  │   └──────────────────┘
image ───► CLIP preprocess (224×224) ───┐         │
                                        ▼         │
                              ┌─────────────────────┐
                              │  CLIP ViT-B/32      │  512-d  L2-normalized
                              │  (laion2b, frozen)  │
                              └──────────┬──────────┘
                                         ▼
                              ┌─────────────────────┐  P_image (27)
                              │  MLP 512→512→256→27 │────────┬────────┐
                              │  (BN + GELU + Drop) │        │        │
                              └─────────────────────┘        │        ▼ (fallback image seule)
                                                             │   ┌──────────────────┐
                                                             │   │  Meta-LR  E2     │
                                                             │   │  (27 → 27)       │
                                                             │   └──────────────────┘
                                                             │
                       (Texte + Image disponibles)           │
                              ┌──────────────────────────────┴──────────────┐
                              │  build_meta_features(P_text, P_image)        │
                              │  → 27 + 27 + 3 + 3 + 1 = 61 features          │
                              │   [P_text | P_image | entropy/margin/maxconf  │
                              │    × 2 | agreement]                           │
                              └─────────────────────┬────────────────────────┘
                                                    ▼
                                  ┌──────────────────────────────────┐
                                  │   MetaMLP  (meta_best.pkl)        │
                                  │   61 → 256 → 128 → 64 → 27        │
                                  │   FC → BN → ReLU → Dropout(0.3)   │
                                  └────────────────┬─────────────────┘
                                                   ▼
                                       27 prdtypecode probas (softmax)
        """,
        language=None,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">Choix techniques</div>', unsafe_allow_html=True)
        st.markdown("""
- **CamemBERT-base** (110 M params) fine-tuné 3 epochs, batch 32, lr 2e-5
- **CLIP ViT-B/32 (laion2B)** : backbone gelé, MLP entraîné from scratch
- **MLP image** : 3 couches denses avec BatchNorm + GELU + Dropout 0.3
- **Meta-stackers mono-modaux** (E1, E2) : Logistic Regression `saga`, `class_weight=balanced`, C=2.0 — utilisés en fallback texte-seul / image-seule
- **Fusion finale** : **MetaMLP** `meta_best.pkl` (NB05) sur 61 features = 27 probs texte + 27 probs image + 3 méta-scalaires texte (entropie, margin, max_conf) + 3 méta-scalaires image + 1 indicateur d'accord — entraîné en OOF 5-fold pour éviter la fuite
- **Pas de TTA / d'augmentation** : focus sur la qualité du backbone
        """)
    with col2:
        st.markdown('<div class="section-title">Limites identifiées</div>', unsafe_allow_html=True)
        st.markdown("""
- **CamemBERT 512 tokens max** : descriptions très longues tronquées (~3% du corpus)
- **CLIP en zéro-shot visuel** : embeddings non spécialisés sur l'iconographie produit Rakuten
- **MetaMLP late-fusion** : pas de co-attention texte↔image au niveau features (fusion uniquement au niveau probas)
- **Pas de calibration de température** post-hoc (Platt scaling pourrait gagner 0.5 pt)
- **Classes 1280 / 1281** restent confondues : signal sémantique trop proche
        """)

    st.markdown('<div class="section-title">Perspectives</div>', unsafe_allow_html=True)
    persp = pd.DataFrame([
        {"Idée": "DINOv2 frozen + linear head",      "Gain attendu": "+1 à 2 pts", "Coût": "Inférence légère"},
        {"Idée": "Fine-tuning CLIP visuel end-to-end","Gain attendu": "+2 à 4 pts","Coût": "GPU recommandé"},
        {"Idée": "Test-Time Augmentation (image)",   "Gain attendu": "+0.5 à 1 pt","Coût": "Re-inférence ×N"},
        {"Idée": "Distillation → modèle unique léger","Gain attendu": "iso F1, latence ÷ 3","Coût": "1 entraînement"},
        {"Idée": "GPT-4o fine-tuning multimodal",    "Gain attendu": "+3 à 5 pts", "Coût": "API payante"},
    ])
    st.dataframe(persp, width="stretch", hide_index=True)

    st.markdown('<div class="section-title">Livrables</div>', unsafe_allow_html=True)
    st.markdown("""
- 📓 **5 notebooks** : `EDA.ipynb`, `modeling.ipynb`, `rakuten.ipynb`, `compare_tfidf.ipynb`, `check.ipynb`
- 🐍 **Scripts modèles** : baseline · TF-IDF traduit · OpenCV · CLIP+MLP · GPT-4o
- 🤖 **Artefacts v2** (`models/v2/`) :
    - `camembert_ft_best.pth` — CamemBERT fine-tuné (442 MB)
    - `clip_mlp_best.pth` — Tête image CLIP+MLP (1.6 MB)
    - `meta_E1.pkl`, `meta_E2.pkl` — Stackers Logistic Regression (texte-seul / image-seule)
    - **`meta_best.pkl` — MetaMLP final (61 → 27)** : fusion utilisée quand les deux modalités sont présentes
    - `label_encoder.pkl`, `ctfidf_centroids.npy` — Outils de support
- 📊 **17 figures** EDA + modélisation
- 🎯 **Cette démo** Streamlit (CamemBERT + CLIP + stacking, sans ré-entraînement)
    """)

    st.markdown("---")
    st.caption("DataScientest · Sprint 2 · Promo nov-2025 — DS Alternance MLE Rakuten")
