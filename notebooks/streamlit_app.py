"""
streamlit_app.py — Rakuten Product Classifier
Lancer : streamlit run streamlit_app.py

Architecture :
    Texte  : CamemBERT fine-tuné (camembert_ft_best.pth)
    Image  : CLIP ViT-B/32 + CLIPHead MLP (clip_mlp_best.pth)
    Meta   : meta_E1_text_only.pkl / meta_E2_image_only.pkl / meta_best.pkl
"""

import os, re, pickle, warnings
os.environ.update({
    "KMP_DUPLICATE_LIB_OK":   "TRUE",
    "OMP_NUM_THREADS":        "1",
    "TOKENIZERS_PARALLELISM": "false",
})
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import plotly.graph_objects as go
import plotly.express as px

from pathlib import Path
from PIL import Image
import streamlit as st

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG PAGE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.set_page_config(
    page_title="Rakuten Classifier",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHEMINS — tous les fichiers dans output/
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUT = Path("output")

CAMEMBERT_PTH   = OUT / "camembert_ft_best.pth"
CLIP_HEAD_PTH   = OUT / "clip_mlp_best.pth"
LABEL_ENC_PKL   = OUT / "label_encoder.pkl"
META_E1_PKL     = OUT / "meta_E1_text_only.pkl"
META_E2_PKL     = OUT / "meta_E2_image_only.pkl"
META_BEST_PKL   = OUT / "meta_best.pkl"
BENCHMARK_CSV   = OUT / "benchmark_final.csv"
STATS_CSV       = OUT / "stats_par_classe.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CAMEMBERT_REPO = "camembert-base"
CLIP_REPO      = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAPPING CLASSES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CLASS_NAMES = {
    10:   "Livres occasion",          40:   "Jeux vidéo",
    50:   "Accessoires gaming",       60:   "Consoles de jeux",
    1140: "Figurines",                1160: "Cartes collection",
    1180: "Jeux de société",          1281: "Jeux enfants",
    1300: "Miniatures véhicules",     1301: "Bébé / Puériculture",
    1302: "Jeux plein air",           1320: "Puériculture & Peluches",
    1560: "Mobilier",                 1920: "Linge maison",
    1940: "Épicerie",                 2060: "Décoration",
    2220: "Animalerie",               2280: "Journaux / Revues",
    2403: "Livres neufs",             2462: "Jeux PC dématérialisés",
    2522: "Papeterie / Bureautique",  2582: "Piscines / Spas",
    2583: "Outillage jardinage",      2585: "Bricolage",
    2705: "Livres BD / Mangas",       2905: "Jeux de rôle",
    1970: "Cuisine / Électroménager",
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ARCHITECTURE CLIP HEAD (doit correspondre à clip_mlp_best.pth)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class CLIPHead(nn.Module):
    """MLP head entraîné sur embeddings CLIP L2-normalisés (512-d → 27 classes)."""
    def __init__(self, in_dim: int = 512, n_classes: int = 27, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),    nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 256),       nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )
    def forward(self, x):
        return self.net(x)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# META MLP (pour unpickle meta_best.pkl si c'est un MetaMLPClassifier NB5)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class MetaMLP(nn.Module):
    def __init__(self, in_dim, hidden, n_classes):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.3)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class MetaMLPClassifier:
    def __init__(self, in_dim=61, hidden=(256,128,64), n_classes=27,
                 lr=1e-3, epochs=100, patience=20, batch=256):
        self.cfg = dict(in_dim=in_dim, hidden=hidden, n_classes=n_classes,
                        lr=lr, epochs=epochs, patience=patience, batch=batch)
        self.model = MetaMLP(in_dim, hidden, n_classes)

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval().to(DEVICE)
        Xt = torch.tensor(np.array(X), dtype=torch.float32).to(DEVICE)
        return F.softmax(self.model(Xt), dim=1).cpu().numpy()

    def predict(self, X):
        return self.predict_proba(X).argmax(1)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHARGEMENT DES MODÈLES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@st.cache_resource(show_spinner="⚙️  Chargement label encoder + meta stackers…")
def load_meta_models():
    """Charge les modèles légers (pkl) — quasi instantané."""
    import io

    class CpuUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "torch.storage" and name == "_load_from_bytes":
                return lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
            return super().find_class(module, name)

    def _pkl(path: Path):
        if not path.exists():
            return None
        with open(path, "rb") as f:
            data = f.read()
        try:
            return CpuUnpickler(io.BytesIO(data)).load()
        except Exception:
            pass
        try:
            return pickle.loads(data)
        except Exception as e:
            st.warning(f"Impossible de charger {path.name} : {e}")
            return None

    return {
        "le":       _pkl(LABEL_ENC_PKL),
        "meta_E1":  _pkl(META_E1_PKL),
        "meta_E2":  _pkl(META_E2_PKL),
        "meta_best":_pkl(META_BEST_PKL),
    }


@st.cache_resource(show_spinner="⬇️  Chargement CamemBERT fine-tuné…")
def load_text_model():
    from transformers import AutoTokenizer, CamembertForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained(CAMEMBERT_REPO)
    model     = CamembertForSequenceClassification.from_pretrained(
        CAMEMBERT_REPO, num_labels=27
    )
    state = torch.load(CAMEMBERT_PTH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval().to(DEVICE)
    return tokenizer, model


@st.cache_resource(show_spinner="⬇️  Chargement CLIP ViT-B/32 + MLP head…")
def load_image_model():
    from transformers import CLIPModel, CLIPProcessor
    clip      = CLIPModel.from_pretrained(CLIP_REPO).eval().to(DEVICE)
    processor = CLIPProcessor.from_pretrained(CLIP_REPO)
    head      = CLIPHead(in_dim=512, n_classes=27)
    state     = torch.load(CLIP_HEAD_PTH, map_location=DEVICE, weights_only=True)
    head.load_state_dict(state)
    head.eval().to(DEVICE)
    return clip, processor, head

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PREPROCESSING TEXTE (identique au collègue)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.lower()
    return re.sub(r"\s+", " ", text).strip()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# INFÉRENCE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@torch.inference_mode()
def get_text_probas(desig: str, desc: str) -> np.ndarray | None:
    """CamemBERT → softmax 27-d."""
    text = clean_text(f"{desig} {desc or ''}")
    if not text:
        return None
    tokenizer, model = load_text_model()
    enc    = tokenizer(text, truncation=True, max_length=256,
                       padding=True, return_tensors="pt").to(DEVICE)
    logits = model(**enc).logits[0]
    return F.softmax(logits, dim=-1).cpu().numpy()


@torch.inference_mode()
def get_image_probas(pil_img: Image.Image) -> np.ndarray:
    """CLIP ViT-B/32 → CLIPHead → softmax 27-d."""
    clip, processor, head = load_image_model()
    inputs = processor(images=pil_img.convert("RGB"),
                       return_tensors="pt").to(DEVICE)

    # get_image_features peut retourner un objet ou un tenseur selon la version
    feats = clip.get_image_features(**inputs)
    if not isinstance(feats, torch.Tensor):
        feats = feats.image_embeds if hasattr(feats, "image_embeds") else feats.pooler_output

    feats    = feats / feats.norm(dim=-1, keepdim=True)
    feats_bn = feats.repeat(2, 1) if feats.shape[0] == 1 else feats
    logits   = head(feats_bn)[0]
    return F.softmax(logits, dim=-1).cpu().numpy()


def build_meta_features(P_txt: np.ndarray, P_img: np.ndarray) -> np.ndarray:
    """
    61 meta-features (NB5) :
    27 probas texte + 27 probas image
    + entropie / marge / max_conf (×2) + accord = 61
    """
    eps = 1e-12
    def _stats(P):
        e = -(P * np.log(P + eps)).sum(1, keepdims=True)
        s = np.sort(P, 1)[:, ::-1]
        return e, (s[:, 0] - s[:, 1]).reshape(-1, 1), s[:, 0].reshape(-1, 1)

    te, tm, tc = _stats(P_txt)
    ie, im, ic = _stats(P_img)
    agree = (P_txt.argmax(1) == P_img.argmax(1)).astype(np.float32).reshape(-1, 1)
    return np.hstack([P_txt, P_img, te, tm, tc, ie, im, ic, agree]).astype(np.float32)


def decode_top(probas: np.ndarray, le, n: int = 5) -> list[dict]:
    idx = np.argsort(probas)[::-1][:n]
    return [
        {"code":  int(le.classes_[i]),
         "name":  CLASS_NAMES.get(int(le.classes_[i]), str(le.classes_[i])),
         "proba": float(probas[i])}
        for i in idx
    ]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CSS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("""
<style>
.pred-box  { padding:1.1rem; border-radius:10px; margin-bottom:.8rem }
.grey-lbl  { font-size:.85rem; color:#888 }
.empty-box { text-align:center; padding:3rem; background:#f8f9fa;
             border-radius:12px; color:#aaa }
</style>
""", unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HEADER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.title("🛍️  Rakuten Product Classifier")
st.caption(
    "Classification multimodale — 27 catégories  |  "
    "Texte : CamemBERT fine-tuné  ·  "
    "Image : CLIP ViT-B/32 + MLP  ·  "
    "Stacking MetaMLP (NB5)"
)

# Modèles légers — chargés immédiatement
mdls = load_meta_models()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ONGLETS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
tab_pred, tab_bench, tab_stats = st.tabs([
    "🔍  Prédiction",
    "📊  Benchmark des modèles",
    "📈  Statistiques par classe",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — PRÉDICTION
# ─────────────────────────────────────────────────────────────────────────────
with tab_pred:
    col_in, col_out = st.columns([1, 1], gap="large")

    # ── Inputs ────────────────────────────────────────────────────────────────
    with col_in:
        st.markdown("#### 📝  Saisir un produit")
        desig  = st.text_input("Désignation *",
                               placeholder="Ex : Harry Potter à l'école des sorciers")
        desc   = st.text_area("Description (optionnel)", height=120,
                               placeholder="Ex : Roman fantastique — J.K. Rowling…")
        img_up = st.file_uploader("📷  Image produit (optionnel)",
                                  type=["jpg", "jpeg", "png"])
        if img_up:
            st.image(img_up, width="stretch")
        else:
            st.info("💡 Ajoutez une image pour activer le **stacking multimodal**")

        btn = st.button("🔍  Classifier", type="primary",
                        disabled=not bool(desig.strip()))

    # ── Résultats ─────────────────────────────────────────────────────────────
    with col_out:
        st.markdown("#### 🎯  Résultats")

        if not btn:
            st.markdown("""
            <div class="empty-box">
                Renseignez une désignation<br>et cliquez sur <b>Classifier</b>
            </div>""", unsafe_allow_html=True)

        else:
            with st.spinner("Analyse en cours…"):

                # — Branche texte —
                P_txt = get_text_probas(desig, desc)

                # — Branche image (optionnelle) —
                has_img = img_up is not None
                if has_img:
                    P_img = get_image_probas(Image.open(img_up))

            le = mdls["le"]

            # Stacking texte seul via meta_E1
            P_txt_stacked = (
                mdls["meta_E1"].predict_proba(P_txt.reshape(1, -1))[0]
                if mdls["meta_E1"] is not None else P_txt
            )

            if has_img:
                # Stacking image seul via meta_E2
                P_img_stacked = (
                    mdls["meta_E2"].predict_proba(P_img.reshape(1, -1))[0]
                    if mdls["meta_E2"] is not None else P_img
                )
                # Fusion meta_best (61 features) ou moyenne
                if mdls["meta_best"] is not None:
                    M = build_meta_features(P_txt.reshape(1, -1), P_img.reshape(1, -1))
                    P_comb = mdls["meta_best"].predict_proba(M)[0]
                else:
                    P_comb = (P_txt_stacked + P_img_stacked) / 2.0

            # Résultat principal
            main = decode_top(P_comb if has_img else P_txt_stacked, le)
            conf  = main[0]["proba"]
            color = "#27ae60" if conf > .8 else "#e67e22" if conf > .5 else "#e74c3c"
            mode  = "Stacking texte + image" if has_img else "Texte seul (CamemBERT)"

            # ── Prédiction principale ──────────────────────────────────────
            st.markdown(f"""
            <div class="pred-box"
                 style="background:{color}18; border-left:5px solid {color}">
                <div style="font-size:1.6rem;font-weight:700">{main[0]['name']}</div>
                <div class="grey-lbl">
                    Confiance : {conf*100:.1f}%
                    &nbsp;|&nbsp; {mode}
                </div>
            </div>""", unsafe_allow_html=True)
            st.progress(conf)

            # ── Comparaison des modalités (si image) ──────────────────────
            if has_img:
                st.markdown("**Comparaison des modalités**")
                top_txt = decode_top(P_txt_stacked, le, 1)[0]
                top_img = decode_top(P_img_stacked, le, 1)[0]
                top_cmb = main[0]
                cmp_df  = pd.DataFrame([
                    {"Modalité": "📝 Texte (CamemBERT)",
                     "Classe":    top_txt["name"],
                     "Confiance": f"{top_txt['proba']*100:.1f}%"},
                    {"Modalité": "🖼️ Image (CLIP+MLP)",
                     "Classe":    top_img["name"],
                     "Confiance": f"{top_img['proba']*100:.1f}%"},
                    {"Modalité": "🔀 Stacking final",
                     "Classe":    top_cmb["name"],
                     "Confiance": f"{top_cmb['proba']*100:.1f}%"},
                ])
                st.dataframe(cmp_df, hide_index=True, use_container_width=True)

            # ── Top 5 ─────────────────────────────────────────────────────
            st.markdown("**Top 5 classes**")
            top5    = main[:5]
            fig_bar = go.Figure(go.Bar(
                x=[r["proba"] * 100 for r in top5[::-1]],
                y=[r["name"]        for r in top5[::-1]],
                orientation="h",
                marker_color=[color] * 5,
                text=[f"{r['proba']*100:.1f}%" for r in top5[::-1]],
                textposition="outside",
            ))
            fig_bar.update_layout(
                xaxis_title="Probabilité (%)", xaxis_range=[0, 118],
                height=260, margin=dict(l=0, r=70, t=5, b=30),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # ── 27 classes complètes ──────────────────────────────────────
            with st.expander("Voir les 27 probabilités complètes"):
                all27 = decode_top(P_comb if has_img else P_txt_stacked, le, 27)
                st.dataframe(
                    pd.DataFrame({"Classe": [r["name"] for r in all27],
                                  "Probabilité (%)": [round(r["proba"]*100, 2) for r in all27]}),
                    hide_index=True, use_container_width=True,
                )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — BENCHMARK
# ─────────────────────────────────────────────────────────────────────────────
with tab_bench:
    st.markdown("#### 📊  F1-score (weighted) — tous les modèles vs SIGIR 2020")
    SIGIR = 0.9194

    if BENCHMARK_CSV.exists():
        bdf = pd.read_csv(BENCHMARK_CSV).sort_values("F1_Test", ascending=True)
        colors_b = [
            "#e74c3c" if str(t) == "Reference" else "#3498db"
            for t in bdf.get("Type", ["Nos modeles"] * len(bdf))
        ]
        fig_b = go.Figure(go.Bar(
            y=bdf["Systeme"], x=bdf["F1_Test"], orientation="h",
            marker_color=colors_b,
            text=[f"{v:.4f}" for v in bdf["F1_Test"]],
            textposition="outside",
        ))
        fig_b.add_shape(type="line",
                        y0=-0.5, y1=len(bdf)-0.5,
                        x0=SIGIR, x1=SIGIR,
                        line=dict(color="#e74c3c", dash="dash", width=2))
        fig_b.add_annotation(y=len(bdf)-0.7, x=SIGIR,
                              text=f"SIGIR 2020 — 1er ({SIGIR})",
                              showarrow=False, xshift=75,
                              font=dict(color="#e74c3c", size=11))
        fig_b.update_layout(
            xaxis_title="F1-score (weighted)", xaxis_range=[0.5, 1.07],
            height=max(320, len(bdf)*46),
            margin=dict(l=0, r=90, t=20, b=30),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_b, use_container_width=True)
        st.dataframe(bdf.sort_values("F1_Test", ascending=False),
                     hide_index=True, use_container_width=True)
    else:
        st.warning(f"⚠️  `benchmark_final.csv` introuvable dans `{OUT}`")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — STATISTIQUES PAR CLASSE
# ─────────────────────────────────────────────────────────────────────────────
with tab_stats:
    st.markdown("#### 📈  Statistiques du dataset Rakuten par classe")

    if STATS_CSV.exists():
        sdf = pd.read_csv(STATS_CSV, index_col=0)
        if "N produits" in sdf.columns:
            fig_d = px.bar(
                sdf.sort_values("N produits", ascending=False),
                x=sdf.sort_values("N produits", ascending=False).index,
                y="N produits",
                color="N produits", color_continuous_scale="Blues",
                title="Distribution des 27 classes",
            )
            fig_d.update_layout(
                xaxis_tickangle=-45, coloraxis_showscale=False,
                height=400, plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                xaxis_title="", margin=dict(b=130),
            )
            st.plotly_chart(fig_d, use_container_width=True)
        st.dataframe(sdf, use_container_width=True)
    else:
        st.warning(f"⚠️  `stats_par_classe.csv` introuvable dans `{OUT}`")