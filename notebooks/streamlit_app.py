"""
================================================================================
  Rakuten France — Classification Multimodale de Produits
  Application de démonstration interactive Streamlit
================================================================================

DESCRIPTION
-----------
Cette application permet d'explorer interactivement la fusion des modèles
VGG16 (image) et CamemBERT (texte) pour la classification de produits
Rakuten en 27 catégories.

PRÉREQUIS — Fichiers nécessaires dans le dossier data/
-------------------------------------------------------
    data/
    ├── df_streamlit_subset.csv     ← subset de 20 produits (Kaggle output)
    ├── df_test.csv                 ← dataset test complet (Kaggle input)
    ├── vgg16_test_probs.npy        ← probabilités VGG16 (Kaggle input/output)
    ├── camembert_test_probs.npy    ← probabilités CamemBERT (Kaggle output)
    └── images_streamlit/           ← extraire images_streamlit.zip ici
        ├── 10/
        ├── 1140/
        └── ...

INSTALLATION
------------
    pip install streamlit pandas numpy pillow

LANCEMENT
---------
    # Depuis la racine du projet
    streamlit run streamlit_app.py

    # L'app s'ouvre automatiquement sur http://localhost:8501

UTILISATION
-----------
    - Sidebar gauche : choisir un scénario de prédiction
        🔀 Fusion gagne seule (image ❌ texte ❌)
        🖼️ Image aide la fusion (image ✅ texte ❌)
        📝 Texte aide la fusion (image ❌ texte ✅)
        ✅ Tous corrects (image ✅ texte ✅)
    - Slider : ajuster le poids image/texte et observer
      les prédictions changer en temps réel
================================================================================
"""








# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import os
import zipfile
from pathlib import Path

# ── Configuration page ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Rakuten — Classification multimodale",
    page_icon="🛍️",
    layout="wide"
)

# ── Chemins ──────────────────────────────────────────────────────────────────
DATA_DIR  = Path("data")
IMAGE_ZIP = DATA_DIR / "images_streamlit.zip"
IMAGE_DIR = DATA_DIR / "images_streamlit"

CODE2LABEL = {
    10: "Livre occasion",       40: "Jeu vidéo",
    50: "Access. Console",      60: "Console de jeu",
    1140: "Figurine",           1160: "Carte Collection",
    1180: "Jeu Plateau",        1280: "Jouet enfant",
    1281: "Jeu société",        1300: "Jouet tech",
    1301: "Chaussettes",        1302: "Jeu extérieur",
    1320: "Bébé",               1560: "Mobilier int.",
    1920: "Chambre",            1940: "Cuisine",
    2060: "Déco intérieure",    2220: "Animal",
    2280: "Revues",             2403: "Magazines",
    2462: "Jeu occasion",       2522: "Bureautique",
    2582: "Mobilier ext.",      2583: "Piscine",
    2585: "Bricolage",          2705: "Livre neuf",
    2905: "Jeu PC",
}

# ── Extraction des images ─────────────────────────────────────────────────────
def extract_images_if_needed():
    if not IMAGE_DIR.exists() and IMAGE_ZIP.exists():
        with zipfile.ZipFile(IMAGE_ZIP, "r") as z:
            z.extractall(DATA_DIR)
    return IMAGE_DIR

# ── Chargement des données ────────────────────────────────────────────────────
@st.cache_data
def load_data():
    # Subset pour la démo
    df_subset = pd.read_csv(DATA_DIR / "df_streamlit_subset.csv", index_col=0)

    # Dataset test complet (pour les prédictions dynamiques)
    df_test   = pd.read_csv(DATA_DIR / "df_test.csv", index_col=0)
    if 'imagefile' not in df_test.columns:
        df_test['imagefile'] = ('image_' + df_test['imageid'].astype(str) +
                                '_product_' + df_test['productid'].astype(str) + '.jpg')

    # Probabilités des modèles (fichiers .npy)
    vgg16_probs     = np.load(DATA_DIR / "vgg16_test_probs.npy")
    camembert_probs = np.load(DATA_DIR / "camembert_test_probs.npy")

    # Classes ordonnées (même ordre que lors de l'entraînement Keras)
    classes_ordered = list(np.sort(
        np.array(df_test["prdtypecode"].unique(), dtype="str")
    ))

    # Mapping index df_test → position dans le tableau de prédictions
    idx_to_pos = {idx: pos for pos, idx in enumerate(df_test.index)}

    return df_subset, df_test, vgg16_probs, camembert_probs, classes_ordered, idx_to_pos

# ── Fusion dynamique ──────────────────────────────────────────────────────────
def get_fusion_pred_for_row(idx, vgg16_probs, camembert_probs,
                             classes_ordered, idx_to_pos, img_w):
    """Calcule la prédiction fusionnée pour une ligne donnée."""
    if idx not in idx_to_pos:
        return None
    pos      = idx_to_pos[idx]
    txt_w    = round(1 - img_w, 2)
    combined = (vgg16_probs[pos] * img_w) + (camembert_probs[pos] * txt_w)
    pred_idx = np.argmax(combined)
    return int(classes_ordered[pred_idx])

def get_fusion_f1_estimate(img_w, vgg16_probs, camembert_probs, df_test):
    """Estime le F1 score pour un poids donné (sur le test set complet)."""
    txt_w    = round(1 - img_w, 2)
    combined = (vgg16_probs * img_w) + (camembert_probs * txt_w)
    pred_idx = np.argmax(combined, axis=1)
    classes  = list(np.sort(np.array(df_test["prdtypecode"].unique(), dtype="str")))
    preds    = [int(classes[c]) for c in pred_idx]
    n_correct = sum(p == r for p, r in zip(preds, df_test["prdtypecode"].values))
    return n_correct / len(preds)

# ── Chargement ────────────────────────────────────────────────────────────────
IMAGE_DIR = extract_images_if_needed()
df_subset, df_test, vgg16_probs, camembert_probs, classes_ordered, idx_to_pos = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Paramètres")
st.sidebar.markdown("---")

# Scénario
scenario_labels = {
    "🔀 Fusion gagne seule (image ❌  texte ❌)": (False, False),
    "🖼️ Image aide la fusion (image ✅  texte ❌)": (True, False),
    "📝 Texte aide la fusion (image ❌  texte ✅)": (False, True),
    "✅ Tous corrects": (True, True),
}
scenario = st.sidebar.selectbox("Scénario de prédiction", list(scenario_labels.keys()))
img_ok, txt_ok = scenario_labels[scenario]

st.sidebar.markdown("---")

# Slider poids
img_weight = st.sidebar.slider(
    "Poids modèle Image (VGG16)",
    min_value=0.0, max_value=1.0, value=0.15, step=0.05
)
txt_weight = round(1 - img_weight, 2)

st.sidebar.markdown(f"""
| Modèle | Poids |
|--------|-------|
| 🖼️ VGG16 | **{img_weight}** |
| 📝 CamemBERT | **{txt_weight}** |
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Performances individuelles")
st.sidebar.markdown(f"""
| Modèle | F1 Test |
|--------|---------|
| TF-IDF + SVC | 0.8121 |
| VGG16 seul | 0.6521 |
| CamemBERT seul | 0.8826 |
| **Fusion (0.15/0.85)** | **~0.89** |
""")

# ── Filtrage du subset ────────────────────────────────────────────────────────
df_filtered = df_subset[
    (df_subset["prediction_correct_image"]  == img_ok) &
    (df_subset["prediction_correct_text"]   == txt_ok) &
    (df_subset["prediction_correct_fusion"] == True)
]

# ── Page principale ───────────────────────────────────────────────────────────
st.title("🛍️ Rakuten — Classification multimodale de produits")
st.markdown(
    "Exploration interactive de la fusion **VGG16** (image) + **CamemBERT** (texte) "
    "pour la classification de produits en 27 catégories."
)
st.divider()

col_a, col_b, col_c = st.columns(3)
col_a.metric("Produits dans ce scénario", len(df_filtered))
col_b.metric("Poids Image / Texte", f"{img_weight} / {txt_weight}")
col_c.metric("Scénario", scenario.split(" ")[0])

st.divider()
st.markdown(f"**Scénario sélectionné :** {scenario}")

if df_filtered.empty:
    st.warning("Aucun produit ne correspond à ce scénario dans le subset de démonstration.")
else:
    for idx, row in df_filtered.iterrows():
        col1, col2 = st.columns([1, 2])

        # ── Colonne image ─────────────────────────────────────────────────────
        with col1:
            # Structure zip : images_streamlit/{prdtypecode}/{imagefile}
            img_path = IMAGE_DIR / "images_streamlit" / \
                       str(row["prdtypecode"]) / row["imagefile"]

            # Fallback si double nesting absent
            if not img_path.exists():
                img_path = IMAGE_DIR / str(row["prdtypecode"]) / row["imagefile"]

            if img_path.exists():
                st.image(Image.open(img_path), width='stretch')
            else:
                st.warning(f"Image non trouvée : {img_path.name}")
                st.caption(f"Chemin tenté : {img_path}")

        # ── Colonne infos + prédictions ───────────────────────────────────────
        with col2:
            cat_label = CODE2LABEL.get(int(row["prdtypecode"]), str(row["prdtypecode"]))
            st.markdown(f"### Catégorie réelle : `{row['prdtypecode']}` — {cat_label}")

            texte = str(row.get("text_fr", row.get("text", "—")))
            st.markdown(f"**Texte produit :**")
            st.info(texte[:300] + ("..." if len(texte) > 300 else ""))

            st.markdown("**Résultats des prédictions :**")

            # Prédiction image seule
            res_img = "✅ Correct" if row["prediction_correct_image"] else "❌ Incorrect"
            # Prédiction texte seule
            res_txt = "✅ Correct" if row["prediction_correct_text"] else "❌ Incorrect"

            # Prédiction fusion dynamique avec le slider
            fusion_pred = get_fusion_pred_for_row(
                idx, vgg16_probs, camembert_probs,
                classes_ordered, idx_to_pos, img_weight
            )
            if fusion_pred is not None:
                fusion_correct = (fusion_pred == int(row["prdtypecode"]))
                fusion_label   = CODE2LABEL.get(fusion_pred, str(fusion_pred))
                res_fusion     = f"{'✅' if fusion_correct else '❌'} `{fusion_pred}` — {fusion_label}"
            else:
                res_fusion = "— (index non trouvé)"

            st.markdown(f"""
| Modèle | Résultat |
|--------|----------|
| 🖼️ VGG16 (image seule) | {res_img} |
| 📝 CamemBERT (texte seul) | {res_txt} |
| 🔀 Fusion ({img_weight} / {txt_weight}) | {res_fusion} |
""")

        st.divider()

# ── Footer ────────────────────────────────────────────────────────────────────
st.caption(
    "Projet Rakuten France Multimodal Product Data Classification — "
    "DataScientest · CamemBERT + VGG16"
)