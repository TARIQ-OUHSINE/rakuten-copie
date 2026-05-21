"""
app.py — Interface Démo Rakuten Classification Multimodale

Lancement :
    cd <racine_projet>
    streamlit run streamlit_app/app.py

Contraintes : ZERO ré-entraînement. Tous les modèles sont chargés depuis
models/clean/ et data/processed_clean/ (sorties des notebooks 1-5).
"""

from __future__ import annotations

import sys
import time
import logging
from pathlib import Path
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'streamlit_app'))

from pipeline import RakutenPipeline, CLASS_NAMES, MODELS_DIR, PROCESSED_DIR, SPLIT_DIR

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Rakuten — Classification Multimodale",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# CSS Professionnel
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    /* Palette */
    :root {
        --primary: #BF0000;
        --primary-light: #ff4444;
        --bg-card: #f8f9fa;
        --border: #e0e0e0;
        --text-muted: #6c757d;
    }

    /* Header principal */
    .main-header {
        background: linear-gradient(135deg, #BF0000 0%, #8B0000 100%);
        color: white;
        padding: 2rem 3rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(191,0,0,0.3);
    }
    .main-header h1 { font-size: 2.2rem; font-weight: 800; margin: 0; }
    .main-header p  { font-size: 1rem; opacity: 0.85; margin: 0.4rem 0 0; }

    /* KPI Cards */
    .kpi-card {
        background: white;
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border-top: 3px solid var(--primary);
    }
    .kpi-value { font-size: 2rem; font-weight: 800; color: var(--primary); }
    .kpi-label { font-size: 0.8rem; color: var(--text-muted); margin-top: 0.3rem; letter-spacing: 0.05em; text-transform: uppercase; }

    /* Prédiction Result */
    .pred-card {
        background: white;
        border: 2px solid var(--primary);
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 4px 16px rgba(191,0,0,0.15);
    }
    .pred-class { font-size: 1.8rem; font-weight: 800; color: var(--primary); }
    .pred-conf  { font-size: 1rem; color: var(--text-muted); margin-top: 0.4rem; }

    /* Section titres */
    .section-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #2c3e50;
        border-left: 4px solid var(--primary);
        padding-left: 0.8rem;
        margin: 1.5rem 0 1rem;
    }

    /* Status badges */
    .badge-ok   { background: #d4edda; color: #155724; padding: 2px 8px; border-radius: 12px; font-size: 0.8rem; }
    .badge-warn { background: #fff3cd; color: #856404; padding: 2px 8px; border-radius: 12px; font-size: 0.8rem; }
    .badge-err  { background: #f8d7da; color: #721c24; padding: 2px 8px; border-radius: 12px; font-size: 0.8rem; }

    /* Masquer le menu Streamlit par défaut */
    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
    header     { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Chargement des ressources (cache Streamlit)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Chargement du pipeline d'inférence...")
def get_pipeline() -> RakutenPipeline:
    pipe = RakutenPipeline()
    pipe.load_static_artifacts()
    return pipe


@st.cache_data(show_spinner=False)
def load_eda_data() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Charge stats_par_classe.csv et train.csv (sample)."""
    stats = None
    sample = None

    stats_path = PROCESSED_DIR / 'stats_par_classe.csv'
    if stats_path.exists():
        stats = pd.read_csv(stats_path)

    train_path = SPLIT_DIR / 'train.csv'
    if train_path.exists():
        sample = pd.read_csv(train_path)

    return stats, sample


@st.cache_data(show_spinner=False)
def load_benchmark() -> pd.DataFrame | None:
    p = MODELS_DIR / 'benchmark_final.csv'
    if p.exists():
        return pd.read_csv(p)
    # Resultats hardcodes si benchmark pas encore genere
    return pd.DataFrame([
        {'Systeme': 'SIGIR 2020 — 1er place',     'F1_Test': 0.9194, 'Type': 'Référence'},
        {'Systeme': 'SIGIR 2020 — Baseline',       'F1_Test': 0.8705, 'Type': 'Référence'},
        {'Systeme': 'Stacking Final (NB5)',         'F1_Test': 0.8541, 'Type': 'Nos modèles'},
        {'Systeme': 'Texte seul — LogReg (NB3)',    'F1_Test': 0.8340, 'Type': 'Nos modèles'},
        {'Systeme': 'Baseline LightGBM (NB2)',      'F1_Test': 0.8050, 'Type': 'Nos modèles'},
        {'Systeme': 'Image seule — DINOv2 (NB4)',   'F1_Test': 0.6117, 'Type': 'Nos modèles'},
    ])


@st.cache_resource
def get_e5_loaded_pipeline() -> RakutenPipeline:
    """Pipeline avec e5-large pre-charge (pour onglet inference)."""
    pipe = get_pipeline()
    pipe.load_e5()
    return pipe


# ---------------------------------------------------------------------------
# Helpers visuels
# ---------------------------------------------------------------------------

def kpi_card(value: str, label: str) -> str:
    return f"""
    <div class="kpi-card">
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
    </div>"""


def make_bar_chart_px(df, x, y, title, color=None, orientation='v'):
    import plotly.express as px
    color_map = {'Référence': '#f39c12', 'Nos modèles': '#BF0000'}
    kwargs = dict(title=title, template='plotly_white', height=420)
    if color and color in df.columns:
        kwargs['color'] = color
        kwargs['color_discrete_map'] = color_map
    if orientation == 'h':
        fig = px.bar(df, x=x, y=y, orientation='h', **kwargs)
    else:
        fig = px.bar(df, x=x, y=y, **kwargs)
    fig.update_layout(
        title_font_size=14, margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    return fig


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown("""
<div class="main-header">
    <h1>🛒 Rakuten France — Classification Multimodale</h1>
    <p>84 916 produits · 27 catégories · Texte + Image · Challenge SIGIR eCommerce 2020</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Exploration des données",
    "🏆 Performances & Benchmark",
    "🔮 Inférence Multimodale",
    "🔍 Interprétabilité",
])

# ===========================================================================
# TAB 1 — EDA
# ===========================================================================

with tab1:
    stats_df, train_sample = load_eda_data()

    # --- KPIs ---
    st.markdown('<div class="section-title">Vue d\'ensemble du dataset</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)

    total = 84_916
    n_classes = 27
    missing_pct = 35.0
    pct_fr = 79.0
    imbalance = 13.0

    with c1: st.markdown(kpi_card(f"{total:,}", "Produits total"), unsafe_allow_html=True)
    with c2: st.markdown(kpi_card(str(n_classes), "Classes"), unsafe_allow_html=True)
    with c3: st.markdown(kpi_card(f"{missing_pct:.0f}%", "Desc. manquantes"), unsafe_allow_html=True)
    with c4: st.markdown(kpi_card(f"{pct_fr:.0f}%", "Produits en Français"), unsafe_allow_html=True)
    with c5: st.markdown(kpi_card(f"{imbalance:.0f}x", "Ratio déséquilibre"), unsafe_allow_html=True)

    st.markdown("---")

    # --- Distributions ---
    if stats_df is not None:
        col_left, col_right = st.columns(2)

        with col_left:
            import plotly.express as px
            st.markdown('<div class="section-title">Répartition des classes</div>', unsafe_allow_html=True)
            df_sorted = stats_df.sort_values('N produits', ascending=True)
            fig = px.bar(
                df_sorted, x='N produits', y='class_name', orientation='h',
                color='N produits', color_continuous_scale='Reds',
                template='plotly_white', height=520,
                labels={'N produits': 'Nb produits', 'class_name': ''},
            )
            fig.update_coloraxes(showscale=False)
            fig.update_layout(margin=dict(l=10, r=20, t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.markdown('<div class="section-title">Descriptions manquantes par classe</div>', unsafe_allow_html=True)
            miss_col = [c for c in stats_df.columns if 'desc' in c.lower() and 'manquante' in c.lower()]
            if miss_col:
                df_miss = stats_df.sort_values(miss_col[0], ascending=True)
                colors = ['#e74c3c' if v > 50 else '#f39c12' if v > 20 else '#2ecc71'
                          for v in df_miss[miss_col[0]]]
                fig2 = px.bar(
                    df_miss, x=miss_col[0], y='class_name', orientation='h',
                    template='plotly_white', height=520,
                    labels={miss_col[0]: '% manquant', 'class_name': ''},
                    color=miss_col[0], color_continuous_scale=['#2ecc71', '#f39c12', '#e74c3c'],
                )
                fig2.update_coloraxes(showscale=False)
                fig2.add_vline(x=35, line_dash='dash', line_color='navy',
                               annotation_text='Moyenne 35%', annotation_position='top right')
                fig2.update_layout(margin=dict(l=10, r=20, t=20, b=20))
                st.plotly_chart(fig2, use_container_width=True)

        # --- Stats table ---
        st.markdown('<div class="section-title">Statistiques détaillées par classe</div>', unsafe_allow_html=True)
        st.dataframe(
            stats_df.set_index('class_name').style
            .background_gradient(cmap='Reds', subset=['N produits'])
            .format(precision=1),
            use_container_width=True,
            height=400,
        )

    else:
        st.info("Les statistiques EDA seront disponibles après exécution du **Notebook 1** (`01_preprocessing_eda.ipynb`).")

    # --- Figures pre-calculees ---
    st.markdown('<div class="section-title">Visualisations EDA pré-calculées</div>', unsafe_allow_html=True)

    FIGURES_DIR = ROOT / 'reports' / 'figures'
    fig_map = [
        ('graph01_class_distribution.png', 'Distribution des classes'),
        ('graph02_language_distribution.png', 'Distribution des langues'),
        ('graph03_text_length_by_class.png', 'Longueur texte par classe'),
        ('graph04_missing_desc_by_class.png', 'Descriptions manquantes'),
        ('graph05_wordclouds.png', 'Nuages de mots'),
        ('graph07_image_quality.png', 'Qualité des images'),
        ('graph08_autocrop_examples.png', 'Auto-crop exemples'),
    ]

    available_figs = [(fname, label) for fname, label in fig_map
                       if (FIGURES_DIR / fname).exists()]

    if available_figs:
        cols = st.columns(min(3, len(available_figs)))
        for i, (fname, label) in enumerate(available_figs):
            col = cols[i % len(cols)]
            with col:
                st.image(str(FIGURES_DIR / fname), caption=label, use_column_width=True)
    else:
        st.info("Exécutez le **Notebook 1** pour générer les figures EDA.")

    # --- Exemples de produits ---
    if train_sample is not None and 'prdtypecode' in train_sample.columns:
        st.markdown('<div class="section-title">Exemples de produits par classe</div>', unsafe_allow_html=True)

        all_classes = sorted(train_sample['prdtypecode'].unique())
        codes_with_names = {str(c): CLASS_NAMES.get(c, str(c)) for c in all_classes}
        selected_label = st.selectbox(
            "Choisir une classe :",
            options=list(codes_with_names.keys()),
            format_func=lambda k: f"{CLASS_NAMES.get(int(k), k)} (code {k})",
            key='eda_class_select'
        )

        if selected_label:
            cls_code = int(selected_label)
            subset = train_sample[train_sample['prdtypecode'] == cls_code].head(6)
            img_dir = SPLIT_DIR / 'train'

            example_cols = st.columns(min(6, len(subset)))
            for col, (_, row) in zip(example_cols, subset.iterrows()):
                img_path = img_dir / str(row.get('image_name', ''))
                with col:
                    if img_path.exists():
                        st.image(str(img_path), use_column_width=True)
                    else:
                        st.markdown("🖼️ Image non disponible")
                    text_preview = str(row.get('text_combined', row.get('text_combined_raw', '')))[:60]
                    st.caption(text_preview + '...')

# ===========================================================================
# TAB 2 — BENCHMARK
# ===========================================================================

with tab2:
    import plotly.express as px
    import plotly.graph_objects as go

    bench = load_benchmark()

    st.markdown('<div class="section-title">Benchmark comparatif — F1 Score (weighted)</div>', unsafe_allow_html=True)

    if bench is not None:
        bench_sorted = bench.sort_values('F1_Test', ascending=False).reset_index(drop=True)

        # --- Graphique principal ---
        type_col = 'Type' if 'Type' in bench_sorted.columns else None
        color_discrete = {'Référence': '#f39c12', 'Nos modèles': '#BF0000'}

        fig = px.bar(
            bench_sorted,
            x='Systeme', y='F1_Test',
            color=type_col,
            color_discrete_map=color_discrete if type_col else {},
            text='F1_Test',
            template='plotly_white',
            height=460,
            labels={'F1_Test': 'F1 Score (weighted)', 'Systeme': ''},
        )
        fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        fig.add_hline(y=0.9194, line_dash='dot', line_color='#27ae60',
                      annotation_text='SIGIR 1er (0.9194)', annotation_position='top right',
                      annotation_font_size=11)
        fig.add_hline(y=0.8705, line_dash='dot', line_color='#f39c12',
                      annotation_text='SIGIR baseline (0.8705)', annotation_position='top right',
                      annotation_font_size=11)
        fig.update_layout(
            xaxis_tickangle=-25,
            yaxis_range=[0.5, 1.0],
            legend_title_text='',
            margin=dict(l=20, r=20, t=60, b=80),
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Tableau interactif ---
        col_l, col_r = st.columns([2, 1])
        with col_l:
            st.markdown('<div class="section-title">Tableau détaillé</div>', unsafe_allow_html=True)
            bench_display = bench_sorted.copy()
            bench_display['F1_Test'] = bench_display['F1_Test'].map(lambda x: f"{x:.4f}")
            best_score = float(bench_sorted.iloc[0]['F1_Test'])
            ref_sigir = 0.9194

            def row_style(row):
                if 'SIGIR' in str(row.get('Systeme', '')):
                    return ['background-color: #fff8e1'] * len(row)
                elif row.name == 0:
                    return ['background-color: #ffe8e8; font-weight: bold'] * len(row)
                return [''] * len(row)

            st.dataframe(bench_display, use_container_width=True, hide_index=True)

        with col_r:
            st.markdown('<div class="section-title">Positionnement</div>', unsafe_allow_html=True)
            our_best = float(bench_sorted[bench_sorted.get('Type', pd.Series()) == 'Nos modèles']['F1_Test'].max()
                             if 'Type' in bench_sorted.columns else bench_sorted.iloc[0]['F1_Test'])

            gap_baseline = our_best - 0.8705
            gap_champion = our_best - 0.9194

            st.metric("Notre meilleur score", f"{our_best:.4f}")
            st.metric("vs SIGIR Baseline",
                      f"{our_best:.4f}",
                      delta=f"{gap_baseline:+.4f}",
                      delta_color="normal")
            st.metric("vs SIGIR Champion",
                      f"{our_best:.4f}",
                      delta=f"{gap_champion:.4f}",
                      delta_color="inverse")

            pct_gap = (1 - our_best / 0.9194) * 100
            st.markdown(f"**Écart vs champion :** {pct_gap:.1f}% de performance restante")

        # --- Radar chart modalités ---
        st.markdown('<div class="section-title">Apport de chaque modalité</div>', unsafe_allow_html=True)

        modality_data = {
            'Modalité': ['Texte seul', 'Image seule', 'Baseline (LightGBM)', 'Stacking Final'],
            'F1_Score': [0.8340, 0.6117, 0.8050, 0.8541],
        }
        try:
            for _, row in bench_sorted.iterrows():
                s = str(row['Systeme'])
                f = float(row['F1_Test']) if isinstance(row['F1_Test'], (int, float)) else float(row['F1_Test'].replace(',', '.'))
                if 'texte' in s.lower() or 'text' in s.lower():
                    modality_data['F1_Score'][0] = f
                elif 'image' in s.lower() or 'dino' in s.lower():
                    modality_data['F1_Score'][1] = f
                elif 'baseline' in s.lower() or 'lgbm' in s.lower():
                    modality_data['F1_Score'][2] = f
                elif 'stacking' in s.lower() or 'final' in s.lower():
                    modality_data['F1_Score'][3] = f
        except Exception:
            pass

        df_mod = pd.DataFrame(modality_data)
        fig_mod = px.bar(
            df_mod, x='Modalité', y='F1_Score',
            color='F1_Score', color_continuous_scale='Reds',
            text='F1_Score', template='plotly_white', height=320,
            labels={'F1_Score': 'F1 Score (weighted)'},
        )
        fig_mod.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        fig_mod.update_coloraxes(showscale=False)
        fig_mod.update_layout(yaxis_range=[0.5, 1.0], margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_mod, use_container_width=True)

    else:
        st.info("Le benchmark sera disponible après exécution du **Notebook 5** (`05_stacking_final.ipynb`).")

# ===========================================================================
# TAB 3 — INFÉRENCE MULTIMODALE
# ===========================================================================

with tab3:
    st.markdown('<div class="section-title">Testez la classification en temps réel</div>', unsafe_allow_html=True)

    # --- Status pipeline ---
    pipe = get_pipeline()
    artifact_status = pipe.load_static_artifacts()

    with st.expander("🔧 Statut du pipeline", expanded=False):
        for name, ok in artifact_status.items():
            if ok is True:
                st.markdown(f'<span class="badge-ok">✓ {name}</span>', unsafe_allow_html=True)
            elif ok == 'baseline_fallback':
                st.markdown(f'<span class="badge-warn">⚠ {name} (fallback baseline)</span>', unsafe_allow_html=True)
            else:
                st.markdown(f'<span class="badge-err">✗ {name} — manquant</span>', unsafe_allow_html=True)

    st.markdown("---")

    # --- Formulaire ---
    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown('<div class="section-title">📝 Saisir les informations produit</div>', unsafe_allow_html=True)

        designation = st.text_input(
            "Désignation (titre du produit) *",
            placeholder="Ex: LEGO Star Wars Millennium Falcon 75192",
            help="Titre principal du produit tel qu'il apparaît sur la plateforme.",
        )

        description = st.text_area(
            "Description (optionnel)",
            placeholder="Ex: Set de construction Lego. Contient 7541 pièces. Pour adultes et enfants 16+...",
            height=140,
            help="Description détaillée du produit. Peut être vide.",
        )

        uploaded_image = st.file_uploader(
            "Image du produit (optionnel)",
            type=['jpg', 'jpeg', 'png', 'webp'],
            help="Image du produit. Si fournie, améliore la précision de la classification.",
        )

        if uploaded_image is not None:
            from PIL import Image as PILImage
            preview_img = PILImage.open(BytesIO(uploaded_image.read()))
            uploaded_image.seek(0)
            st.image(preview_img.resize((200, 200)), caption="Aperçu", width=200)
        else:
            preview_img = None

        st.markdown("")
        col_btn1, col_btn2 = st.columns([2, 1])
        with col_btn1:
            predict_btn = st.button(
                "🔮 Classifier ce produit",
                type="primary",
                use_container_width=True,
                disabled=not designation.strip(),
            )
        with col_btn2:
            if st.button("🎲 Exemple aléatoire", use_container_width=True):
                # Charger un produit aléatoire depuis le jeu de test
                train_path = SPLIT_DIR / 'val.csv'
                if train_path.exists():
                    sample_df = pd.read_csv(train_path, nrows=500)
                    rand_row = sample_df.sample(1).iloc[0]
                    raw_col = 'text_combined_raw' if 'text_combined_raw' in sample_df.columns else 'text_combined'
                    raw_text = str(rand_row.get(raw_col, ''))
                    # Découpage approximatif designation/description
                    parts = raw_text.split('. ', 1)
                    st.session_state['demo_desig'] = parts[0][:120]
                    st.session_state['demo_desc'] = parts[1][:200] if len(parts) > 1 else ''
                    st.rerun()

    # Auto-remplir depuis session_state (bouton exemple)
    if 'demo_desig' in st.session_state:
        designation = st.session_state.pop('demo_desig')
        description = st.session_state.pop('demo_desc', '')

    # --- Résultat ---
    with col_result:
        st.markdown('<div class="section-title">🎯 Résultat de la classification</div>', unsafe_allow_html=True)

        if predict_btn and designation.strip():
            if not any(artifact_status.values()):
                st.error("Aucun modèle disponible. Veuillez d'abord exécuter les notebooks 1 à 5.")
            else:
                with st.spinner("Analyse en cours..."):
                    t0 = time.perf_counter()

                    # Charger l'image si fournie
                    pil_image = None
                    if uploaded_image is not None:
                        from PIL import Image as PILImage
                        pil_image = PILImage.open(BytesIO(uploaded_image.read())).convert('RGB')

                    try:
                        result = pipe.predict(
                            designation=designation,
                            description=description,
                            image=pil_image,
                            use_deep_text=True,
                            use_deep_image=(pil_image is not None),
                        )
                        elapsed = time.perf_counter() - t0

                        # Stocker pour Tab 4
                        st.session_state['last_result'] = result
                        st.session_state['last_designation'] = designation
                        st.session_state['last_description'] = description
                        st.session_state['last_image'] = pil_image

                        # Carte résultat
                        st.markdown(f"""
                        <div class="pred-card">
                            <div style="font-size: 3rem;">
                                {"📦" if result['confidence'] > 0.8 else "🤔"}
                            </div>
                            <div class="pred-class">{result['predicted_name']}</div>
                            <div class="pred-conf">Code Rakuten : {result['predicted_code']}</div>
                        </div>
                        """, unsafe_allow_html=True)

                        st.markdown("")

                        # Jauge de confiance
                        conf = result['confidence']
                        conf_color = '#2ecc71' if conf > 0.8 else '#f39c12' if conf > 0.5 else '#e74c3c'
                        st.markdown(f"**Confiance : {conf*100:.1f}%**")
                        st.progress(conf)

                        # Top-3 classes
                        st.markdown("**Top-3 classes probables :**")
                        import plotly.graph_objects as go
                        top3_names = [t[0] for t in result['top3']]
                        top3_probs = [t[1] for t in result['top3']]
                        colors_top3 = [conf_color if i == 0 else '#95a5a6' for i in range(3)]

                        fig_top3 = go.Figure(go.Bar(
                            y=top3_names[::-1], x=top3_probs[::-1],
                            orientation='h',
                            marker_color=colors_top3[::-1],
                            text=[f'{p*100:.1f}%' for p in top3_probs[::-1]],
                            textposition='outside',
                        ))
                        fig_top3.update_layout(
                            xaxis=dict(range=[0, 1], tickformat='.0%'),
                            margin=dict(l=10, r=80, t=10, b=10),
                            height=160, template='plotly_white',
                        )
                        st.plotly_chart(fig_top3, use_container_width=True)

                        # Meta-info
                        st.caption(f"⏱ Temps d'inférence : {elapsed*1000:.0f} ms  |  Features : {', '.join(result['features_used'])}")

                    except Exception as exc:
                        st.error(f"Erreur lors de la prédiction : {exc}")
                        st.info("Vérifiez que les notebooks 1-5 ont été exécutés et que les artefacts sont en place.")

        elif not predict_btn:
            st.markdown("""
            <div style="text-align: center; padding: 3rem 1rem; color: #6c757d;">
                <div style="font-size: 4rem;">🔮</div>
                <p style="margin-top: 1rem;">Remplissez le formulaire et cliquez sur <strong>Classifier ce produit</strong></p>
                <p style="font-size: 0.85rem;">Le modèle analysera le texte et l'image pour prédire la catégorie.</p>
            </div>
            """, unsafe_allow_html=True)

# ===========================================================================
# TAB 4 — INTERPRÉTABILITÉ
# ===========================================================================

with tab4:
    st.markdown('<div class="section-title">Comment le modèle prend sa décision ?</div>', unsafe_allow_html=True)

    last_result = st.session_state.get('last_result')
    last_desig  = st.session_state.get('last_designation', '')
    last_desc   = st.session_state.get('last_description', '')
    last_image  = st.session_state.get('last_image')

    if last_result is None:
        st.info("💡 Effectuez une prédiction dans l'onglet **Inférence Multimodale** pour voir l'interprétabilité.")
        st.markdown("---")
        st.markdown("**Ce que vous verrez ici :**")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
            **📝 Interprétabilité Texte**
            - Mots qui ont le plus influencé la décision
            - Coefficient LogReg positif (favorable à la classe)
            - Coefficient LogReg négatif (défavorable)
            - Visualisation avec barres colorées
            """)
        with col_b:
            st.markdown("""
            **🖼️ Interprétabilité Image (GradCAM)**
            - Zones de l'image les plus discriminantes
            - Heatmap de chaleur superposée
            - Calculé via ResNet50 fine-tuné
            - Explique "où le modèle regarde"
            """)
    else:
        pipe_interp = get_pipeline()
        predicted_idx  = last_result['predicted_idx']
        predicted_name = last_result['predicted_name']
        confidence     = last_result['confidence']

        # --- En-tête résultat ---
        st.success(f"Analyse de la prédiction : **{predicted_name}** (confiance {confidence*100:.1f}%)")

        col_text, col_img = st.columns(2, gap="large")

        # ── Texte ──────────────────────────────────────────────────────────
        with col_text:
            st.markdown('<div class="section-title">📝 Interprétabilité Texte</div>', unsafe_allow_html=True)

            word_scores = pipe_interp.highlight_important_words(
                text=last_desig + ' ' + last_desc,
                predicted_class_idx=predicted_idx,
                top_n=15,
            )

            if word_scores:
                import plotly.graph_objects as go
                words = [w for w, _ in word_scores]
                scores = [s for _, s in word_scores]
                colors = ['#2ecc71' if s > 0 else '#e74c3c' for s in scores]

                fig_words = go.Figure(go.Bar(
                    x=scores, y=words, orientation='h',
                    marker_color=colors,
                    text=[f'{s:+.3f}' for s in scores],
                    textposition='outside',
                ))
                fig_words.update_layout(
                    xaxis_title='Coefficient LogReg',
                    yaxis_title='',
                    template='plotly_white',
                    height=400,
                    margin=dict(l=10, r=60, t=20, b=20),
                )
                fig_words.add_vline(x=0, line_color='black', line_width=1)
                st.plotly_chart(fig_words, use_container_width=True)

                # Texte annoté
                st.markdown("**Texte analysé avec importance des mots :**")
                full_text = (last_desig + ' ' + last_desc).strip()
                score_map = {w: s for w, s in word_scores}

                from pipeline import preprocess_text, build_combined_raw
                tokens_clean = preprocess_text(build_combined_raw(last_desig, last_desc)).split()
                raw_tokens = full_text.split()

                highlighted_parts = []
                for tok in raw_tokens[:80]:
                    clean_tok = preprocess_text(tok)
                    if clean_tok in score_map:
                        s = score_map[clean_tok]
                        if s > 0.05:
                            highlighted_parts.append(f'<mark style="background:#d4edda;border-radius:3px;padding:1px 3px"><b>{tok}</b></mark>')
                        elif s < -0.05:
                            highlighted_parts.append(f'<mark style="background:#f8d7da;border-radius:3px;padding:1px 3px">{tok}</mark>')
                        else:
                            highlighted_parts.append(tok)
                    else:
                        highlighted_parts.append(tok)

                html_text = ' '.join(highlighted_parts)
                if len(raw_tokens) > 80:
                    html_text += ' <em>(...)</em>'

                st.markdown(
                    f'<div style="background:#f8f9fa;padding:1rem;border-radius:8px;'
                    f'border:1px solid #dee2e6;font-size:0.9rem;line-height:1.8">'
                    f'{html_text}</div>',
                    unsafe_allow_html=True
                )
                st.caption(
                    '<span style="background:#d4edda;padding:2px 6px;border-radius:3px">Vert</span> = favorable &nbsp;'
                    '<span style="background:#f8d7da;padding:2px 6px;border-radius:3px">Rouge</span> = défavorable',
                    unsafe_allow_html=True
                )
            else:
                st.info("Exécutez le **Notebook 3** pour générer `clf_text_full.pkl`.")

        # ── Image ──────────────────────────────────────────────────────────
        with col_img:
            st.markdown('<div class="section-title">🖼️ Interprétabilité Image (GradCAM)</div>', unsafe_allow_html=True)

            if last_image is not None:
                with st.spinner("Calcul GradCAM..."):
                    cam = pipe_interp.gradcam(last_image, predicted_idx)

                if cam is not None:
                    try:
                        import cv2 as _cv2
                    except ImportError:
                        _cv2 = None
                    import plotly.graph_objects as go
                    from PIL import Image as PILImage

                    img_resized = last_image.resize((224, 224))
                    img_arr = np.array(img_resized)

                    if _cv2 is not None:
                        cam_resized = _cv2.resize(cam, (224, 224))
                        heatmap_raw = _cv2.applyColorMap((cam_resized * 255).astype(np.uint8), _cv2.COLORMAP_JET)
                        heatmap_rgb = _cv2.cvtColor(heatmap_raw, _cv2.COLOR_BGR2RGB)
                    else:
                        # Fallback sans cv2 : colormap matplotlib
                        import matplotlib.cm as mpl_cm
                        from PIL import Image as PILImage
                        cam_resized_pil = np.array(PILImage.fromarray(
                            (cam * 255).astype(np.uint8)).resize((224, 224))) / 255.0
                        cmap = mpl_cm.get_cmap('jet')
                        heatmap_rgb = (cmap(cam_resized_pil)[:, :, :3] * 255).astype(np.uint8)
                        cam_resized = cam_resized_pil
                    overlay = (0.55 * img_arr + 0.45 * heatmap_rgb).clip(0, 255).astype(np.uint8)

                    col_i1, col_i2, col_i3 = st.columns(3)
                    with col_i1:
                        st.image(img_resized, caption="Image originale", use_column_width=True)
                    with col_i2:
                        st.image(heatmap_rgb, caption="Zones d'attention", use_column_width=True)
                    with col_i3:
                        st.image(overlay, caption="Superposition", use_column_width=True)

                    # Statistiques de la heatmap
                    st.markdown("**Analyse des zones d'attention :**")
                    h, w = cam_resized.shape
                    center_cam = cam_resized[h//4:3*h//4, w//4:3*w//4].mean()
                    border_cam = (cam_resized.mean() * cam_resized.size - cam_resized[h//4:3*h//4, w//4:3*w//4].sum()) / \
                                  (cam_resized.size - (h//2) * (w//2))
                    focus = "centre" if center_cam > border_cam else "bords"
                    st.markdown(f"→ Le modèle se concentre principalement sur le **{focus}** de l'image.")
                    st.markdown(f"→ Attention maximale : {cam.max():.2f} | Attention moyenne : {cam.mean():.2f}")
                else:
                    st.warning("GradCAM non disponible. Vérifiez que `resnet50_ft.pth` existe dans `models/clean/`.")
            else:
                st.info("Soumettez une image dans l'onglet **Inférence** pour voir la carte GradCAM.")

        # --- Explication vulgarisée ---
        st.markdown("---")
        st.markdown('<div class="section-title">🎓 Comment le modèle décide-t-il ?</div>', unsafe_allow_html=True)

        col_v1, col_v2, col_v3 = st.columns(3)
        with col_v1:
            st.markdown("""
            **Étape 1 — Analyse Texte**

            Le modèle transforme le titre et la description en vecteur numérique.
            Il compare ce vecteur avec les "profils textuels" de chaque catégorie
            (c-TF-IDF) et avec 1024 dimensions sémantiques (e5-large).
            """)
        with col_v2:
            st.markdown("""
            **Étape 2 — Analyse Image**

            DINOv2 (pré-entraîné sur 142M d'images) extrait 768 caractéristiques
            visuelles. Pas besoin de fine-tuning : le modèle "voit" déjà très bien
            les produits e-commerce.
            """)
        with col_v3:
            st.markdown("""
            **Étape 3 — Fusion (Stacking)**

            Un méta-classifieur combine les probabilités des deux branches.
            Il exploite aussi des méta-features : l'incertitude (entropie), la
            marge entre les top-2 classes, et l'accord texte/image.
            """)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#6c757d;font-size:0.8rem;">'
    'Rakuten Classification Multimodale — Challenge SIGIR eCommerce 2020 · '
    'Xavier · Tingfen · Veyss'
    '</p>',
    unsafe_allow_html=True
)