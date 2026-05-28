Project Name
==============================

This repo is a Starting Pack for DS projects. You can rearrange the structure to make it fits your project.

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data               <- Should be in your computer but not on Github (only in .gitignore)
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's name, and a short `-` delimited description, e.g.
    │                         `1.0-alban-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, links, and all other explanatory materials.
    │
    ├── reports            <- The reports that you'll make during this project as PDF
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   ├── visualization  <- Scripts to create exploratory and results oriented visualizations
    │   │   └── visualize.py

--------

Streamlit demo app
------------

A demo app (inference only — no retraining) is provided in `src/streamlit/app.py`.
It exposes the full multimodal pipeline (CamemBERT fine-tuned + CLIP ViT-B/32 + MLP +
Logistic Regression stackers) through five tabs: project context, data exploration,
modeling results, live inference, and methodology.

### Architecture v2

| Branch  | Backbone                                | Head                       | Output     |
|---------|-----------------------------------------|----------------------------|------------|
| Text    | `camembert-base` (110 M params, fine-tuned) | Classification 768 → 27   | 27 probs  |
| Image   | `CLIP ViT-B/32` laion2b (frozen)        | MLP 512 → 512 → 256 → 27   | 27 probs  |
| Stacker | Logistic Regression (`saga`, balanced)  | 27 → 27 per modality       | 27 probs  |
| Fusion  | Arithmetic mean of stacked probs        | —                          | 27 probs  |

### Artifacts (`models/v2/`)

- `camembert_ft_best.pth` — fine-tuned CamemBERT (442 MB)
- `clip_mlp_best.pth` — image MLP head (1.6 MB)
- `meta_E1.pkl`, `meta_E2.pkl` — per-modality Logistic Regression stackers
- `label_encoder.pkl` — sklearn `LabelEncoder` over the 27 `prdtypecode` classes
- `ctfidf_centroids.npy` — class-based TF-IDF centroids (support artifact)
- `baseline_lgbm.pkl`, `meta_best.pkl` — additional artifacts (not currently wired up)

### Data requirements

The app reads from two locations on disk. None of this data is committed — it must be
present before launching.

**`models/v2/` — model artifacts (required)**

Symlinked from `data/streamlit_data/`. Provide either of:

- The two source archives `data/streamlit_data/models et data.zip` and
  `data/streamlit_data/model text et image best.zip`, which the project setup script
  unzips and symlinks into `models/v2/`, or
- The eight files listed in the *Artifacts* section above placed directly in `models/v2/`.

**`data/` — dataset files used by the EDA and demo tabs**

| File                      | Size    | Used by                                  | Required? |
|---------------------------|---------|------------------------------------------|-----------|
| `train_cleaned.pkl`       | ~91 MB  | Tab 2 (class stats), Tab 4 (sample products) | **Yes** |
| `train_translated.pkl`    | ~138 MB | Tab 2 (language distribution chart)      | Optional — chart hides if absent |

**`docs/figures/` — pre-rendered EDA + modeling figures**

Tabs 2 and 3 display PNGs from `docs/figures/` (fallback: `reports/figures/`). Missing
figures are silently skipped, so the app still runs with a partial set. Filenames the
app looks for include `fig3_text_length.png`, `fig4_image_grid.png`, `fig5_wordclouds.png`,
`fig6_image_sizes.png`, `mod_fig1a_lgb_text_confusion.png`, `mod_fig1b_lgb_text_per_class_f1.png`,
`mod_fig_translated_confusion.png`, `mod_fig_translated_per_class_f1.png`,
`mod_fig3_opencv_features.png`, `mod_fig4_ablation_opencv.png`,
`mod_fig5_feature_importance.png`, `mod_fig6_shap.png`.

### Running the app

```bash
pip install -r requirements.txt
streamlit run src/streamlit/app.py
```

The first prediction in the **Démo Inférence** tab triggers a one-time download and load
of CamemBERT (~440 MB) and CLIP ViT-B/32 (~600 MB) from the Hugging Face Hub. Subsequent
predictions reuse the cached models in-session.

### Demo tab

The user supplies a product **title**, **description**, and optionally an **image**.
The app shows the top-3 prediction for each branch:

- `Texte — CamemBERT + Meta-E1`
- `Image — CLIP+MLP + Meta-E2`
- `Fusion — Stacking E1 + E2`

The fusion card is the primary prediction when both modalities are available.

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
