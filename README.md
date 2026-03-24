Rakuten Product Classification
==============================

Multimodal e-commerce product classification using text and image features. The goal is to predict the product type code (`prdtypecode`) for listings on the Rakuten France marketplace across 27 categories.

## Approach

The project follows a progressive modeling strategy across four phases:

| Phase | Model | Features | Script |
|-------|-------|----------|--------|
| 1 — Baseline | TF-IDF + LightGBM | Text only | `models/baseline_model.py` |
| 2 — Optimization | TF-IDF + OpenCV + LightGBM | Text + image (color histogram, HOG, LBP) | `models/advanced_model.py` |
| 3 — Advanced | CLIP ViT-B/32 + MLP | Multimodal CLIP embeddings | `models/clip_mlp.py` / `models/optimized_clip_mlp.py` |
| 4 — Fine-tuning | GPT-4o Vision | Multimodal (text + image) | `models/gpt4o_finetune.py` |

## Project Organization

    ├── data/                  <- Local only (gitignored)
    │   ├── train_cleaned.pkl
    │   ├── test_cleaned.pkl
    │   ├── img_feats_*.npy   <- Cached image features
    │   └── *.csv             <- Raw data and predictions
    │
    ├── models/                <- Training scripts and saved artifacts
    │   ├── baseline_model.py
    │   ├── advanced_model.py
    │   ├── clip_mlp.py
    │   ├── optimized_clip_mlp.py
    │   ├── gpt4o_finetune.py
    │   ├── lgb_text_model.pkl
    │   ├── lgb_combined_model.pkl
    │   ├── tfidf_vectorizer.pkl
    │   └── label_encoder.pkl
    │
    ├── notebooks/             <- Jupyter notebooks
    │   ├── EDA.ipynb          <- Exploratory data analysis
    │   ├── modeling.ipynb     <- Model development
    │   └── rakuten.ipynb      <- End-to-end pipeline
    │
    ├── docs/                  <- Documentation and figures
    │   ├── figures/           <- Model evaluation plots (confusion matrices, SHAP, etc.)
    │   └── knowledge_base.md
    │
    ├── reports/
    │   └── figures/           <- EDA figures (class distribution, word clouds, etc.)
    │
    ├── src/                   <- Source code modules
    │   ├── features/
    │   ├── models/
    │   ├── streamlit/
    │   └── visualization/
    │
    └── requirements.txt

## Usage

```bash
# Phase 1: text-only baseline
python models/baseline_model.py

# Phase 2: text + image (OpenCV features)
python models/advanced_model.py

# Phase 3: CLIP + MLP
python models/clip_mlp.py            # standard
python models/optimized_clip_mlp.py  # optimized (mixed precision, batched)

# Phase 4: GPT-4o fine-tuning
python models/gpt4o_finetune.py prepare
python models/gpt4o_finetune.py train
python models/gpt4o_finetune.py predict --model <model_id>
```

## Data

Training and test data should be placed in the `data/` directory (gitignored). Images should be in `images/image_train/` and `images/image_test/`.

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
