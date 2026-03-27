# Approach Comparison: Text Translation for TF-IDF + LightGBM

## 1. Motivation

The Rakuten France catalogue contains multilingual product text (predominantly French, with German, English, and other languages). This experiment tests whether **translating all text to French** before TF-IDF extraction improves classification, compared to a **multilingual baseline** that processes text as-is with multi-language stopword removal.

---

## 2. Pipeline Overview

| Step | Baseline (Multilingual) | Translated (French-only) |
|------|------------------------|--------------------------|
| **Text input** | `text_clean` (designation + description) | `text_clean` (designation + description) |
| **Language detection** | — | langdetect per row |
| **Translation** | — | Non-French → French via Google Translate (deep-translator) |
| **Post-processing** | Lowercase, strip HTML/punctuation/digits, remove accents | Lowercase, strip punctuation/digits |
| **Stopwords** | French + English + German (NLTK) | French only (NLTK) |
| **Vectorizer** | TF-IDF | TF-IDF |
| **Classifier** | LightGBM | LightGBM |

---

## 3. Model Configuration

| Parameter | Value (both models) |
|-----------|---------------------|
| max_features | 20,000 |
| ngram_range | (1, 2) |
| sublinear_tf | True |
| min_df | 2 |
| n_estimators | 1,000 |
| learning_rate | 0.05 |
| num_leaves | 127 |
| class_weight | balanced |
| early_stopping_rounds | 50 |

> **Note:** Both models use identical hyperparameters to ensure a fair comparison. The only difference is the text preprocessing pipeline (multilingual vs. translated-to-French).

---

## 4. Dataset Split (shared)

| Set | Samples |
|-----|---------|
| Train | 67,932 |
| Validation | 16,984 |
| Classes | 27 |

Split: 80/20 stratified, `random_state=42`.

---

## 5. Overall Results

| Metric | Baseline | Translated | Delta |
|--------|----------|------------|-------|
| **Weighted F1** | **0.8184** | 0.8107 | **−0.0077** |
| Accuracy | 0.81 | 0.81 | 0.00 |
| Macro Avg F1 | 0.80 | 0.79 | −0.01 |
| Best Iteration | 109 | 106 | — |

**Verdict:** Translation **did not improve** classification. The baseline outperforms by +0.77 pp on weighted F1, using the same model configuration.

---

## 6. Per-Class F1 Comparison

| Code | Category | Baseline F1 | Translated F1 | Delta |
|------|----------|-------------|---------------|-------|
| 10 | Livres | 0.46 | 0.44 | −0.02 |
| 40 | Jeux vidéo PC/consoles | 0.68 | 0.64 | −0.04 |
| 50 | Accessoires gaming | 0.82 | 0.80 | −0.02 |
| 60 | Consoles de jeux vidéo | 0.88 | 0.83 | **−0.05** |
| 1140 | Produits dérivés/figurines geek | 0.77 | 0.76 | −0.01 |
| 1160 | Cartes collectionnables | 0.91 | 0.91 | 0.00 |
| 1180 | Figurines jeux de société | 0.62 | 0.62 | 0.00 |
| 1280 | Jouets/peluches/poupées | 0.74 | 0.74 | 0.00 |
| 1281 | Jeux de société/cartes | 0.60 | 0.59 | −0.01 |
| 1300 | Petites voitures/maquettes | 0.94 | 0.94 | 0.00 |
| 1301 | Accessoires petite enfance/jeux | 0.92 | 0.92 | 0.00 |
| 1302 | Jeux d'extérieur | 0.81 | 0.81 | 0.00 |
| 1320 | Sacs femmes/accessoires enfance | 0.80 | 0.80 | 0.00 |
| 1560 | Mobilier/déco maison | 0.85 | 0.85 | 0.00 |
| 1920 | Linge de maison | 0.93 | 0.92 | −0.01 |
| 1940 | Nourriture/épicerie | 0.87 | 0.82 | **−0.05** |
| 2060 | Lampes/accessoires déco | 0.84 | 0.82 | −0.02 |
| 2220 | Accessoires animaux | 0.84 | 0.84 | 0.00 |
| 2280 | Magazines | 0.70 | 0.69 | −0.01 |
| 2403 | Livres et BDs | 0.74 | 0.73 | −0.01 |
| 2462 | Consoles/jeux vidéo | 0.80 | 0.79 | −0.01 |
| 2522 | Papeterie/rangement bureau | 0.91 | 0.91 | 0.00 |
| 2582 | Mobilier extérieur/accessoires | 0.78 | 0.76 | −0.02 |
| 2583 | Accessoires piscine | 0.97 | 0.97 | 0.00 |
| 2585 | Outillage/jardinage | 0.80 | 0.80 | 0.00 |
| 2705 | BDs et livres | 0.69 | 0.70 | +0.01 |
| 2905 | Jeux en téléchargement | 0.99 | 0.99 | 0.00 |

**Winners (translated):** Only 2705 — BDs et livres (+0.01).
**Losers (translated):** 60 — Consoles de jeux vidéo (−0.05), 1940 — Nourriture/épicerie (−0.05), 40 — Jeux vidéo PC/consoles (−0.04).

---

## 7. Analysis

### Why translation did not help

1. **Loss of discriminative foreign terms.** German product names (e.g. "Notizbuch", "Punktraster") and English brand terms (e.g. "Nintendo", "Gamepad") are category-specific signals. Translation replaces them with common French words, diluting their discriminative power in the TF-IDF space.

2. **Translation noise.** Google Translate introduces errors on short, keyword-heavy product titles — especially for domain-specific vocabulary (gaming, collectibles). These errors add noise without adding signal.

3. **Dataset is already majority French (~70%).** Only ~30% of rows are non-French, limiting the upside of translation. The multi-language stopword removal in the baseline handles the remaining 30% adequately.

4. **TF-IDF is language-agnostic by nature.** TF-IDF does not need a unified language — it weights terms by frequency. A German word appearing only in book-category products is already a strong signal without translation.

### Consistently hard classes (both models)

| Code | Category | F1 Range | Likely Cause |
|------|----------|----------|-------------|
| 10 | Livres | 0.44–0.46 | Overlaps with 2403 (Livres et BDs) and 2705 (BDs et livres) |
| 1281 | Jeux de société/cartes | 0.59–0.60 | Overlaps with 1160 (Cartes collectionnables) and 1180 (Figurines jeux de société) |
| 1180 | Figurines jeux de société | 0.62 | Small class (153 val samples), shares vocabulary with 1140 and 1281 |

---

## 8. Conclusion & Recommendations

| Aspect | Recommendation |
|--------|---------------|
| **Translation** | Not recommended for TF-IDF pipelines on this dataset. Foreign terms carry discriminative signal. |
| **Baseline model** | Keep as the text-only baseline (weighted F1 = 0.8185). |
| **Next steps** | Use CamemBERT/FlauBERT which handle multilingual input natively via subword tokenisation, or add image features (Rendu 2 fusion). |
| **Hard classes** | Classes 10/2403/2705 (all book-related) need image features or hierarchical classification to disambiguate. |

---

## 9. Reproducibility

```bash
# Baseline
python models/baseline_model.py

# Translated
pip install langdetect deep-translator
python models/tfidf_translated.py
```

**Artefacts:**
- `models/lgb_text_model.pkl` — baseline model
- `models/lgb_translated_model.pkl` — translated model
- `models/tfidf_vectorizer.pkl` / `models/tfidf_translated_vectorizer.pkl`
- `docs/figures/mod_fig1a_lgb_text_confusion.png` — baseline confusion matrix
- `docs/figures/mod_fig_translated_confusion.png` — translated confusion matrix
- `docs/figures/mod_fig1b_lgb_text_per_class_f1.png` — baseline per-class F1
- `docs/figures/mod_fig_translated_per_class_f1.png` — translated per-class F1
- `docs/figures/mod_fig_lang_distribution.png` — language distribution
