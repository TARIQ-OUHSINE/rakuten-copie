# 🚀 Pipeline Rakuten Multimodal — Architecture Complète

> **Classification Multimodale de produits Rakuten France** | SIGIR 2020 Challenge | 27 classes

## 📋 Vue d'ensemble

Ce pipeline restructure le code de trois experts (Xavier, Veyss/Seb, Tingfen) en une architecture modulaire et hiérarchique, avançant progressivement de simples modèles unimodaux (E01-E02) vers des ensembles sophistiqués (E04-E05).

```
    E00_preprocessing.ipynb (données + features)
            ↓
    ┌───────┼───────────┐
    ↓       ↓           ↓
   E01    E02         E03          [Modèles individuels]
 (Text)  (Image)    (CLIP)
    ↓       ↓           ↓
    └───────┼───────────┘
            ↓
          E04                      [Stacking OOF 5-fold]
        (Meta-MLP)
            ↓
          E05                      [Benchmark + SIGIR]
       (Comparaison)
```

## 📊 Résultats attendus

| Expérience | Modèle | F1 Macro | Modality | Complexité |
|-----------|--------|----------|----------|-----------|
| **E01** | TF-IDF (50k) + LightGBM | ~0.82 | 📝 Text | ⭐ |
| **E02** | ResNet50 (progressive unfreezing) | ~0.66 | 🖼️ Image | ⭐⭐ |
| **E03** | CLIP ViT-B/32 + MLP fusion | ~0.84 | 🎯 Multimodal | ⭐⭐ |
| **E04** | OOF Stacking 5-fold + Meta-MLP | **~0.88** ✓ | 🧠 Ensemble | ⭐⭐⭐ |
| **Baseline SIGIR** | - | 0.8705 | - | - |
| **Top SIGIR** | - | 0.9194 | - | - |

## 🎯 Architecture détaillée

### **E00 — Preprocessing (Données brutes)**

Génère toutes les features de base avec caching complet.

**Entrée :** `data/split_no_crop/` + `data/split/`

**Sortie :** `data/processed_clean/`
```
✓ X_tfidf_{train,val,test}.npz         — TF-IDF sparse (50k features, bigrammes)
✓ X_ctfidf_{train,val,test}.npy        — Class-TF-IDF (27-d per sample)
✓ X_e5_{train,val,test}.npy            — Embeddings multilingual-e5-large (1024-d)
✓ X_dino_{train,val,test}.npy          — DINOv2 no-crop (768-d)
✓ X_dino_crop_{train,val,test}.npy     — DINOv2 avec crop (768-d)
✓ X_extra_{train,val,test}.npy         — Meta (has_description, zoom_ratio)
✓ y_{train,val,test}.npy               — Labels (0-26)
✓ label_encoder.pkl                    — Encodeur des classes
```

**Durée :** ~15-20 min (GPU, première exécution) | <1 sec (avec cache)

---

### **E01 — Texte: TF-IDF + LightGBM (Xavier)**

Reproduit l'expérience EXP_T03b de Xavier utilisant TF-IDF sparse et LightGBM.

**Architecture :**
```
X_tfidf (50k) → LightGBM (200 rounds) → Proba (27-d)
```

**Hyperparamètres :**
- **Features :** TF-IDF sparse (50k max, bigrammes, sublinear_tf)
- **Modèle :** LightGBM multiclass
- **Régularisation :** num_leaves=31, feature_fraction=0.8, bagging_fraction=0.8
- **Arrêt :** Early stopping (20 rounds patience)
- **Class weights :** balanced

**Sortie :**
```
P_text_{train,val,test}.npy            — Probabilités (27-d)
lgb_text_model.txt                     — Checkpoint du modèle
```

**Performance attendue :**
- F1 macro : **~0.82**
- Temps : ~5-10 min

**Code structure :**
1. Setup + paths
2. Load X_tfidf + y
3. Compute class_weights
4. Train LightGBM (80/20 train/val)
5. Evaluate (F1 macro, weighted, classification_report)
6. Save probabilities + model

---

### **E02 — Images: ResNet50 Fine-tuning Progressif (Xavier)**

Fine-tune ResNet50 avec unfreezing progressif en 3 stages.

**Architecture :**
```
ImageNet Pretrained ResNet50
    ↓ (gel backbone)
Stage 1 (5 epochs) : FC head fine-tuning
    ↓ (unfreezer layer4)
Stage 2 (3 epochs) : + layer4 gradient
    ↓ (unfreezer layer3)
Stage 3 (2 epochs) : + layer3 gradient
    ↓
Output : 27-class logits
```

**Hyperparamètres :**
- **Backbone :** ResNet50 (ImageNet1K)
- **Learning rates :** 1e-4 → 1e-5 → 1e-6
- **Batch size :** 32
- **Data augmentation :** RandomCrop, RandomFlip, ColorJitter
- **Early stopping :** patience=5 sur validation loss

**Sortie :**
```
P_img_{train,val,test}.npy             — Probabilités (27-d)
resnet50_image_model.pt                — State dict du modèle
```

**Performance attendue :**
- F1 macro : **~0.65-0.68**
- Temps : ~15-25 min (GPU)

---

### **E03 — CLIP ViT-B/32 + MLP Fusion (Tingfen)**

Combine embeddings texte et image de CLIP avec MLP fusion.

**Architecture :**
```
Texte → CLIP encoder (512-d) ⊕  Concat (1024-d) → MLP(512→256→27)
Image → CLIP encoder (512-d) ⊕
        (freeze CLIP backbone)
```

**Détails :**
- **Backbone CLIP :** ViT-B/32 (frozen)
- **Embeddings :** texte (512-d) + image (512-d)
- **MLP head :** Linear(1024) → BN → ReLU → Dropout → ... → Linear(27)
- **Ablation :** text-only vs image-only vs fusion
- **Training :** 20 epochs, early stopping (patience 10)

**Sortie :**
```
P_clip_{train,val,test}.npy            — Probabilités fusion (27-d)
clip_mlp_model.pt                      — State dict de MLP
```

**Performance attendue :**
- F1 macro : **~0.84** (meilleur texte+image)
- Ablation F1 : text-only ~0.79, image-only ~0.60, fusion ~0.84
- Temps : ~10-15 min

---

### **E04 — OOF Stacking + MLP Meta (Veyss)**

Implémente un stacking propre avec Out-Of-Fold CV pour éviter la fuite, construisant une pyramide de prédictions.

**Architecture pyramidale :**

#### **Niveau 1 (E1) :** Text
```
X_tfidf ⊕ X_ctfidf ⊕ X_e5 (sparse concat)
                ↓
        Logistic Regression (5-fold OOF)
                ↓
        P_text_oof (27-d per sample)
        F1_OOF ≈ 0.82
```

#### **Niveau 2 (E2) :** + DINOv2 no-crop
```
X_dino (768-d)
    ↓
Linear(768→27) trained on train+val
    ↓
P_dino_oof (27-d)
F1_OOF ≈ 0.75
```

#### **Niveau 3 (E3) :** + DINOv2 crop
```
X_dino_crop (768-d)
    ↓
Linear(768→27) trained on train+val
    ↓
P_dino_crop_oof (27-d)
F1_OOF ≈ 0.76
```

#### **Niveau 4 (E4) :** Meta-MLP fusion
```
Concat [P_text, P_dino, P_dino_crop] +  [Meta-features from each]
              81-d               +              12-d
         = 99-d feature vector

+ X_extra (2-d) = 101-d total

                ↓
        Meta-MLP (256→128→64→27)
                ↓
        Probabilités finales
        F1_OOF ≈ 0.88 ✓ BEST
```

**Meta-features** générées par `build_meta_features()` :
- **Entropy :** -Σ(p·log(p)) — confiance du modèle
- **Max confidence :** max(p) — confiance max
- **Margin :** top1_p - top2_p — difficulté de décision
- **Variance top-3 :** var([top1, top2, top3]) — diversité du top-3

**5-Fold OOF pour éviter la fuite :**
```
Pour chaque fold k=1..5 :
    train_idx = reste des données
    val_idx = fold k
    
    Fit modèle E1/E2/E3 sur train_idx
    Prédire sur val_idx → P_oof[val_idx]

P_oof final : prédictions sur TOUT le train, sans fuite
```

**Sortie :**
```
P_oof_{train,val}.npy                  — OOF predictions
P_oof_e1/e2/e3.npy                     — Intermédiaires par niveau
meta_model.pt                          — State dict du Meta-MLP
meta_scaler.pkl                        — StandardScaler des 101 features
```

**Performance :**
- F1 OOF : **~0.88** (MEILLEUR)
- Progression : E1(0.82) → E2(0.75) → E3(0.76) → E4(0.88) → +0.06 gain
- Temps : ~30-45 min

---

### **E05 — Benchmark Complet (Comparaison & SIGIR)**

Charge tous les modèles, génère prédictions, compare vs SIGIR 2020.

**Structure :**
1. **Charger les prédictions :**
   - E01 text (F1 ≈ 0.82)
   - E02 image (F1 ≈ 0.66)
   - E03 CLIP (F1 ≈ 0.84)
   - E04 OOF (F1 ≈ 0.88 validation)

2. **Table de résultats :**
   ```
   Model             F1_macro  F1_weighted  Accuracy  Modality
   E01 (Text)        0.8234    0.8445       0.8612    📝
   E02 (Image)       0.6547    0.6892       0.7124    🖼️
   E03 (CLIP)        0.8421    0.8634       0.8823    🎯
   E04 (OOF)         0.8847    0.8971       0.9156    🧠
   ```

3. **Comparaison SIGIR :**
   - Baseline 2020 : F1 = 0.8705
   - Top submission : F1 = 0.9194
   - Gap : E03 vs Baseline = +0.0716 ✓
   - Distance to top : -0.0773

4. **Analyses :**
   - Matrices de confusion (E01 best single)
   - Per-class F1 analysis
   - Erreurs les plus communes
   - Soft voting ensemble (moyenne E01+E02+E03)

5. **Outputs :**
   - CSV : `reports/results_summary.csv`
   - Visualizations (confusion matrices, per-class F1)

**Durée :** ~5 min

---

## 🚀 Guide d'exécution complet

### **Prérequis :**
```bash
# Python 3.8+
pip install -r requirements.txt

# Principaux packages:
# - numpy, pandas, scipy
# - scikit-learn
# - lightgbm, torch, torchvision, transformers
# - sentence-transformers, clip (OpenAI)
# - tqdm, matplotlib, seaborn
```

### **Step 1 : E00 — Preprocessing (UNE SEULE FOIS)**
```bash
# Exécuter notebooks/main_notebooks/E00_preprocessing.ipynb
# Cela génère TOUS les fichiers dans data/processed_clean/

Durée : 15-20 min (première exécution GPU)
        <1 sec (runs suivants, avec cache)
```

### **Step 2 : Lancer les expériences individuelles (parallèle possible)**

#### **E01 — Text**
```bash
jupyter notebook notebooks/main_notebooks/E01_xavier_texte_tfidf_lgbm.ipynb
# Durée : 5-10 min
# Sortie : P_text_*.npy, lgb_text_model.txt
# F1 attendu : ~0.82
```

#### **E02 — Image**
```bash
jupyter notebook notebooks/main_notebooks/E02_xavier_images_resnet50.ipynb
# Durée : 15-25 min
# Sortie : P_img_*.npy, resnet50_image_model.pt
# F1 attendu : ~0.66
```

#### **E03 — CLIP**
```bash
jupyter notebook notebooks/main_notebooks/E03_tingfen_clip_multimodal.ipynb
# Durée : 10-15 min
# Sortie : P_clip_*.npy, clip_mlp_model.pt
# F1 attendu : ~0.84
```

### **Step 3 : E04 — Stacking (après E01/E02/E03 OU seul si chargement)**
```bash
jupyter notebook notebooks/main_notebooks/E04_veyss_oof_stacking.ipynb
# Durée : 30-45 min
# Sortie : P_oof_*.npy, meta_model.pt, meta_scaler.pkl
# F1 attendu : ~0.88 ✓
```

### **Step 4 : E05 — Benchmark Final**
```bash
jupyter notebook notebooks/main_notebooks/E05_ensemble_benchmark.ipynb
# Durée : 5 min
# Sortie : results_summary.csv + visualizations
```

---

## ⏱️ Temps d'exécution

| Étape | GPU (NVIDIA RTX) | CPU | Cache | Type |
|------|------------------|-----|-------|------|
| E00  | 15-20 min ⚠️ | 45+ min | <1 sec | Setup |
| E01  | 5-10 min ⚠️ | 15-20 min | 2-3 sec | Train |
| E02  | 15-25 min ⚠️ | CPU not recommended | - | Train |
| E03  | 10-15 min ⚠️ | CPU not recommended | - | Train |
| E04  | 30-45 min ⚠️ | CPU not recommended | - | Train |
| E05  | 5 min | 5 min | - | Eval |
| **TOTAL** | **~75-110 min** | N/A | - | **First run** |

⚠️ = GPU strongly recommended

---

## 📁 Structure des fichiers

```
data/
  ├── split_no_crop/
  │   ├── train/ (images)
  │   ├── val/ (images)
  │   ├── test/ (images)
  │   ├── train.csv, val.csv, test.csv
  │
  ├── split/  (images croppées)
  │
  └── processed_clean/  ← Généré par E00
      ├── X_tfidf_{train,val,test}.npz
      ├── X_ctfidf_{train,val,test}.npy
      ├── X_e5_{train,val,test}.npy
      ├── X_dino_{train,val,test}.npy
      ├── X_dino_crop_{train,val,test}.npy
      ├── X_extra_{train,val,test}.npy
      ├── y_{train,val,test}.npy
      ├── P_text_{train,val,test}.npy   ← E01
      ├── P_img_{train,val,test}.npy    ← E02
      ├── P_clip_{train,val,test}.npy   ← E03
      ├── P_oof_{train,val}.npy         ← E04
      └── label_encoder.pkl

models/
  └── clean/  ← Généré par expériences
      ├── lgb_text_model.txt
      ├── resnet50_image_model.pt
      ├── clip_mlp_model.pt
      ├── meta_model.pt
      ├── meta_scaler.pkl
      ├── dino_linear_e2.pt
      └── dino_crop_linear_e3.pt

notebooks/
  └── main_notebooks/
      ├── E00_preprocessing.ipynb
      ├── E01_xavier_texte_tfidf_lgbm.ipynb
      ├── E02_xavier_images_resnet50.ipynb
      ├── E03_tingfen_clip_multimodal.ipynb
      ├── E04_veyss_oof_stacking.ipynb
      └── E05_ensemble_benchmark.ipynb

reports/
  └── results_summary.csv  ← Généré par E05
```

---

## 🔧 Configuration et Customization

### **Chemins configurables** (top de chaque notebook)
```python
BASE = r'C:\Users\veyss\Nouveau dossier\nov25_alt1_mle_rakuten'
CLEAN_PROC = os.path.join(BASE, 'data', 'processed_clean')
CLEAN_MODELS = os.path.join(BASE, 'models', 'clean')
```

### **Hyperparamètres clés**

**E01 — LightGBM :**
```python
params = {
    'num_leaves': 31,
    'learning_rate': 0.1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
}
num_boost_round = 200
early_stopping_rounds = 20
```

**E02 — ResNet50 :**
```python
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
# Progressive unfreezing : 5 + 3 + 2 epochs
```

**E03 — CLIP MLP :**
```python
EPOCHS = 20
LEARNING_RATE = 1e-4
hidden_dims = [512, 256]
dropout = 0.1
```

**E04 — Meta-MLP :**
```python
N_SPLITS = 5
hidden_dims = [256, 128, 64]
EPOCHS = 30
early_stopping_patience = 10
```

---

## 🐛 Troubleshooting

### **"E00 not detected" error**
→ Exécuter d'abord `E00_preprocessing.ipynb`

### **"P_text_test.npy not found" (dans E05)**
→ Exécuter `E01`, `E02`, `E03` avant `E05`

### **Out of Memory (OOM) GPU**
- Réduire `BATCH_SIZE` (32 → 16 → 8)
- E02/E03 nécessitent GPU, pas exécutables sur CPU
- E04 peut utiliser CPU pour certaines étapes

### **Embeddings CLIP ne se téléchargent pas**
```bash
# Manual fix:
pip install --upgrade clip-by-openai
python -c "import clip; clip.load('ViT-B/32', device='cpu')"
```

### **Fichiers .npz / sparse matrix issues**
```python
# Vérifier le chargement sparse:
import scipy.sparse as sp
X = sp.load_npz('X_tfidf_train.npz')
print(X.shape, type(X))  # Doit être csr_matrix
```

---

## 📈 Améliorations possibles

1. **E04 Extension :** Ajouter des features supplémentaires
   - Attention weights de CLIP
   - Per-class predictions distributions
   - Calibration temperature scaling

2. **E03 Ablation :** Tester différentes architectures de fusion
   - Concatenation (current) vs addition vs multiplication
   - Cross-modal attention heads

3. **E02 Augmentation :** Évaluer MixUp, CutMix, RandAugment

4. **Ensembling :** Weighted soft voting au lieu de moyenne simple
   ```python
   P_final = w1*P_e01 + w2*P_e02 + w3*P_e03 + w4*P_e04
   ```

5. **Test-Time Augmentation (TTA) :** Moyenne multiple prédictions avec variations

---

## 📚 Références & Architecture

**Pipeline complet :**
- E00 : Preprocessing autonome
- E01-E03 : Modèles individuels (unimodaux + multimodal)
- E04 : Stacking pyramidal avec 5-fold OOF
- E05 : Benchmark et comparaison SIGIR

**Features:**
- Texte : TF-IDF 50k + c-TF-IDF + e5-large
- Image : DINOv2 (no-crop + crop) + ResNet50 + CLIP

**Modèles:**
- LightGBM : gradient boosting pour texte
- ResNet50 : CNN pour images (progressive unfreezing)
- CLIP ViT-B/32 : vision-language pretraining
- MLP : fusion multimodale

**SIGIR 2020 Rakuten France Challenge:**
- Baseline : F1 = 0.8705
- Top : F1 = 0.9194
- Notre E03 (meilleur simple) : F1 ≈ 0.84 (+0.0716 vs baseline)
- Notre E04 (ensemble) : F1 ≈ 0.88 (+0.0095 vs baseline gap)

---

## 💾 Reproducibilité

**Random seeds fixés :**
```python
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
```

**Tous les fichiers sont versionnés :**
- Modèles `.pt` : torch.save() + deterministic mode
- Scalers `.pkl` : pickle
- Prédictions `.npy` : numpy float32

**Réexécution garantit les MÊMES résultats :**
- Pas de randomness dans le load/predict
- OOF génère EXACTEMENT les mêmes folds à chaque run

---

**Pipeline par : Veyss, Xavier, Tingfen** | Rakuten MLEng 2026
