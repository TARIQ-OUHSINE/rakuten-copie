


## What is `chi2_contingency`?

The code tests whether missing `description` values are independent of `prdtypecode` (product category).

### 1. **Creating the Contingency Table**
```python
contingency = pd.crosstab(train['prdtypecode'], train['description'].isna())
```
This creates a two-way table:
- Rows: each `prdtypecode` category
- Columns: `False` (description present) and `True` (description missing)
- Cells: counts of products in each category × missingness combination

### 2. **The Chi-Square Test Logic**

`chi2_contingency` performs a chi-square test of independence:

- Null hypothesis (H₀): `prdtypecode` and missingness are independent
- Alternative hypothesis (H₁): they are dependent

Steps:
1. Compute expected frequencies under independence:
   - `expected[i,j] = (row_total[i] × col_total[j]) / grand_total`
2. Compute the chi-square statistic:
   - `χ² = Σ (observed[i,j] - expected[i,j])² / expected[i,j]`
   - Larger values indicate stronger deviation from independence
3. Calculate the p-value:
   - Compare `χ²` to the chi-square distribution with `dof = (rows-1) × (cols-1)`
   - The p-value is the probability of observing this `χ²` (or larger) if H₀ is true

### 3. **Why p < 0.05 Indicates Dependence**

- p-value: probability of seeing the observed data (or more extreme) if H₀ is true
- p < 0.05 (α = 0.05): less than 5% chance under independence, so we reject H₀
- Interpretation: the missingness pattern is not random; it depends on product category

### Example Interpretation

If `p < 0.05`:
- Some categories are more likely to have missing descriptions than others
- Missingness is non-random (MNAR or MAR, not MCAR)
- This matters for modeling: you may need to handle missingness differently across categories

If `p ≥ 0.05`:
- Missingness appears random across categories (MCAR)
- Less concern about category-specific bias

This helps assess whether missing descriptions introduce bias that could affect model performance.

## What is the Kruskal-Wallis Test?

The **Kruskal-Wallis test** is a non-parametric test that checks whether multiple groups have the same distribution. It’s the non-parametric version of one-way ANOVA.

### In Your Code Context

```python
groups = [train.loc[train['prdtypecode'] == c, 'combined_wc'].values for c in cats]
stat, p = kruskal(*groups)
```

This tests whether the combined word count (`combined_wc`) distributions differ across product categories.

### How It Works

1. Null hypothesis (H₀): All groups have the same distribution (medians are equal).
2. Alternative hypothesis (H₁): At least one group has a different distribution.

Steps:
1. Rank all values across groups (smallest = 1, next = 2, etc.).
2. Compute the test statistic H:
   - Compare the average rank in each group to the overall average rank.
   - H increases when group ranks differ more.
3. Calculate the p-value:
   - Compare H to the chi-square distribution (with `k-1` degrees of freedom, where `k` = number of groups).
   - The p-value is the probability of observing this H (or larger) if H₀ is true.

### Why Use Kruskal-Wallis Instead of ANOVA?

- Non-parametric: doesn’t assume normality.
- Robust to outliers.
- Works with ordinal or continuous data.
- Good for skewed distributions (word counts are often skewed).

### Interpretation

- p < 0.05: Reject H₀ → word count distributions differ across categories.
- p ≥ 0.05: Fail to reject H₀ → no strong evidence of differences.

In your case, with `p = 0.00e+00` (p ≪ 0.05), word count distributions differ significantly across categories, so text length is a useful feature for classification.

### Comparison to Chi-Square Test

- Chi-square (your earlier test): Tests independence between two categorical variables (category vs. missingness).
- Kruskal-Wallis: Tests whether a continuous variable (word count) differs across categorical groups (categories).



## Simple TF-IDF Calculation Example

### Input: 3 Product Documents

```python
# Let's say we have 3 products from different categories:
documents = [
    "livre roman français",           # Document 1: Book category
    "chaussure sport running",        # Document 2: Shoe category  
    "livre histoire ancienne"         # Document 3: Book category
]
```

### Step 1: Term Frequency (TF)

Count how many times each word appears in each document:

| Word | Doc 1 | Doc 2 | Doc 3 |
|------|-------|-------|-------|
| livre | 1 | 0 | 1 |
| roman | 1 | 0 | 0 |
| français | 1 | 0 | 0 |
| chaussure | 0 | 1 | 0 |
| sport | 0 | 1 | 0 |
| running | 0 | 1 | 0 |
| histoire | 0 | 0 | 1 |
| ancienne | 0 | 0 | 1 |

### Step 2: Inverse Document Frequency (IDF)

IDF measures how rare a word is across all documents:
- **IDF = log(total_documents / documents_containing_word)**

| Word | Documents with word | IDF = log(3 / count) |
|------|---------------------|----------------------|
| livre | 2 (Doc 1, Doc 3) | log(3/2) = 0.405 |
| roman | 1 (Doc 1) | log(3/1) = 1.099 |
| français | 1 (Doc 1) | log(3/1) = 1.099 |
| chaussure | 1 (Doc 2) | log(3/1) = 1.099 |
| sport | 1 (Doc 2) | log(3/1) = 1.099 |
| running | 1 (Doc 2) | log(3/1) = 1.099 |
| histoire | 1 (Doc 3) | log(3/1) = 1.099 |
| ancienne | 1 (Doc 3) | log(3/1) = 1.099 |

### Step 3: TF-IDF = TF × IDF

Multiply TF by IDF for each word in each document:

**Document 1 ("livre roman français"):**
- livre: 1 × 0.405 = **0.405**
- roman: 1 × 1.099 = **1.099**
- français: 1 × 1.099 = **1.099**

**Document 2 ("chaussure sport running"):**
- chaussure: 1 × 1.099 = **1.099**
- sport: 1 × 1.099 = **1.099**
- running: 1 × 1.099 = **1.099**

**Document 3 ("livre histoire ancienne"):**
- livre: 1 × 0.405 = **0.405**
- histoire: 1 × 1.099 = **1.099**
- ancienne: 1 × 1.099 = **1.099**

### Step 4: Normalized TF-IDF Matrix

After normalization (sklearn normalizes by default), the output matrix `X_tfidf` looks like:

```
        livre  roman  français  chaussure  sport  running  histoire  ancienne
Doc 1   0.33   0.58   0.58      0.00       0.00   0.00    0.00      0.00
Doc 2   0.00   0.00   0.00      0.58       0.58   0.58    0.00      0.00
Doc 3   0.33   0.00   0.00      0.00       0.00   0.00    0.58      0.58
```

### In Your Code Context

When you do:
```python
mean_score = X_tfidf[mask].mean(axis=0).A1
```

For category with documents 1 and 3 (both books):
- Mean TF-IDF for "livre": (0.33 + 0.33) / 2 = **0.33**
- Mean TF-IDF for "roman": (0.58 + 0.00) / 2 = **0.29**
- Mean TF-IDF for "histoire": (0.00 + 0.58) / 2 = **0.29**

So "livre" would be the top term for the book category because it appears in both book documents, while "roman" and "histoire" are specific to individual documents.

### Key Insight

- Words that appear in many documents (like "livre") get lower IDF → lower TF-IDF
- Words that are unique to a category get higher IDF → higher TF-IDF
- This is why category-specific terms (like "chaussure" for shoes) score higher and help distinguish categories.




## Overview: What OpenCV (HOG + color hist + LBP)

### 1. **Color Histogram (HSV)** — 96 dimensions
**What it captures:** Color distribution

- Converts image to HSV (Hue, Saturation, Value)
- Counts how many pixels fall into each color bin
- Produces a histogram per channel (H, S, V)

**Example:**
- A red product → high counts in red hue bins
- A gray product → low saturation, spread across value bins
- A colorful product → high saturation counts

**Why HSV?** More intuitive than RGB for describing color appearance.

**Use case:** Distinguishing products by dominant colors (e.g., red shoes vs. blue shoes).

---

### 2. **HOG (Histogram of Oriented Gradients)** — 3,780 dimensions
**What it captures:** Shape and edge patterns

- Divides the image into small cells
- In each cell, finds edge directions (gradients)
- Counts how many edges point in each direction
- Aggregates across blocks

**Example:**
- A rectangular box → strong horizontal and vertical edges
- A round object → edges in many directions
- Text → strong horizontal edges

**Use case:** Detecting object shapes and structures (e.g., rectangular books vs. round objects).

---

### 3. **LBP (Local Binary Pattern)** — 26 dimensions
**What it captures:** Texture patterns

- For each pixel, compares it to its neighbors
- Creates a binary pattern (brighter/darker than neighbors)
- Counts how often each pattern appears

**Example:**
- Smooth surface → uniform patterns
- Rough/fabric texture → varied patterns
- Stripes → repeating patterns

**Use case:** Distinguishing materials and surfaces (e.g., smooth plastic vs. textured fabric).

---

## Combined Feature Vector (3,902 dimensions total)

The final vector concatenates all three:
```
[Color info (96) | Shape info (3780) | Texture info (26)]
```

This gives a compact representation that combines:
- **Color** (what color is it?)
- **Shape** (what shape is it?)
- **Texture** (what does the surface look like?)

---

## Real-World Example

For a **red leather shoe**:
- **Color histogram:** High values in red hue bins
- **HOG:** Strong edges around the shoe outline
- **LBP:** Patterns typical of leather texture

For a **blue book**:
- **Color histogram:** High values in blue hue bins
- **HOG:** Strong rectangular edges
- **LBP:** Smooth paper texture patterns

---

## Why This Approach?

- **Traditional computer vision:** Hand-crafted features (before deep learning)
- **Interpretable:** Each part has a clear meaning
- **Fast:** Computes quickly
- **Fixed size:** Always 3,902 numbers per image (good for machine learning)

**Limitation:** Less powerful than deep learning (CNNs), but useful as a baseline or when you need interpretable features.

This feature vector can be fed into a classifier (e.g., Random Forest, SVM) to predict product categories.



## TF-IDF Dimensionality: Why We Use 20K Instead of PCA

### Problem

The TF-IDF vectorizer was set to `max_features=50_000`, producing a 50K-dimensional sparse feature matrix. Combined with 8,222 image features, the total input to LightGBM was ~58K features.

### Why PCA Is the Wrong Solution

- **Standard PCA requires dense matrices.** Materializing the covariance matrix for 50K features would require ~18 GB of RAM — impractical.
- **TruncatedSVD** (the sparse-compatible alternative, i.e. LSA) converts output to dense, breaking the sparse pipeline and losing memory/speed benefits.
- **Interpretability is destroyed.** Feature importance and SHAP analysis map importances back to actual words. After SVD you'd have `svd_component_0` instead of readable terms.

### Decision: Reduce `max_features` to 20,000

Change in `models/baseline_model.py`:
```python
# Before
max_features=50_000
# After
max_features=20_000
```

### Reasons

1. **Preserves sparsity and interpretability** — vocabulary terms remain readable in SHAP/importance plots.
2. **Minimal F1 impact** — most signal is in high-frequency terms. Result: weighted F1 = **0.8184** (fast convergence at iteration 109).
3. **LightGBM handles high-dimensional sparse data natively** via `feature_fraction=0.3` (evaluates 30% of features per tree) — explicit dimensionality reduction is redundant.
4. **60% vocabulary reduction** with zero architectural complexity added.

### Rule of Thumb

For tree-based models on TF-IDF: prefer reducing `max_features` over post-hoc dimensionality reduction. Only reach for TruncatedSVD if you need semantic compression (e.g. for a neural model) and are willing to sacrifice interpretability.


### Score
50K features → weighted F1: 0.8244                                                                             
20K features → weighted F1: 0.8184     

---

## `ThreadPoolExecutor` vs `multiprocessing` for Parallel LightGBM Training

### The Core Question: What Releases the GIL?

Python's Global Interpreter Lock (GIL) prevents true parallel execution of **Python bytecode** across threads. However, the GIL is **released** whenever a thread calls into C/C++ extensions that explicitly drop it. LightGBM's training loop is implemented in C++ with OpenMP and releases the GIL during `fit()`. This is the key fact that determines the right choice.

### Why `ThreadPoolExecutor` Wins Here

| Factor | `ThreadPoolExecutor` | `multiprocessing.Pool` |
|--------|----------------------|------------------------|
| GIL during LightGBM `fit()` | Released → true parallelism | Not relevant (separate processes) |
| Data transfer to workers | Zero-copy (shared memory) | Pickle + IPC pipe (expensive) |
| Sparse matrix serialization | Not needed | Required per worker call |
| Memory usage | Shared (1× dataset) | Duplicated per process (N× dataset) |
| Startup overhead | Low | High (process fork + reimport) |
| OpenMP thread control | `n_jobs` per model | `n_jobs` per model |

### The Serialization Problem with `multiprocessing`

In this project, each ablation group receives a large sparse matrix:

- TF-IDF matrix: ~68K rows × 18K+ columns (sparse CSR)
- + image feature slice: up to 8,100 additional columns

`multiprocessing` must **pickle** these matrices and send them through an OS pipe to each worker process. For large sparse matrices, pickling alone can take longer than the training itself, eliminating any parallelism benefit.

`ThreadPoolExecutor` workers share the same Python process memory — no copying at all.

### CPU Core Distribution

Both approaches need careful `n_jobs` tuning to avoid contention. With N cores and K parallel workers:

```python
n_workers  = max(1, min(K, N // 2))   # e.g. 4 workers on 8-core machine
n_jobs_per = max(1, N // n_workers)    # e.g. 2 OpenMP threads per model
```

This ensures the total OpenMP threads across all models equals N, preventing over-subscription.

### When `multiprocessing` Is the Better Choice

| Use case | Preferred tool |
|----------|---------------|
| Pure Python loops (GIL-bound) | `multiprocessing` |
| Small data, many short tasks | `multiprocessing` |
| C-extension work that releases the GIL (LightGBM, NumPy, sklearn) | `ThreadPoolExecutor` |
| Large data that cannot be cheaply serialized | `ThreadPoolExecutor` |
| Need full process isolation (crash safety) | `multiprocessing` |

### Summary

For parallelizing multiple LightGBM `fit()` calls on large sparse matrices, `ThreadPoolExecutor` is strictly better: LightGBM's C++ core releases the GIL (real parallelism), and threads share memory (no serialization cost). `multiprocessing` would only win if the work were GIL-bound pure Python or the data were small enough that pickling is negligible.

---

## What is an Ablation Study?

An **ablation study** is a method to understand which components of your model contribute most to performance by systematically removing (ablating) features one at a time and measuring the impact.

### Code Context

The ablation study in `advanced_model.py` tests 5 different feature combinations:

1. **TF-IDF only** — Text features only (baseline)
2. **TF-IDF + color hist** — Text + color histogram (96 features)
3. **TF-IDF + HOG** — Text + HOG descriptor (8,100 features)
4. **TF-IDF + LBP** — Text + LBP texture (26 features)
5. **TF-IDF + all OpenCV** — Text + all image features (8,222 features)

### Purpose

- **Measure contribution** of each feature group individually
- **Identify which features help most** — which image features (color, shape, texture) are most useful?
- **Find redundant features** — are all features necessary, or can some be removed?
- **Validate design decisions** — do image features actually improve over text-only?

### Example Results

```
TF-IDF only            F1=0.8195  ← Baseline (text only)
TF-IDF + color hist    F1=0.8317  ← +0.012 improvement
TF-IDF + HOG           F1=0.8369  ← +0.017 improvement (best single feature)
TF-IDF + LBP           F1=0.8200  ← +0.0005 (minimal contribution)
TF-IDF + all OpenCV    F1=0.8159  ← Full model (may be lower due to overfitting)
```

### Why It's Called "Ablation"

In medical research, "ablation" means removing tissue to study its function. In machine learning, it means removing features/components to see what happens to performance.

### What the Code Does

1. **Trains 5 separate models** — one for each feature combination
2. **Evaluates each on validation set** — measures weighted F1 score
3. **Compares results** — shows which features contribute most
4. **Creates visualization** — bar chart showing F1 scores for each combination

### Key Insights from Ablation

- **HOG (shape features) provides the biggest boost** — shape is more discriminative than color or texture
- **Color histogram helps moderately** — useful but not as strong as shape
- **LBP (texture) has minimal impact** — may be redundant or less informative for this task
- **Combined features may not always be best** — sometimes individual features work better than all together (due to overfitting or feature interactions)

### When to Use Ablation Studies

- **Feature engineering** — testing which hand-crafted features matter
- **Model architecture** — testing which layers/components are necessary
- **Hyperparameter tuning** — understanding which settings matter most
- **Interpretability** — explaining which parts of your model drive performance

This helps you understand which image features (color, shape, texture) are most useful for your product classification task and guides future feature engineering decisions.

---

## Workflow Summary: Project Phases and Artifacts

This project follows a **3-phase progressive modeling approach**, where each phase builds on the previous one's artifacts.

### Phase 1: Baseline (Text-Only)
**File:** `models/baseline_model.py`

**Input:**
- `data/train_cleaned.pkl` — preprocessed training data
- `data/test_cleaned.pkl` — preprocessed test data

**Process:**
1. Creates `LabelEncoder` to map product categories (10, 40, 50, ...) → numeric labels (0, 1, 2, ...)
2. Fits `TfidfVectorizer` on training text (max_features=20,000)
3. Trains LightGBM classifier on TF-IDF features only

**Output Artifacts:**
- `models/label_encoder.pkl` — category → label mapping (reused in Phases 2 & 3)
- `models/tfidf_vectorizer.pkl` — fitted TF-IDF vectorizer (reused in Phase 2)
- `models/lgb_text_model.pkl` — trained text-only model
- `docs/figures/mod_fig1a_lgb_text_confusion.png` — confusion matrix
- `docs/figures/mod_fig1b_lgb_text_per_class_f1.png` — per-class F1 scores

**Performance:** Baseline weighted F1 score (text-only)

---

### Phase 2: Optimization (Text + Image Features)
**File:** `models/advanced_model.py`

**Input:**
- `data/train_cleaned.pkl` — preprocessed training data
- `data/test_cleaned.pkl` — preprocessed test data
- `images/image_train/` — product images
- `models/label_encoder.pkl` — **from Phase 1** (required)
- `models/tfidf_vectorizer.pkl` — **from Phase 1** (required)

**Process:**
1. Loads Phase 1 artifacts (label encoder, TF-IDF vectorizer)
2. Extracts OpenCV image features:
   - Color histogram (HSV, 96-d)
   - HOG descriptor (8,100-d)
   - LBP texture (26-d)
   - Total: 8,222 image features
3. Combines TF-IDF (20K) + image features (8,222) = 28,222 total features
4. Trains LightGBM on combined features
5. Runs ablation study to measure feature contributions

**Output Artifacts:**
- `models/lgb_combined_model.pkl` — trained text + image model
- `data/img_feats_train.npy` — cached image features (train)
- `data/img_feats_val.npy` — cached image features (validation)
- `docs/figures/mod_fig3_opencv_features.png` — feature visualization
- `docs/figures/mod_fig4_ablation_opencv.png` — ablation results
- `docs/figures/mod_fig5_feature_importance.png` — feature importance
- `docs/figures/mod_fig6_shap.png` — SHAP values

**Performance:** Optimized weighted F1 score (text + image)

---

### Phase 3: Advanced (CLIP Multimodal)
**File:** `models/phase3_clip_mlp.py`

**Input:**
- `data/train_cleaned.pkl` — preprocessed training data
- `data/test_cleaned.pkl` — preprocessed test data
- `images/image_train/` — product images
- `models/label_encoder.pkl` — **from Phase 1** (required)

**Process:**
1. Loads CLIP ViT-B/32 pretrained model (OpenAI)
2. Extracts image embeddings (512-d) and text embeddings (512-d) using CLIP
3. Concatenates image + text embeddings (1,024-d total)
4. Trains MLP classifier head on CLIP embeddings
5. Runs ablation study (image-only, text-only, combined)

**Output Artifacts:**
- `models/clip_mlp_head.pth` — trained MLP classifier head
- `data/test_predictions.csv` — final test set predictions
- `docs/figures/mod_fig7_clip_training.png` — training curves
- `docs/figures/mod_fig8_clip_ablation.png` — ablation results

**Performance:** Advanced weighted F1 score (multimodal deep learning)

---

### Artifact Dependency Graph

```
Phase 1 (baseline_model.py)
  ├─ Creates: label_encoder.pkl ──────────┐
  ├─ Creates: tfidf_vectorizer.pkl ───────┤
  └─ Creates: lgb_text_model.pkl          │
                                          │
Phase 2 (advanced_model.py)               │
  ├─ Loads:  label_encoder.pkl ───────────┘
  ├─ Loads:  tfidf_vectorizer.pkl ────────┘
  └─ Creates: lgb_combined_model.pkl
                                          │
Phase 3 (phase3_clip_mlp.py)             │
  └─ Loads:  label_encoder.pkl ──────────┘
```

### Key Points

1. **Phase 1 must run first** — creates foundational artifacts (label encoder, TF-IDF vectorizer)
2. **Phase 2 depends on Phase 1** — reuses label encoder and TF-IDF vectorizer for consistency
3. **Phase 3 depends on Phase 1** — reuses label encoder for consistent category mapping
4. **Phases 2 and 3 are independent** — can run in parallel after Phase 1 completes
5. **Artifacts ensure reproducibility** — same preprocessing pipeline across all phases

### Running the Pipeline

```bash
# Step 1: Run Phase 1 (creates foundational artifacts)
python models/baseline_model.py

# Step 2: Run Phase 2 (uses Phase 1 artifacts)
python models/advanced_model.py

# Step 3: Run Phase 3 (uses Phase 1 artifacts)
python models/phase3_clip_mlp.py
```

This workflow ensures consistent preprocessing, enables fair model comparison, and allows each phase to build incrementally on previous work.

---

## Optimizing CLIP Embedding Extraction: Batched Inference

### Problem with the Original Approach

`clip_mlp.py` extracts embeddings one sample at a time inside a Python `for` loop:

```python
for _, row in df.iterrows():
    img_tensor = preprocess(img).unsqueeze(0)   # batch_size=1
    tok = tokenizer([text])
    img_feat = clip_model.encode_image(img_tensor)  # one GPU call per sample
```

This is extremely slow because:
- GPU parallelism is almost entirely wasted at batch size 1
- Python loop overhead accumulates over 84K samples
- Image decoding and preprocessing block the GPU between calls
- `num_workers=0` means CPU preprocessing is single-threaded

### Solution: `CLIPDataset` + `DataLoader` Batched Pipeline

The fix replaces the row loop with a `torch.utils.data.Dataset` that returns
preprocessed tensors, fed through a `DataLoader` with parallel workers:

```python
class CLIPDataset(Dataset):
    def __getitem__(self, i):
        img = preprocess(Image.open(path).convert("RGB"))  # [3, 224, 224]
        tok = tokenizer([text])[0]                         # [77]
        return img, tok

loader = DataLoader(CLIPDataset(df), batch_size=256, num_workers=4, pin_memory=True)

img_embs, txt_embs = [], []
for imgs, toks in loader:
    with torch.no_grad(), torch.cuda.amp.autocast():
        img_feats = clip_model.encode_image(imgs.to(DEVICE))   # [256, 512]
        txt_feats = clip_model.encode_text(toks.to(DEVICE))    # [256, 512]
    img_embs.append(img_feats.cpu().float().numpy())
    txt_embs.append(txt_feats.cpu().float().numpy())
```

### Why Each Component Matters

| Optimization | Mechanism | Speedup |
|---|---|---|
| **Batch size 256** | GPU processes 256 samples per forward pass — matrix ops fully saturate CUDA cores | 10–50× vs batch=1 |
| **`num_workers=4`** | Worker processes load and preprocess images on CPU **in parallel** while GPU computes | 1.5–2× |
| **`pin_memory=True`** | Tensors allocated in pinned (page-locked) RAM → faster CPU→GPU DMA transfer | ~10–20% |
| **`torch.cuda.amp.autocast()`** | Uses float16 for CLIP forward pass → 2× memory reduction, faster tensor cores on GPU | 1.5–2× on GPU |

### Key Insight: Why batch_size=1 Wastes the GPU

A GPU contains thousands of CUDA cores that execute in parallel. With batch_size=1,
only a tiny fraction of those cores are active during each forward pass.
With batch_size=256, the same cores process 256 samples simultaneously — the
wall-clock time per sample drops by ~100× compared to serial execution.

### Handling Broken Images

The batched approach uses a collate-level fallback: the `Dataset.__getitem__`
catches `Exception` and returns a zero tensor, so a single corrupt image never
crashes the full extraction:

```python
def __getitem__(self, i):
    try:
        img = preprocess(Image.open(path).convert("RGB"))
    except Exception:
        img = torch.zeros(3, 224, 224)
    return img, tok
```

### Additional Training Optimizations

Beyond embedding extraction, three more changes speed up MLP training:

1. **`num_workers=4` in training DataLoaders** — prefetches next batch while
   GPU trains on current batch, eliminating CPU bottleneck between iterations.

2. **Mixed precision training with `GradScaler`** — `autocast()` runs forward
   pass in float16; `GradScaler` rescales gradients to prevent underflow before
   the optimizer step. Standard pattern:
   ```python
   scaler = torch.cuda.amp.GradScaler()
   with torch.cuda.amp.autocast():
       loss = criterion(model(x), y)
   scaler.scale(loss).backward()
   scaler.step(optimizer)
   scaler.update()
   ```

3. **Parallel ablation with `ThreadPoolExecutor`** — the 3 modality configs
   (text-only, image-only, combined) are independent and can train concurrently.
   `ThreadPoolExecutor` is preferred over `multiprocessing` here because PyTorch
   releases the GIL during C++ GPU ops, giving true thread-level parallelism
   with zero data-copying overhead (see ThreadPoolExecutor vs multiprocessing
   section above).

    When multiprocessing would be the right choice here:

  - If training ran on CPU only (GIL never released → threads can't truly parallelize Python loops)
  - If the data were small (pickling cost negligible)
  - If you needed crash isolation (a segfault in one worker doesn't kill the others)

---

## How `baseline_model.py` and `tfidf_translated.py` Work

Both models classify Rakuten products into categories using **TF-IDF + LightGBM**, but they differ in how they handle multilingual text.

### `baseline_model.py` — Naive Multilingual Approach

**Pipeline:** `text_clean` → TF-IDF → LightGBM → predict

1. Loads pre-cleaned data (`train_cleaned.pkl`) with a `text_clean` column
2. Builds a **TF-IDF** vectorizer (20K features, unigrams+bigrams) directly on the raw multilingual text, using stopwords from **French + English + German** combined
3. Trains a **LightGBM** classifier with early stopping (1000 trees max, stops after 50 rounds without improvement, balanced class weights)
4. Evaluates with weighted F1, saves confusion matrix + per-class F1 plots

**Key limitation:** The TF-IDF vocabulary is fragmented across languages — the same product concept in French vs English creates different tokens, diluting the signal.

---

### `tfidf_translated.py` — Translation-First Approach

**Pipeline:** detect language → translate to French → normalize → TF-IDF → LightGBM → predict

Adds 3 preprocessing steps before TF-IDF:

1. **Language detection** — uses `langdetect` to tag each row with its language (e.g. `fr`, `en`, `de`)
2. **Translation** — translates non-French text to French via Google Translate (`deep-translator`), supporting 9 source languages (en, de, es, it, pt, nl, ja, ko, zh-cn). Rate-limited with 50ms delays, truncates at 4500 chars
3. **Text normalization** — lowercases, strips punctuation/digits, collapses whitespace on the translated text
4. Builds TF-IDF on the unified French text (`text_fr_clean`) using **French-only** stopwords
5. Same LightGBM classifier as baseline

**Key advantage:** All text is in one language, so TF-IDF features are semantically unified — "shoes" and "chaussures" both become "chaussures".

It also **caches** the translated data to `data/train_translated.pkl` / `test_translated.pkl` so the expensive translation step only runs once.

---

### Comparison

| Aspect | baseline_model.py | tfidf_translated.py |
|---|---|---|
| Text input | raw multilingual `text_clean` | translated French `text_fr_clean` |
| Stopwords | FR + EN + DE | FR only |
| Extra deps | none | `langdetect`, `deep-translator` |
| Speed | fast | slow first run (translation), fast after cache |
| Hypothesis | multilingual TF-IDF is "good enough" | unifying language improves feature quality |

---

## Weighted F1 vs Macro F1

### Definitions

- **Macro F1** — Compute F1 for each class independently, then take the **unweighted average**. Every class counts equally regardless of size.
  - `Macro F1 = (1/N) × Σ F1_class_i`

- **Weighted F1** — Compute F1 for each class independently, then take the **weighted average** by each class's support (number of samples). Larger classes have more influence.
  - `Weighted F1 = Σ (support_i / total_samples) × F1_class_i`

### Example from This Project

| Metric | Baseline | Translated |
|--------|----------|------------|
| Weighted F1 | 0.8184 | 0.8107 |
| Macro F1 | 0.80 | 0.79 |

The gap (Weighted > Macro) indicates that **smaller classes perform worse** than larger ones. For example:
- Class 2583 (support=2,042, F1=0.97) pulls weighted F1 **up**
- Class 10 (support=623, F1=0.44) pulls macro F1 **down** but has less impact on weighted F1

### When to Use Which

| Metric | Use when... |
|--------|-------------|
| **Weighted F1** | You care about overall accuracy proportional to real-world class frequency |
| **Macro F1** | Every class matters equally, regardless of size (e.g., rare categories are just as important) |

### Key Insight

If Weighted F1 ≈ Macro F1, the model performs evenly across classes. If Weighted F1 >> Macro F1, the model struggles on minority classes — consider class balancing, oversampling, or per-class tuning.