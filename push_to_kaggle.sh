#!/usr/bin/env bash
# Usage :
#   bash push_to_kaggle.sh dataset    -> upload/update le dataset de features
#   bash push_to_kaggle.sh notebook 01_baseline   -> push un notebook
#   bash push_to_kaggle.sh all        -> push dataset + tous les notebooks

set -e

KAGGLE_USER="TON_USERNAME_KAGGLE"
DATASET_NAME="rakuten-multimodal-features"
NB_DIR="notebooks"
DATA_DIR="kaggle_dataset"

cmd=$1

upload_dataset() {
    echo "=== Upload dataset ==="
    # Copier les fichiers de features dans kaggle_dataset/
    mkdir -p "$DATA_DIR/data/processed" "$DATA_DIR/data/split" "$DATA_DIR/models"
    cp data/processed/X_tfidf_*.npz  "$DATA_DIR/data/processed/" 2>/dev/null || true
    cp data/processed/y_*.npy         "$DATA_DIR/data/processed/" 2>/dev/null || true
    cp data/processed/X_st_*.npy      "$DATA_DIR/data/processed/" 2>/dev/null || true
    cp data/processed/X_img_*.npy     "$DATA_DIR/data/processed/" 2>/dev/null || true
    cp data/split/*.csv               "$DATA_DIR/data/split/"
    cp models/*.pkl                   "$DATA_DIR/models/"
    # Upload
    if kaggle datasets list --user "$KAGGLE_USER" | grep -q "$DATASET_NAME"; then
        echo "Mise a jour du dataset..."
        kaggle datasets version -p "$DATA_DIR" -m "update"
    else
        echo "Creation du dataset..."
        kaggle datasets create -p "$DATA_DIR"
    fi
    echo "Dataset OK"
}

push_notebook() {
    nb=$1
    nb_base="${nb%.ipynb}"
    meta_file="$NB_DIR/${nb_base}-kernel-metadata.json"
    if [ ! -f "$NB_DIR/$nb" ]; then
        echo "Notebook introuvable : $nb"; exit 1
    fi
    # Copier le metadata sous le nom attendu par l'API
    cp "$meta_file" "$NB_DIR/kernel-metadata.json"
    echo "=== Push $nb ==="
    kaggle kernels push -p "$NB_DIR"
    rm "$NB_DIR/kernel-metadata.json"
    echo "Notebook $nb envoye. Suivi : kaggle kernels status $KAGGLE_USER/${nb_base}"
}

pull_outputs() {
    nb=$1
    nb_base="${nb%.ipynb}"
    echo "=== Pull outputs $nb ==="
    kaggle kernels output "$KAGGLE_USER/${nb_base}" -p data/processed/
    echo "Outputs recuperes dans data/processed/"
}

case "$cmd" in
    dataset)
        upload_dataset
        ;;
    notebook)
        push_notebook "$2"
        ;;
    pull)
        pull_outputs "$2"
        ;;
    all)
        upload_dataset
        for nb in 01_baseline.ipynb 02_st_experiments.ipynb 02_stacking.ipynb 03_fusion_cond.ipynb 04_deep.ipynb; do
            push_notebook "$nb"
        done
        ;;
    *)
        echo "Usage : bash push_to_kaggle.sh [dataset|notebook <file>|pull <file>|all]"
        ;;
esac
