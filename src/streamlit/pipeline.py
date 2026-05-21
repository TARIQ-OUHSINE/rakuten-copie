"""
pipeline.py — Inference pipeline Rakuten (ZERO re-entrainement).

Charge les artefacts pre-sauvegardes (notebooks 1-5) et expose
une interface unique : RakutenPipeline.predict(designation, description, image).
"""

from __future__ import annotations

import re
import sys
import pickle
import logging
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# MetaMLP + MetaMLPClassifier — définis AVANT pickle.load de meta_best.pkl
# Le pickle référence __main__.MetaMLP (nn.Module) et __main__.MetaMLPClassifier
# ---------------------------------------------------------------------------

import torch
import torch.nn as nn


class MetaMLP(nn.Module):
    """Module PyTorch interne du méta-classifieur (FC → BN → ReLU → Dropout)."""

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
    """Wrapper sklearn-like autour de MetaMLP.
    Attributs sérialisés dans le pickle : cfg (dict) + model (MetaMLP).
    """

    def __init__(self, in_dim: int = 61, hidden=(256, 128, 64), n_classes: int = 27,
                 lr: float = 1e-3, epochs: int = 100, patience: int = 20, batch: int = 256):
        self.cfg = {
            'in_dim': in_dim, 'hidden': hidden, 'n_classes': n_classes,
            'lr': lr, 'epochs': epochs, 'patience': patience, 'batch': batch,
        }
        self.model = MetaMLP(in_dim, hidden, n_classes)

    def fit(self, X, y, X_val=None, y_val=None):
        from torch.utils.data import DataLoader, TensorDataset

        cfg = self.cfg
        device = torch.device('cpu')
        net = self.model.to(device)
        opt = torch.optim.Adam(net.parameters(), lr=cfg['lr'])
        criterion = nn.CrossEntropyLoss()

        Xt = torch.tensor(np.array(X), dtype=torch.float32)
        yt = torch.tensor(np.array(y), dtype=torch.long)
        dl = DataLoader(TensorDataset(Xt, yt), batch_size=cfg['batch'], shuffle=True)

        best_loss, patience_ctr, best_state = float('inf'), 0, None
        Xv = yv = None
        if X_val is not None:
            Xv = torch.tensor(np.array(X_val), dtype=torch.float32).to(device)
            yv = torch.tensor(np.array(y_val), dtype=torch.long).to(device)

        for _ in range(cfg['epochs']):
            net.train()
            for xb, yb in dl:
                opt.zero_grad()
                criterion(net(xb.to(device)), yb.to(device)).backward()
                opt.step()

            if Xv is not None:
                net.eval()
                with torch.no_grad():
                    val_loss = criterion(net(Xv), yv).item()
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = {k: v.clone() for k, v in net.state_dict().items()}
                    patience_ctr = 0
                else:
                    patience_ctr += 1
                    if patience_ctr >= cfg['patience']:
                        break

        if best_state is not None:
            net.load_state_dict(best_state)
        self.model = net
        return self

    def predict_proba(self, X):
        import torch.nn.functional as F
        self.model.eval()
        with torch.no_grad():
            Xt = torch.tensor(np.array(X), dtype=torch.float32)
            return F.softmax(self.model(Xt), dim=1).numpy()

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def score(self, X, y):
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chemins (relatifs a la racine du projet)
# ---------------------------------------------------------------------------

def _find_root() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in [here, here.parent, here.parent.parent]:
        if (candidate / 'data' / 'split_no_crop').exists():
            return candidate
    raise FileNotFoundError("Impossible de trouver la racine du projet.")


ROOT = _find_root()
MODELS_DIR = ROOT / 'models' / 'clean'
PROCESSED_DIR = ROOT / 'data' / 'processed_clean'
SPLIT_DIR = ROOT / 'data' / 'split_no_crop2'

# ---------------------------------------------------------------------------
# Constantes de preprocessing (identiques a NB1)
# ---------------------------------------------------------------------------

_HTML_TAG = re.compile(r'<[^>]+>', re.IGNORECASE)
_HTML_ENTITY = re.compile(r'&[a-zA-Z]+;|&#\d+;')
_MULTIPLE_SPC = re.compile(r'\s+')
_NON_ALPHA = re.compile(r'[^a-z\s]')

STOPWORDS: set[str] = set()
try:
    import nltk
    from nltk.corpus import stopwords as nltk_sw
    for lang in ['french', 'english', 'german', 'spanish']:
        STOPWORDS.update(nltk_sw.words(lang))
    STOPWORDS.update(['cm', 'mm', 'ml', 'kg', 'g', 'l', 'n', 'x', 'ref',
                       'lot', 'pack', 'set', 'new', 'neuf', 'article', 'produit'])
except Exception:
    pass

try:
    from nltk.stem.snowball import SnowballStemmer
    _STEMMER = SnowballStemmer('french')
except Exception:
    _STEMMER = None


def clean_html(text: str) -> str:
    if not isinstance(text, str):
        return ''
    text = _HTML_TAG.sub(' ', text)
    text = _HTML_ENTITY.sub(' ', text)
    return _MULTIPLE_SPC.sub(' ', text).strip()


def preprocess_text(text: str) -> str:
    """Pipeline baseline NB1 : minuscules → non-alpha → stopwords → stemming."""
    if not isinstance(text, str) or not text.strip():
        return ''
    text = text.lower()
    text = _NON_ALPHA.sub(' ', text)
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 2]
    if _STEMMER:
        tokens = [_STEMMER.stem(t) for t in tokens]
    return ' '.join(tokens)


def build_combined_raw(designation: str, description: str) -> str:
    """Texte brut (pour e5-large) : HTML clean seulement."""
    return (clean_html(designation) + ' ' + clean_html(description)).strip()


def build_text_combined(designation: str, description: str) -> str:
    """Texte nettoyé (pour TF-IDF) : preprocessing complet."""
    raw = build_combined_raw(designation, description)
    return preprocess_text(raw)


# ---------------------------------------------------------------------------
# Pipeline principale
# ---------------------------------------------------------------------------

class RakutenPipeline:
    """
    Pipeline d'inference zero-entrainement.

    Charge les artefacts depuis models/clean/ et data/processed_clean/.
    Les modeles gelés (e5, DINOv2) sont charges a la demande (lazy).
    """

    def __init__(self):
        self._tfidf = None
        self._le = None
        self._centroids = None
        self._meta_clf = None
        self._e5_model = None
        self._dino_model = None
        self._resnet = None
        self._class_names: Optional[dict] = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Chargement des artefacts statiques (pkl, npy)
    # ------------------------------------------------------------------

    def load_static_artifacts(self) -> dict[str, bool]:
        """Charge les artefacts non-neuronaux. Retourne un dict {name: ok}."""
        status = {}

        # TF-IDF vectorizer
        p = MODELS_DIR / 'tfidf_vectorizer.pkl'
        if p.exists():
            with open(p, 'rb') as f:
                self._tfidf = pickle.load(f)
            status['tfidf'] = True
        else:
            status['tfidf'] = False
            log.warning('tfidf_vectorizer.pkl manquant — executer NB2')

        # LabelEncoder
        p = MODELS_DIR / 'label_encoder.pkl'
        if p.exists():
            with open(p, 'rb') as f:
                self._le = pickle.load(f)
            status['label_encoder'] = True
        else:
            status['label_encoder'] = False
            log.warning('label_encoder.pkl manquant — executer NB1')

        # c-TF-IDF centroids
        p = MODELS_DIR / 'ctfidf_centroids.npy'
        if p.exists():
            self._centroids = np.load(str(p))
            status['ctfidf'] = True
        else:
            status['ctfidf'] = False
            log.warning('ctfidf_centroids.npy manquant — executer NB3')

        # Meta-classifieur
        p = MODELS_DIR / 'meta_best.pkl'
        if p.exists():
            # meta_best.pkl référence __main__.MetaMLP + __main__.MetaMLPClassifier
            sys.modules['__main__'].MetaMLP = MetaMLP
            sys.modules['__main__'].MetaMLPClassifier = MetaMLPClassifier
            with open(p, 'rb') as f:
                self._meta_clf = pickle.load(f)
            status['meta_clf'] = True
        else:
            # Fallback : baseline LightGBM
            p2 = MODELS_DIR / 'baseline_lgbm.pkl'
            if p2.exists():
                with open(p2, 'rb') as f:
                    self._meta_clf = pickle.load(f)
                status['meta_clf'] = 'baseline_fallback'
                log.warning('meta_best.pkl absent — fallback sur baseline_lgbm')
            else:
                status['meta_clf'] = False
                log.warning('Aucun classifieur disponible — executer NB5')

        self._loaded = any(status.values())
        return status

    # ------------------------------------------------------------------
    # Chargement des modeles gelés (lazy)
    # ------------------------------------------------------------------

    def load_e5(self):
        """Charge multilingual-e5-large (lazy, mis en cache par Streamlit)."""
        if self._e5_model is None:
            from sentence_transformers import SentenceTransformer
            log.info('Chargement multilingual-e5-large...')
            self._e5_model = SentenceTransformer('intfloat/multilingual-e5-large')
        return self._e5_model

    def load_dino(self):
        """Charge DINOv2-base frozen (lazy)."""
        if self._dino_model is None:
            import torch
            from transformers import AutoModel
            log.info('Chargement DINOv2-base...')
            self._dino_model = AutoModel.from_pretrained('facebook/dinov2-base')
            self._dino_model.eval()
            for p in self._dino_model.parameters():
                p.requires_grad = False
        return self._dino_model

    def load_resnet(self):
        """Charge ResNet50 fine-tune si disponible, sinon pretrained ImageNet."""
        if self._resnet is None:
            import torch
            import torchvision.models as tv_models
            import torch.nn as nn

            class _ResNet50(nn.Module):
                def __init__(self, n_classes=27):
                    super().__init__()
                    backbone = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V2)
                    in_features = backbone.fc.in_features
                    backbone.fc = nn.Identity()
                    self.backbone = backbone
                    self.head = nn.Sequential(
                        nn.Dropout(0.3), nn.Linear(in_features, 512),
                        nn.GELU(), nn.Dropout(0.3), nn.Linear(512, n_classes),
                    )
                def forward(self, x):
                    return self.head(self.backbone(x))

            model = _ResNet50(n_classes=27)
            ft_path = MODELS_DIR / 'resnet50_ft.pth'
            if ft_path.exists():
                model.load_state_dict(torch.load(str(ft_path), map_location='cpu'))
                log.info('ResNet50 fine-tune charge.')
            else:
                log.info('ResNet50 pretrained ImageNet (pas de fine-tune disponible).')
            model.eval()
            self._resnet = model
        return self._resnet

    # ------------------------------------------------------------------
    # Extraction de features
    # ------------------------------------------------------------------

    def _tfidf_features(self, text: str) -> np.ndarray:
        if self._tfidf is None:
            return np.zeros((1, 1))
        return self._tfidf.transform([text])  # sparse (1, n_vocab)

    def _ctfidf_features(self, tfidf_vec) -> np.ndarray:
        if self._centroids is None:
            return np.zeros((1, 27))
        from sklearn.metrics.pairwise import cosine_similarity
        return cosine_similarity(tfidf_vec, self._centroids).astype(np.float32)  # (1, 27)

    def _e5_features(self, combined_raw: str) -> np.ndarray:
        model = self.load_e5()
        emb = model.encode(
            ['query: ' + combined_raw],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return emb.astype(np.float32)  # (1, 1024)

    def _dino_features(self, image) -> np.ndarray:
        """Extrait le token [CLS] DINOv2 depuis une PIL.Image."""
        import torch
        import torchvision.transforms as T

        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        dino = self.load_dino()
        with torch.no_grad():
            img_t = transform(image.convert('RGB')).unsqueeze(0)
            out = dino(pixel_values=img_t)
            cls = out.last_hidden_state[:, 0, :].numpy()  # (1, 768)
        return cls.astype(np.float32)

    # ------------------------------------------------------------------
    # Meta-features (meme logique que NB5)
    # ------------------------------------------------------------------

    @staticmethod
    def _scalar_meta(P: np.ndarray) -> np.ndarray:
        """entropy, margin, max_conf pour un vecteur de probas (1, n_classes)."""
        eps = 1e-12
        entropy = -(P * np.log(P + eps)).sum(axis=1, keepdims=True)
        sorted_p = np.sort(P, axis=1)[:, ::-1]
        margin = (sorted_p[:, 0] - sorted_p[:, 1]).reshape(-1, 1)
        max_conf = sorted_p[:, 0].reshape(-1, 1)
        return np.hstack([entropy, margin, max_conf]).astype(np.float32)

    def _build_meta(self, P_text: np.ndarray, P_image: np.ndarray) -> np.ndarray:
        agreement = (P_text.argmax(1) == P_image.argmax(1)).astype(np.float32).reshape(-1, 1)
        return np.hstack([
            P_text, P_image,
            self._scalar_meta(P_text),
            self._scalar_meta(P_image),
            agreement,
        ])  # (1, 27+27+3+3+1) = (1, 61)

    # ------------------------------------------------------------------
    # Pipeline complete
    # ------------------------------------------------------------------

    def predict(
        self,
        designation: str,
        description: str,
        image=None,  # PIL.Image ou None
        use_deep_text: bool = True,
        use_deep_image: bool = True,
    ) -> dict:
        """
        Execute la pipeline complete et retourne un dict avec :
          - probas        : np.ndarray (27,) probabilites de chaque classe
          - predicted_idx : int
          - predicted_code: int (prdtypecode original)
          - predicted_name: str
          - top3          : list[(class_name, proba)]
          - confidence    : float
          - features_used : list[str]
        """
        combined_raw = build_combined_raw(designation, description)
        text_clean = build_text_combined(designation, description)

        features_used = []

        # ---- Branche Texte ----
        tfidf_vec = self._tfidf_features(text_clean)
        ctfidf_vec = self._ctfidf_features(tfidf_vec)  # (1, 27)
        features_used.append('TF-IDF')
        features_used.append('c-TF-IDF')

        if use_deep_text:
            try:
                e5_vec = self._e5_features(combined_raw)  # (1, 1024)
                features_used.append('multilingual-e5-large')
                text_dense = np.hstack([
                    tfidf_vec.toarray(), ctfidf_vec, e5_vec
                ]).astype(np.float32)
            except Exception as exc:
                log.warning(f'e5 indisponible ({exc}), fallback TF-IDF seul')
                text_dense = np.hstack([
                    tfidf_vec.toarray(), ctfidf_vec
                ]).astype(np.float32)
        else:
            text_dense = np.hstack([
                tfidf_vec.toarray(), ctfidf_vec
            ]).astype(np.float32)

        # ---- Branche Image ----
        if image is not None and use_deep_image:
            try:
                image_vec = self._dino_features(image)  # (1, 768)
                features_used.append('DINOv2')
            except Exception as exc:
                log.warning(f'DINOv2 indisponible ({exc})')
                image_vec = np.zeros((1, 768), dtype=np.float32)
        else:
            image_vec = np.zeros((1, 27), dtype=np.float32)  # fallback neutral

        # ---- Prédiction ----
        if self._meta_clf is None:
            raise RuntimeError("Aucun classifieur disponible. Executez les notebooks 1-5.")

        # Stratégie : si meta_best est disponible avec texte+image, l'utiliser
        # Sinon, utiliser le texte seul si le clf l'attend
        try:
            meta_features = self._build_meta(
                self._get_text_probas(text_dense),
                self._get_image_probas(image_vec),
            )
            probas = self._meta_clf.predict_proba(meta_features)[0]
        except Exception:
            # Fallback : texte seul via le classifieur
            try:
                probas = self._meta_clf.predict_proba(text_dense)[0]
            except Exception as exc:
                raise RuntimeError(f"Echec de prediction : {exc}")

        predicted_idx = int(probas.argmax())
        predicted_code = int(self._le.inverse_transform([predicted_idx])[0]) if self._le else predicted_idx
        predicted_name = CLASS_NAMES.get(predicted_code, f'Classe {predicted_code}')

        sorted_idx = probas.argsort()[::-1]
        top3 = []
        for i in sorted_idx[:3]:
            code = int(self._le.inverse_transform([i])[0]) if self._le else i
            name = CLASS_NAMES.get(code, f'Classe {code}')
            top3.append((name, float(probas[i])))

        return {
            'probas': probas,
            'predicted_idx': predicted_idx,
            'predicted_code': predicted_code,
            'predicted_name': predicted_name,
            'top3': top3,
            'confidence': float(probas[predicted_idx]),
            'features_used': features_used,
        }

    def _get_text_probas(self, text_dense: np.ndarray) -> np.ndarray:
        """Obtient les probas texte depuis le classifieur texte (NB3 ou fallback)."""
        # Tenter de charger P_text directement depuis le cache NB3
        # (chemin standard depuis NB3 : clf_text_full.pkl)
        clf_text_path = MODELS_DIR / 'clf_text_full.pkl'
        if clf_text_path.exists():
            if not hasattr(self, '_clf_text') or self._clf_text is None:
                with open(clf_text_path, 'rb') as f:
                    self._clf_text = pickle.load(f)
            return self._clf_text.predict_proba(text_dense)
        # Fallback : probas uniformes (si seul meta_clf sans branche texte)
        n_classes = 27
        return np.full((1, n_classes), 1.0 / n_classes, dtype=np.float32)

    def _get_image_probas(self, image_vec: np.ndarray) -> np.ndarray:
        """Obtient les probas image depuis un classifieur image linéaire."""
        # Tentative depuis clf_dino_linear.pkl
        clf_dino_path = MODELS_DIR / 'dino_linear_full.pth'
        if clf_dino_path.exists():
            try:
                import torch, torch.nn as nn
                if not hasattr(self, '_clf_dino_lin') or self._clf_dino_lin is None:
                    head = nn.Linear(768, 27)
                    state = torch.load(str(clf_dino_path), map_location='cpu')
                    # Extraire seulement les poids de la tete lineaire
                    if isinstance(state, dict) and 'weight' in state:
                        head.load_state_dict(state)
                    self._clf_dino_lin = head
                    self._clf_dino_lin.eval()
                with torch.no_grad():
                    t = torch.tensor(image_vec, dtype=torch.float32)
                    p = torch.softmax(self._clf_dino_lin(t), dim=1).numpy()
                return p
            except Exception:
                pass
        # Fallback uniforme
        n_classes = 27
        return np.full((1, n_classes), 1.0 / n_classes, dtype=np.float32)

    # ------------------------------------------------------------------
    # Interprétabilité Texte
    # ------------------------------------------------------------------

    def highlight_important_words(
        self, text: str, predicted_class_idx: int, top_n: int = 10
    ) -> list[tuple[str, float]]:
        """
        Retourne une liste (mot, score) ou score > 0 = favorable, < 0 = defavorable
        pour la classe predite, base sur les coefficients TF-IDF du LogReg.
        """
        if self._tfidf is None:
            return []

        # Chercher un LogReg (clf_text_full.pkl)
        clf_path = MODELS_DIR / 'clf_text_full.pkl'
        if not clf_path.exists():
            return []

        with open(clf_path, 'rb') as f:
            clf = pickle.load(f)

        if not hasattr(clf, 'coef_'):
            return []

        feature_names = np.array(self._tfidf.get_feature_names_out())
        coef = clf.coef_[predicted_class_idx]  # (n_features,)

        # Tokeniser le texte d'entree
        tokens = set(preprocess_text(build_combined_raw(text, '')).split())
        word_scores = []
        for tok in tokens:
            indices = np.where(feature_names == tok)[0]
            if len(indices) > 0:
                score = float(coef[indices[0]])
                word_scores.append((tok, score))

        word_scores.sort(key=lambda x: abs(x[1]), reverse=True)
        return word_scores[:top_n]

    # ------------------------------------------------------------------
    # GradCAM
    # ------------------------------------------------------------------

    def gradcam(self, image, predicted_class_idx: int) -> Optional[np.ndarray]:
        """
        Calcule la heatmap GradCAM (H x W, valeurs 0-1) via ResNet50.
        Retourne None si le modele n'est pas disponible.
        """
        import torch
        import torch.nn.functional as F
        import torchvision.transforms as T

        transform = T.Compose([
            T.Resize((224, 224)), T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        resnet = self.load_resnet()
        activations = {}
        gradients = {}

        def fwd_hook(_, __, output):
            activations['feat'] = output.detach()

        def bwd_hook(_, __, grad_out):
            gradients['grad'] = grad_out[0].detach()

        target_layer = resnet.backbone.layer4[-1]
        h1 = target_layer.register_forward_hook(fwd_hook)
        h2 = target_layer.register_full_backward_hook(bwd_hook)

        try:
            img_t = transform(image.convert('RGB')).unsqueeze(0)
            resnet.eval()
            logits = resnet(img_t)
            resnet.zero_grad()
            logits[0, predicted_class_idx].backward()

            weights = gradients['grad'].mean(dim=(2, 3), keepdim=True)
            cam = (weights * activations['feat']).sum(dim=1).squeeze()
            cam = F.relu(cam).cpu().numpy()
            cam = cam - cam.min()
            if cam.max() > 0:
                cam = cam / cam.max()
            return cam
        except Exception as e:
            log.warning(f'GradCAM echoue : {e}')
            return None
        finally:
            h1.remove()
            h2.remove()


# ---------------------------------------------------------------------------
# Mapping codes -> noms de classes
# ---------------------------------------------------------------------------

CLASS_NAMES: dict[int, str] = {
    10:   'Livres occasion',        40:   'Jeux vidéo',
    50:   'Accessoires gaming',     60:   'Consoles de jeux',
    1140: 'Figurines',              1160: 'Cartes collection',
    1180: 'Jeux de société',        1281: 'Jeux enfants',
    1300: 'Miniatures véhicules',   1301: 'Bébé / Puériculture',
    1302: 'Jeux plein air',         1320: 'Puériculture & Peluches',
    1560: 'Mobilier',               1920: 'Linge maison',
    1940: 'Épicerie',               2060: 'Décoration',
    2220: 'Animalerie',             2280: 'Journaux / Revues',
    2403: 'Livres neufs',           2462: 'Jeux PC dématérialisés',
    2522: 'Papeterie / Bureautique',2582: 'Piscines / Spas',
    2583: 'Outillage jardinage',    2585: 'Bricolage',
    2705: 'Livres BD / Mangas',     2905: 'Jeux de rôle',
    1970: 'Cuisine / Électroménager',
}