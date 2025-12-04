# Mamba-Quaternion-Lite

**Un Modèle SSD Sélectif Multi-États avec Dynamique Scalaire et Projections Quaternioniques**

## Description

Mamba-Quaternion-Lite est une variante hybride du modèle Mamba-2 qui combine :

- **Dynamique interne scalaire réelle** (préservant la commutativité)
- **Injections et projections quaternioniques** (B_t et C_t)
- **Scan parallèle standard de Mamba-2** (efficacité maximale)
- **Sélectivité S6** (gating content-dépendant)

Cette architecture offre une expressivité géométrique accrue grâce aux quaternions tout en conservant la simplicité et la rapidité de Mamba-2.

## Caractéristiques

✅ **Dynamique scalaire** : A_t reste scalaire → scan parallèle standard
✅ **Projections quaternioniques** : B_t et C_t introduisent rotations et couplages multicanaux
✅ **Optimisé pour la performance** : Mixed precision, gradient checkpointing
✅ **Efficace en VRAM** : Optimisations mémoire
✅ **Compatible Tensor Cores** : Opérations quaternioniques optimisées GPU

## Architecture

```
Input → Embedding → [Mamba-Quaternion-Lite Block] × N → LM Head → Output

Mamba-Quaternion-Lite Block:
├── Projection d'entrée (expansion + gating S6)
├── Convolution causale 1D
├── Calcul des paramètres dynamiques (Δ_t, B_t, C_t, D_t)
├── SSM avec scan parallèle
│   ├── A_t : scalaire (dynamique)
│   ├── B_t : quaternionique (injection)
│   ├── C_t : quaternionique (projection)
│   └── h_t : états réels
└── Projection de sortie
```

## Installation

```bash
# Cloner le dépôt
git clone <repo-url>
cd mambaliteclaude

# Installer les dépendances
pip install -r requirements.txt
```

## Utilisation

### Entraînement sur TinyStories

Le script `train.py` entraîne le modèle sur le dataset TinyStories avec un tokenizer BPE (vocabulaire = 2048).

```bash
python train.py
```

### Configuration

Les hyperparamètres peuvent être modifiés dans la classe `Config` du fichier `train.py` :

```python
class Config:
    # Dataset
    max_seq_len = 256
    vocab_size = 2048

    # Modèle
    d_model = 256
    n_layers = 6
    d_state = 16
    expand_factor = 2
    dropout = 0.1

    # Entraînement
    batch_size = 16
    gradient_accumulation_steps = 4
    max_steps = 50000
    learning_rate = 3e-4

    # Optimisations
    use_amp = True  # Mixed precision
```

### Utilisation du Modèle

```python
from mamba_quaternion_lite import MambaQuaternionLiteModel
import torch

# Créer le modèle
model = MambaQuaternionLiteModel(
    vocab_size=2048,
    d_model=256,
    n_layers=6,
    d_state=16,
)

# Entraînement
model.train()
input_ids = torch.randint(0, 2048, (4, 128))  # (batch, seq_len)
targets = torch.randint(0, 2048, (4, 128))
loss, logits = model(input_ids, targets)

# Génération
model.eval()
generated = model.generate(
    input_ids[:, :10],  # Contexte initial
    max_new_tokens=100,
    temperature=0.8,
    top_k=50
)
```

## Optimisations Implémentées

### Performance
- ✅ **Mixed Precision (AMP)** : Entraînement en float16
- ✅ **Gradient Accumulation** : Batch effectif plus grand
- ✅ **Efficient DataLoading** : Multi-workers, pin memory
- ✅ **Cosine LR Schedule** : Avec warmup
- ✅ **Weight Tying** : Embedding ↔ LM Head

### VRAM
- ✅ **Scan parallèle optimisé** : Complexité O(log T)
- ✅ **Opérations quaternioniques efficaces** : Matrices 4×4
- ✅ **In-place operations** : Quand possible
- ✅ **set_to_none=True** : Pour optimizer.zero_grad()

## Détails Techniques

### Scan Parallèle

Le scan parallèle est rendu possible par la nature **scalaire** de A_t :

```
h_t = A_t * h_{t-1} + b_t

où:
- A_t ∈ ℝ (scalaire, commutatif)
- b_t ∈ ℍ^(C×K) (quaternionique)
- h_t ∈ ℝ^(C×K) (états réels)
```

### Quaternions

Représentation : q = a + bi + cj + dk ↔ [a, b, c, d] ∈ ℝ^4

Opérations :
- Multiplication : Non-commutative mais associative
- Utilisée dans B_t (injection) et C_t (projection)
- Permet rotations et couplages multicanaux

### Discrétisation Bilinéaire

```
z_{t,c,k} = Δ_{t,c} · Λ_{c,k}

A_{t,c,k} = (1 + z/2) / (1 - z/2)
```

Garantit la stabilité numérique (Λ < 0).

## Structure du Projet

```
mambaliteclaude/
├── mamba_quaternion_lite/
│   ├── __init__.py
│   ├── quaternion.py          # Opérations quaternioniques
│   ├── mamba_quat_lite.py     # Bloc Mamba-Quaternion-Lite
│   └── model.py               # Modèle complet
├── train.py                   # Script d'entraînement
├── requirements.txt
└── README.md
```

## Paramètres du Modèle

Modèle par défaut (6 couches, d_model=256) :
- **Paramètres entraînables** : ~10-15M
- **VRAM (batch=16, seq=256)** : ~4-6 GB
- **Tokens/s (V100)** : ~15k-20k

## Résultats Attendus

Sur TinyStories (50k steps) :
- **Loss de validation** : ~2.0-2.5
- **Perplexité** : ~7-12
- **Qualité de génération** : Histoires cohérentes courtes

## Papier de Référence

Voir la description complète de l'architecture dans le document fourni :

> **Mamba-Quaternion-Lite : Un Modèle SSD Sélectif Multi-États avec Dynamique Scalaire et Projections Quaternioniques**

Principales innovations :
1. Dynamique scalaire (A_t) → scan parallèle standard
2. Projections quaternioniques (B_t, C_t) → expressivité géométrique
3. Hybridation optimale : simplicité de Mamba-2 + richesse des quaternions

## Licence

MIT

## Citation

```bibtex
@article{mamba-quaternion-lite,
  title={Mamba-Quaternion-Lite: Un Modèle SSD Sélectif Multi-États avec Dynamique Scalaire et Projections Quaternioniques},
  year={2024}
}
```

## Contact

Pour questions et suggestions : [votre email]
