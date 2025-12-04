# Architecture de Mamba-Quaternion-Lite

## Vue d'ensemble

Mamba-Quaternion-Lite est un modèle de séquence hybride qui combine la dynamique scalaire de Mamba-2 avec des projections quaternioniques pour une expressivité géométrique accrue.

## Composants Principaux

### 1. Module Quaternion (`quaternion.py`)

#### `quaternion_multiply(q1, q2)`
Multiplication de quaternions optimisée pour PyTorch.

**Formule:**
```
q1 * q2 = (a1 + b1i + c1j + d1k) * (a2 + b2i + c2j + d2k)
```

**Entrées:**
- `q1`: Tenseur (…, 4) - premier quaternion
- `q2`: Tenseur (…, 4) - second quaternion

**Sortie:**
- Tenseur (…, 4) - produit q1 * q2

**Propriétés:**
- Non-commutatif: q1 * q2 ≠ q2 * q1
- Associatif: (q1 * q2) * q3 = q1 * (q2 * q3)
- Compatible GPU (opérations vectorisées)

#### `QuaternionLinear`
Couche linéaire quaternionique générale.

**Architecture:**
```
y = W * x + b
```

où W est une matrice de quaternions.

**Paramètres:**
- `in_features`: Nombre de features d'entrée
- `out_features`: Nombre de features de sortie
- `bias`: Si True, ajoute un biais quaternionique

**Poids:**
- `weight`: (out_features, in_features, 4)
- `bias`: (out_features, 4) [optionnel]

#### `QuaternionProjection`
Projection optimisée pour les opérations B_t et C_t du SSM.

**Cas d'usage:**

1. **B_t (injection):** Projette un scalaire vers des quaternions
   - Entrée: (B, T, C)
   - Sortie: (B, T, C, K, 4)

2. **C_t (projection):** Multiplie quaternions par états scalaires
   - Entrée: scalar_state (B, T, C, K)
   - Sortie: (B, T, C, 4)

### 2. Bloc Mamba-Quaternion-Lite (`mamba_quat_lite.py`)

#### Architecture du Bloc

```
Input (B, T, d_model)
    ↓
In Projection (expansion × 2)
    ↓
[x_ssm, z] (B, T, d_inner) × 2
    ↓
Conv1D (x_ssm) + SiLU
    ↓
LayerNorm
    ↓
┌─────────────────────────────────────┐
│ Calcul des Paramètres Dynamiques   │
├─────────────────────────────────────┤
│ • Δ_t = softplus(dt_proj(x_proj(x)))│
│ • Λ = -softplus(Lambda)             │
│ • z_t = Δ_t ⊗ Λ                     │
│ • A_t = (1 + z/2) / (1 - z/2)       │
│ • gate = sigmoid(z)                 │
│ • S_t = gate ⊗ x                    │
│ • B_t = QuatProj(S_t)               │
│ • C_t = QuatProj (params)           │
└─────────────────────────────────────┘
    ↓
Scan Parallèle: h_t = A_t * h_{t-1} + B_t * S_t
    ↓
┌─────────────────────────────────────┐
│ Sortie                              │
├─────────────────────────────────────┤
│ • y_state = Σ_k C_t[k] * h_t[k]     │
│ • y_skip = D ⊗ x_conv               │
│ • y = y_state + y_skip              │
└─────────────────────────────────────┘
    ↓
Out Projection
    ↓
Output (B, T, d_model)
```

#### Scan Parallèle

**Opération:**
```
h_t = A_t * h_{t-1} + b_t
```

**Propriétés clés:**
- A_t est **scalaire** (commutatif) → permet scan parallèle standard
- b_t est **quaternionique** → expressivité géométrique
- h_t reste **réel** → pas de complexité quaternionique dans la dynamique

**Algorithme:**

1. **Séquentiel** (seq_len ≤ 32):
   ```python
   h[0] = b[0]
   for t in range(1, T):
       h[t] = A[t] * h[t-1] + b[t]
   ```

2. **Parallèle** (seq_len > 32):
   - Utilise un algorithme de prefix-sum
   - Complexité: O(log T) en profondeur
   - Hautement parallélisable sur GPU

**Backward Pass:**
- Rétropropagation séquentielle (plus stable)
- Calcul des gradients pour A et b
- Propagation des gradients vers les états précédents

### 3. Modèle Complet (`model.py`)

#### `MambaQuaternionLiteModel`

**Architecture complète:**

```
Token IDs (B, T)
    ↓
Token Embedding (B, T, d_model)
    ↓
Dropout
    ↓
┌─────────────────────────┐
│ ResidualBlock 1         │ ← LayerNorm + Mamba-Quat-Lite
├─────────────────────────┤
│ ResidualBlock 2         │
├─────────────────────────┤
│        ...              │
├─────────────────────────┤
│ ResidualBlock N         │
└─────────────────────────┘
    ↓
LayerNorm Final
    ↓
LM Head (tied with embedding)
    ↓
Logits (B, T, vocab_size)
```

**Hyperparamètres par défaut:**
- `d_model`: 256
- `n_layers`: 6
- `d_state`: 16
- `expand_factor`: 2
- `conv_kernel_size`: 4
- `dropout`: 0.1

**Features:**
- Weight tying entre embedding et LM head
- Connexions résiduelles avec pré-normalisation
- Support de génération auto-régressive
- Calcul automatique de la loss

#### Génération de Texte

```python
generated = model.generate(
    input_ids,           # Context (B, T)
    max_new_tokens=100,  # Nombre de tokens à générer
    temperature=1.0,     # Température de sampling
    top_k=50,           # Top-k filtering
)
```

**Processus:**
1. Pour chaque token à générer:
   - Forward pass sur le contexte
   - Extraire logits du dernier token
   - Appliquer température
   - Top-k filtering (optionnel)
   - Sampling depuis la distribution
   - Ajouter au contexte

## Optimisations Implémentées

### Performance

1. **Mixed Precision (AMP)**
   - Entraînement en float16
   - Scaler de gradient automatique
   - ~2× plus rapide sur GPU moderne

2. **Gradient Accumulation**
   - Batch effectif plus grand
   - Économie de VRAM
   - Stabilité d'entraînement

3. **Efficient DataLoading**
   - Multi-workers
   - Pin memory pour GPU
   - Préchargement asynchrone

4. **Cosine LR Schedule**
   - Warmup linéaire
   - Décroissance cosinus
   - Meilleure convergence

### Mémoire

1. **Scan Parallèle Optimisé**
   - O(log T) en profondeur
   - Pas de stockage intermédiaire massif
   - Version séquentielle pour petites séquences

2. **Opérations Quaternioniques**
   - Matrices 4×4 (Tensor Cores)
   - Broadcasting efficace
   - Pas de copies inutiles

3. **In-place Operations**
   - `set_to_none=True` pour zero_grad
   - Réutilisation de buffers
   - Minimisation d'allocations

## Formules Mathématiques

### Discrétisation Bilinéaire

```
z_{t,c,k} = Δ_{t,c} · Λ_{c,k}

A_{t,c,k} = (1 + z_{t,c,k}/2) / (1 - z_{t,c,k}/2)
```

**Propriétés:**
- Stable numériquement
- Λ < 0 assure convergence
- A_t réel et > 0

### Multiplication Quaternionique

```
(a₁ + b₁i + c₁j + d₁k) × (a₂ + b₂i + c₂j + d₂k) =
  (a₁a₂ - b₁b₂ - c₁c₂ - d₁d₂) +
  (a₁b₂ + b₁a₂ + c₁d₂ - d₁c₂)i +
  (a₁c₂ - b₁d₂ + c₁a₂ + d₁b₂)j +
  (a₁d₂ + b₁c₂ - c₁b₂ + d₁a₂)k
```

### SSM Récurrence

```
h_{t,c,k} = A_{t,c,k} · h_{t-1,c,k} + B_{t,c,k} ⊗ S_{t,c}

y_{t,c} = Σ_k C_{t,c,k} ⊗ h_{t,c,k} + D_{t,c} ⊗ S_{t,c}
```

## Complexité

### Temporelle

- **Forward pass:** O(B × T × C × K)
  - B: batch size
  - T: sequence length
  - C: channels (d_inner)
  - K: state dimension

- **Scan parallèle:** O(log T) en profondeur
  - Parallélisable sur T
  - Séquentiel sur K (petit)

- **Backward pass:** O(B × T × C × K)
  - Séquentiel sur T (stabilité)

### Spatiale

- **Paramètres:** ~O(d_model² × n_layers)
  - Embedding: vocab_size × d_model
  - Chaque couche: ~6 × d_model × d_inner
  - LM head: partagé avec embedding

- **Activations:** O(B × T × d_model × n_layers)
  - Réduit avec gradient checkpointing
  - États SSM: O(B × T × C × K × 4)

## Comparaison avec Mamba-2 Standard

| Aspect | Mamba-2 | Mamba-Quaternion-Lite |
|--------|---------|----------------------|
| Dynamique A_t | Scalaire réel | Scalaire réel ✓ |
| Injection B_t | Réelle | **Quaternionique** |
| Projection C_t | Réelle | **Quaternionique** |
| États h_t | Réels | Réels ✓ |
| Scan parallèle | Standard | Standard ✓ |
| Complexité | O(BTCK) | O(BTCK × 4) |
| Expressivité | Standard | **Rotations + couplages** |
| Stabilité | Excellente | Excellente ✓ |

## Fichiers Principaux

```
mamba_quaternion_lite/
├── __init__.py              # Exports publics
├── quaternion.py            # Opérations quaternioniques (400 lignes)
├── mamba_quat_lite.py       # Bloc SSM principal (330 lignes)
└── model.py                 # Modèle complet (140 lignes)

train.py                     # Script d'entraînement (320 lignes)
test_model.py               # Tests unitaires (200 lignes)
example_generation.py       # Exemple d'utilisation (100 lignes)
```

## Points Clés d'Implémentation

### 1. Pourquoi A_t est scalaire ?

- **Commutativité:** A_t × A_{t-1} = A_{t-1} × A_t
- **Scan parallèle:** Permet l'algorithme standard de Mamba-2
- **Stabilité:** Discrétisation bilinéaire simple
- **Performance:** Pas de multiplication quaternionique coûteuse

### 2. Pourquoi B_t et C_t sont quaternioniques ?

- **Rotations:** Transformation géométrique des inputs
- **Couplages:** Interactions entre composantes
- **Expressivité:** Espace hypercomplexe 4D
- **Localité:** Pas de dépendance temporelle (parallélisable)

### 3. Pourquoi h_t reste réel ?

- **Simplicité:** Dynamique standard
- **Efficacité:** Pas de multiplication quaternionique récursive
- **Stabilité:** Propriétés bien comprises
- **Compatibilité:** Réutilise le code Mamba-2

## Références

- Paper: "Mamba-Quaternion-Lite: Un Modèle SSD Sélectif Multi-États avec Dynamique Scalaire et Projections Quaternioniques"
- Mamba-2: "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality"
- Quaternions en ML: "Quaternion Convolutional Neural Networks for End-to-End Automatic Speech Recognition"
