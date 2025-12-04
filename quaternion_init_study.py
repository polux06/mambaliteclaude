"""
Stratégies avancées d'initialisation pour les réseaux quaternioniques.

Basé sur les recherches :
- "Quaternion Convolutional Neural Networks" (Gaudet & Maida, 2018)
- "Deep Quaternion Networks" (Parcollet et al., 2019)
- "Quaternion Neural Networks" (Zhu et al., 2018)

Les quaternions unitaires (norme = 1) forment une sphère S³ dans R⁴.
"""

import torch
import torch.nn as nn
import math


# ============================================================================
# 1. Initialisation Standard (Actuelle)
# ============================================================================

def quaternion_glorot_init(tensor, gain=1.0):
    """
    Initialisation de Glorot/Xavier adaptée aux quaternions.

    Variance ajustée pour les 4 composantes.

    Args:
        tensor: Tensor de forme (..., 4)
        gain: Facteur de gain

    Returns:
        Tensor initialisé
    """
    if tensor.ndim < 2:
        raise ValueError("Tensor must have at least 2 dimensions")

    fan_in = tensor.shape[0]
    fan_out = tensor.shape[1] if tensor.ndim > 2 else 1

    # Variance pour Glorot, divisée par 4 pour les quaternions
    std = gain * math.sqrt(2.0 / ((fan_in + fan_out) * 4))

    with torch.no_grad():
        tensor.normal_(0, std)

    return tensor


# ============================================================================
# 2. Initialisation sur la Sphère S³ (RECOMMANDÉ)
# ============================================================================

def quaternion_sphere_init(tensor, gain=1.0):
    """
    Initialisation sur la sphère S³ (quaternions unitaires).

    Les quaternions unitaires préservent les rotations et ont de meilleures
    propriétés géométriques. Cette méthode est recommandée pour les couches
    de projection où les rotations sont importantes.

    Args:
        tensor: Tensor de forme (..., 4)
        gain: Échelle de la norme (typiquement proche de 1)

    Returns:
        Tensor initialisé avec quaternions unitaires

    Références:
        - "Quaternion Convolutional Neural Networks" (Gaudet & Maida, 2018)
    """
    with torch.no_grad():
        # Générer des quaternions aléatoires
        tensor.normal_(0, 1)

        # Normaliser pour obtenir des quaternions unitaires (norme = 1)
        # norme = sqrt(a² + b² + c² + d²)
        norms = torch.sqrt((tensor ** 2).sum(dim=-1, keepdim=True))
        tensor /= norms

        # Appliquer le gain
        tensor *= gain

    return tensor


# ============================================================================
# 3. Initialisation Hybride (Réelle vs Imaginaire)
# ============================================================================

def quaternion_hybrid_init(tensor, gain=1.0, real_bias=0.0):
    """
    Initialisation hybride avec biais vers la partie réelle.

    Initialise la partie réelle avec une variance plus grande et un biais,
    ce qui peut aider à la convergence initiale.

    Args:
        tensor: Tensor de forme (..., 4) où [a, b, c, d] = [real, i, j, k]
        gain: Échelle globale
        real_bias: Biais pour la partie réelle (typiquement 0.5-1.0)

    Returns:
        Tensor initialisé
    """
    if tensor.shape[-1] != 4:
        raise ValueError("Last dimension must be 4 for quaternions")

    with torch.no_grad():
        # Partie réelle (a) : variance plus grande + biais
        fan_in = tensor.shape[0]
        std_real = gain * math.sqrt(4.0 / (fan_in * 4))  # 2x la variance
        tensor[..., 0].normal_(real_bias, std_real)

        # Parties imaginaires (b, c, d) : variance standard
        std_imag = gain * math.sqrt(2.0 / (fan_in * 4))
        tensor[..., 1:].normal_(0, std_imag)

    return tensor


# ============================================================================
# 4. Initialisation Indépendante (Composantes IID)
# ============================================================================

def quaternion_independent_init(tensor, gain=1.0):
    """
    Initialisation avec composantes indépendantes et identiquement distribuées.

    Chaque composante (a, b, c, d) est initialisée indépendamment avec la même
    distribution. Simple mais peut ne pas préserver les propriétés géométriques.

    Args:
        tensor: Tensor de forme (..., 4)
        gain: Échelle

    Returns:
        Tensor initialisé
    """
    fan_in = tensor.shape[0]
    std = gain * math.sqrt(1.0 / fan_in)

    with torch.no_grad():
        tensor.normal_(0, std)

    return tensor


# ============================================================================
# 5. Initialisation Unifiée (He pour Quaternions)
# ============================================================================

def quaternion_he_init(tensor, gain=1.0, mode='fan_in'):
    """
    Initialisation de He (Kaiming) adaptée aux quaternions.

    Recommandée pour les activations non-linéaires (ReLU, SiLU, etc.).

    Args:
        tensor: Tensor de forme (..., 4)
        gain: Facteur de gain
        mode: 'fan_in', 'fan_out', ou 'fan_avg'

    Returns:
        Tensor initialisé
    """
    if tensor.ndim < 2:
        raise ValueError("Tensor must have at least 2 dimensions")

    fan_in = tensor.shape[0]
    fan_out = tensor.shape[1] if tensor.ndim > 2 else 1

    if mode == 'fan_in':
        fan = fan_in
    elif mode == 'fan_out':
        fan = fan_out
    elif mode == 'fan_avg':
        fan = (fan_in + fan_out) / 2
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # Variance pour He, adaptée aux quaternions
    std = gain * math.sqrt(2.0 / (fan * 4))

    with torch.no_grad():
        tensor.normal_(0, std)

    return tensor


# ============================================================================
# 6. Initialisation pour Skip Connections
# ============================================================================

def quaternion_identity_init(tensor):
    """
    Initialise avec des quaternions identité (1, 0, 0, 0).

    Utile pour les skip connections et les connexions résiduelles.
    Les quaternions identité correspondent à "pas de rotation".

    Args:
        tensor: Tensor de forme (..., 4)

    Returns:
        Tensor initialisé
    """
    with torch.no_grad():
        tensor.zero_()
        tensor[..., 0] = 1.0  # Partie réelle = 1

    return tensor


# ============================================================================
# 7. Initialisation pour Biases
# ============================================================================

def quaternion_bias_init(tensor, mode='zero'):
    """
    Initialisation des biais quaternioniques.

    Args:
        tensor: Tensor de forme (..., 4)
        mode: 'zero', 'identity', ou 'small'

    Returns:
        Tensor initialisé
    """
    with torch.no_grad():
        if mode == 'zero':
            tensor.zero_()
        elif mode == 'identity':
            tensor.zero_()
            tensor[..., 0] = 1.0
        elif mode == 'small':
            tensor.normal_(0, 0.01)
        else:
            raise ValueError(f"Invalid mode: {mode}")

    return tensor


# ============================================================================
# Analyse des Initialisations
# ============================================================================

def compare_initializations(shape=(128, 64, 4), num_samples=10000):
    """
    Compare les différentes stratégies d'initialisation.

    Args:
        shape: Forme du tensor à initialiser
        num_samples: Nombre d'échantillons pour les statistiques

    Returns:
        Dict avec les statistiques de chaque méthode
    """
    methods = {
        'glorot': quaternion_glorot_init,
        'sphere': quaternion_sphere_init,
        'hybrid': quaternion_hybrid_init,
        'independent': quaternion_independent_init,
        'he': quaternion_he_init,
    }

    results = {}

    for name, init_fn in methods.items():
        tensor = torch.empty(shape)
        init_fn(tensor)

        # Calculer les statistiques
        norms = torch.sqrt((tensor ** 2).sum(dim=-1))

        results[name] = {
            'mean': tensor.mean().item(),
            'std': tensor.std().item(),
            'norm_mean': norms.mean().item(),
            'norm_std': norms.std().item(),
            'min': tensor.min().item(),
            'max': tensor.max().item(),
        }

    return results


# ============================================================================
# Recommandations
# ============================================================================

def get_recommended_init(layer_type):
    """
    Retourne l'initialisation recommandée selon le type de couche.

    Args:
        layer_type: Type de couche ('projection', 'linear', 'bias', 'skip')

    Returns:
        Fonction d'initialisation recommandée

    Recommandations:
    ----------------

    1. **Projections B_t et C_t (SSM)**: quaternion_sphere_init
       - Préserve les rotations
       - Meilleures propriétés géométriques
       - Stable pendant l'entraînement

    2. **Couches linéaires**: quaternion_glorot_init ou quaternion_he_init
       - Glorot: pour activations symétriques (tanh, sigmoid)
       - He: pour activations non-symétriques (ReLU, SiLU)

    3. **Skip connections**: quaternion_identity_init
       - Commence avec l'identité (pas de transformation)
       - Apprend progressivement

    4. **Biais**: quaternion_bias_init(mode='zero')
       - Commence à zéro
       - Évite les biais initiaux

    5. **Premières couches**: quaternion_hybrid_init
       - Biais vers la partie réelle
       - Aide à la convergence initiale
    """
    recommendations = {
        'projection': quaternion_sphere_init,
        'linear': quaternion_glorot_init,
        'bias': lambda t: quaternion_bias_init(t, mode='zero'),
        'skip': quaternion_identity_init,
        'first_layer': quaternion_hybrid_init,
        'with_relu': quaternion_he_init,
    }

    return recommendations.get(layer_type, quaternion_glorot_init)


# ============================================================================
# Tests
# ============================================================================

def test_initializations():
    """Teste toutes les initialisations."""
    print("="*80)
    print("TEST DES INITIALISATIONS QUATERNIONIQUES")
    print("="*80)

    shape = (128, 64, 4)

    print(f"\nShape: {shape}")
    print(f"Nombre d'éléments: {128 * 64 * 4:,}")

    stats = compare_initializations(shape)

    print("\n" + "="*80)
    print("STATISTIQUES")
    print("="*80)

    print(f"\n{'Méthode':<15} {'Mean':<10} {'Std':<10} {'Norm Mean':<12} {'Norm Std':<12}")
    print("-"*80)

    for name, stat in stats.items():
        print(f"{name:<15} {stat['mean']:>9.4f} {stat['std']:>9.4f} "
              f"{stat['norm_mean']:>11.4f} {stat['norm_std']:>11.4f}")

    print("\n" + "="*80)
    print("RECOMMANDATIONS")
    print("="*80)

    print("""
    1. Projections SSM (B_t, C_t):     quaternion_sphere_init ✓
       - Norme = 1 (unitaire)
       - Préserve les rotations
       - Meilleure stabilité

    2. Couches linéaires:              quaternion_glorot_init ou quaternion_he_init
       - Variance adaptée au fan-in/fan-out
       - Glorot: activations symétriques
       - He: activations non-symétriques (SiLU)

    3. Skip connections:               quaternion_identity_init
       - Commence avec identité
       - Apprend progressivement

    4. Biais:                          quaternion_bias_init(mode='zero')
       - Commence à zéro
       - Standard pour les biais

    5. Premières couches:              quaternion_hybrid_init
       - Biais vers partie réelle
       - Aide convergence initiale
    """)

    print("="*80)

    # Test de propriétés
    print("\nTEST DES PROPRIÉTÉS")
    print("="*80)

    # Test 1: Sphère S³
    tensor_sphere = torch.empty(100, 50, 4)
    quaternion_sphere_init(tensor_sphere)
    norms_sphere = torch.sqrt((tensor_sphere ** 2).sum(dim=-1))

    print(f"\nSphere Init:")
    print(f"  Norme moyenne: {norms_sphere.mean():.6f} (doit être ≈ 1.0)")
    print(f"  Norme std: {norms_sphere.std():.6f} (doit être ≈ 0.0)")
    print(f"  Norme min: {norms_sphere.min():.6f}")
    print(f"  Norme max: {norms_sphere.max():.6f}")

    # Test 2: Identité
    tensor_identity = torch.empty(10, 4)
    quaternion_identity_init(tensor_identity)

    print(f"\nIdentity Init:")
    print(f"  Partie réelle moyenne: {tensor_identity[..., 0].mean():.6f} (doit être = 1.0)")
    print(f"  Parties imaginaires: {tensor_identity[..., 1:].abs().sum():.6f} (doit être = 0.0)")

    print("\n" + "="*80)


if __name__ == "__main__":
    test_initializations()
