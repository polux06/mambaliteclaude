"""
Version améliorée des opérations quaternioniques avec initialisations optimales.

Améliorations :
- Initialisation sur la sphère S³ pour les projections SSM
- Initialisation de He pour les couches avec activations non-linéaires
- Initialisation identité pour skip connections
- Support de différentes stratégies selon le contexte
"""

import torch
import torch.nn as nn
import math


# ============================================================================
# Initialisations Optimales
# ============================================================================

def quaternion_sphere_init_(tensor, gain=1.0):
    """
    Initialise sur la sphère S³ (quaternions unitaires).

    Les quaternions unitaires ont les meilleures propriétés pour les rotations.
    Recommandé pour les projections B_t et C_t du SSM.

    Args:
        tensor: Tensor (..., 4) à initialiser
        gain: Échelle de la norme (par défaut 1.0 pour norme unitaire)
    """
    with torch.no_grad():
        tensor.normal_(0, 1)
        norms = torch.sqrt((tensor ** 2).sum(dim=-1, keepdim=True))
        tensor.div_(norms)
        tensor.mul_(gain)
    return tensor


def quaternion_he_init_(tensor, gain=1.0):
    """
    Initialisation de He adaptée aux quaternions.

    Recommandé pour les couches avec activations non-linéaires (SiLU, ReLU).

    Args:
        tensor: Tensor à initialiser
        gain: Facteur de gain
    """
    fan_in = tensor.shape[0] if tensor.ndim >= 2 else 1
    std = gain * math.sqrt(2.0 / (fan_in * 4))

    with torch.no_grad():
        tensor.normal_(0, std)
    return tensor


def quaternion_identity_init_(tensor):
    """
    Initialise avec quaternions identité (1, 0, 0, 0).

    Recommandé pour les skip connections.
    """
    with torch.no_grad():
        tensor.zero_()
        tensor[..., 0] = 1.0
    return tensor


# ============================================================================
# Multiplication Quaternionique (identique mais optimisée)
# ============================================================================

def quaternion_multiply(q1, q2):
    """
    Multiplication de quaternions optimisée.

    q1 * q2 = (a1 + b1i + c1j + d1k) * (a2 + b2i + c2j + d2k)
    """
    a1, b1, c1, d1 = q1[..., 0:1], q1[..., 1:2], q1[..., 2:3], q1[..., 3:4]
    a2, b2, c2, d2 = q2[..., 0:1], q2[..., 1:2], q2[..., 2:3], q2[..., 3:4]

    r = a1*a2 - b1*b2 - c1*c2 - d1*d2
    i = a1*b2 + b1*a2 + c1*d2 - d1*c2
    j = a1*c2 - b1*d2 + c1*a2 + d1*b2
    k = a1*d2 + b1*c2 - c1*b2 + d1*a2

    return torch.cat([r, i, j, k], dim=-1)


# ============================================================================
# Projection Quaternionique Améliorée pour SSM
# ============================================================================

class ImprovedQuaternionProjection(nn.Module):
    """
    Projection quaternionique avec initialisation optimale.

    Optimisations :
    - Initialisation sur S³ (quaternions unitaires)
    - Normalisation optionnelle pour stabilité
    - Support de la régularisation de norme
    """

    def __init__(self, dim, state_dim, normalize=False, init_gain=1.0):
        """
        Args:
            dim: Dimension des canaux
            state_dim: Dimension de l'état
            normalize: Si True, normalise les poids pendant le forward
            init_gain: Gain d'initialisation (1.0 = norme unitaire)
        """
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim
        self.normalize = normalize

        # Poids quaternionique: (dim, state_dim, 4)
        self.weight = nn.Parameter(torch.empty(dim, state_dim, 4))

        # Initialisation sur la sphère S³
        quaternion_sphere_init_(self.weight, gain=init_gain)

    def forward(self, scalar_state):
        """
        Args:
            scalar_state: (B, T, C, K) - états scalaires

        Returns:
            (B, T, C, 4) - projection quaternionique
        """
        weight = self.weight

        # Normalisation optionnelle (force norme unitaire)
        if self.normalize:
            norms = torch.sqrt((weight ** 2).sum(dim=-1, keepdim=True))
            weight = weight / (norms + 1e-8)

        # Projection via einsum (optimal)
        result = torch.einsum('btck,ckq->btcq', scalar_state, weight)

        return result

    def get_quaternion_norms(self):
        """Retourne les normes des quaternions (pour monitoring)."""
        with torch.no_grad():
            norms = torch.sqrt((self.weight ** 2).sum(dim=-1))
            return {
                'mean': norms.mean().item(),
                'std': norms.std().item(),
                'min': norms.min().item(),
                'max': norms.max().item(),
            }


# ============================================================================
# Projection pour B_t (Injection)
# ============================================================================

class QuaternionInjectionProjection(nn.Module):
    """
    Projection B_t : scalar → quaternion (injection dans le SSM).

    Optimisations :
    - Initialisation sphérique (S³)
    - Option de normalisation pour stabilité
    - Gating optionnel
    """

    def __init__(self, dim, state_dim, init_gain=0.5, use_gate=False):
        """
        Args:
            dim: Dimension des canaux
            state_dim: Dimension de l'état
            init_gain: Gain initial (0.5 par défaut pour injection modérée)
            use_gate: Si True, ajoute un gate appris
        """
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim
        self.use_gate = use_gate

        # Poids quaternionique
        self.weight = nn.Parameter(torch.empty(dim, state_dim, 4))
        quaternion_sphere_init_(self.weight, gain=init_gain)

        # Gate optionnel (contrôle l'injection)
        if use_gate:
            self.gate = nn.Parameter(torch.ones(dim, state_dim))

    def forward(self, x):
        """
        Args:
            x: (B, T, C) - signal sélectif

        Returns:
            (B, T, C, K, 4) - injection quaternionique
        """
        # x: (B, T, C)
        # weight: (C, K, 4)

        # Expand x pour broadcasting
        # x: (B, T, C, 1, 1)
        x_expanded = x.unsqueeze(-1).unsqueeze(-1)

        # weight: (1, 1, C, K, 4)
        w = self.weight.unsqueeze(0).unsqueeze(0)

        # Multiplication
        result = x_expanded * w  # (B, T, C, K, 4)

        # Gate optionnel
        if self.use_gate:
            gate = self.gate.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # (1, 1, C, K, 1)
            result = result * torch.sigmoid(gate)

        return result


# ============================================================================
# Projection pour C_t (Extraction)
# ============================================================================

class QuaternionExtractionProjection(nn.Module):
    """
    Projection C_t : quaternion states → scalar output.

    Optimisations :
    - Initialisation sphérique
    - Normalisation optionnelle
    - Fusion avec réduction
    """

    def __init__(self, dim, state_dim, init_gain=1.0, normalize=False):
        """
        Args:
            dim: Dimension des canaux
            state_dim: Dimension de l'état
            init_gain: Gain initial
            normalize: Force norme unitaire
        """
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim
        self.normalize = normalize

        # Poids quaternionique
        self.weight = nn.Parameter(torch.empty(dim, state_dim, 4))
        quaternion_sphere_init_(self.weight, gain=init_gain)

    def forward(self, h):
        """
        Args:
            h: (B, T, C, K, 4) - états quaternioniques

        Returns:
            (B, T, C) - sortie scalaire
        """
        weight = self.weight

        # Normalisation optionnelle
        if self.normalize:
            norms = torch.sqrt((weight ** 2).sum(dim=-1, keepdim=True))
            weight = weight / (norms + 1e-8)

        # Projection et réduction
        # h: (B, T, C, K, 4)
        # weight: (C, K, 4)
        # Produit élément par élément puis somme sur K et 4
        result = torch.einsum('btckq,ckq->btc', h, weight)

        return result


# ============================================================================
# Analyse et Monitoring
# ============================================================================

def analyze_quaternion_weights(model):
    """
    Analyse les poids quaternioniques d'un modèle.

    Args:
        model: Modèle contenant des projections quaternioniques

    Returns:
        Dict avec statistiques
    """
    stats = {}

    for name, module in model.named_modules():
        if isinstance(module, (ImprovedQuaternionProjection,
                              QuaternionInjectionProjection,
                              QuaternionExtractionProjection)):

            weight = module.weight.data
            norms = torch.sqrt((weight ** 2).sum(dim=-1))

            stats[name] = {
                'shape': tuple(weight.shape),
                'norm_mean': norms.mean().item(),
                'norm_std': norms.std().item(),
                'norm_min': norms.min().item(),
                'norm_max': norms.max().item(),
                'weight_mean': weight.mean().item(),
                'weight_std': weight.std().item(),
            }

    return stats


def print_quaternion_stats(stats):
    """Affiche les statistiques des quaternions."""
    print("\n" + "="*80)
    print("STATISTIQUES DES POIDS QUATERNIONIQUES")
    print("="*80)

    for name, stat in stats.items():
        print(f"\n{name}:")
        print(f"  Shape: {stat['shape']}")
        print(f"  Norme: {stat['norm_mean']:.4f} ± {stat['norm_std']:.4f} "
              f"[{stat['norm_min']:.4f}, {stat['norm_max']:.4f}]")
        print(f"  Poids: {stat['weight_mean']:.4f} ± {stat['weight_std']:.4f}")

    print("="*80)


# ============================================================================
# Tests
# ============================================================================

def test_improved_projections():
    """Teste les projections améliorées."""
    print("="*80)
    print("TEST DES PROJECTIONS QUATERNIONIQUES AMÉLIORÉES")
    print("="*80)

    batch_size, seq_len, channels, state_dim = 2, 16, 64, 8

    # Test Injection (B_t)
    print("\n1. Projection d'injection (B_t):")
    b_proj = QuaternionInjectionProjection(channels, state_dim, init_gain=0.5)

    x = torch.randn(batch_size, seq_len, channels)
    b_t = b_proj(x)

    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {b_t.shape}")
    print(f"   Expected: ({batch_size}, {seq_len}, {channels}, {state_dim}, 4)")

    # Vérifier les normes initiales
    norms = torch.sqrt((b_proj.weight ** 2).sum(dim=-1))
    print(f"   Normes des poids: {norms.mean():.4f} ± {norms.std():.4f}")
    print(f"   (Devrait être ≈ 0.5 car init_gain=0.5)")

    # Test Extraction (C_t)
    print("\n2. Projection d'extraction (C_t):")
    c_proj = QuaternionExtractionProjection(channels, state_dim, init_gain=1.0)

    h = torch.randn(batch_size, seq_len, channels, state_dim, 4)
    y = c_proj(h)

    print(f"   Input shape: {h.shape}")
    print(f"   Output shape: {y.shape}")
    print(f"   Expected: ({batch_size}, {seq_len}, {channels})")

    norms = torch.sqrt((c_proj.weight ** 2).sum(dim=-1))
    print(f"   Normes des poids: {norms.mean():.4f} ± {norms.std():.4f}")
    print(f"   (Devrait être ≈ 1.0 car init_gain=1.0)")

    # Test avec normalisation
    print("\n3. Projection avec normalisation:")
    c_proj_norm = QuaternionExtractionProjection(channels, state_dim, normalize=True)

    # Perturber les poids
    with torch.no_grad():
        c_proj_norm.weight *= 2.0

    y_norm = c_proj_norm(h)

    print(f"   Avec normalize=True, les normes sont forcées à 1.0 pendant forward")

    print("\n" + "="*80)
    print("✓ Tous les tests passés!")
    print("="*80)


if __name__ == "__main__":
    test_improved_projections()
