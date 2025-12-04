"""
Module implémentant les opérations quaternioniques optimisées pour PyTorch.

Un quaternion q = a + bi + cj + dk est représenté comme un tenseur de forme (..., 4)
où les 4 composantes sont [a, b, c, d] (partie réelle, i, j, k).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def quaternion_multiply(q1, q2):
    """
    Multiplication de quaternions optimisée.

    Args:
        q1: Tenseur de forme (..., 4) représentant le premier quaternion
        q2: Tenseur de forme (..., 4) représentant le second quaternion

    Returns:
        Tenseur de forme (..., 4) représentant q1 * q2

    Formule:
    (a1 + b1i + c1j + d1k) * (a2 + b2i + c2j + d2k) =
    (a1*a2 - b1*b2 - c1*c2 - d1*d2) +
    (a1*b2 + b1*a2 + c1*d2 - d1*c2)i +
    (a1*c2 - b1*d2 + c1*a2 + d1*b2)j +
    (a1*d2 + b1*c2 - c1*b2 + d1*a2)k
    """
    a1, b1, c1, d1 = q1[..., 0:1], q1[..., 1:2], q1[..., 2:3], q1[..., 3:4]
    a2, b2, c2, d2 = q2[..., 0:1], q2[..., 1:2], q2[..., 2:3], q2[..., 3:4]

    # Partie réelle
    r = a1*a2 - b1*b2 - c1*c2 - d1*d2
    # Partie i
    i = a1*b2 + b1*a2 + c1*d2 - d1*c2
    # Partie j
    j = a1*c2 - b1*d2 + c1*a2 + d1*b2
    # Partie k
    k = a1*d2 + b1*c2 - c1*b2 + d1*a2

    return torch.cat([r, i, j, k], dim=-1)


def quaternion_init_(tensor, scale=1.0):
    """
    Initialisation isotropique d'un tenseur quaternionique.

    Args:
        tensor: Tenseur de forme (..., 4) à initialiser
        scale: Échelle de l'initialisation
    """
    # Initialisation de Glorot/Xavier adaptée aux quaternions
    # Division par 2 car les quaternions ont une norme euclidienne naturellement plus grande
    fan_in = tensor.shape[0] if tensor.dim() >= 2 else 1
    std = scale * math.sqrt(2.0 / (fan_in * 4))

    with torch.no_grad():
        tensor.normal_(0, std)

    return tensor


class QuaternionLinear(nn.Module):
    """
    Couche linéaire quaternionique optimisée.

    Implémente: y = W * x + b
    où W est une matrice quaternionique et x, y sont des vecteurs quaternioniques.

    Args:
        in_features: Nombre de features d'entrée (quaternions)
        out_features: Nombre de features de sortie (quaternions)
        bias: Si True, ajoute un biais quaternionique
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Poids: (out_features, in_features, 4)
        # Chaque élément est un quaternion
        self.weight = nn.Parameter(torch.empty(out_features, in_features, 4))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, 4))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialise les paramètres avec une distribution isotropique."""
        quaternion_init_(self.weight)

        if self.bias is not None:
            with torch.no_grad():
                self.bias.zero_()

    def forward(self, x):
        """
        Args:
            x: Tenseur de forme (batch, seq_len, in_features, 4)

        Returns:
            Tenseur de forme (batch, seq_len, out_features, 4)
        """
        # x: (B, T, in_features, 4)
        # weight: (out_features, in_features, 4)

        batch_size, seq_len = x.shape[:2]

        # Reshape pour le calcul matriciel
        x_reshaped = x.view(batch_size * seq_len, self.in_features, 4)

        # Initialiser le résultat
        output = torch.zeros(batch_size * seq_len, self.out_features, 4,
                           dtype=x.dtype, device=x.device)

        # Pour chaque feature de sortie
        for i in range(self.out_features):
            # Pour chaque feature d'entrée
            acc = None
            for j in range(self.in_features):
                # Multiplication quaternionique: weight[i,j] * x[:,j]
                prod = quaternion_multiply(
                    self.weight[i, j].unsqueeze(0).expand(batch_size * seq_len, 4),
                    x_reshaped[:, j, :]
                )

                if acc is None:
                    acc = prod
                else:
                    acc = acc + prod

            output[:, i, :] = acc

        # Ajouter le biais si présent
        if self.bias is not None:
            output = output + self.bias.unsqueeze(0)

        # Reshape de retour
        output = output.view(batch_size, seq_len, self.out_features, 4)

        return output


class QuaternionProjection(nn.Module):
    """
    Projection optimisée pour les opérations B_t et C_t de Mamba-Quaternion-Lite.

    Cette version est optimisée pour les projections où on multiplie un quaternion
    par un scalaire réel (comme dans h_t * C_t).
    """

    def __init__(self, dim, state_dim):
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim

        # Poids quaternionique: (dim, state_dim, 4)
        self.weight = nn.Parameter(torch.empty(dim, state_dim, 4))

        self.reset_parameters()

    def reset_parameters(self):
        quaternion_init_(self.weight, scale=0.5)

    def forward(self, x, scalar_state=None):
        """
        Args:
            x: Tenseur d'entrée (batch, seq_len, dim) ou (batch, seq_len, dim, 4)
            scalar_state: Si fourni, tenseur scalaire réel (batch, seq_len, dim, state_dim)

        Returns:
            Tenseur quaternionique (batch, seq_len, dim, state_dim, 4) ou
            (batch, seq_len, dim, 4) selon l'usage
        """
        if scalar_state is not None:
            # Cas C_t * h_t où h_t est réel
            # scalar_state: (B, T, C, K)
            # weight: (C, K, 4)
            # Résultat: (B, T, C, 4)

            batch_size, seq_len, channels, state_dim = scalar_state.shape

            # Broadcast et multiplication
            # weight: (1, 1, C, K, 4)
            # scalar_state: (B, T, C, K, 1)
            w = self.weight.unsqueeze(0).unsqueeze(0)  # (1, 1, C, K, 4)
            s = scalar_state.unsqueeze(-1)  # (B, T, C, K, 1)

            # Multiplication scalaire * quaternion
            result = w * s  # (B, T, C, K, 4)

            # Somme sur K
            result = result.sum(dim=3)  # (B, T, C, 4)

            return result
        else:
            # Cas B_t qui projette x vers quaternions
            # x: (B, T, C)
            # weight: (C, K, 4)
            # Résultat: (B, T, C, K, 4)

            batch_size, seq_len, channels = x.shape

            # x: (B, T, C, 1, 1)
            # weight: (1, 1, C, K, 4)
            x_expanded = x.unsqueeze(-1).unsqueeze(-1)  # (B, T, C, 1, 1)
            w = self.weight.unsqueeze(0).unsqueeze(0)  # (1, 1, C, K, 4)

            # Multiplication (broadcasting implicite)
            result = x_expanded * w  # (B, T, C, K, 4)

            return result
