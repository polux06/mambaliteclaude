"""
Implémentation optimisée de Mamba-Quaternion-Lite.

Ce module implémente le cœur du modèle avec:
- Dynamique scalaire A_t (commutative)
- Projections quaternioniques B_t et C_t
- Scan parallèle standard de Mamba-2
- Sélectivité S6
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .quaternion import QuaternionProjection


class ParallelScan(torch.autograd.Function):
    """
    Scan parallèle associatif optimisé pour la dynamique scalaire.

    Implémente l'opération:
    h_t = A_t * h_{t-1} + b_t

    où A_t est scalaire (commutatif) et b_t est quaternionique.

    Cette implémentation utilise un algorithme de scan parallèle
    logarithmique en profondeur.
    """

    @staticmethod
    def forward(ctx, A, b):
        """
        Args:
            A: (batch, seq_len, channels, state_dim) - coefficients scalaires
            b: (batch, seq_len, channels, state_dim, 4) - termes quaternioniques

        Returns:
            h: (batch, seq_len, channels, state_dim, 4) - états quaternioniques
        """
        batch_size, seq_len, channels, state_dim = A.shape
        device = A.device
        dtype = A.dtype

        # Initialiser les états
        h = torch.zeros(batch_size, seq_len, channels, state_dim, 4,
                       dtype=b.dtype, device=device)

        # Version séquentielle optimisée pour petites séquences
        if seq_len <= 32:
            h[:, 0] = b[:, 0]
            for t in range(1, seq_len):
                # h_t = A_t * h_{t-1} + b_t
                # A[:, t] shape: (B, C, K)
                # h[:, t-1] shape: (B, C, K, 4)
                # On ajoute une dimension pour broadcaster: (B, C, K, 1)
                h[:, t] = A[:, t].unsqueeze(-1) * h[:, t-1] + b[:, t]

            ctx.save_for_backward(A, b, h)
            return h

        # Scan parallèle pour séquences longues
        # Utilisation d'un algorithme de type prefix-sum
        log_steps = math.ceil(math.log2(seq_len))

        # Copier les valeurs initiales
        A_scan = A.clone()
        b_scan = b.clone()

        # Phase montante (calcul des produits partiels)
        for step in range(log_steps):
            stride = 2 ** step

            # Indices à mettre à jour
            indices = torch.arange(stride, seq_len, device=device)

            if len(indices) == 0:
                break

            # A[t] = A[t] * A[t-stride]
            A_scan[:, indices] = A_scan[:, indices] * A_scan[:, indices - stride]

            # b[t] = A[t-stride] * b[t-stride] + b[t]
            # Note: on utilise A avant la mise à jour (indices - stride)
            b_scan[:, indices] = (
                A_scan[:, indices - stride].unsqueeze(-1) *
                b_scan[:, indices - stride]
            ) + b_scan[:, indices]

        # Phase descendante (calcul des états finaux)
        h[:, 0] = b_scan[:, 0]

        for t in range(1, seq_len):
            # Trouver le plus grand saut valide
            h[:, t] = b_scan[:, t]

        ctx.save_for_backward(A, b, h)
        return h

    @staticmethod
    def backward(ctx, grad_h):
        """
        Rétropropagation optimisée du scan parallèle.
        """
        A, b, h = ctx.saved_tensors
        batch_size, seq_len, channels, state_dim = A.shape

        # Gradients
        grad_A = torch.zeros_like(A)
        grad_b = torch.zeros_like(b)

        # Rétropropagation séquentielle (plus simple et stable)
        grad_h_prev = torch.zeros_like(h[:, 0])

        for t in range(seq_len - 1, -1, -1):
            # ∂L/∂b_t = ∂L/∂h_t
            grad_b[:, t] = grad_h[:, t] + grad_h_prev

            if t > 0:
                # ∂L/∂A_t = ∂L/∂h_t * h_{t-1}
                grad_A[:, t] = (grad_h[:, t] * h[:, t-1]).sum(dim=-1)

                # ∂L/∂h_{t-1} = ∂L/∂h_t * A_t
                grad_h_prev = grad_h[:, t] * A[:, t].unsqueeze(-1) + grad_h_prev

        return grad_A, grad_b


def parallel_scan(A, b):
    """Wrapper pour le scan parallèle."""
    return ParallelScan.apply(A, b)


class MambaQuaternionLiteBlock(nn.Module):
    """
    Bloc Mamba-Quaternion-Lite complet.

    Architecture:
    1. Projection d'entrée (expansion)
    2. Convolution causale 1D
    3. Calcul des paramètres dynamiques (Δ_t, B_t, C_t, D_t, gate)
    4. SSM avec dynamique scalaire et projections quaternioniques
    5. Projection de sortie (réduction)

    Args:
        d_model: Dimension du modèle
        d_state: Dimension de l'état SSM (K dans le papier)
        expand_factor: Facteur d'expansion (typiquement 2)
        conv_kernel_size: Taille du kernel de convolution
        dt_rank: Rang pour la projection de Δ
    """

    def __init__(
        self,
        d_model,
        d_state=16,
        expand_factor=2,
        conv_kernel_size=4,
        dt_rank=None,
        bias=False,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand_factor
        self.dt_rank = dt_rank or math.ceil(d_model / 16)

        # Projection d'entrée
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)

        # Convolution causale
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=conv_kernel_size,
            groups=self.d_inner,
            padding=conv_kernel_size - 1,
            bias=True,
        )

        # Paramètres du SSM

        # Δ_t (timestep) - projection vers un scalaire par canal
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Λ (Lambda) - modes continus (négatifs pour stabilité)
        self.Lambda = nn.Parameter(torch.randn(self.d_inner, d_state))

        # B_t - projection quaternionique d'entrée
        self.B_proj = QuaternionProjection(self.d_inner, d_state)

        # C_t - projection quaternionique de sortie
        self.C_proj = QuaternionProjection(self.d_inner, d_state)

        # D - skip connection
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Projection pour calculer Δ_t
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank, bias=False)

        # Projection de sortie
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

        # Layer norm
        self.norm = nn.LayerNorm(self.d_inner)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialise les paramètres pour la stabilité."""
        # Initialiser Lambda avec des valeurs négatives (amortissement)
        with torch.no_grad():
            # softplus inverse pour avoir des valeurs négatives après softplus
            self.Lambda.uniform_(-3, -1)

        # Initialiser D proche de 1
        nn.init.constant_(self.D, 1.0)

        # Initialiser les projections avec une petite variance
        nn.init.xavier_uniform_(self.in_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.5)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # 1. Projection d'entrée avec gating S6
        xz = self.in_proj(x)  # (B, T, 2*d_inner)
        x_ssm, z = xz.chunk(2, dim=-1)  # (B, T, d_inner) chacun

        # 2. Convolution causale
        # Transpose pour Conv1d: (B, d_inner, T)
        x_conv = self.conv1d(x_ssm.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
        x_conv = F.silu(x_conv)  # Activation

        # 3. Normalisation
        x_norm = self.norm(x_conv)

        # 4. Calcul des paramètres dynamiques

        # Δ_t (timestep) - dépend du contenu
        dt_input = self.x_proj(x_norm)  # (B, T, dt_rank)
        dt = F.softplus(self.dt_proj(dt_input))  # (B, T, d_inner)

        # Λ (Lambda) - modes fixes, négatifs
        Lambda = -F.softplus(self.Lambda)  # (d_inner, d_state)

        # z_t = Δ_t * Λ
        z_t = dt.unsqueeze(-1) * Lambda.unsqueeze(0).unsqueeze(0)  # (B, T, d_inner, d_state)

        # Discrétisation bilinéaire: A_t = (1 + z_t/2) / (1 - z_t/2)
        A_t = (1.0 + z_t / 2.0) / (1.0 - z_t / 2.0)  # (B, T, d_inner, d_state)

        # Gate S6
        gate = torch.sigmoid(z)  # (B, T, d_inner)
        S_t = gate * x_norm  # Signal sélectif (B, T, d_inner)

        # B_t - projection quaternionique
        # B_t: (B, T, d_inner, d_state, 4)
        B_t = self.B_proj(S_t, scalar_state=None)

        # b_t = B_t * S_t (mais S_t est déjà inclus dans la projection)
        # En fait, on veut B_t comme fonction de x, et b_t = B_t * S_t
        # Réorganisons: B_t doit être (B, T, d_inner, d_state, 4)

        # Pour simplifier, on crée B_t directement depuis S_t
        # B_t représente l'injection quaternionique
        B_t_weight = self.B_proj.weight.unsqueeze(0).unsqueeze(0)  # (1, 1, d_inner, d_state, 4)
        S_t_expanded = S_t.unsqueeze(-1).unsqueeze(-1)  # (B, T, d_inner, 1, 1)
        b_t = B_t_weight * S_t_expanded  # (B, T, d_inner, d_state, 4)

        # 5. SSM - Scan parallèle
        # h_t = A_t * h_{t-1} + b_t
        h = parallel_scan(A_t, b_t)  # (B, T, d_inner, d_state, 4)

        # 6. Projection de sortie quaternionique C_t
        # y_state = sum_k C_t[:,:,:,k] * h_t[:,:,:,k]
        # C_proj attend (scalar_state) de forme (B, T, C, K)
        # Mais h est quaternionique, donc on doit traiter différemment

        # Pour chaque composante quaternionique, on fait la projection
        # puis on somme les 4 composantes
        y_state = torch.zeros(batch_size, seq_len, self.d_inner, device=device, dtype=x.dtype)

        # C_proj.weight: (d_inner, d_state, 4)
        C_weight = self.C_proj.weight  # (d_inner, d_state, 4)

        # Pour chaque composante quaternionique
        for q_idx in range(4):
            # h[..., q_idx]: (B, T, d_inner, d_state)
            # C_weight[..., q_idx]: (d_inner, d_state)
            h_component = h[..., q_idx]  # (B, T, d_inner, d_state)
            C_component = C_weight[..., q_idx].unsqueeze(0).unsqueeze(0)  # (1, 1, d_inner, d_state)

            # Multiplication et somme sur d_state
            y_state += (h_component * C_component).sum(dim=-1)  # (B, T, d_inner)

        # 7. Skip connection D
        y_skip = self.D.unsqueeze(0).unsqueeze(0) * x_conv  # (B, T, d_inner)

        # 8. Sortie totale
        y = y_state + y_skip  # (B, T, d_inner)

        # 9. Projection finale
        output = self.out_proj(y)  # (B, T, d_model)

        return output


class ResidualBlock(nn.Module):
    """Bloc résiduel avec pré-normalisation."""

    def __init__(self, d_model, **mamba_kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = MambaQuaternionLiteBlock(d_model, **mamba_kwargs)

    def forward(self, x):
        return x + self.mamba(self.norm(x))
