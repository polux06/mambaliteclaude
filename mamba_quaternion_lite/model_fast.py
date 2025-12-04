"""
Modèle complet Mamba-Quaternion-Lite ultra-optimisé.
"""

import torch
import torch.nn as nn
from .mamba_quat_lite_fast import FastResidualBlock, enable_tf32, compile_model


class MambaQuaternionLiteFastModel(nn.Module):
    """
    Modèle de langage ultra-optimisé basé sur Mamba-Quaternion-Lite.

    Optimisations:
    - Blocs optimisés avec kernels Triton
    - Gradient checkpointing sélectif
    - Compatible torch.compile
    - Fused operations
    - Memory-efficient
    """

    def __init__(
        self,
        vocab_size,
        d_model=256,
        n_layers=6,
        d_state=16,
        expand_factor=2,
        conv_kernel_size=4,
        max_seq_len=512,
        dropout=0.1,
        use_gradient_checkpointing=False,
        checkpoint_every_n_layers=2,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.checkpoint_every_n_layers = checkpoint_every_n_layers

        # Embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Blocs Mamba optimisés
        self.layers = nn.ModuleList([
            FastResidualBlock(
                d_model=d_model,
                d_state=d_state,
                expand_factor=expand_factor,
                conv_kernel_size=conv_kernel_size,
                use_fast_path=True,
            )
            for _ in range(n_layers)
        ])

        # Normalisation finale
        self.norm_f = nn.LayerNorm(d_model)

        # LM head (weight tying)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        # Initialisation
        self.apply(self._init_weights)

        # Activer TF32 automatiquement
        enable_tf32()

    def _init_weights(self, module):
        """Initialisation optimisée."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, input_ids, targets=None):
        """
        Forward pass optimisé.

        Args:
            input_ids: (B, T)
            targets: (B, T) optionnel

        Returns:
            loss, logits si targets fourni
            logits sinon
        """
        # Embeddings
        x = self.token_embedding(input_ids)
        x = self.dropout(x)

        # Passer par les couches avec gradient checkpointing sélectif
        for i, layer in enumerate(self.layers):
            if (
                self.training
                and self.use_gradient_checkpointing
                and i % self.checkpoint_every_n_layers == 0
            ):
                # Gradient checkpointing pour cette couche
                x = torch.utils.checkpoint.checkpoint(
                    layer,
                    x,
                    use_reentrant=False
                )
            else:
                x = layer(x)

        # Normalisation finale
        x = self.norm_f(x)

        # Logits
        logits = self.lm_head(x)

        # Loss
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-1,
            )
            return loss, logits

        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens,
        temperature=1.0,
        top_k=None,
        top_p=None,
    ):
        """
        Génération optimisée avec top-k et top-p (nucleus) sampling.

        Args:
            input_ids: (B, T)
            max_new_tokens: int
            temperature: float
            top_k: int optionnel
            top_p: float optionnel (nucleus sampling)

        Returns:
            (B, T + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Logits
            logits = self(input_ids)
            logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    torch.softmax(sorted_logits, dim=-1), dim=-1
                )

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                # Scatter sorted tensors to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = -float('Inf')

            # Sampling
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Ajouter à la séquence
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    def count_parameters(self):
        """Compte les paramètres entraînables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_num_params(self, non_embedding=True):
        """
        Retourne le nombre de paramètres (avec option d'exclure embeddings).
        """
        n_params = self.count_parameters()
        if non_embedding:
            n_params -= self.token_embedding.weight.numel()
        return n_params

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """
        Estime le MFU (Model FLOPs Utilization).

        Compare les FLOPs effectifs vs FLOPs théoriques du GPU.

        Args:
            fwdbwd_per_iter: Nombre de tokens traités par itération
            dt: Temps par itération (secondes)

        Returns:
            MFU en pourcentage
        """
        N = self.get_num_params(non_embedding=True)
        L, H, Q, T = self.n_layers, self.d_model, self.d_model, fwdbwd_per_iter

        # Approximation des FLOPs par token
        # Forward: ~2N, Backward: ~4N (approximation)
        flops_per_token = 6 * N
        flops_per_iter = flops_per_token * T

        # FLOPs effectifs
        flops_achieved = flops_per_iter / dt  # FLOPs/sec

        # FLOPs théoriques du GPU (exemple: A100 = 312 TFLOPS en FP16)
        # Ajuster selon votre GPU
        flops_promised = 312e12  # A100
        # Pour V100: 125e12
        # Pour RTX 3090: 71e12

        mfu = flops_achieved / flops_promised

        return mfu


def create_optimized_model(
    vocab_size,
    d_model=256,
    n_layers=6,
    compile=True,
    compile_mode='default',
    **kwargs
):
    """
    Crée un modèle optimisé avec toutes les optimisations activées.

    Args:
        vocab_size: Taille du vocabulaire
        d_model: Dimension du modèle
        n_layers: Nombre de couches
        compile: Si True, compile avec torch.compile
        compile_mode: Mode de compilation ('default', 'reduce-overhead', 'max-autotune')
        **kwargs: Arguments supplémentaires pour le modèle

    Returns:
        Modèle optimisé
    """
    model = MambaQuaternionLiteFastModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        **kwargs
    )

    print(f"Created model with {model.count_parameters():,} parameters")

    if compile and hasattr(torch, 'compile'):
        model = compile_model(model, mode=compile_mode)

    return model
