"""
Modèle complet Mamba-Quaternion-Lite pour la modélisation de langage.
"""

import torch
import torch.nn as nn
from .mamba_quat_lite import ResidualBlock


class MambaQuaternionLiteModel(nn.Module):
    """
    Modèle de langage basé sur Mamba-Quaternion-Lite.

    Args:
        vocab_size: Taille du vocabulaire
        d_model: Dimension du modèle
        n_layers: Nombre de couches Mamba
        d_state: Dimension de l'état SSM
        expand_factor: Facteur d'expansion dans les blocs
        conv_kernel_size: Taille du kernel de convolution
        max_seq_len: Longueur maximale de séquence (pour positional encoding optionnel)
        dropout: Taux de dropout
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
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers

        # Embedding de tokens
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Blocs Mamba-Quaternion-Lite
        self.layers = nn.ModuleList([
            ResidualBlock(
                d_model=d_model,
                d_state=d_state,
                expand_factor=expand_factor,
                conv_kernel_size=conv_kernel_size,
            )
            for _ in range(n_layers)
        ])

        # Normalisation finale
        self.norm_f = nn.LayerNorm(d_model)

        # Tête de langage (LM head)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Partage des poids entre embedding et lm_head (weight tying)
        self.lm_head.weight = self.token_embedding.weight

        # Initialisation
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialisation des poids."""
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
        Args:
            input_ids: (batch, seq_len) - indices des tokens
            targets: (batch, seq_len) - targets pour le calcul de la loss (optionnel)

        Returns:
            Si targets fourni: (loss, logits)
            Sinon: logits
        """
        batch_size, seq_len = input_ids.shape

        # Embeddings
        x = self.token_embedding(input_ids)  # (B, T, d_model)
        x = self.dropout(x)

        # Passer par tous les blocs Mamba
        for layer in self.layers:
            x = layer(x)

        # Normalisation finale
        x = self.norm_f(x)

        # Logits
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Calcul de la loss si targets fourni
        if targets is not None:
            # Reshape pour cross_entropy
            loss = nn.functional.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-1,
            )
            return loss, logits

        return logits

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=None):
        """
        Génération de texte auto-régressive.

        Args:
            input_ids: (batch, seq_len) - contexte initial
            max_new_tokens: nombre de tokens à générer
            temperature: température pour le sampling
            top_k: si fourni, ne considère que les top_k tokens les plus probables

        Returns:
            (batch, seq_len + max_new_tokens) - séquence générée
        """
        for _ in range(max_new_tokens):
            # Logits pour le dernier token
            logits = self(input_ids)  # (B, T, vocab_size)
            logits = logits[:, -1, :] / temperature  # (B, vocab_size)

            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Sampling
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Ajouter à la séquence
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    def count_parameters(self):
        """Compte le nombre de paramètres entraînables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
