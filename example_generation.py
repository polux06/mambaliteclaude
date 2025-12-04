"""
Exemple d'utilisation de Mamba-Quaternion-Lite pour la génération de texte.
"""

import torch
from mamba_quaternion_lite import MambaQuaternionLiteModel


def main():
    print("="*60)
    print("MAMBA-QUATERNION-LITE - EXAMPLE GENERATION")
    print("="*60)

    # Configuration
    vocab_size = 100
    d_model = 128
    n_layers = 4

    print(f"\nConfiguration:")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Model dimension: {d_model}")
    print(f"  Number of layers: {n_layers}")

    # Créer le modèle
    print("\nInitializing model...")
    model = MambaQuaternionLiteModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        d_state=16,
        expand_factor=2,
    )

    model.eval()

    print(f"Model parameters: {model.count_parameters():,}")

    # Contexte initial (séquence aléatoire)
    context = torch.randint(0, vocab_size, (1, 10))

    print(f"\nInput context (10 tokens): {context[0].tolist()}")

    # Génération
    print("\nGenerating 20 new tokens...")

    with torch.no_grad():
        # Génération avec température basse (plus déterministe)
        print("\n1. Low temperature (0.5) - More deterministic:")
        generated_low = model.generate(
            context.clone(),
            max_new_tokens=20,
            temperature=0.5,
            top_k=10,
        )
        print(f"   {generated_low[0].tolist()}")

        # Génération avec température moyenne
        print("\n2. Medium temperature (1.0) - Balanced:")
        generated_med = model.generate(
            context.clone(),
            max_new_tokens=20,
            temperature=1.0,
            top_k=20,
        )
        print(f"   {generated_med[0].tolist()}")

        # Génération avec température haute (plus créatif)
        print("\n3. High temperature (1.5) - More creative:")
        generated_high = model.generate(
            context.clone(),
            max_new_tokens=20,
            temperature=1.5,
            top_k=50,
        )
        print(f"   {generated_high[0].tolist()}")

    print("\n" + "="*60)
    print("Generation complete!")
    print("="*60)

    # Exemple de forward pass avec calcul de loss
    print("\n\nExample forward pass with loss computation:")
    print("-"*60)

    model.train()

    # Créer un batch de données
    batch_size = 4
    seq_len = 32
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Forward pass
    loss, logits = model(input_ids, targets)

    print(f"Input shape: {input_ids.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")

    # Backward pass
    loss.backward()

    print("\nBackward pass completed successfully!")
    print(f"All {sum(1 for p in model.parameters() if p.grad is not None)} "
          f"parameters have gradients.")

    print("\n" + "="*60)
    print("Example completed successfully! ✓")
    print("="*60)


if __name__ == "__main__":
    main()
