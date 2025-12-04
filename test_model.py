"""
Script de test pour vérifier que le modèle Mamba-Quaternion-Lite fonctionne correctement.
"""

import torch
from mamba_quaternion_lite import MambaQuaternionLiteModel


def test_forward_pass():
    """Test du forward pass du modèle."""
    print("Testing forward pass...")

    # Créer un petit modèle
    model = MambaQuaternionLiteModel(
        vocab_size=1000,
        d_model=64,
        n_layers=2,
        d_state=8,
        expand_factor=2,
        max_seq_len=32,
        dropout=0.0,
    )

    print(f"Model created with {model.count_parameters():,} parameters")

    # Créer des données de test
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    targets = torch.randint(0, 1000, (batch_size, seq_len))

    # Forward pass avec targets (calcul de loss)
    print("\nForward pass with targets...")
    loss, logits = model(input_ids, targets)

    print(f"Loss: {loss.item():.4f}")
    print(f"Logits shape: {logits.shape}")
    assert logits.shape == (batch_size, seq_len, 1000), "Incorrect logits shape!"

    # Forward pass sans targets
    print("\nForward pass without targets...")
    logits = model(input_ids)
    print(f"Logits shape: {logits.shape}")
    assert logits.shape == (batch_size, seq_len, 1000), "Incorrect logits shape!"

    print("✓ Forward pass test passed!")


def test_backward_pass():
    """Test du backward pass (gradient flow)."""
    print("\n" + "="*60)
    print("Testing backward pass...")

    model = MambaQuaternionLiteModel(
        vocab_size=500,
        d_model=32,
        n_layers=2,
        d_state=4,
        expand_factor=2,
    )

    # Données de test
    input_ids = torch.randint(0, 500, (2, 8))
    targets = torch.randint(0, 500, (2, 8))

    # Forward + backward
    loss, _ = model(input_ids, targets)
    loss.backward()

    # Vérifier que les gradients existent
    has_grads = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for _ in model.parameters())

    print(f"Parameters with gradients: {has_grads}/{total_params}")
    assert has_grads == total_params, "Some parameters don't have gradients!"

    print("✓ Backward pass test passed!")


def test_generation():
    """Test de la génération de texte."""
    print("\n" + "="*60)
    print("Testing text generation...")

    model = MambaQuaternionLiteModel(
        vocab_size=100,
        d_model=32,
        n_layers=2,
        d_state=4,
    )

    model.eval()

    # Contexte initial
    context = torch.randint(0, 100, (1, 5))

    # Générer
    with torch.no_grad():
        generated = model.generate(
            context,
            max_new_tokens=10,
            temperature=1.0,
            top_k=20,
        )

    print(f"Context shape: {context.shape}")
    print(f"Generated shape: {generated.shape}")
    print(f"Context: {context[0].tolist()}")
    print(f"Generated: {generated[0].tolist()}")

    assert generated.shape[1] == context.shape[1] + 10, "Incorrect generation length!"

    print("✓ Generation test passed!")


def test_quaternion_operations():
    """Test des opérations quaternioniques."""
    print("\n" + "="*60)
    print("Testing quaternion operations...")

    from mamba_quaternion_lite.quaternion import quaternion_multiply, QuaternionLinear

    # Test de multiplication
    q1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # Quaternion réel
    q2 = torch.tensor([[0.0, 1.0, 0.0, 0.0]])  # Quaternion imaginaire i

    result = quaternion_multiply(q1, q2)
    print(f"1 * i = {result}")
    assert torch.allclose(result, q2), "Quaternion multiplication error!"

    # Test de la couche linéaire quaternionique
    qlinear = QuaternionLinear(in_features=4, out_features=2, bias=True)

    x = torch.randn(2, 3, 4, 4)  # (batch, seq, features, quaternion)
    y = qlinear(x)

    print(f"QuaternionLinear input shape: {x.shape}")
    print(f"QuaternionLinear output shape: {y.shape}")
    assert y.shape == (2, 3, 2, 4), "QuaternionLinear output shape error!"

    print("✓ Quaternion operations test passed!")


def test_memory_efficiency():
    """Test de l'efficacité mémoire."""
    print("\n" + "="*60)
    print("Testing memory efficiency...")

    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory test")
        return

    device = torch.device("cuda")

    model = MambaQuaternionLiteModel(
        vocab_size=2048,
        d_model=256,
        n_layers=6,
        d_state=16,
    ).to(device)

    # Mesurer la mémoire
    torch.cuda.reset_peak_memory_stats()

    batch_size = 8
    seq_len = 128
    input_ids = torch.randint(0, 2048, (batch_size, seq_len), device=device)
    targets = torch.randint(0, 2048, (batch_size, seq_len), device=device)

    # Forward pass
    loss, _ = model(input_ids, targets)

    forward_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB

    # Backward pass
    loss.backward()

    total_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB

    print(f"Peak memory (forward): {forward_memory:.2f} MB")
    print(f"Peak memory (forward+backward): {total_memory:.2f} MB")
    print(f"Model parameters: {model.count_parameters():,}")

    print("✓ Memory test passed!")


def run_all_tests():
    """Exécute tous les tests."""
    print("="*60)
    print("MAMBA-QUATERNION-LITE TEST SUITE")
    print("="*60)

    test_quaternion_operations()
    test_forward_pass()
    test_backward_pass()
    test_generation()
    test_memory_efficiency()

    print("\n" + "="*60)
    print("ALL TESTS PASSED! ✓")
    print("="*60)


if __name__ == "__main__":
    run_all_tests()
