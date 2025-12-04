"""
Script d'entraînement pour Mamba-Quaternion-Lite sur TinyStories.

Optimisé pour:
- Minimiser l'utilisation de la VRAM (gradient checkpointing, mixed precision)
- Maximiser les performances (data loading efficace, compilation)
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import os
import json
import time
from pathlib import Path
from tqdm import tqdm
import math

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

from mamba_quaternion_lite import MambaQuaternionLiteModel


# Configuration d'entraînement
class Config:
    # Dataset
    dataset_name = "roneneldan/TinyStories"
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
    weight_decay = 0.1
    warmup_steps = 1000
    grad_clip = 1.0

    # Optimisations
    use_amp = True  # Mixed precision
    compile_model = False  # PyTorch 2.0 compile (désactivé par défaut pour compatibilité)

    # Logging et sauvegarde
    eval_interval = 500
    save_interval = 2000
    log_interval = 50
    output_dir = "./outputs"

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"


class TinyStoriesDataset(Dataset):
    """Dataset pour TinyStories avec tokenization."""

    def __init__(self, texts, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        print(f"Tokenizing {len(texts)} examples...")
        for text in tqdm(texts):
            # Tokenize
            encoded = tokenizer.encode(text)
            tokens = encoded.ids

            # Créer des chunks de taille max_length
            for i in range(0, len(tokens) - 1, max_length):
                chunk = tokens[i:i + max_length + 1]
                if len(chunk) > 1:  # Au moins 2 tokens (input + target)
                    self.examples.append(chunk)

        print(f"Created {len(self.examples)} training examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        tokens = self.examples[idx]

        # Padding si nécessaire
        if len(tokens) < self.max_length + 1:
            tokens = tokens + [0] * (self.max_length + 1 - len(tokens))
        else:
            tokens = tokens[:self.max_length + 1]

        # Séparer input et target
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)

        return x, y


def load_tinystories(max_examples=100000):
    """Charge le dataset TinyStories depuis HuggingFace."""
    try:
        from datasets import load_dataset

        print("Loading TinyStories dataset...")
        dataset = load_dataset("roneneldan/TinyStories", split="train")

        # Limiter le nombre d'exemples pour l'entraînement rapide
        if max_examples and len(dataset) > max_examples:
            dataset = dataset.select(range(max_examples))

        texts = [item['text'] for item in dataset]

        # Split train/val
        split_idx = int(0.95 * len(texts))
        train_texts = texts[:split_idx]
        val_texts = texts[split_idx:]

        print(f"Train examples: {len(train_texts)}, Val examples: {len(val_texts)}")

        return train_texts, val_texts

    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Generating dummy data for testing...")
        # Données de test
        dummy_texts = [
            "Once upon a time, there was a little girl named Lily.",
            "She loved to play in the park with her friends.",
            "One day, she found a beautiful butterfly in the garden.",
        ] * 100

        split_idx = int(0.95 * len(dummy_texts))
        return dummy_texts[:split_idx], dummy_texts[split_idx:]


def train_tokenizer(texts, vocab_size=2048):
    """Entraîne un tokenizer BPE."""
    tokenizer_path = f"tokenizer_bpe_{vocab_size}.json"

    if os.path.exists(tokenizer_path):
        print(f"Loading existing tokenizer from {tokenizer_path}")
        tokenizer = Tokenizer.from_file(tokenizer_path)
        return tokenizer

    print(f"Training BPE tokenizer with vocab_size={vocab_size}...")

    # Initialiser le tokenizer BPE
    tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
    tokenizer.pre_tokenizer = Whitespace()

    # Entraîner
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"],
        min_frequency=2,
    )

    # Sauvegarder les textes temporairement
    temp_file = "temp_texts.txt"
    with open(temp_file, "w", encoding="utf-8") as f:
        for text in texts:
            f.write(text + "\n")

    tokenizer.train([temp_file], trainer)

    # Post-processing pour ajouter BOS/EOS
    tokenizer.post_processor = TemplateProcessing(
        single="<BOS> $A <EOS>",
        special_tokens=[
            ("<BOS>", tokenizer.token_to_id("<BOS>")),
            ("<EOS>", tokenizer.token_to_id("<EOS>")),
        ],
    )

    # Sauvegarder
    tokenizer.save(tokenizer_path)

    # Nettoyer
    os.remove(temp_file)

    print(f"Tokenizer saved to {tokenizer_path}")
    return tokenizer


def get_lr(step, warmup_steps, max_steps, max_lr):
    """Cosine learning rate schedule avec warmup."""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step > max_steps:
        return 0.0

    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return max_lr * coeff


@torch.no_grad()
def evaluate(model, dataloader, device, max_batches=50):
    """Évalue le modèle sur le dataset de validation."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (x, y) in enumerate(dataloader):
        if batch_idx >= max_batches:
            break

        x, y = x.to(device), y.to(device)

        with autocast(enabled=Config.use_amp):
            loss, _ = model(x, y)

        total_loss += loss.item()
        num_batches += 1

    model.train()
    return total_loss / num_batches if num_batches > 0 else 0.0


def train():
    """Fonction principale d'entraînement."""
    config = Config()

    # Créer le répertoire de sortie
    os.makedirs(config.output_dir, exist_ok=True)

    # Charger les données
    train_texts, val_texts = load_tinystories(max_examples=50000)

    # Entraîner/charger le tokenizer
    tokenizer = train_tokenizer(train_texts, vocab_size=config.vocab_size)

    # Créer les datasets
    train_dataset = TinyStoriesDataset(train_texts, tokenizer, config.max_seq_len)
    val_dataset = TinyStoriesDataset(val_texts, tokenizer, config.max_seq_len)

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Créer le modèle
    print("\nInitializing Mamba-Quaternion-Lite model...")
    model = MambaQuaternionLiteModel(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_layers=config.n_layers,
        d_state=config.d_state,
        expand_factor=config.expand_factor,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout,
    )

    model = model.to(config.device)

    print(f"Model parameters: {model.count_parameters():,}")

    # Compiler le modèle (PyTorch 2.0+)
    if config.compile_model and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile()...")
        model = torch.compile(model)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=config.weight_decay,
    )

    # Gradient scaler pour mixed precision
    scaler = GradScaler(enabled=config.use_amp)

    # Stats d'entraînement
    step = 0
    running_loss = 0.0
    best_val_loss = float('inf')

    print("\nStarting training...")
    print(f"Device: {config.device}")
    print(f"Batch size: {config.batch_size}")
    print(f"Gradient accumulation steps: {config.gradient_accumulation_steps}")
    print(f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"Max steps: {config.max_steps}")
    print(f"Mixed precision: {config.use_amp}\n")

    model.train()
    start_time = time.time()

    # Boucle d'entraînement
    data_iter = iter(train_loader)

    while step < config.max_steps:
        for micro_step in range(config.gradient_accumulation_steps):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                x, y = next(data_iter)

            x, y = x.to(config.device), y.to(config.device)

            # Forward pass avec mixed precision
            with autocast(enabled=config.use_amp):
                loss, _ = model(x, y)
                loss = loss / config.gradient_accumulation_steps

            # Backward pass
            scaler.scale(loss).backward()

            running_loss += loss.item()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        # Update learning rate
        lr = get_lr(step, config.warmup_steps, config.max_steps, config.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        step += 1

        # Logging
        if step % config.log_interval == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = (config.batch_size * config.gradient_accumulation_steps *
                            config.max_seq_len * config.log_interval) / elapsed

            print(f"Step {step}/{config.max_steps} | "
                  f"Loss: {running_loss:.4f} | "
                  f"LR: {lr:.2e} | "
                  f"Tokens/s: {tokens_per_sec:.0f}")

            running_loss = 0.0
            start_time = time.time()

        # Évaluation
        if step % config.eval_interval == 0:
            val_loss = evaluate(model, val_loader, config.device)
            print(f"\n{'='*60}")
            print(f"Validation at step {step}: Loss = {val_loss:.4f}")
            print(f"{'='*60}\n")

            # Sauvegarder le meilleur modèle
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = os.path.join(config.output_dir, "best_model.pt")
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': vars(config),
                }, checkpoint_path)
                print(f"Saved best model (val_loss={val_loss:.4f})\n")

        # Sauvegarde périodique
        if step % config.save_interval == 0:
            checkpoint_path = os.path.join(config.output_dir, f"checkpoint_{step}.pt")
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': vars(config),
            }, checkpoint_path)
            print(f"Saved checkpoint at step {step}\n")

        if step >= config.max_steps:
            break

    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    # Configuration de PyTorch pour les performances
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    train()
