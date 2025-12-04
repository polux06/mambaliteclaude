"""
Script d'entraînement ULTRA-OPTIMISÉ pour Mamba-Quaternion-Lite.

Optimisations inspirées de l'équipe Mamba-2:
- Kernels Triton pour le scan parallèle
- Fused operations (Conv+SiLU, Linear+gate)
- torch.compile pour JIT compilation
- Gradient accumulation avec scaling optimisé
- DataLoader avec prefetching
- Mixed precision optimisée (BF16 si disponible)
- Gradient checkpointing sélectif
- Profilage et monitoring de saturation GPU
- Optimisations mémoire avancées
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import time
from pathlib import Path
from tqdm import tqdm
import math

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

# Import du modèle optimisé
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mamba_quaternion_lite.model_fast import create_optimized_model
from mamba_quaternion_lite.mamba_quat_lite_fast import enable_tf32


# ============================================================================
# Configuration Optimisée
# ============================================================================

class OptimizedConfig:
    """Configuration ultra-optimisée pour saturation GPU maximale."""

    # Dataset
    dataset_name = "roneneldan/TinyStories"
    max_seq_len = 512  # Augmenté pour mieux saturer le GPU
    vocab_size = 2048

    # Modèle
    d_model = 512  # Augmenté de 256
    n_layers = 8  # Augmenté de 6
    d_state = 16
    expand_factor = 2
    dropout = 0.1

    # Entraînement - Optimisé pour saturation GPU
    batch_size = 32  # Augmenté de 16
    gradient_accumulation_steps = 2  # Réduit car batch_size augmenté
    max_steps = 50000
    learning_rate = 3e-4
    weight_decay = 0.1
    warmup_steps = 1000
    grad_clip = 1.0

    # Optimisations avancées
    use_amp = True
    amp_dtype = 'bfloat16'  # BF16 si disponible, sinon FP16
    use_torch_compile = True
    compile_mode = 'max-autotune'  # 'default', 'reduce-overhead', ou 'max-autotune'
    use_gradient_checkpointing = True
    checkpoint_every_n_layers = 2
    num_workers = 8  # Augmenté pour prefetching
    pin_memory = True
    prefetch_factor = 4  # Prefetching agressif

    # Profilage
    profile = False
    profile_steps = 10

    # Logging
    eval_interval = 500
    save_interval = 2000
    log_interval = 10  # Plus fréquent pour monitoring
    output_dir = "./outputs_optimized"

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# Dataset (identique mais avec optimisations de chargement)
# ============================================================================

class OptimizedTinyStoriesDataset(Dataset):
    """Dataset optimisé avec pré-tokenization en mémoire."""

    def __init__(self, texts, tokenizer, max_length):
        self.max_length = max_length
        self.examples = []

        print(f"Tokenizing {len(texts)} examples...")
        for text in tqdm(texts, desc="Tokenizing"):
            encoded = tokenizer.encode(text)
            tokens = encoded.ids

            # Créer des chunks
            for i in range(0, len(tokens) - 1, max_length):
                chunk = tokens[i:i + max_length + 1]
                if len(chunk) > 1:
                    # Padding immédiat pour éviter le padding dynamique
                    if len(chunk) < max_length + 1:
                        chunk = chunk + [0] * (max_length + 1 - len(chunk))
                    else:
                        chunk = chunk[:max_length + 1]

                    # Séparer input/target
                    self.examples.append({
                        'input': chunk[:-1],
                        'target': chunk[1:],
                    })

        print(f"Created {len(self.examples)} training examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        return (
            torch.tensor(example['input'], dtype=torch.long),
            torch.tensor(example['target'], dtype=torch.long),
        )


def load_tinystories(max_examples=100000):
    """Charge TinyStories avec caching."""
    cache_file = "tinystories_cache.pt"

    if os.path.exists(cache_file):
        print(f"Loading cached dataset from {cache_file}...")
        data = torch.load(cache_file)
        return data['train'], data['val']

    try:
        from datasets import load_dataset

        print("Loading TinyStories dataset...")
        dataset = load_dataset("roneneldan/TinyStories", split="train")

        if max_examples and len(dataset) > max_examples:
            dataset = dataset.select(range(max_examples))

        texts = [item['text'] for item in dataset]

        # Split
        split_idx = int(0.95 * len(texts))
        train_texts = texts[:split_idx]
        val_texts = texts[split_idx:]

        # Cache
        print(f"Caching dataset to {cache_file}...")
        torch.save({
            'train': train_texts,
            'val': val_texts,
        }, cache_file)

        return train_texts, val_texts

    except Exception as e:
        print(f"Error loading dataset: {e}")
        # Dummy data
        dummy = ["Once upon a time."] * 100
        split_idx = int(0.95 * len(dummy))
        return dummy[:split_idx], dummy[split_idx:]


def train_tokenizer(texts, vocab_size=2048):
    """Entraîne le tokenizer (identique)."""
    tokenizer_path = f"tokenizer_bpe_{vocab_size}.json"

    if os.path.exists(tokenizer_path):
        print(f"Loading tokenizer from {tokenizer_path}")
        return Tokenizer.from_file(tokenizer_path)

    print(f"Training BPE tokenizer...")
    tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"],
        min_frequency=2,
    )

    temp_file = "temp_texts.txt"
    with open(temp_file, "w", encoding="utf-8") as f:
        for text in texts:
            f.write(text + "\n")

    tokenizer.train([temp_file], trainer)

    tokenizer.post_processor = TemplateProcessing(
        single="<BOS> $A <EOS>",
        special_tokens=[
            ("<BOS>", tokenizer.token_to_id("<BOS>")),
            ("<EOS>", tokenizer.token_to_id("<EOS>")),
        ],
    )

    tokenizer.save(tokenizer_path)
    os.remove(temp_file)

    return tokenizer


# ============================================================================
# Utilitaires d'optimisation
# ============================================================================

def get_lr(step, warmup_steps, max_steps, max_lr):
    """Cosine LR schedule avec warmup."""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step > max_steps:
        return 0.0

    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return max_lr * coeff


class ThroughputMonitor:
    """Monitore le throughput et la saturation GPU."""

    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        self.start_time = time.time()
        self.num_tokens = 0
        self.num_steps = 0

    def update(self, batch_size, seq_len):
        self.num_tokens += batch_size * seq_len
        self.num_steps += 1

    def get_stats(self):
        elapsed = time.time() - self.start_time
        if elapsed == 0:
            return {}

        tokens_per_sec = self.num_tokens / elapsed
        steps_per_sec = self.num_steps / elapsed

        stats = {
            'tokens_per_sec': tokens_per_sec,
            'steps_per_sec': steps_per_sec,
            'elapsed': elapsed,
        }

        # Saturation GPU (estimation)
        if torch.cuda.is_available():
            gpu_util = torch.cuda.utilization()
            mem_used = torch.cuda.max_memory_allocated() / 1024**3  # GB
            mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            stats['gpu_utilization'] = gpu_util
            stats['vram_used_gb'] = mem_used
            stats['vram_total_gb'] = mem_total
            stats['vram_percent'] = (mem_used / mem_total) * 100

        return stats


@torch.no_grad()
def evaluate(model, dataloader, device, max_batches=50, amp_enabled=True, amp_dtype=torch.float16):
    """Évaluation optimisée."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (x, y) in enumerate(dataloader):
        if batch_idx >= max_batches:
            break

        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        with torch.amp.autocast(device_type='cuda' if device == 'cuda' else 'cpu',
                               enabled=amp_enabled,
                               dtype=amp_dtype):
            loss, _ = model(x, y)

        total_loss += loss.item()
        num_batches += 1

    model.train()
    return total_loss / num_batches if num_batches > 0 else 0.0


# ============================================================================
# Boucle d'entraînement ultra-optimisée
# ============================================================================

def train():
    """Boucle d'entraînement avec toutes les optimisations."""
    config = OptimizedConfig()

    # Créer répertoire de sortie
    os.makedirs(config.output_dir, exist_ok=True)

    # Activer TF32
    enable_tf32()

    # Déterminer le dtype pour AMP
    if config.amp_dtype == 'bfloat16' and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
        print("✓ Using BFloat16 for mixed precision")
    else:
        amp_dtype = torch.float16
        print("✓ Using Float16 for mixed precision")

    # Charger données
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    train_texts, val_texts = load_tinystories(max_examples=100000)

    # Tokenizer
    tokenizer = train_tokenizer(train_texts, vocab_size=config.vocab_size)

    # Datasets
    train_dataset = OptimizedTinyStoriesDataset(train_texts, tokenizer, config.max_seq_len)
    val_dataset = OptimizedTinyStoriesDataset(val_texts, tokenizer, config.max_seq_len)

    # DataLoaders optimisés
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=True if config.num_workers > 0 else False,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=config.pin_memory,
    )

    # Créer modèle optimisé
    print("\n" + "="*80)
    print("CREATING OPTIMIZED MODEL")
    print("="*80)

    model = create_optimized_model(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_layers=config.n_layers,
        d_state=config.d_state,
        expand_factor=config.expand_factor,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout,
        compile=config.use_torch_compile,
        compile_mode=config.compile_mode,
        use_gradient_checkpointing=config.use_gradient_checkpointing,
        checkpoint_every_n_layers=config.checkpoint_every_n_layers,
    )

    model = model.to(config.device)

    print(f"Parameters: {model.count_parameters():,}")
    print(f"Non-embedding parameters: {model.get_num_params(non_embedding=True):,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=config.weight_decay,
        fused=True if config.device == 'cuda' else False,  # Fused AdamW
    )

    # Gradient scaler
    scaler = torch.amp.GradScaler('cuda' if config.device == 'cuda' else 'cpu',
                                   enabled=config.use_amp)

    # Monitoring
    throughput_monitor = ThroughputMonitor(config)

    # Stats
    step = 0
    running_loss = 0.0
    best_val_loss = float('inf')

    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    print(f"Device: {config.device}")
    print(f"Batch size: {config.batch_size}")
    print(f"Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"Sequence length: {config.max_seq_len}")
    print(f"Mixed precision: {config.use_amp} ({amp_dtype})")
    print(f"Torch compile: {config.use_torch_compile}")
    print(f"Gradient checkpointing: {config.use_gradient_checkpointing}")
    print("="*80 + "\n")

    model.train()
    data_iter = iter(train_loader)

    # Profiler (optionnel)
    profiler = None
    if config.profile:
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=config.profile_steps),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        profiler.start()

    # Boucle principale
    while step < config.max_steps:
        epoch_start = time.time()

        for micro_step in range(config.gradient_accumulation_steps):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                x, y = next(data_iter)

            x = x.to(config.device, non_blocking=True)
            y = y.to(config.device, non_blocking=True)

            # Forward avec AMP
            with torch.amp.autocast(device_type='cuda' if config.device == 'cuda' else 'cpu',
                                   enabled=config.use_amp,
                                   dtype=amp_dtype):
                loss, _ = model(x, y)
                loss = loss / config.gradient_accumulation_steps

            # Backward
            scaler.scale(loss).backward()

            running_loss += loss.item()

            # Monitoring
            throughput_monitor.update(config.batch_size, config.max_seq_len)

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        # Update LR
        lr = get_lr(step, config.warmup_steps, config.max_steps, config.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        step += 1

        # Profiler
        if profiler:
            profiler.step()
            if step >= config.profile_steps:
                profiler.stop()
                profiler = None
                print("✓ Profiling completed")

        # Logging
        if step % config.log_interval == 0:
            stats = throughput_monitor.get_stats()

            log_str = f"Step {step}/{config.max_steps} | Loss: {running_loss:.4f} | LR: {lr:.2e}"

            if 'tokens_per_sec' in stats:
                log_str += f" | Tokens/s: {stats['tokens_per_sec']:.0f}"

            if 'vram_percent' in stats:
                log_str += f" | VRAM: {stats['vram_percent']:.1f}%"

            if 'gpu_utilization' in stats:
                log_str += f" | GPU: {stats['gpu_utilization']}%"

            print(log_str)

            running_loss = 0.0
            throughput_monitor.reset()

        # Évaluation
        if step % config.eval_interval == 0:
            val_loss = evaluate(model, val_loader, config.device,
                              amp_enabled=config.use_amp, amp_dtype=amp_dtype)

            # Estimer MFU
            mfu = model.estimate_mfu(
                fwdbwd_per_iter=config.batch_size * config.gradient_accumulation_steps * config.max_seq_len,
                dt=time.time() - epoch_start
            )

            print(f"\n{'='*80}")
            print(f"Validation @ step {step}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"MFU: {mfu*100:.2f}%")
            print(f"{'='*80}\n")

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
                print(f"✓ Saved best model (val_loss={val_loss:.4f})\n")

        # Sauvegarde périodique
        if step % config.save_interval == 0:
            checkpoint_path = os.path.join(config.output_dir, f"checkpoint_{step}.pt")
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': vars(config),
            }, checkpoint_path)
            print(f"✓ Checkpoint saved at step {step}\n")

        if step >= config.max_steps:
            break

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("="*80)


if __name__ == "__main__":
    train()
