# =============================================================================
# BLOCK 2: BACKBONE FINE-TUNING (PEFT)
# =============================================================================
# Notebook cell — runs after Block 1 (init), before Block 3 (run).
# All names from Block 1 are already in scope. Do NOT add imports for
# symbols defined in other blocks.
#
# Defines:
#   backbone_finetune()            - full FT with pretraining loss (per hospital)
#   classification_head_finetune() - full FT + classification head (per outcome)
#   EncounterDataset, collate_encounters
#
# Testing:
#   TEST_OUTCOMES=tachypnea python -m mint.five.fig_one.notebook.test_fig1_notebook
# =============================================================================

import copy
import gc
import math
import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class EncounterDataset(Dataset):
    """Dataset that converts a tokens DataFrame into model-ready sequences.

    Each encounter becomes one training sample: token IDs + times, truncated
    to max_len and shifted by 1 for next-token prediction.
    """

    def __init__(self, tokens_df, max_len=128, seed=42):
        self.max_len = max_len
        self.seed = seed
        self.encounters = []

        for enc_key, grp in tokens_df.groupby("encounter_key"):
            grp_sorted = grp.sort_values("t")
            token_ids = grp_sorted["token_id"].values.astype(np.int64)
            times = grp_sorted["t"].values.astype(np.float32)
            if len(token_ids) < 3:
                continue
            self.encounters.append((token_ids, times))

    def __len__(self):
        return len(self.encounters)

    def __getitem__(self, idx):
        token_ids, times = self.encounters[idx]

        # +1 shift for padding token (token 0 = padding in model)
        token_ids = token_ids + 1

        # Truncate to max_len + 1 (need 1 extra for shifted targets)
        seq_len = self.max_len + 1
        if len(token_ids) > seq_len:
            # Take the last seq_len tokens (most recent history)
            token_ids = token_ids[-seq_len:]
            times = times[-seq_len:]

        # Shift: x = tokens[:-1], y = tokens[1:], a = times[:-1], b = times[1:]
        x = torch.tensor(token_ids[:-1], dtype=torch.long)
        y = torch.tensor(token_ids[1:], dtype=torch.long)
        a = torch.tensor(times[:-1], dtype=torch.float32)
        b = torch.tensor(times[1:], dtype=torch.float32)

        return x, a, y, b


def collate_encounters(batch):
    """Pad variable-length sequences to the longest in the batch."""
    xs, as_, ys, bs = zip(*batch)
    max_t = max(len(x) for x in xs)

    x_pad = torch.zeros(len(xs), max_t, dtype=torch.long)
    a_pad = torch.zeros(len(xs), max_t, dtype=torch.float32)
    y_pad = torch.full((len(xs), max_t), -1, dtype=torch.long)
    b_pad = torch.zeros(len(xs), max_t, dtype=torch.float32)

    for i, (x, a, y, b) in enumerate(zip(xs, as_, ys, bs)):
        L = len(x)
        x_pad[i, :L] = x
        a_pad[i, :L] = a
        y_pad[i, :L] = y
        b_pad[i, :L] = b

    return x_pad, a_pad, y_pad, b_pad


def backbone_finetune(model, tokens_train_df, tokens_val_df, output_dir,
                      logger, device=None, max_len=128, min_encounters=200,
                      batch_size=8, grad_accum_steps=4, lr=3e-5,
                      weight_decay=0.1, warmup_steps=50, patience=3,
                      dropout=0.2, seed=42):
    """Full-parameter fine-tune of the Delphi model on local hospital data.

    Uses the original pretraining loss: loss_ce + loss_dt.

    Args:
        model: Pretrained Delphi model (not modified — deepcopy is used).
        tokens_train_df: DataFrame with [encounter_key, name, t, token_id].
        tokens_val_df: DataFrame with [encounter_key, name, t, token_id].
        output_dir: Path to save checkpoint.
        logger: Logger instance.
        device: torch device (defaults to CPU).
        max_len: Max sequence length for training.
        min_encounters: Skip FT if fewer train encounters than this.
        batch_size: Micro-batch size.
        grad_accum_steps: Gradient accumulation steps (effective batch = batch_size * grad_accum_steps).
        lr: Peak learning rate.
        weight_decay: AdamW weight decay.
        warmup_steps: Linear warmup steps.
        patience: Early stopping patience (number of evals without improvement).
        dropout: Dropout rate during fine-tuning.
        seed: Random seed.

    Returns:
        Fine-tuned model in eval mode, or None if skipped.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Check minimum encounter threshold
    n_train_enc = tokens_train_df["encounter_key"].nunique()
    n_val_enc = tokens_val_df["encounter_key"].nunique()
    if n_train_enc < min_encounters:
        logger.info(f"  PEFT skipped: only {n_train_enc} train encounters (min={min_encounters})")
        return None

    logger.info(f"  PEFT: Starting backbone fine-tuning")
    logger.info(f"    Train encounters: {n_train_enc}, Val encounters: {n_val_enc}")
    logger.info(f"    Config: lr={lr}, batch={batch_size}, grad_accum={grad_accum_steps}, "
                f"max_len={max_len}, dropout={dropout}")

    # Build datasets
    train_ds = EncounterDataset(tokens_train_df, max_len=max_len, seed=seed)
    val_ds = EncounterDataset(tokens_val_df, max_len=max_len, seed=seed)

    if len(train_ds) < min_encounters:
        logger.info(f"  PEFT skipped: only {len(train_ds)} valid train sequences (min={min_encounters})")
        return None

    logger.info(f"    Train sequences: {len(train_ds)}, Val sequences: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_encounters, drop_last=True,
                              generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_encounters)

    # Compute training schedule
    effective_batch = batch_size * grad_accum_steps
    steps_per_epoch = max(1, len(train_ds) // effective_batch)
    max_steps = steps_per_epoch * 3  # ~3 epochs
    eval_every = max(50, max_steps // 6)

    logger.info(f"    Steps/epoch: {steps_per_epoch}, Max steps: {max_steps}, Eval every: {eval_every}")

    # Deep copy model and set training config
    ft_model = copy.deepcopy(model)
    ft_model.config.dropout = dropout
    # Update dropout in all submodules
    for module in ft_model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = dropout
    ft_model.config.mask_ties = True
    ft_model.train()
    ft_model.to(device)

    # Optimizer (using model's own configure_optimizers)
    optimizer = ft_model.configure_optimizers(
        weight_decay=weight_decay,
        learning_rate=lr,
        betas=(0.9, 0.99),
        device_type=device.type,
    )

    # Cosine LR schedule with warmup
    def get_lr(step):
        if step < warmup_steps:
            return lr * step / warmup_steps
        decay_ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return lr * max(coeff, 0.1)

    # Training loop
    best_val_loss = float("inf")
    evals_without_improvement = 0
    global_step = 0
    train_losses = []
    t_start = time.time()

    data_iter = iter(train_loader)

    for step in range(max_steps):
        # Get batch (cycle through data)
        try:
            x, a, y, b = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, a, y, b = next(data_iter)

        x, a, y, b = x.to(device), a.to(device), y.to(device), b.to(device)

        # Forward pass
        logits, loss_dict, _ = ft_model(x, a, targets=y, targets_age=b)
        loss = (loss_dict["loss_ce"] + loss_dict["loss_dt"]) / grad_accum_steps
        loss.backward()
        train_losses.append(loss.item() * grad_accum_steps)

        # Gradient accumulation step
        if (step + 1) % grad_accum_steps == 0 or step == max_steps - 1:
            torch.nn.utils.clip_grad_norm_(ft_model.parameters(), 1.0)
            # Update LR
            current_lr = get_lr(global_step)
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        # Evaluation
        if (step + 1) % eval_every == 0 or step == max_steps - 1:
            ft_model.eval()
            val_losses = []
            with torch.no_grad():
                for vx, va, vy, vb in val_loader:
                    vx, va, vy, vb = vx.to(device), va.to(device), vy.to(device), vb.to(device)
                    _, vloss_dict, _ = ft_model(vx, va, targets=vy, targets_age=vb)
                    val_losses.append((vloss_dict["loss_ce"] + vloss_dict["loss_dt"]).item())

            val_loss = np.mean(val_losses) if val_losses else float("inf")
            train_loss_avg = np.mean(train_losses[-eval_every:])
            elapsed = time.time() - t_start

            logger.info(f"    Step {step+1}/{max_steps} (epoch {(step+1)/steps_per_epoch:.2f}) | "
                        f"train_loss={train_loss_avg:.4f} | val_loss={val_loss:.4f} | "
                        f"lr={current_lr:.2e} | elapsed={elapsed:.0f}s")

            epochs_completed = (step + 1) / steps_per_epoch

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                evals_without_improvement = 0
                # Save best checkpoint
                ckpt_path = output_dir / "peft" / "backbone_ft.pt"
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model": ft_model.state_dict(),
                    "model_args": {
                        "block_size": ft_model.config.block_size,
                        "vocab_size": ft_model.config.vocab_size,
                        "n_layer": ft_model.config.n_layer,
                        "n_head": ft_model.config.n_head,
                        "n_embd": ft_model.config.n_embd,
                        "dropout": model.config.dropout,  # save original dropout
                        "bias": ft_model.config.bias,
                    },
                    "step": step + 1,
                    "val_loss": val_loss,
                    "train_loss": train_loss_avg,
                }, str(ckpt_path))
                logger.info(f"    Saved best checkpoint (val_loss={val_loss:.4f})")
            else:
                evals_without_improvement += 1
                if evals_without_improvement >= patience and epochs_completed >= 1.0:
                    logger.info(f"    Early stopping at step {step+1} ({epochs_completed:.2f} epochs, patience={patience})")
                    break

            ft_model.train()

    # Load best checkpoint back
    ckpt_path = output_dir / "peft" / "backbone_ft.pt"
    if ckpt_path.exists():
        best_ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        ft_model.load_state_dict(best_ckpt["model"])
        logger.info(f"    Loaded best checkpoint (val_loss={best_ckpt['val_loss']:.4f}, step={best_ckpt['step']})")

    # Reset dropout to original and switch to eval
    original_dropout = model.config.dropout
    ft_model.config.dropout = original_dropout
    for module in ft_model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = original_dropout
    ft_model.eval()
    ft_model.to(torch.device("cpu"))

    # Clean up optimizer state
    del optimizer
    gc.collect()

    final_epochs = (step + 1) / steps_per_epoch
    elapsed_total = time.time() - t_start
    logger.info(f"  PEFT complete: {final_epochs:.2f} epochs in {elapsed_total:.0f}s, best_val_loss={best_val_loss:.4f}")

    return ft_model


class DelphiWithHead(torch.nn.Module):
    """Delphi backbone + binary classification head on top of last hidden state."""

    def __init__(self, backbone, n_embd):
        super().__init__()
        self.backbone = backbone
        self.head = torch.nn.Linear(n_embd, 1)

    def forward(self, idx, age, lengths):
        self.backbone.config.return_reps = True
        logits, _, reps, _ = self.backbone(idx, age)
        self.backbone.config.return_reps = False
        # Pool: take the representation at the last valid token position
        batch_idx = torch.arange(idx.size(0), device=idx.device)
        pooled = reps[batch_idx, lengths, :]
        return self.head(pooled).squeeze(-1)


def classification_head_finetune(model, train_cases, val_cases, max_len,
                                 output_dir, outcome, logger, device=None,
                                 batch_size=8, grad_accum_steps=4, lr=3e-5,
                                 weight_decay=0.1, warmup_steps=50,
                                 patience=3, dropout=0.2, seed=42,
                                 min_cases=30):
    """Fine-tune MINT backbone + classification head end-to-end on labeled cases.

    Trains per-outcome using binary cross-entropy. Returns predicted probabilities
    on the test set (passed separately via predict), or None if skipped.

    Args:
        model: Pretrained Delphi model (not modified — deepcopy is used).
        train_cases: dict with 'positive' and 'negative' case lists.
        val_cases: dict with 'positive' and 'negative' case lists.
        max_len: Max sequence length.
        output_dir: Path to save checkpoint.
        outcome: Task name (for logging/saving).
        logger: Logger instance.
        device: torch device.
        batch_size: Micro-batch size.
        grad_accum_steps: Gradient accumulation steps.
        lr: Peak learning rate.
        weight_decay: AdamW weight decay.
        warmup_steps: Linear warmup steps.
        patience: Early stopping patience.
        dropout: Dropout during fine-tuning.
        seed: Random seed.
        min_cases: Minimum total train cases to proceed.

    Returns:
        Trained DelphiWithHead in eval mode, or None if skipped.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_train = train_cases["positive"] + train_cases["negative"]
    all_val = val_cases["positive"] + val_cases["negative"]

    n_train = len(all_train)
    n_val = len(all_val)
    n_pos_train = len(train_cases["positive"])

    if n_train < min_cases:
        logger.info(f"    ClassHead skipped for {outcome}: only {n_train} train cases (min={min_cases})")
        return None

    if n_pos_train < 1 or (n_train - n_pos_train) < 1:
        logger.info(f"    ClassHead skipped for {outcome}: only one class in train")
        return None

    logger.info(f"    ClassHead: Fine-tuning backbone + head for {outcome}")
    logger.info(f"      Train: {n_train} ({n_pos_train} pos), Val: {n_val}")

    # Build datasets using CaseDataset from Block 1 (already in scope)
    # We need a dummy age_to_pos/neg since CaseDataset requires them but we only need events/times/labels
    dummy_pos = {}
    dummy_neg = {}
    train_ds = CaseDataset(all_train, dummy_pos, dummy_neg, max_len=max_len)
    val_ds = CaseDataset(all_val, dummy_pos, dummy_neg, max_len=max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, drop_last=len(train_ds) > batch_size,
                              generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn)

    # Training schedule
    effective_batch = batch_size * grad_accum_steps
    steps_per_epoch = max(1, len(train_ds) // effective_batch)
    max_steps = steps_per_epoch * 5  # up to 5 epochs
    eval_every = max(20, max_steps // 8)

    logger.info(f"      Steps/epoch: {steps_per_epoch}, Max steps: {max_steps}, Eval every: {eval_every}")

    # Build model with head
    ft_backbone = copy.deepcopy(model)
    ft_backbone.config.dropout = dropout
    for module in ft_backbone.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = dropout
    ft_backbone.train()

    clf_model = DelphiWithHead(ft_backbone, ft_backbone.config.n_embd)
    clf_model.to(device)

    # Optimizer: backbone params + head params
    decay_params = []
    no_decay_params = []
    for name, param in clf_model.named_parameters():
        if "bias" in name or "ln_" in name or "LayerNorm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=lr, betas=(0.9, 0.99))

    # Class weighting for BCE
    pos_weight = torch.tensor([(n_train - n_pos_train) / max(n_pos_train, 1)], device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Cosine LR with warmup
    def get_lr(step):
        if step < warmup_steps:
            return lr * step / warmup_steps
        decay_ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return lr * max(coeff, 0.1)

    # Training loop
    best_val_loss = float("inf")
    evals_without_improvement = 0
    global_step = 0
    train_losses = []
    t_start = time.time()
    data_iter = iter(train_loader)

    for step in range(max_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        events = batch["events"].to(device)
        times = batch["times"].to(device)
        labels = batch["labels"].float().to(device)
        lengths = batch["lengths"].to(device) - 1  # last valid index

        logits = clf_model(events, times, lengths)
        loss = loss_fn(logits, labels) / grad_accum_steps
        loss.backward()
        train_losses.append(loss.item() * grad_accum_steps)

        if (step + 1) % grad_accum_steps == 0 or step == max_steps - 1:
            torch.nn.utils.clip_grad_norm_(clf_model.parameters(), 1.0)
            current_lr = get_lr(global_step)
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        # Evaluation
        if (step + 1) % eval_every == 0 or step == max_steps - 1:
            clf_model.eval()
            val_losses = []
            with torch.no_grad():
                for vbatch in val_loader:
                    v_events = vbatch["events"].to(device)
                    v_times = vbatch["times"].to(device)
                    v_labels = vbatch["labels"].float().to(device)
                    v_lengths = vbatch["lengths"].to(device) - 1
                    v_logits = clf_model(v_events, v_times, v_lengths)
                    val_losses.append(loss_fn(v_logits, v_labels).item())

            val_loss = np.mean(val_losses) if val_losses else float("inf")
            train_loss_avg = np.mean(train_losses[-eval_every:])
            elapsed = time.time() - t_start
            epochs_completed = (step + 1) / steps_per_epoch

            logger.info(f"      Step {step+1}/{max_steps} (epoch {epochs_completed:.2f}) | "
                        f"train_loss={train_loss_avg:.4f} | val_loss={val_loss:.4f} | "
                        f"lr={current_lr:.2e} | elapsed={elapsed:.0f}s")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                evals_without_improvement = 0
                ckpt_path = output_dir / "peft" / f"classhead_{outcome}.pt"
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(clf_model.state_dict(), str(ckpt_path))
                logger.info(f"      Saved best checkpoint (val_loss={val_loss:.4f})")
            else:
                evals_without_improvement += 1
                if evals_without_improvement >= patience and epochs_completed >= 1.0:
                    logger.info(f"      Early stopping at step {step+1} ({epochs_completed:.2f} epochs, patience={patience})")
                    break

            clf_model.train()

    # Load best checkpoint
    ckpt_path = output_dir / "peft" / f"classhead_{outcome}.pt"
    if ckpt_path.exists():
        clf_model.load_state_dict(torch.load(str(ckpt_path), map_location=device, weights_only=False))

    # Reset dropout and eval mode
    original_dropout = model.config.dropout
    clf_model.backbone.config.dropout = original_dropout
    for module in clf_model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = original_dropout
    clf_model.eval()
    clf_model.to(torch.device("cpu"))

    del optimizer
    gc.collect()

    final_epochs = (step + 1) / steps_per_epoch
    elapsed_total = time.time() - t_start
    logger.info(f"    ClassHead complete: {final_epochs:.2f} epochs in {elapsed_total:.0f}s, best_val_loss={best_val_loss:.4f}")

    return clf_model


def classification_head_predict(clf_model, dataloader, device=None):
    """Run inference with a trained DelphiWithHead model.

    Returns probabilities and labels in the same order as the dataloader.
    """
    if device is None:
        device = torch.device("cpu")

    clf_model.eval()
    clf_model.to(device)
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            events = batch["events"].to(device)
            times = batch["times"].to(device)
            labels = batch["labels"]
            lengths = batch["lengths"].to(device) - 1
            logits = clf_model(events, times, lengths)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    clf_model.to(torch.device("cpu"))
    return np.asarray(all_probs, dtype=np.float32), np.asarray(all_labels)
