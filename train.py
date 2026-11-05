"""Train a MINT (Delphi-style) model. Hyperparameters come from a JSON config; data paths,
output directory and seed are command-line arguments."""
import argparse
import json
import logging
import math
import os
import time
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import torch

from model import Delphi, DelphiConfig
from utils import get_p2i, get_batch, load_demographic_token_ids

PTDTYPES = {'float32': torch.float32, 'float64': torch.float64,
            'bfloat16': torch.bfloat16, 'float16': torch.float16}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='JSON file with "model" and training settings')
    parser.add_argument('--train-npy', required=True)
    parser.add_argument('--val-npy', required=True)
    parser.add_argument('--vocab', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--log', required=True, help='path to the training log file')
    parser.add_argument('--resume', action='store_true', help='resume from <out-dir>/ckpt.pt')
    return parser.parse_args()


def get_lr(it, cfg):
    """Cosine learning-rate decay with linear warmup."""
    if not cfg.decay_lr:
        return cfg.learning_rate
    if it < cfg.warmup_iters:
        return cfg.learning_rate * it / cfg.warmup_iters
    if it > cfg.lr_decay_iters:
        return cfg.min_lr
    decay_ratio = (it - cfg.warmup_iters) / (cfg.lr_decay_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


def batch(data, p2i, cfg, demographic_token_ids, train):
    ix = torch.randint(len(p2i), (cfg.batch_size,))
    return get_batch(ix, data, p2i, block_size=cfg.model['block_size'], device=cfg.device,
                     padding='random' if train else 'regular', select=cfg.select,
                     no_event_token_rate=cfg.no_event_token_rate, cut_batch=True,
                     demographic_token_ids=demographic_token_ids)


@torch.no_grad()
def estimate_loss(model, splits, cfg, ctx, demographic_token_ids):
    """Mean [cross-entropy, time] loss over eval_iters batches for each split."""
    out = {}
    model.eval()
    for split, (data, p2i) in splits.items():
        losses = torch.zeros(cfg.eval_iters, 2)
        for k in range(cfg.eval_iters):
            X, A, Y, B = batch(data, p2i, cfg, demographic_token_ids, train=False)
            with ctx:
                _, loss, _ = model(X, A, Y, B, validation_loss_mode=True)
            losses[k] = torch.stack([loss['loss_ce'], loss['loss_dt']])
        out[split] = losses.mean(0)
    model.train()
    return out


def main():
    args = parse_args()
    with open(args.config) as f:
        config = json.load(f)
    cfg = SimpleNamespace(**config)
    logging.basicConfig(filename=args.log, filemode='w', level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, 'ckpt.pt')
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision('high')
    device_type = 'cuda' if 'cuda' in cfg.device else 'cpu'
    ptdtype = PTDTYPES[cfg.dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    torch.set_default_dtype(ptdtype)

    train_data = np.load(args.train_npy).astype(np.uint32)
    val_data = np.load(args.val_npy).astype(np.uint32)
    train_p2i = get_p2i(train_data)
    train_p2i = train_p2i[:int(cfg.data_fraction * len(train_p2i))]
    splits = {'train': (train_data, train_p2i), 'val': (val_data, get_p2i(val_data))}
    demographic_token_ids = load_demographic_token_ids(args.vocab)

    model = Delphi(DelphiConfig(**cfg.model))
    optimizer = model.configure_optimizers(cfg.weight_decay, cfg.learning_rate, (cfg.beta1, cfg.beta2), device_type)
    iter_num, best_val_loss = 0, 1e9
    if args.resume:
        checkpoint = torch.load(ckpt_path, map_location=cfg.device, weights_only=False)
        assert checkpoint['model_args'] == cfg.model, 'model config differs from checkpoint'
        model.load_state_dict({k.removeprefix('_orig_mod.'): v for k, v in checkpoint['model'].items()})
        optimizer.load_state_dict(checkpoint['optimizer'])
        iter_num, best_val_loss = checkpoint['iter_num'], checkpoint['best_val_loss']
    model.to(cfg.device)
    if cfg.compile:
        model = torch.compile(model)
    scaler = torch.amp.GradScaler(device_type, enabled=(cfg.dtype == 'float16'))

    def save_checkpoint(path, val_loss):
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'model_args': cfg.model, 'iter_num': iter_num,
                    'best_val_loss': val_loss, 'config': config}, path)
        logging.info(f"saving checkpoint to {path}")

    if cfg.wandb_log:
        import wandb
        wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name, config=config)

    X, A, Y, B = batch(train_data, train_p2i, cfg, demographic_token_ids, train=True)
    t0 = time.time()
    while iter_num <= cfg.max_iters:
        metrics = {"iter": iter_num}
        lr = get_lr(iter_num, cfg)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if iter_num % cfg.eval_interval == 0 and iter_num > 0:
            losses = estimate_loss(model, splits, cfg, ctx, demographic_token_ids)
            val_loss = losses['val'].sum().item()
            logging.info(f"step {iter_num}: train loss {losses['train'].sum().item():.4f}, val loss {val_loss:.4f}")
            metrics.update({"train/agg_loss": losses['train'].sum().item(), "val/loss": val_loss,
                            "val/loss_ce": losses['val'][0].item(), "val/loss_dt": losses['val'][1].item()})
            if cfg.always_save_checkpoint or val_loss < best_val_loss:
                save_checkpoint(ckpt_path, val_loss)
            best_val_loss = min(best_val_loss, val_loss)
            if iter_num % 10_000 == 0:
                save_checkpoint(os.path.join(args.out_dir, f'ckpt_{iter_num}.pt'), best_val_loss)

        if iter_num == 0 and cfg.eval_only:
            break

        # forward/backward with gradient accumulation; the next batch is fetched during the forward pass
        for _ in range(cfg.gradient_accumulation_steps):
            with ctx:
                _, loss, _ = model(X, A, Y, B)
            X, A, Y, B = batch(train_data, train_p2i, cfg, demographic_token_ids, train=True)
            loss = loss['loss_ce'] + loss['loss_dt']
            scaler.scale(loss).backward()
        if cfg.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        t1 = time.time()
        if iter_num % cfg.log_interval == 0:
            logging.info(f"iter {iter_num}: loss {loss.item():.4f}, time {(t1 - t0) * 1000:.2f}ms")
            metrics.update({"train/loss": loss.item(), "lr": lr})
        t0 = t1
        if cfg.wandb_log and len(metrics) > 1:
            wandb.log(metrics)
        iter_num += 1


if __name__ == "__main__":
    main()
