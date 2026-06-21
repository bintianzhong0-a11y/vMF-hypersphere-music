# -*- coding: utf-8 -*-
"""
Train function-aware vMF Hypersphere Music Representation.

Adds:
- function loss
- function_transition loss
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from torch.utils.data import DataLoader, random_split

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import scripts.train_vmf_hypersphere as base

# Patch base dataset class keys so collate_fn loads these labels.
for k in ["function", "function_transition"]:
    if k not in base.CLASS_KEYS:
        base.CLASS_KEYS.append(k)

from models.vmf_conformer_function import VMFConformerFunction
from models.vmf_losses import MultiTaskVMFLoss, masked_cross_entropy


def infer_meta_function(dataset, max_scan: int = 128) -> Dict[str, int]:
    meta = base.infer_meta(dataset, max_scan=max_scan)
    meta["num_function"] = 4
    meta["num_function_transition"] = 16

    n = min(len(dataset), max_scan)

    for i in range(n):
        item = dataset[i]

        if "function" in item:
            y = item["function"]
            valid = y[y >= 0]
            if valid.numel() > 0:
                meta["num_function"] = max(
                    meta["num_function"],
                    int(valid.max().item()) + 1,
                )

        if "function_transition" in item:
            y = item["function_transition"]
            valid = y[y >= 0]
            if valid.numel() > 0:
                meta["num_function_transition"] = max(
                    meta["num_function_transition"],
                    int(valid.max().item()) + 1,
                )

    return meta


@torch.no_grad()
def accuracy(logits, target, mask):
    return base.accuracy(logits, target, mask)


def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    grad_clip: float = 1.0,
    function_weight: float = 0.7,
    function_transition_weight: float = 0.5,
):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    steps = 0

    acc_sum = {
        "root_acc": 0.0,
        "template_acc": 0.0,
        "triad_acc": 0.0,
        "seventh_acc": 0.0,
        "beat_acc": 0.0,
        "bar_acc": 0.0,
        "function_acc": 0.0,
        "function_transition_acc": 0.0,
    }
    acc_count = {k: 0 for k in acc_sum}

    for batch in loader:
        batch = base.move_to_device(batch, device)

        x = batch["x"]
        valid_mask = batch["mask"].bool()
        padding_mask = ~valid_mask
        targets = base.get_targets(batch)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            outputs = model(x, padding_mask=padding_mask)
            loss, loss_dict = criterion(outputs, targets, mask=valid_mask)

            if "function" in targets and "function_logits" in outputs:
                loss_function = masked_cross_entropy(
                    outputs["function_logits"],
                    targets["function"],
                    mask=valid_mask,
                    ignore_index=base.IGNORE_INDEX,
                )
                loss = loss + function_weight * loss_function

            if "function_transition" in targets and "function_transition_logits" in outputs:
                loss_function_transition = masked_cross_entropy(
                    outputs["function_transition_logits"],
                    targets["function_transition"],
                    mask=valid_mask,
                    ignore_index=base.IGNORE_INDEX,
                )
                loss = loss + function_transition_weight * loss_function_transition

            if is_train:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        total_loss += float(loss.detach().cpu())
        steps += 1

        specs = [
            ("root_acc", "root_logits", "root"),
            ("template_acc", "template_logits", "template"),
            ("triad_acc", "triad_logits", "triad"),
            ("seventh_acc", "seventh_logits", "seventh"),
            ("beat_acc", "beat_logits", "beat"),
            ("bar_acc", "bar_logits", "bar"),
            ("function_acc", "function_logits", "function"),
            ("function_transition_acc", "function_transition_logits", "function_transition"),
        ]

        for acc_name, logit_key, target_key in specs:
            if logit_key in outputs and target_key in targets:
                a = accuracy(outputs[logit_key], targets[target_key], valid_mask)
                if a is not None:
                    acc_sum[acc_name] += a
                    acc_count[acc_name] += 1

    metrics = {"loss": total_loss / max(steps, 1)}

    for k in acc_sum:
        if acc_count[k] > 0:
            metrics[k] = acc_sum[k] / acc_count[k]

    return metrics


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--max_files", type=int, default=None)

    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--conv_kernel_size", type=int, default=15)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_name", type=str, default="vmf_conformer_function_full")
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--function_weight", type=float, default=0.7)
    parser.add_argument("--function_transition_weight", type=float, default=0.5)

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("[device]", device)

    dataset = base.VMFPTDataset(
        data_dir=args.data_dir,
        seq_len=args.seq_len,
        stride=args.stride,
        max_files=args.max_files,
    )

    meta = infer_meta_function(dataset)
    print("[meta]")
    print(json.dumps(meta, indent=2, ensure_ascii=False))

    n_total = len(dataset)
    n_val = max(1, int(n_total * args.val_ratio))
    n_train = n_total - n_val

    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=base.collate_fn,
        num_workers=args.num_workers,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=base.collate_fn,
        num_workers=args.num_workers,
    )

    model = VMFConformerFunction(
        input_dim=meta["input_dim"],
        mu_dim=meta["mu_dim"],
        d_model=args.d_model,
        num_layers=args.num_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        conv_kernel_size=args.conv_kernel_size,
        num_root=meta["num_root"],
        num_template=meta["num_template"],
        num_triad=meta["num_triad"],
        num_seventh=meta["num_seventh"],
        num_beat=meta["num_beat"],
        num_bar=meta["num_bar"],
        num_function=meta["num_function"],
        num_function_transition=meta["num_function_transition"],
    ).to(device)

    criterion = MultiTaskVMFLoss(
        weights={
            "mu": 1.0,
            "kappa": 0.01,
            "root": 1.0,
            "chord_like": 0.5,
            "template": 1.0,
            "triad": 0.5,
            "seventh": 0.5,
            "beat": 0.3,
            "bar": 0.3,
            "onset": 0.5,
            "velocity": 0.3,
            "timing": 0.2,
            "duration": 0.2,
        },
        ignore_index=base.IGNORE_INDEX,
        regression_loss="smooth_l1",
        use_vmf_nll=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    ckpt_dir = PROJECT_DIR / "checkpoints"
    result_dir = PROJECT_DIR / "results"
    ckpt_dir.mkdir(exist_ok=True)
    result_dir.mkdir(exist_ok=True)

    best_val = float("inf")
    rows = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            grad_clip=args.grad_clip,
            function_weight=args.function_weight,
            function_transition_weight=args.function_transition_weight,
        )

        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer=None,
            device=device,
            grad_clip=args.grad_clip,
            function_weight=args.function_weight,
            function_transition_weight=args.function_transition_weight,
        )

        elapsed = time.time() - t0

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "elapsed_sec": elapsed,
        }

        for k, v in train_metrics.items():
            if k != "loss":
                row[f"train_{k}"] = v

        for k, v in val_metrics.items():
            if k != "loss":
                row[f"val_{k}"] = v

        rows.append(row)

        print(
            f"[epoch {epoch:03d}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"time={elapsed:.1f}s"
        )

        for k, v in val_metrics.items():
            if k != "loss":
                print(f"  val_{k}: {v:.4f}")

        last_path = ckpt_dir / f"{args.save_name}_last.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "meta": meta,
                "args": vars(args),
                "epoch": epoch,
                "val_metrics": val_metrics,
                "function_label_map": {
                    "T": 0,
                    "D": 1,
                    "SD": 2,
                    "OTHER": 3,
                },
            },
            last_path,
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_path = ckpt_dir / f"{args.save_name}_best.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "meta": meta,
                    "args": vars(args),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "function_label_map": {
                        "T": 0,
                        "D": 1,
                        "SD": 2,
                        "OTHER": 3,
                    },
                },
                best_path,
            )
            print("  saved best:", best_path)

        log_path = result_dir / f"{args.save_name}_train_log.csv"
        fieldnames = sorted(set().union(*[r.keys() for r in rows]))

        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print("[done]")
    print("best_val:", best_val)
    print("checkpoint best:", ckpt_dir / f"{args.save_name}_best.pt")
    print("checkpoint last:", ckpt_dir / f"{args.save_name}_last.pt")


if __name__ == "__main__":
    main()
