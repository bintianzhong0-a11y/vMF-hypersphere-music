# -*- coding: utf-8 -*-
"""
Train Conformer-mediated block-level function transition.

Uses pretrained:
    vmf_conformer_function_pop1k7_1000_best.pt

Adds:
    block_transition_head

Target:
    P(f_b | H_{b-1}, f_{b-1}, phrase_pos_b, root_{b-1}, template_{b-1})
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from models.vmf_conformer_block_transition import VMFConformerBlockTransition


FUNCTION_NAMES = {
    0: "T",
    1: "D",
    2: "SD",
    3: "OTHER",
}

IGNORE_INDEX = -100


def mode_int(x: torch.Tensor, minlength: int) -> int:
    x = x.detach().cpu().long()
    x = x[x >= 0]
    if x.numel() == 0:
        return 0
    return int(torch.bincount(x, minlength=minlength).argmax().item())


def get_1d(obj: Dict[str, Any], key: str, T: int, default: int = 0) -> torch.Tensor:
    if key in obj and torch.is_tensor(obj[key]):
        y = obj[key].detach().cpu().long().flatten()
        if y.numel() >= T:
            return y[:T]
        pad = torch.full((T - y.numel(),), default, dtype=torch.long)
        return torch.cat([y, pad], dim=0)
    return torch.full((T,), default, dtype=torch.long)


class VMFBlockTransitionDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        events_per_block: int = 8,
        num_blocks: int = 16,
        block_stride: int = 4,
        phrase_period: int = 4,
        max_files: int | None = None,
        min_blocks: int = 8,
    ):
        self.data_dir = Path(data_dir)
        self.files = sorted(self.data_dir.rglob("*.pt"))

        if max_files is not None:
            self.files = self.files[:max_files]

        self.events_per_block = events_per_block
        self.num_blocks = num_blocks
        self.block_stride = block_stride
        self.phrase_period = phrase_period
        self.min_blocks = min_blocks

        self.index = []

        for file_idx, path in enumerate(self.files):
            try:
                obj = torch.load(path, map_location="cpu", weights_only=False)
            except Exception:
                continue

            if "x" not in obj or "function" not in obj:
                continue

            T = int(obj["x"].shape[0])
            n_blocks = T // events_per_block

            if n_blocks < min_blocks:
                continue

            max_start = max(0, n_blocks - num_blocks)

            for b0 in range(0, max_start + 1, block_stride):
                b1 = min(b0 + num_blocks, n_blocks)
                if b1 - b0 >= min_blocks:
                    self.index.append((file_idx, b0, b1))

        print("[dataset]")
        print("files:", len(self.files))
        print("chunks:", len(self.index))
        print("events_per_block:", self.events_per_block)
        print("num_blocks:", self.num_blocks)
        print("block_stride:", self.block_stride)

        if len(self.index) == 0:
            raise RuntimeError("No chunks found. Check data_dir/events_per_block/min_blocks.")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        file_idx, b0, b1 = self.index[idx]
        path = self.files[file_idx]
        obj = torch.load(path, map_location="cpu", weights_only=False)

        x = obj["x"].detach().cpu().float()
        T_total = int(x.shape[0])

        f = get_1d(obj, "function", T_total, default=0)
        root = get_1d(obj, "root", T_total, default=0)
        template = get_1d(obj, "template", T_total, default=0)

        e0 = b0 * self.events_per_block
        e1 = b1 * self.events_per_block

        x_chunk = x[e0:e1]

        block_f = []
        block_root = []
        block_template = []

        for b in range(b0, b1):
            a = b * self.events_per_block
            z = (b + 1) * self.events_per_block

            block_f.append(mode_int(f[a:z], minlength=4))
            block_root.append(mode_int(root[a:z], minlength=12))
            block_template.append(mode_int(template[a:z], minlength=16))

        block_f = torch.tensor(block_f, dtype=torch.long)
        block_root = torch.tensor(block_root, dtype=torch.long)
        block_template = torch.tensor(block_template, dtype=torch.long)

        # Transitions:
        # H_{b-1}, f_{b-1}, root_{b-1}, template_{b-1}, phrase_pos_b -> f_b
        prev_function = block_f[:-1]
        target_function = block_f[1:]
        prev_root = block_root[:-1]
        prev_template = block_template[:-1].clamp(0, 15)

        phrase_positions = torch.tensor(
            [
                (b % self.phrase_period)
                for b in range(b0 + 1, b1)
            ],
            dtype=torch.long,
        )

        return {
            "x": x_chunk,
            "prev_function": prev_function,
            "target_function": target_function,
            "phrase_pos": phrase_positions,
            "prev_root": prev_root,
            "prev_template": prev_template,
            "path": str(path),
            "b0": b0,
            "b1": b1,
        }


def collate_fn(batch):
    B = len(batch)
    input_dim = batch[0]["x"].shape[-1]

    max_T = max(item["x"].shape[0] for item in batch)
    max_N = max(item["target_function"].shape[0] for item in batch)

    x = torch.zeros(B, max_T, input_dim, dtype=torch.float32)
    padding_mask = torch.ones(B, max_T, dtype=torch.bool)

    prev_function = torch.full((B, max_N), IGNORE_INDEX, dtype=torch.long)
    target_function = torch.full((B, max_N), IGNORE_INDEX, dtype=torch.long)
    phrase_pos = torch.zeros(B, max_N, dtype=torch.long)
    prev_root = torch.zeros(B, max_N, dtype=torch.long)
    prev_template = torch.zeros(B, max_N, dtype=torch.long)
    block_mask = torch.zeros(B, max_N, dtype=torch.bool)

    for i, item in enumerate(batch):
        T = item["x"].shape[0]
        N = item["target_function"].shape[0]

        x[i, :T] = item["x"]
        padding_mask[i, :T] = False

        prev_function[i, :N] = item["prev_function"]
        target_function[i, :N] = item["target_function"]
        phrase_pos[i, :N] = item["phrase_pos"]
        prev_root[i, :N] = item["prev_root"]
        prev_template[i, :N] = item["prev_template"]
        block_mask[i, :N] = True

    return {
        "x": x,
        "padding_mask": padding_mask,
        "prev_function": prev_function,
        "target_function": target_function,
        "phrase_pos": phrase_pos,
        "prev_root": prev_root,
        "prev_template": prev_template,
        "block_mask": block_mask,
    }


def build_model_from_pretrained(
    checkpoint_path: str,
    phrase_period: int,
    block_emb_dim: int,
    block_hidden_dim: int,
    device: torch.device,
):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    meta = ckpt["meta"]
    args = ckpt.get("args", {})

    model = VMFConformerBlockTransition(
        input_dim=meta["input_dim"],
        mu_dim=meta["mu_dim"],
        d_model=args.get("d_model", 128),
        num_layers=args.get("num_layers", 4),
        n_heads=args.get("n_heads", 4),
        dropout=args.get("dropout", 0.1),
        conv_kernel_size=args.get("conv_kernel_size", 15),
        num_root=meta["num_root"],
        num_template=meta["num_template"],
        num_triad=meta["num_triad"],
        num_seventh=meta["num_seventh"],
        num_beat=meta["num_beat"],
        num_bar=meta["num_bar"],
        num_function=meta.get("num_function", 4),
        num_function_transition=meta.get("num_function_transition", 16),
        phrase_period=phrase_period,
        block_emb_dim=block_emb_dim,
        block_hidden_dim=block_hidden_dim,
        num_block_template=16,
    )

    missing, unexpected = model.load_state_dict(
        ckpt["model_state_dict"],
        strict=False,
    )

    print("[load pretrained]")
    print("missing:", missing[:20], "..." if len(missing) > 20 else "")
    print("unexpected:", unexpected[:20], "..." if len(unexpected) > 20 else "")

    model.to(device)

    return model, meta, args


def set_freeze_base(model, freeze_base: bool):
    if not freeze_base:
        for p in model.parameters():
            p.requires_grad = True
        return

    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("block_transition_head")


@torch.no_grad()
def compute_metrics(logits, target, mask):
    """
    logits:
        [B, N, 4]
    target:
        [B, N]
    mask:
        [B, N]
    """
    pred = logits.argmax(dim=-1)

    valid = mask & (target >= 0)
    if valid.sum().item() == 0:
        return {
            "acc": 0.0,
            "macro_f1": 0.0,
            "confusion": torch.zeros(4, 4, dtype=torch.long),
            "class_f1": {FUNCTION_NAMES[i]: 0.0 for i in range(4)},
        }

    y = target[valid].detach().cpu()
    p = pred[valid].detach().cpu()

    acc = float((y == p).float().mean().item())

    confusion = torch.zeros(4, 4, dtype=torch.long)
    for yy, pp in zip(y, p):
        confusion[int(yy), int(pp)] += 1

    f1s = []
    for c in range(4):
        tp = confusion[c, c].item()
        fp = confusion[:, c].sum().item() - tp
        fn = confusion[c, :].sum().item() - tp

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        f1s.append(f1)

    return {
        "acc": acc,
        "macro_f1": sum(f1s) / len(f1s),
        "confusion": confusion,
        "class_f1": {
            FUNCTION_NAMES[i]: f1s[i]
            for i in range(4)
        },
    }


def run_epoch(
    model,
    loader,
    optimizer,
    device,
    events_per_block: int,
    grad_clip: float,
):
    is_train = optimizer is not None
    model.train(is_train)

    ce = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    total_loss = 0.0
    total_items = 0

    acc_sum = 0.0
    f1_sum = 0.0
    batches = 0

    total_conf = torch.zeros(4, 4, dtype=torch.long)

    for batch in loader:
        batch = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            out = model.forward_block_transition(
                x=batch["x"],
                prev_function=batch["prev_function"].clamp_min(0),
                phrase_pos=batch["phrase_pos"],
                prev_root=batch["prev_root"],
                prev_template=batch["prev_template"],
                padding_mask=batch["padding_mask"],
                events_per_block=events_per_block,
            )

            logits = out["block_transition_logits"]
            target = batch["target_function"][:, :logits.shape[1]]
            mask = batch["block_mask"][:, :logits.shape[1]]

            loss = ce(
                logits.reshape(-1, logits.shape[-1]),
                target.reshape(-1),
            )

            if is_train:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        valid_count = int(mask.sum().item())
        total_loss += float(loss.detach().cpu()) * max(valid_count, 1)
        total_items += max(valid_count, 1)

        m = compute_metrics(logits.detach(), target.detach(), mask.detach())
        acc_sum += m["acc"]
        f1_sum += m["macro_f1"]
        total_conf += m["confusion"]
        batches += 1

    # recompute global f1 from total confusion
    f1s = []
    for c in range(4):
        tp = total_conf[c, c].item()
        fp = total_conf[:, c].sum().item() - tp
        fn = total_conf[c, :].sum().item() - tp

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        f1s.append(f1)

    return {
        "loss": total_loss / max(total_items, 1),
        "acc": acc_sum / max(batches, 1),
        "macro_f1": sum(f1s) / len(f1s),
        "confusion": total_conf.tolist(),
        "class_f1": {
            FUNCTION_NAMES[i]: f1s[i]
            for i in range(4)
        },
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--pretrained_checkpoint", type=str, required=True)
    parser.add_argument("--save_name", type=str, default="vmf_conformer_block_transition")

    parser.add_argument("--events_per_block", type=int, default=8)
    parser.add_argument("--num_blocks", type=int, default=16)
    parser.add_argument("--block_stride", type=int, default=4)
    parser.add_argument("--phrase_period", type=int, default=4)
    parser.add_argument("--min_blocks", type=int, default=8)
    parser.add_argument("--max_files", type=int, default=None)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--block_emb_dim", type=int, default=32)
    parser.add_argument("--block_hidden_dim", type=int, default=128)
    parser.add_argument("--freeze_base", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("[device]", device)

    dataset = VMFBlockTransitionDataset(
        data_dir=args.data_dir,
        events_per_block=args.events_per_block,
        num_blocks=args.num_blocks,
        block_stride=args.block_stride,
        phrase_period=args.phrase_period,
        max_files=args.max_files,
        min_blocks=args.min_blocks,
    )

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
        collate_fn=collate_fn,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    model, meta, pretrained_args = build_model_from_pretrained(
        checkpoint_path=args.pretrained_checkpoint,
        phrase_period=args.phrase_period,
        block_emb_dim=args.block_emb_dim,
        block_hidden_dim=args.block_hidden_dim,
        device=device,
    )

    set_freeze_base(model, args.freeze_base)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    print("[params]")
    print("trainable:", trainable)
    print("total:", total)
    print("freeze_base:", args.freeze_base)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    ckpt_dir = PROJECT_DIR / "checkpoints"
    result_dir = PROJECT_DIR / "results"
    ckpt_dir.mkdir(exist_ok=True)
    result_dir.mkdir(exist_ok=True)

    best_val_loss = float("inf")
    rows = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            events_per_block=args.events_per_block,
            grad_clip=args.grad_clip,
        )

        val_metrics = run_epoch(
            model,
            val_loader,
            optimizer=None,
            device=device,
            events_per_block=args.events_per_block,
            grad_clip=args.grad_clip,
        )

        elapsed = time.time() - t0

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_macro_f1": val_metrics["macro_f1"],
            "elapsed_sec": elapsed,
        }

        for k, v in val_metrics["class_f1"].items():
            row[f"val_f1_{k}"] = v

        rows.append(row)

        print(
            f"[epoch {epoch:03d}] "
            f"train_loss={row['train_loss']:.4f} "
            f"train_acc={row['train_acc']:.4f} "
            f"train_macro_f1={row['train_macro_f1']:.4f} "
            f"val_loss={row['val_loss']:.4f} "
            f"val_acc={row['val_acc']:.4f} "
            f"val_macro_f1={row['val_macro_f1']:.4f} "
            f"time={elapsed:.1f}s"
        )
        print("  val_class_f1:", val_metrics["class_f1"])

        payload = {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "meta": meta,
            "pretrained_args": pretrained_args,
            "events_per_block": args.events_per_block,
            "num_blocks": args.num_blocks,
            "phrase_period": args.phrase_period,
            "function_names": FUNCTION_NAMES,
            "val_metrics": val_metrics,
        }

        last_path = ckpt_dir / f"{args.save_name}_last.pt"
        torch.save(payload, last_path)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_path = ckpt_dir / f"{args.save_name}_best.pt"
            torch.save(payload, best_path)
            print("  saved best:", best_path)

        log_path = result_dir / f"{args.save_name}_train_log.csv"
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print("[done]")
    print("best_val_loss:", best_val_loss)
    print("best:", ckpt_dir / f"{args.save_name}_best.pt")
    print("last:", ckpt_dir / f"{args.save_name}_last.pt")


if __name__ == "__main__":
    main()
