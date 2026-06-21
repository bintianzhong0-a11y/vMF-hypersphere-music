# -*- coding: utf-8 -*-
"""
Minimal stable trainer for vMF Hypersphere Music Representation.

Expected .pt keys:
    x, mu,
    root, chord_like, template, triad, seventh, beat, bar, onset,
    velocity, timing, duration

This script works with toy_vmf_processed and later real vmf_processed datasets.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, random_split

# Make repository root importable even when called by absolute path
PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from models.vmf_conformer import VMFConformer
from models.vmf_losses import MultiTaskVMFLoss

IGNORE_INDEX = -100


CLASS_KEYS = ["root", "template", "triad", "seventh", "beat", "bar"]
BINARY_KEYS = ["chord_like", "onset"]
REG_KEYS = ["velocity", "timing", "duration"]
CONT_KEYS = ["mu"] + BINARY_KEYS + REG_KEYS


def safe_load(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def ensure_2d_x(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().cpu()
    if not torch.is_floating_point(x):
        x = x.float()

    if x.dim() == 1:
        x = x.unsqueeze(-1)
    elif x.dim() == 2:
        pass
    elif x.dim() == 3:
        # [T, N, D] -> [T, D]
        x = x.mean(dim=1)
    elif x.dim() > 3:
        x = x.reshape(x.shape[0], -1)

    return x.float()


def ensure_class(y: torch.Tensor) -> torch.Tensor:
    y = y.detach().cpu()
    if y.dim() == 0:
        y = y.view(1)
    elif y.dim() >= 2:
        y = y.reshape(y.shape[0], -1)[:, 0]
    return y.long()


def ensure_float_seq(y: torch.Tensor) -> torch.Tensor:
    y = y.detach().cpu()
    if not torch.is_floating_point(y):
        y = y.float()

    if y.dim() == 0:
        y = y.view(1)
    elif y.dim() == 1:
        pass
    elif y.dim() == 2:
        pass
    elif y.dim() >= 3:
        y = y.reshape(y.shape[0], -1)

    return y.float()


class VMFPTDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        seq_len: int = 128,
        stride: Optional[int] = None,
        max_files: Optional[int] = None,
    ):
        self.data_dir = Path(data_dir)
        self.seq_len = int(seq_len)
        self.stride = int(stride) if stride is not None else int(seq_len)

        files = sorted(self.data_dir.rglob("*.pt"))
        if max_files is not None:
            files = files[: int(max_files)]

        if len(files) == 0:
            raise FileNotFoundError(f"No .pt files found in: {self.data_dir}")

        self.files = files
        self.index: List[Tuple[Path, int, int]] = []

        print(f"[dataset] files: {len(files)}")

        for p in files:
            try:
                obj = safe_load(p)
                if "x" not in obj:
                    print(f"[skip] no x key: {p}")
                    continue

                x = ensure_2d_x(obj["x"])
                T = x.shape[0]

                if T <= 0:
                    continue

                if T <= self.seq_len:
                    self.index.append((p, 0, T))
                else:
                    for s in range(0, T - self.seq_len + 1, self.stride):
                        self.index.append((p, s, s + self.seq_len))

            except Exception as e:
                print(f"[skip] {p}: {e}")

        if len(self.index) == 0:
            raise RuntimeError("No usable chunks found.")

        print(f"[dataset] chunks: {len(self.index)}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path, start, end = self.index[idx]
        obj = safe_load(path)

        x = ensure_2d_x(obj["x"])
        x = x[start:end]

        item: Dict[str, Any] = {
            "x": x,
            "path": str(path),
        }

        for key in CLASS_KEYS:
            if key in obj:
                y = ensure_class(obj[key])
                item[key] = y[start:end]

        for key in BINARY_KEYS:
            if key in obj:
                y = ensure_float_seq(obj[key])
                if y.dim() >= 2:
                    y = y.reshape(y.shape[0], -1)[:, 0]
                item[key] = y[start:end].float()

        for key in REG_KEYS:
            if key in obj:
                y = ensure_float_seq(obj[key])
                if y.dim() == 1:
                    y = y.unsqueeze(-1)
                item[key] = y[start:end].float()

        if "mu" in obj:
            mu = ensure_2d_x(obj["mu"])
            item["mu"] = mu[start:end]
        else:
            # self-supervised fallback
            mu = x / (torch.linalg.norm(x, dim=-1, keepdim=True) + 1e-8)
            item["mu"] = mu

        return item


def pad_tensor_list(
    tensors: List[torch.Tensor],
    fill_value: float = 0.0,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    max_len = max(t.shape[0] for t in tensors)
    trailing = tensors[0].shape[1:]

    if dtype is None:
        dtype = tensors[0].dtype

    out = torch.full(
        (len(tensors), max_len, *trailing),
        fill_value=fill_value,
        dtype=dtype,
    )

    for i, t in enumerate(tensors):
        out[i, : t.shape[0]] = t.to(dtype=dtype)

    return out


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    xs = [b["x"] for b in batch]
    x = pad_tensor_list(xs, fill_value=0.0, dtype=torch.float32)

    lengths = torch.tensor([b["x"].shape[0] for b in batch], dtype=torch.long)
    T = x.shape[1]
    mask = torch.arange(T).unsqueeze(0) < lengths.unsqueeze(1)

    out: Dict[str, Any] = {
        "x": x,
        "mask": mask,
        "lengths": lengths,
        "paths": [b["path"] for b in batch],
    }

    all_keys = sorted(set().union(*[set(b.keys()) for b in batch]))
    all_keys = [k for k in all_keys if k not in {"x", "path"}]

    for key in all_keys:
        examples = [b[key] for b in batch if key in b]
        if len(examples) == 0:
            continue

        example = examples[0]
        seqs = []

        for b in batch:
            if key in b:
                seqs.append(b[key])
            else:
                shape = (b["x"].shape[0], *example.shape[1:])
                if key in CLASS_KEYS:
                    seqs.append(torch.full(shape, IGNORE_INDEX, dtype=torch.long))
                else:
                    seqs.append(torch.zeros(shape, dtype=torch.float32))

        if key in CLASS_KEYS:
            out[key] = pad_tensor_list(seqs, fill_value=IGNORE_INDEX, dtype=torch.long)
        else:
            out[key] = pad_tensor_list(seqs, fill_value=0.0, dtype=torch.float32)

    return out


def infer_meta(dataset: VMFPTDataset, max_scan: int = 128) -> Dict[str, int]:
    base = {
        "input_dim": 10,
        "mu_dim": 10,
        "num_root": 12,
        "num_template": 9,
        "num_triad": 6,
        "num_seventh": 6,
        "num_beat": 4,
        "num_bar": 4,
    }

    n = min(len(dataset), max_scan)

    for i in range(n):
        item = dataset[i]

        base["input_dim"] = int(item["x"].shape[-1])
        base["mu_dim"] = int(item["mu"].shape[-1])

        key_to_meta = {
            "root": "num_root",
            "template": "num_template",
            "triad": "num_triad",
            "seventh": "num_seventh",
            "beat": "num_beat",
            "bar": "num_bar",
        }

        for k, meta_k in key_to_meta.items():
            if k in item:
                y = item[k]
                valid = y[y >= 0]
                if valid.numel() > 0:
                    base[meta_k] = max(base[meta_k], int(valid.max().item()) + 1)

    return base


def move_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def get_targets(batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    skip = {"x", "mask", "lengths", "paths"}
    return {k: v for k, v in batch.items() if k not in skip and torch.is_tensor(v)}


@torch.no_grad()
def accuracy(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> Optional[float]:
    pred = logits.argmax(dim=-1)
    valid = mask.bool() & target.ne(IGNORE_INDEX)

    if valid.sum().item() == 0:
        return None

    return float((pred[valid] == target[valid]).float().mean().item())


def run_epoch(
    model: VMFConformer,
    loader: DataLoader,
    criterion: MultiTaskVMFLoss,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
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
    }
    acc_count = {k: 0 for k in acc_sum}

    for batch in loader:
        batch = move_to_device(batch, device)

        x = batch["x"]
        valid_mask = batch["mask"].bool()
        padding_mask = ~valid_mask

        targets = get_targets(batch)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            outputs = model(x, padding_mask=padding_mask)
            loss, loss_dict = criterion(outputs, targets, mask=valid_mask)

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
        ]

        for acc_name, logit_key, target_key in specs:
            if logit_key in outputs and target_key in targets:
                a = accuracy(outputs[logit_key], targets[target_key], valid_mask)
                if a is not None:
                    acc_sum[acc_name] += a
                    acc_count[acc_name] += 1

    metrics = {
        "loss": total_loss / max(steps, 1)
    }

    for k in acc_sum:
        if acc_count[k] > 0:
            metrics[k] = acc_sum[k] / acc_count[k]

    return metrics


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--max_files", type=int, default=None)

    parser.add_argument("--d_model", type=int, default=96)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--conv_kernel_size", type=int, default=15)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_name", type=str, default="vmf_conformer_toy_test")
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

    dataset = VMFPTDataset(
        data_dir=args.data_dir,
        seq_len=args.seq_len,
        stride=args.stride,
        max_files=args.max_files,
    )

    meta = infer_meta(dataset)
    print("[meta]")
    print(json.dumps(meta, indent=2, ensure_ascii=False))

    n_total = len(dataset)
    n_val = max(1, int(n_total * args.val_ratio))
    n_train = n_total - n_val

    if n_train <= 0:
        n_train = n_total
        n_val = 0

    if n_val > 0:
        train_set, val_set = random_split(
            dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(args.seed),
        )
    else:
        train_set = dataset
        val_set = dataset

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    model = VMFConformer(
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
        ignore_index=IGNORE_INDEX,
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
        )

        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer=None,
            device=device,
            grad_clip=args.grad_clip,
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
    print("log:", result_dir / f"{args.save_name}_train_log.csv")


if __name__ == "__main__":
    main()
