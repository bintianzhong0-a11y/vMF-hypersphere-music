# -*- coding: utf-8 -*-
"""
Train block-level function transition model.

Target:
    P(f_t | f_{t-1}, phrase_pos, prev_root, prev_template)

Also saves empirical transition table:
    P(f_t | f_{t-1}, phrase_pos)
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))


FUNCTION_NAMES = {
    0: "T",
    1: "D",
    2: "SD",
    3: "OTHER",
}


class BlockFunctionPairDataset(Dataset):
    def __init__(self, dataset_pt: str):
        obj = torch.load(dataset_pt, map_location="cpu", weights_only=False)

        self.sequences = obj["sequences"]
        self.phrase_period = int(obj.get("phrase_period", 4))
        self.samples = []

        for seq_idx, item in enumerate(self.sequences):
            f = item["function"].long()
            root = item["root"].long()
            template = item["template"].long()

            B = int(f.numel())

            for t in range(1, B):
                self.samples.append({
                    "seq_idx": seq_idx,
                    "t": t,
                    "prev_function": int(f[t - 1].item()),
                    "target_function": int(f[t].item()),
                    "phrase_pos": int(t % self.phrase_period),
                    "prev_root": int(root[t - 1].item()) % 12,
                    "prev_template": int(template[t - 1].item()),
                })

        print("[block dataset]")
        print("sequences:", len(self.sequences))
        print("samples:", len(self.samples))
        print("phrase_period:", self.phrase_period)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        return {
            "prev_function": torch.tensor(s["prev_function"], dtype=torch.long),
            "phrase_pos": torch.tensor(s["phrase_pos"], dtype=torch.long),
            "prev_root": torch.tensor(s["prev_root"], dtype=torch.long),
            "prev_template": torch.tensor(s["prev_template"], dtype=torch.long),
            "target_function": torch.tensor(s["target_function"], dtype=torch.long),
        }


class BlockFunctionTransitionMLP(nn.Module):
    def __init__(
        self,
        num_function: int = 4,
        phrase_period: int = 4,
        num_root: int = 12,
        num_template: int = 16,
        emb_dim: int = 32,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_function = num_function
        self.phrase_period = phrase_period
        self.num_root = num_root
        self.num_template = num_template

        self.emb_prev_function = nn.Embedding(num_function, emb_dim)
        self.emb_phrase_pos = nn.Embedding(phrase_period, emb_dim)
        self.emb_root = nn.Embedding(num_root, emb_dim)
        self.emb_template = nn.Embedding(num_template, emb_dim)

        in_dim = emb_dim * 4

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_function),
        )

    def forward(self, prev_function, phrase_pos, prev_root, prev_template):
        e = torch.cat([
            self.emb_prev_function(prev_function),
            self.emb_phrase_pos(phrase_pos),
            self.emb_root(prev_root),
            self.emb_template(prev_template),
        ], dim=-1)

        return self.net(e)


def compute_empirical_transition(dataset: BlockFunctionPairDataset, smoothing: float = 1.0):
    phrase_period = dataset.phrase_period

    counts = torch.full((4, phrase_period, 4), float(smoothing))

    for s in dataset.samples:
        prev_f = int(s["prev_function"])
        phrase_pos = int(s["phrase_pos"])
        target = int(s["target_function"])
        counts[prev_f, phrase_pos, target] += 1.0

    probs = counts / counts.sum(dim=-1, keepdim=True)
    log_probs = torch.log(probs + 1e-12)

    return counts, probs, log_probs


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    total = 0
    correct = 0
    total_loss = 0.0
    ce = nn.CrossEntropyLoss()

    confusion = torch.zeros(4, 4, dtype=torch.long)

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        logits = model(
            batch["prev_function"],
            batch["phrase_pos"],
            batch["prev_root"],
            batch["prev_template"],
        )

        target = batch["target_function"]
        loss = ce(logits, target)

        pred = logits.argmax(dim=-1)

        total += target.numel()
        correct += int((pred == target).sum().item())
        total_loss += float(loss.detach().cpu()) * target.numel()

        for y, p in zip(target.detach().cpu(), pred.detach().cpu()):
            confusion[int(y), int(p)] += 1

    acc = correct / max(total, 1)
    loss = total_loss / max(total, 1)

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

    macro_f1 = sum(f1s) / len(f1s)

    return {
        "loss": loss,
        "acc": acc,
        "macro_f1": macro_f1,
        "confusion": confusion.tolist(),
        "class_f1": {
            FUNCTION_NAMES[i]: f1s[i]
            for i in range(4)
        },
    }


def train_one_epoch(model, loader, optimizer, device):
    model.train()

    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    correct = 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad(set_to_none=True)

        logits = model(
            batch["prev_function"],
            batch["phrase_pos"],
            batch["prev_root"],
            batch["prev_template"],
        )

        target = batch["target_function"]
        loss = ce(logits, target)

        loss.backward()
        optimizer.step()

        pred = logits.argmax(dim=-1)

        total_loss += float(loss.detach().cpu()) * target.numel()
        total += target.numel()
        correct += int((pred == target).sum().item())

    return {
        "loss": total_loss / max(total, 1),
        "acc": correct / max(total, 1),
    }


def print_transition_table(probs: torch.Tensor):
    phrase_period = probs.shape[1]

    print("\n[empirical transition table]")
    for prev in range(4):
        for pos in range(phrase_period):
            row = probs[prev, pos]
            row_str = ", ".join(
                f"{FUNCTION_NAMES[i]}={float(row[i]):.3f}"
                for i in range(4)
            )
            print(f"prev={FUNCTION_NAMES[prev]:5s}, phrase_pos={pos}: {row_str}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--save_name", type=str, default="block_function_transition")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--emb_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.1)
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

    dataset = BlockFunctionPairDataset(args.dataset)

    counts, probs, log_probs = compute_empirical_transition(dataset, smoothing=1.0)
    print_transition_table(probs)

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
        num_workers=0,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = BlockFunctionTransitionMLP(
        num_function=4,
        phrase_period=dataset.phrase_period,
        num_root=12,
        num_template=16,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
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

        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)

        elapsed = time.time() - t0

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_macro_f1": val_metrics["macro_f1"],
            "elapsed_sec": elapsed,
        }

        rows.append(row)

        print(
            f"[epoch {epoch:03d}] "
            f"train_loss={row['train_loss']:.4f} "
            f"train_acc={row['train_acc']:.4f} "
            f"val_loss={row['val_loss']:.4f} "
            f"val_acc={row['val_acc']:.4f} "
            f"val_macro_f1={row['val_macro_f1']:.4f} "
            f"time={elapsed:.1f}s"
        )

        payload = {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "phrase_period": dataset.phrase_period,
            "model_config": {
                "num_function": 4,
                "phrase_period": dataset.phrase_period,
                "num_root": 12,
                "num_template": 16,
                "emb_dim": args.emb_dim,
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
            },
            "empirical_counts": counts,
            "empirical_probs": probs,
            "empirical_log_probs": log_probs,
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
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("[done]")
    print("best_val_loss:", best_val_loss)
    print("best checkpoint:", ckpt_dir / f"{args.save_name}_best.pt")
    print("log:", log_path)


if __name__ == "__main__":
    main()
