# -*- coding: utf-8 -*-
"""
Evaluate function-aware vMF checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import scripts.train_vmf_hypersphere as base

for k in ["function", "function_transition"]:
    if k not in base.CLASS_KEYS:
        base.CLASS_KEYS.append(k)

from models.vmf_conformer_function import VMFConformerFunction
from models.vmf_losses import MultiTaskVMFLoss, masked_cross_entropy


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

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

        outputs = model(x, padding_mask=padding_mask)
        loss, loss_dict = criterion(outputs, targets, mask=valid_mask)

        if "function" in targets and "function_logits" in outputs:
            loss = loss + 0.7 * masked_cross_entropy(
                outputs["function_logits"],
                targets["function"],
                mask=valid_mask,
                ignore_index=base.IGNORE_INDEX,
            )

        if "function_transition" in targets and "function_transition_logits" in outputs:
            loss = loss + 0.5 * masked_cross_entropy(
                outputs["function_transition_logits"],
                targets["function_transition"],
                mask=valid_mask,
                ignore_index=base.IGNORE_INDEX,
            )

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
                a = base.accuracy(outputs[logit_key], targets[target_key], valid_mask)
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
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--out_json", type=str, default="results/eval_vmf_function_metrics.json")
    parser.add_argument("--out_csv", type=str, default="results/eval_vmf_function_metrics.csv")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("[device]", device)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    meta = ckpt["meta"]
    ckpt_args = ckpt.get("args", {})

    print("[checkpoint]", args.checkpoint)
    print("[meta]")
    print(json.dumps(meta, indent=2, ensure_ascii=False))

    dataset = base.VMFPTDataset(
        data_dir=args.data_dir,
        seq_len=args.seq_len,
        stride=args.stride,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=base.collate_fn,
        num_workers=args.num_workers,
    )

    model = VMFConformerFunction(
        input_dim=meta["input_dim"],
        mu_dim=meta["mu_dim"],
        d_model=ckpt_args.get("d_model", 128),
        num_layers=ckpt_args.get("num_layers", 4),
        n_heads=ckpt_args.get("n_heads", 4),
        dropout=ckpt_args.get("dropout", 0.1),
        conv_kernel_size=ckpt_args.get("conv_kernel_size", 15),
        num_root=meta["num_root"],
        num_template=meta["num_template"],
        num_triad=meta["num_triad"],
        num_seventh=meta["num_seventh"],
        num_beat=meta["num_beat"],
        num_bar=meta["num_bar"],
        num_function=meta.get("num_function", 4),
        num_function_transition=meta.get("num_function_transition", 16),
    )

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

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

    metrics = evaluate(model, loader, criterion, device)

    print("[metrics]")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    out_json = PROJECT_DIR / args.out_json
    out_csv = PROJECT_DIR / args.out_csv
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)

    print("[saved]", out_json)
    print("[saved]", out_csv)


if __name__ == "__main__":
    main()
