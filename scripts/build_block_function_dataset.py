# -*- coding: utf-8 -*-
"""
Build block-level function sequence dataset.

Input:
    data/vmf_processed_pop1k7_1000/*.pt

Each .pt has note/event-level labels:
    function: T/D/SD/OTHER
    root
    template

This script converts them into block-level sequences:
    f_0, f_1, ..., f_B

Default:
    events_per_block = 8
    phrase_period = 4

Function labels:
    0: T
    1: D
    2: SD
    3: OTHER
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Dict, Any

import torch


FUNCTION_NAMES = {
    0: "T",
    1: "D",
    2: "SD",
    3: "OTHER",
}


def mode_int(x: torch.Tensor, minlength: int) -> int:
    x = x.detach().cpu().long()
    x = x[x >= 0]
    if x.numel() == 0:
        return 0
    counts = torch.bincount(x, minlength=minlength)
    return int(torch.argmax(counts).item())


def tensor_or_default(obj: Dict[str, Any], key: str, T: int, default: int = 0) -> torch.Tensor:
    if key in obj and torch.is_tensor(obj[key]):
        y = obj[key].detach().cpu().long()
        if y.numel() >= T:
            return y[:T]
        pad = torch.full((T - y.numel(),), default, dtype=torch.long)
        return torch.cat([y, pad], dim=0)
    return torch.full((T,), default, dtype=torch.long)


def convert_file_to_blocks(path: Path, events_per_block: int, min_blocks: int):
    obj = torch.load(path, map_location="cpu", weights_only=False)

    if "function" not in obj:
        return None

    f = obj["function"].detach().cpu().long()
    T = int(f.numel())

    if T < events_per_block * min_blocks:
        return None

    root = tensor_or_default(obj, "root", T, default=0)
    template = tensor_or_default(obj, "template", T, default=0)

    n_blocks = T // events_per_block

    f_blocks = []
    root_blocks = []
    template_blocks = []

    for b in range(n_blocks):
        a = b * events_per_block
        z = (b + 1) * events_per_block

        f_blocks.append(mode_int(f[a:z], minlength=4))
        root_blocks.append(mode_int(root[a:z], minlength=12))
        template_blocks.append(mode_int(template[a:z], minlength=16))

    if len(f_blocks) < min_blocks:
        return None

    return {
        "function": torch.tensor(f_blocks, dtype=torch.long),
        "root": torch.tensor(root_blocks, dtype=torch.long),
        "template": torch.tensor(template_blocks, dtype=torch.long),
        "path": str(path),
        "n_events": T,
        "n_blocks": len(f_blocks),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_pt", type=str, required=True)
    parser.add_argument("--out_csv", type=str, default=None)
    parser.add_argument("--events_per_block", type=int, default=8)
    parser.add_argument("--phrase_period", type=int, default=4)
    parser.add_argument("--min_blocks", type=int, default=8)
    parser.add_argument("--max_files", type=int, default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    files = sorted(data_dir.rglob("*.pt"))

    if args.max_files is not None:
        files = files[: args.max_files]

    print("[data_dir]", data_dir)
    print("[num files]", len(files))

    sequences = []
    skipped = 0

    for i, p in enumerate(files):
        item = convert_file_to_blocks(
            p,
            events_per_block=args.events_per_block,
            min_blocks=args.min_blocks,
        )

        if item is None:
            skipped += 1
            continue

        sequences.append(item)

        if len(sequences) <= 5 or len(sequences) % 100 == 0:
            names = [FUNCTION_NAMES[int(v)] for v in item["function"][:16].tolist()]
            print("[seq]", len(sequences), "blocks:", item["n_blocks"], "func:", " - ".join(names))

    out = {
        "sequences": sequences,
        "events_per_block": args.events_per_block,
        "phrase_period": args.phrase_period,
        "function_names": FUNCTION_NAMES,
        "label_map": {
            "T": 0,
            "D": 1,
            "SD": 2,
            "OTHER": 3,
        },
    }

    out_pt = Path(args.out_pt)
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_pt)

    print("[done]")
    print("saved sequences:", len(sequences))
    print("skipped:", skipped)
    print("out_pt:", out_pt)

    out_csv = args.out_csv
    if out_csv is None:
        out_csv = str(out_pt).replace(".pt", "_summary.csv")

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(
            fcsv,
            fieldnames=["idx", "path", "n_events", "n_blocks", "function_sequence"],
        )
        writer.writeheader()

        for i, item in enumerate(sequences):
            funcs = [FUNCTION_NAMES[int(v)] for v in item["function"].tolist()]
            writer.writerow({
                "idx": i,
                "path": item["path"],
                "n_events": item["n_events"],
                "n_blocks": item["n_blocks"],
                "function_sequence": "-".join(funcs),
            })

    print("out_csv:", out_csv)


if __name__ == "__main__":
    main()
