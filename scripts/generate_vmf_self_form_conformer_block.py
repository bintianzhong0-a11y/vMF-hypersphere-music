# -*- coding: utf-8 -*-
"""
Self-forming MIDI generation using Conformer-mediated block-level function transition.

Uses:
    vmf_conformer_block_transition_pop1k7_1000_finetune_from_head_best.pt

Generation:
    note-level vMF context
        -> Conformer hidden states
        -> average last generated block
        -> block_transition_head
        -> next function
        -> chord selection
        -> vMF pitch generation
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from models.vmf_conformer_block_transition import VMFConformerBlockTransition
from models.vmf_coordinates import make_note_direction

from scripts.generate_vmf_self_form import (
    FUNCTION_NAMES,
    KEY_TO_PC,
    choose_chord_for_function,
    choose_next_pitch,
    render_midi,
    compute_stats,
)


def sample_from_logits(logits: torch.Tensor, temperature: float = 1.0) -> int:
    if temperature <= 0:
        return int(torch.argmax(logits).item())

    probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def function_name_to_id(name: str) -> int:
    name = name.upper()
    if name == "T":
        return 0
    if name == "D":
        return 1
    if name == "SD":
        return 2
    if name == "OTHER":
        return 3
    return 0


def quality_to_template_id(quality: str) -> int:
    if quality == "maj":
        return 1
    if quality == "min":
        return 2
    if quality == "dim":
        return 3
    if quality == "aug":
        return 4
    if quality == "sus":
        return 5
    if quality == "7":
        return 6
    if quality == "maj7":
        return 7
    if quality == "min7":
        return 8
    return 0



def build_generation_context(
    pitches: List[int],
    ticks: List[int],
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Build [1, T, D] vMF note-direction context from generated notes.

    Important:
        models/vmf_coordinates.py expects torch.Tensor inputs.
        Therefore MIDI pitch / previous pitch / onset_tick are wrapped as tensors.
    """
    use_pitches = pitches[-seq_len:]
    use_ticks = ticks[-seq_len:]

    feats = []

    for i, p in enumerate(use_pitches):
        if i == 0:
            prev_p = use_pitches[i]
        else:
            prev_p = use_pitches[i - 1]

        onset_tick = int(use_ticks[i]) if i < len(use_ticks) else i * 120

        midi_pitch = torch.tensor([int(p)], dtype=torch.long)
        prev_midi_pitch = torch.tensor([int(prev_p)], dtype=torch.long)
        onset_tick_tensor = torch.tensor([int(onset_tick)], dtype=torch.long)

        # Current vmf_coordinates.py signature uses midi_pitch / prev_midi_pitch.
        e = make_note_direction(
            midi_pitch=midi_pitch,
            prev_midi_pitch=prev_midi_pitch,
            onset_tick=onset_tick_tensor,
        )

        if not torch.is_tensor(e):
            e = torch.tensor(e, dtype=torch.float32)
        else:
            e = e.detach().float().cpu()

        # If e is [1, D], squeeze to [D].
        if e.ndim == 2 and e.shape[0] == 1:
            e = e[0]

        feats.append(e)

    x = torch.stack(feats, dim=0).unsqueeze(0).to(device)
    return x

def load_conformer_block_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    meta = ckpt["meta"]
    pretrained_args = ckpt.get("pretrained_args", {})
    train_args = ckpt.get("args", {})

    model = VMFConformerBlockTransition(
        input_dim=meta["input_dim"],
        mu_dim=meta["mu_dim"],
        d_model=pretrained_args.get("d_model", 128),
        num_layers=pretrained_args.get("num_layers", 4),
        n_heads=pretrained_args.get("n_heads", 4),
        dropout=pretrained_args.get("dropout", 0.1),
        conv_kernel_size=pretrained_args.get("conv_kernel_size", 15),
        num_root=meta["num_root"],
        num_template=meta["num_template"],
        num_triad=meta["num_triad"],
        num_seventh=meta["num_seventh"],
        num_beat=meta["num_beat"],
        num_bar=meta["num_bar"],
        num_function=meta.get("num_function", 4),
        num_function_transition=meta.get("num_function_transition", 16),
        phrase_period=int(ckpt.get("phrase_period", train_args.get("phrase_period", 4))),
        block_emb_dim=int(train_args.get("block_emb_dim", 32)),
        block_hidden_dim=int(train_args.get("block_hidden_dim", 128)),
        num_block_template=16,
    )

    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=True)

    print("[load conformer block transition]")
    print("missing:", missing)
    print("unexpected:", unexpected)

    model.to(device)
    model.eval()

    return model, ckpt


@torch.no_grad()

@torch.no_grad()

@torch.no_grad()

@torch.no_grad()

@torch.no_grad()
def choose_next_function_conformer_block(
    model: VMFConformerBlockTransition,
    pitches: List[int],
    ticks: List[int],
    prev_function: int,
    phrase_pos: int,
    prev_root: int,
    prev_template: int,
    device: torch.device,
    seq_len: int,
    events_per_block: int,
    temperature: float,
    max_same_function: int,
    function_history: List[int],
    target_sd_ratio: float,
    sd_coverage_weight: float,
    target_t_ratio: float,
    t_overuse_penalty: float,
    phrase_start_d_penalty: float,
    phrase_start_sd_bonus: float,
    d_sd_balance_weight: float,
    d_over_sd_penalty: float,
    target_d_ratio: float,
    target_other_ratio: float,
    target_distribution_weight: float,
    total_blocks: int,
    quota_distribution_weight: float,
    quota_overuse_penalty: float,
) -> Tuple[int, Dict]:
    """
    Conformer block transition decoding with:
      1. learned Conformer logits
      2. target distribution control
      3. quota-aware control using remaining block count
      4. SD coverage and D/SD balance control
    """
    x = build_generation_context(
        pitches=pitches,
        ticks=ticks,
        seq_len=seq_len,
        device=device,
    )

    base_outputs = model.forward(
        x,
        padding_mask=None,
        return_hidden=True,
    )

    hidden = base_outputs["hidden"]

    if hidden.shape[1] >= events_per_block:
        last_block_hidden = hidden[:, -events_per_block:, :].mean(dim=1)
    else:
        last_block_hidden = hidden.mean(dim=1)

    pf = torch.tensor([[prev_function]], dtype=torch.long, device=device)
    pp = torch.tensor([[phrase_pos]], dtype=torch.long, device=device)
    pr = torch.tensor([[prev_root % 12]], dtype=torch.long, device=device)
    pt = torch.tensor([[prev_template]], dtype=torch.long, device=device)

    logits = model.block_transition_head(
        block_hidden_prev=last_block_hidden.unsqueeze(1),
        prev_function=pf,
        phrase_pos=pp,
        prev_root=pr,
        prev_template=pt,
    )[0, 0].detach().cpu()

    adjusted = logits.clone()

    # Current counts.
    counts = torch.zeros(4, dtype=torch.float32)
    for f in function_history:
        if int(f) in [0, 1, 2, 3]:
            counts[int(f)] += 1.0

    n_hist = max(len(function_history), 1)
    current_ratios = counts / float(n_hist)

    current_t_ratio = float(current_ratios[0])
    current_d_ratio = float(current_ratios[1])
    current_sd_ratio = float(current_ratios[2])
    current_other_ratio = float(current_ratios[3])

    # Target distribution.
    target = torch.tensor(
        [
            float(target_t_ratio),
            float(target_d_ratio),
            float(target_sd_ratio),
            float(target_other_ratio),
        ],
        dtype=torch.float32,
    )
    target = target / target.sum().clamp_min(1e-8)

    # A. Ratio-based target distribution control.
    ratio_gap = target - current_ratios
    distribution_adjustment = float(target_distribution_weight) * ratio_gap
    adjusted += distribution_adjustment

    # B. Quota-aware control.
    # For 16 blocks, target_counts roughly means:
    # T≈7.4, D≈2.7, SD≈3.2, OTHER≈2.7, etc.
    total_blocks_f = max(float(total_blocks), 1.0)
    target_counts = target * total_blocks_f

    remaining_including_current = max(int(total_blocks) - len(function_history), 1)

    shortage = target_counts - counts
    quota_signal = shortage / float(remaining_including_current)

    # If a class is under quota, raise it.
    # If it is over quota, this naturally lowers it.
    quota_adjustment = float(quota_distribution_weight) * quota_signal
    adjusted += quota_adjustment

    # Additional overuse penalty for already overshot classes.
    overuse = torch.clamp(counts - target_counts, min=0.0) / total_blocks_f
    quota_overuse_adjustment = -float(quota_overuse_penalty) * overuse
    adjusted += quota_overuse_adjustment

    # C. Additional SD coverage control.
    sd_deficit = max(0.0, float(target_sd_ratio) - float(current_sd_ratio))
    sd_bonus = float(sd_coverage_weight) * sd_deficit
    adjusted[2] += sd_bonus

    # D. T overuse control.
    t_excess = max(0.0, float(current_t_ratio) - float(target_t_ratio))
    t_penalty = float(t_overuse_penalty) * t_excess
    adjusted[0] -= t_penalty

    # E. D/SD balance control.
    d_minus_sd = float(current_d_ratio) - float(current_sd_ratio)
    d_sd_gap = max(0.0, d_minus_sd)

    d_sd_bonus = float(d_sd_balance_weight) * d_sd_gap
    d_penalty = float(d_over_sd_penalty) * d_sd_gap

    adjusted[2] += d_sd_bonus
    adjusted[1] -= d_penalty

    # F. Phrase start control.
    phrase_d_penalty_applied = 0.0
    phrase_sd_bonus_applied = 0.0

    if int(phrase_pos) == 0:
        adjusted[1] -= float(phrase_start_d_penalty)
        adjusted[2] += float(phrase_start_sd_bonus)
        phrase_d_penalty_applied = float(phrase_start_d_penalty)
        phrase_sd_bonus_applied = float(phrase_start_sd_bonus)

    # G. Same-function safeguard.
    run_len = 0
    if len(function_history) > 0:
        for f in reversed(function_history):
            if int(f) == int(prev_function):
                run_len += 1
            else:
                break

    if max_same_function > 0 and run_len >= max_same_function:
        adjusted[prev_function] = -1.0e9

    next_function = sample_from_logits(adjusted, temperature=temperature)

    raw_probs = torch.softmax(logits, dim=-1)
    adjusted_probs = torch.softmax(adjusted, dim=-1)

    info = {
        "prev_function": int(prev_function),
        "prev_function_name": FUNCTION_NAMES[int(prev_function)],
        "phrase_pos": int(phrase_pos),
        "prev_root": int(prev_root),
        "prev_template": int(prev_template),
        "raw_logits": logits.tolist(),
        "raw_probs": raw_probs.tolist(),
        "adjusted_logits": adjusted.tolist(),
        "adjusted_probs": adjusted_probs.tolist(),
        "same_function_run_len": int(run_len),
        "history_counts": {
            "T": int(counts[0].item()),
            "D": int(counts[1].item()),
            "SD": int(counts[2].item()),
            "OTHER": int(counts[3].item()),
        },
        "current_ratios": {
            "T": current_t_ratio,
            "D": current_d_ratio,
            "SD": current_sd_ratio,
            "OTHER": current_other_ratio,
        },
        "target_ratios": {
            "T": float(target[0]),
            "D": float(target[1]),
            "SD": float(target[2]),
            "OTHER": float(target[3]),
        },
        "target_counts": {
            "T": float(target_counts[0]),
            "D": float(target_counts[1]),
            "SD": float(target_counts[2]),
            "OTHER": float(target_counts[3]),
        },
        "shortage": {
            "T": float(shortage[0]),
            "D": float(shortage[1]),
            "SD": float(shortage[2]),
            "OTHER": float(shortage[3]),
        },
        "quota_signal": quota_signal.tolist(),
        "quota_adjustment": quota_adjustment.tolist(),
        "quota_overuse_adjustment": quota_overuse_adjustment.tolist(),
        "target_distribution_weight": float(target_distribution_weight),
        "quota_distribution_weight": float(quota_distribution_weight),
        "quota_overuse_penalty": float(quota_overuse_penalty),
        "target_sd_ratio": float(target_sd_ratio),
        "sd_deficit": float(sd_deficit),
        "sd_bonus_applied": float(sd_bonus),
        "target_t_ratio": float(target_t_ratio),
        "t_excess": float(t_excess),
        "t_penalty_applied": float(t_penalty),
        "d_minus_sd": float(d_minus_sd),
        "d_sd_gap": float(d_sd_gap),
        "d_sd_bonus_applied": float(d_sd_bonus),
        "d_penalty_applied": float(d_penalty),
        "phrase_start_d_penalty_applied": float(phrase_d_penalty_applied),
        "phrase_start_sd_bonus_applied": float(phrase_sd_bonus_applied),
        "selected_function": int(next_function),
        "selected_function_name": FUNCTION_NAMES[int(next_function)],
    }

    return int(next_function), info

def make_seed_context(key_pc: int, pitch_min: int, pitch_max: int):
    """
    8 notes seed so the first Conformer block is meaningful.
    """
    pcs = [
        key_pc,
        (key_pc + 4) % 12,
        (key_pc + 7) % 12,
        key_pc,
        (key_pc + 7) % 12,
        (key_pc + 4) % 12,
        key_pc,
        (key_pc + 2) % 12,
    ]

    pitches = []
    for pc in pcs:
        p = 60 + pc
        while p > pitch_max:
            p -= 12
        while p < pitch_min:
            p += 12
        pitches.append(int(p))

    ticks = [i * 120 for i in range(len(pitches))]
    return pitches, ticks


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out_midi", type=str, required=True)
    parser.add_argument("--out_json", type=str, required=True)

    parser.add_argument("--key", type=str, default="C")
    parser.add_argument("--blocks", type=int, default=16)
    parser.add_argument("--steps_per_block", type=int, default=8)
    parser.add_argument("--events_per_block", type=int, default=8)
    parser.add_argument("--phrase_period", type=int, default=4)

    parser.add_argument("--start_function", type=str, default="T")

    parser.add_argument("--temperature_function", type=float, default=0.85)
    parser.add_argument("--temperature_chord", type=float, default=0.70)
    parser.add_argument("--temperature_pitch", type=float, default=0.50)

    parser.add_argument("--max_same_function", type=int, default=2)
    parser.add_argument("--target_sd_ratio", type=float, default=0.0)
    parser.add_argument("--sd_coverage_weight", type=float, default=0.0)
    parser.add_argument("--target_t_ratio", type=float, default=0.70)
    parser.add_argument("--t_overuse_penalty", type=float, default=0.0)
    parser.add_argument("--phrase_start_d_penalty", type=float, default=0.0)
    parser.add_argument("--phrase_start_sd_bonus", type=float, default=0.0)
    parser.add_argument("--d_sd_balance_weight", type=float, default=0.0)
    parser.add_argument("--d_over_sd_penalty", type=float, default=0.0)
    parser.add_argument("--target_d_ratio", type=float, default=0.1378)
    parser.add_argument("--target_other_ratio", type=float, default=0.2079)
    parser.add_argument("--target_distribution_weight", type=float, default=0.0)
    parser.add_argument("--quota_distribution_weight", type=float, default=0.0)
    parser.add_argument("--quota_overuse_penalty", type=float, default=0.0)

    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--tempo", type=float, default=120.0)

    parser.add_argument("--pitch_min", type=int, default=48)
    parser.add_argument("--pitch_max", type=int, default=84)

    parser.add_argument("--agreement_weight", type=float, default=2.5)
    parser.add_argument("--chord_weight", type=float, default=1.4)
    parser.add_argument("--repeat_penalty", type=float, default=0.9)
    parser.add_argument("--leap_penalty", type=float, default=0.25)
    parser.add_argument("--same_dir_penalty", type=float, default=0.25)

    parser.add_argument("--same_chord_penalty", type=float, default=1.2)
    parser.add_argument("--same_root_penalty", type=float, default=0.4)

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

    key_pc = KEY_TO_PC.get(args.key, 0)

    model, ckpt = load_conformer_block_model(args.checkpoint, device=device)

    pitches, ticks = make_seed_context(
        key_pc=key_pc,
        pitch_min=args.pitch_min,
        pitch_max=args.pitch_max,
    )

    prev_function = function_name_to_id(args.start_function)
    prev_root = key_pc
    prev_template = 1

    prev_chord_root = None
    prev_chord_quality = None

    function_history = []
    notes = []
    block_infos = []

    global_step = 0

    for b in range(args.blocks):
        phrase_pos = b % args.phrase_period

        next_function, f_info = choose_next_function_conformer_block(
            model=model,
            pitches=pitches,
            ticks=ticks,
            prev_function=prev_function,
            phrase_pos=phrase_pos,
            prev_root=prev_root,
            prev_template=prev_template,
            device=device,
            seq_len=args.seq_len,
            events_per_block=args.events_per_block,
            temperature=args.temperature_function,
            max_same_function=args.max_same_function,
            function_history=function_history,
            target_sd_ratio=args.target_sd_ratio,
            sd_coverage_weight=args.sd_coverage_weight,
            target_t_ratio=args.target_t_ratio,
            t_overuse_penalty=args.t_overuse_penalty,
            phrase_start_d_penalty=args.phrase_start_d_penalty,
            phrase_start_sd_bonus=args.phrase_start_sd_bonus,
            d_sd_balance_weight=args.d_sd_balance_weight,
            d_over_sd_penalty=args.d_over_sd_penalty,
            target_d_ratio=args.target_d_ratio,
            target_other_ratio=args.target_other_ratio,
            target_distribution_weight=args.target_distribution_weight,
            total_blocks=args.blocks,
            quota_distribution_weight=args.quota_distribution_weight,
            quota_overuse_penalty=args.quota_overuse_penalty,
        )

        root_pc, quality, c_info = choose_chord_for_function(
            model=model,
            pitches=pitches,
            ticks=ticks,
            function_id=next_function,
            key_pc=key_pc,
            device=device,
            seq_len=args.seq_len,
            temperature=args.temperature_chord,
            prev_chord_root=prev_chord_root,
            prev_chord_quality=prev_chord_quality,
            same_chord_penalty=args.same_chord_penalty,
            same_root_penalty=args.same_root_penalty,
        )

        template_id = quality_to_template_id(quality)

        cname = c_info["candidate_chords"][c_info["selected_index"]]["name"]

        block_infos.append({
            "block": int(b),
            "phrase_pos": int(phrase_pos),
            "function": int(next_function),
            "function_name": FUNCTION_NAMES[int(next_function)],
            "root_pc": int(root_pc),
            "quality": str(quality),
            "template_id": int(template_id),
            "chord": cname,
            "function_debug": f_info,
            "chord_debug": c_info,
        })

        for s in range(args.steps_per_block):
            onset_tick = global_step * 120

            pitch = choose_next_pitch(
                model=model,
                pitches=pitches,
                ticks=ticks,
                root_pc=root_pc,
                quality=quality,
                device=device,
                seq_len=args.seq_len,
                onset_tick=onset_tick,
                pitch_min=args.pitch_min,
                pitch_max=args.pitch_max,
                temperature=args.temperature_pitch,
                agreement_weight=args.agreement_weight,
                chord_weight=args.chord_weight,
                repeat_penalty=args.repeat_penalty,
                leap_penalty=args.leap_penalty,
                same_dir_penalty=args.same_dir_penalty,
            )

            chord_pcs = {
                root_pc,
                (root_pc + 3) % 12,
                (root_pc + 4) % 12,
                (root_pc + 7) % 12,
            }

            vel = 84 if pitch % 12 in chord_pcs else 76

            pitches.append(int(pitch))
            ticks.append(int(onset_tick))

            notes.append({
                "step": int(global_step),
                "pitch": int(pitch),
                "velocity": int(vel),
                "function": int(next_function),
                "function_name": FUNCTION_NAMES[int(next_function)],
                "root_pc": int(root_pc),
                "quality": str(quality),
                "chord": cname,
                "is_block_start": bool(s == 0),
                "block_steps": int(args.steps_per_block),
            })

            global_step += 1

        function_history.append(int(next_function))

        prev_function = int(next_function)
        prev_root = int(root_pc)
        prev_template = int(template_id)
        prev_chord_root = int(root_pc)
        prev_chord_quality = str(quality)

    render_midi(
        notes=notes,
        out_midi=args.out_midi,
        tempo=args.tempo,
        add_block_chords=True,
    )

    stats = compute_stats(notes)
    stats.update({
        "checkpoint": args.checkpoint,
        "out_midi": args.out_midi,
        "key": args.key,
        "blocks": args.blocks,
        "steps_per_block": args.steps_per_block,
        "events_per_block": args.events_per_block,
        "phrase_period": args.phrase_period,
        "temperature_function": args.temperature_function,
        "temperature_chord": args.temperature_chord,
        "temperature_pitch": args.temperature_pitch,
        "max_same_function": args.max_same_function,
        "target_sd_ratio": args.target_sd_ratio,
        "sd_coverage_weight": args.sd_coverage_weight,
        "target_t_ratio": args.target_t_ratio,
        "t_overuse_penalty": args.t_overuse_penalty,
        "phrase_start_d_penalty": args.phrase_start_d_penalty,
        "phrase_start_sd_bonus": args.phrase_start_sd_bonus,
        "d_sd_balance_weight": args.d_sd_balance_weight,
        "d_over_sd_penalty": args.d_over_sd_penalty,
        "target_d_ratio": args.target_d_ratio,
        "target_other_ratio": args.target_other_ratio,
        "target_distribution_weight": args.target_distribution_weight,
        "quota_distribution_weight": args.quota_distribution_weight,
        "quota_overuse_penalty": args.quota_overuse_penalty,
        "block_infos": block_infos,
    })

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("[saved midi]", args.out_midi)
    print("[saved json]", args.out_json)
    print("[function sequence]", " - ".join(stats["function_name_sequence"]))
    print("[chord sequence]", " - ".join(stats["chord_sequence"]))
    print(json.dumps({
        "n_notes": stats["n_notes"],
        "unique_pitch": stats["unique_pitch"],
        "pitch_min": stats["pitch_min"],
        "pitch_max": stats["pitch_max"],
        "repeat_rate": stats["repeat_rate"],
        "avg_abs_interval": stats["avg_abs_interval"],
        "leap_rate_gt_7": stats["leap_rate_gt_7"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
