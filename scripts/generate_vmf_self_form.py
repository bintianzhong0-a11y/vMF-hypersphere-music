# -*- coding: utf-8 -*-
"""
Self-forming vMF accompaniment generation.

This script does NOT require a fixed chord progression such as C,F,G,C.
Instead, it uses a function-aware checkpoint to generate a harmonic function
sequence first:

    T / D / SD / OTHER

Then it maps the generated function sequence to chords and generates MIDI notes
using vMF directional agreement.

Function labels:
    0: T
    1: D
    2: SD
    3: OTHER

Transition labels:
    prev_function * 4 + current_function
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import pretty_midi

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from models.vmf_conformer_function import VMFConformerFunction
from models.vmf_coordinates import make_note_direction


FUNCTION_NAMES = {
    0: "T",
    1: "D",
    2: "SD",
    3: "OTHER",
}

PC_TO_NAME_SHARP = {
    0: "C",
    1: "C#",
    2: "D",
    3: "Eb",
    4: "E",
    5: "F",
    6: "F#",
    7: "G",
    8: "Ab",
    9: "A",
    10: "Bb",
    11: "B",
}

KEY_TO_PC = {
    "C": 0, "C#": 1, "Db": 1,
    "D": 2, "D#": 3, "Eb": 3,
    "E": 4,
    "F": 5, "F#": 6, "Gb": 6,
    "G": 7, "G#": 8, "Ab": 8,
    "A": 9, "A#": 10, "Bb": 10,
    "B": 11,
}


def sample_from_logits(logits: torch.Tensor, temperature: float = 1.0) -> int:
    if temperature <= 0:
        return int(torch.argmax(logits).item())

    probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def chord_tones(root_pc: int, quality: str) -> List[int]:
    if quality == "maj":
        rel = [0, 4, 7]
    elif quality == "min":
        rel = [0, 3, 7]
    elif quality == "dim":
        rel = [0, 3, 6]
    elif quality == "sus":
        rel = [0, 5, 7]
    elif quality == "7":
        rel = [0, 4, 7, 10]
    elif quality == "maj7":
        rel = [0, 4, 7, 11]
    elif quality == "min7":
        rel = [0, 3, 7, 10]
    else:
        rel = [0, 4, 7]

    return [(root_pc + r) % 12 for r in rel]


def quality_to_template(quality: str) -> int:
    """
    Match preprocess labels:
        0 other
        1 maj
        2 min
        3 dim
        4 aug
        5 sus
        6 dom7
        7 maj7
        8 min7
    """
    return {
        "maj": 1,
        "min": 2,
        "dim": 3,
        "sus": 5,
        "7": 6,
        "maj7": 7,
        "min7": 8,
    }.get(quality, 0)


def function_candidate_chords(function_id: int, key_pc: int) -> List[Tuple[int, str]]:
    """
    Returns candidate chords for each function in a major-key style interpretation.

    T:
        I, vi, iii
    SD:
        IV, ii
    D:
        V, V7, vii°
    OTHER:
        bVII, bVI, iiø-ish alternatives
    """
    if function_id == 0:      # T
        rels = [(0, "maj"), (9, "min"), (4, "min")]
    elif function_id == 2:    # SD
        rels = [(5, "maj"), (2, "min"), (5, "maj7")]
    elif function_id == 1:    # D
        rels = [(7, "maj"), (7, "7"), (11, "dim")]
    else:                     # OTHER
        rels = [(10, "maj"), (8, "maj"), (3, "maj"), (6, "dim")]

    return [((key_pc + rel) % 12, quality) for rel, quality in rels]


def chord_name(root_pc: int, quality: str) -> str:
    name = PC_TO_NAME_SHARP[root_pc]
    if quality == "maj":
        return name
    if quality == "min":
        return name + "m"
    if quality == "dim":
        return name + "dim"
    if quality == "sus":
        return name + "sus"
    if quality == "7":
        return name + "7"
    if quality == "maj7":
        return name + "maj7"
    if quality == "min7":
        return name + "m7"
    return name


def build_context_tensor(
    pitches: List[int],
    ticks: List[int],
    seq_len: int,
    ticks_per_beat: int = 480,
    beats_per_bar: int = 4,
) -> torch.Tensor:
    if len(pitches) == 0:
        pitches = [60, 64, 67, 72]
        ticks = [0, 120, 240, 360]

    use_pitches = pitches[-seq_len:]
    use_ticks = ticks[-seq_len:]

    prev = [use_pitches[0]] + use_pitches[:-1]

    x = make_note_direction(
        midi_pitch=torch.tensor(use_pitches, dtype=torch.long),
        prev_midi_pitch=torch.tensor(prev, dtype=torch.long),
        onset_tick=torch.tensor(use_ticks, dtype=torch.long),
        ticks_per_beat=ticks_per_beat,
        beats_per_bar=beats_per_bar,
        include_register=True,
    ).float()

    return x.unsqueeze(0)


def load_model(checkpoint: str, device: torch.device):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    meta = ckpt["meta"]
    args = ckpt.get("args", {})

    model = VMFConformerFunction(
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
    )

    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device)
    model.eval()

    return model, meta, args







@torch.no_grad()
def choose_next_function(
    model,
    pitches: List[int],
    ticks: List[int],
    prev_function: int,
    device: torch.device,
    seq_len: int,
    temperature: float,
    transition_weight: float,
    function_weight: float,
    function_history: List[int],
    same_function_penalty: float,
    function_run_penalty: float,
    max_same_function: int,
    dominant_to_tonic_bonus: float,
    block_index: int,
    phrase_period: int,
    phrase_start_d_penalty: float,
    phrase_start_t_bonus: float,
    phrase_start_sd_bonus: float,
    subdominant_bonus: float,
    tonic_to_subdominant_bonus: float,
    other_penalty: float,
    phrase_start_after_d_t_bonus: float,
    phrase_start_extra_sd_bonus: float,
    phrase_start_other_penalty: float,
) -> Tuple[int, Dict]:
    x = build_context_tensor(pitches, ticks, seq_len=seq_len).to(device)
    padding_mask = torch.zeros(x.shape[:2], dtype=torch.bool, device=device)

    outputs = model(x, padding_mask=padding_mask)

    f_logits = outputs["function_logits"][0, -1].detach().cpu()
    ft_logits = outputs["function_transition_logits"][0, -1].detach().cpu()

    # transition class = prev_function * 4 + current_function
    trans_scores = torch.stack([
        ft_logits[prev_function * 4 + cur]
        for cur in range(4)
    ])

    combined = transition_weight * trans_scores + function_weight * f_logits

    # Same-function suppression.
    run_len = 0
    if len(function_history) > 0:
        for f in reversed(function_history):
            if int(f) == int(prev_function):
                run_len += 1
            else:
                break

        penalty = same_function_penalty + function_run_penalty * max(0, run_len - 1)
        combined[prev_function] -= penalty

        if max_same_function > 0 and run_len >= max_same_function:
            combined[prev_function] = -1.0e9
    else:
        penalty = 0.0

    # Normal D -> T resolution bias.
    d_to_t_bonus_applied = 0.0
    if int(prev_function) == 1:  # D
        combined[0] += dominant_to_tonic_bonus  # T
        d_to_t_bonus_applied = float(dominant_to_tonic_bonus)

    # Normal SD boost.
    combined[2] += subdominant_bonus

    # T -> SD boost.
    t_to_sd_bonus_applied = 0.0
    if int(prev_function) == 0:  # T
        combined[2] += tonic_to_subdominant_bonus
        t_to_sd_bonus_applied = float(tonic_to_subdominant_bonus)

    # Avoid too much OTHER.
    combined[3] -= other_penalty

    # Phrase-start bias:
    # At every 4-bar phrase start, avoid D/OTHER and favor T/SD.
    phrase_start_bias_applied = False
    phrase_start_after_d_t_bonus_applied = 0.0
    phrase_start_extra_sd_bonus_applied = 0.0
    phrase_start_other_penalty_applied = 0.0

    if phrase_period > 0 and int(block_index) % int(phrase_period) == 0:
        phrase_start_bias_applied = True

        # Existing phrase-start prior.
        combined[1] -= phrase_start_d_penalty   # D down
        combined[0] += phrase_start_t_bonus     # T up
        combined[2] += phrase_start_sd_bonus    # SD up

        # New: at phrase start, make SD more likely.
        combined[2] += phrase_start_extra_sd_bonus
        phrase_start_extra_sd_bonus_applied = float(phrase_start_extra_sd_bonus)

        # New: at phrase start, suppress OTHER.
        combined[3] -= phrase_start_other_penalty
        phrase_start_other_penalty_applied = float(phrase_start_other_penalty)

        # New: if previous function is D, strongly resolve to T at phrase start.
        if int(prev_function) == 1:
            combined[0] += phrase_start_after_d_t_bonus
            phrase_start_after_d_t_bonus_applied = float(phrase_start_after_d_t_bonus)

            # Also make D continuation less attractive at the phrase head.
            combined[1] -= 0.5 * phrase_start_after_d_t_bonus

    next_function = sample_from_logits(combined, temperature=temperature)

    info = {
        "function_logits": f_logits.tolist(),
        "transition_scores_from_prev": trans_scores.tolist(),
        "combined_scores_after_penalty": combined.tolist(),
        "prev_function": int(prev_function),
        "same_function_run_len": int(run_len),
        "same_function_penalty_applied": float(penalty),
        "dominant_to_tonic_bonus_applied": d_to_t_bonus_applied,
        "tonic_to_subdominant_bonus_applied": float(t_to_sd_bonus_applied),
        "block_index": int(block_index),
        "phrase_period": int(phrase_period),
        "phrase_start_bias_applied": bool(phrase_start_bias_applied),
        "phrase_start_d_penalty": float(phrase_start_d_penalty) if phrase_start_bias_applied else 0.0,
        "phrase_start_t_bonus": float(phrase_start_t_bonus) if phrase_start_bias_applied else 0.0,
        "phrase_start_sd_bonus": float(phrase_start_sd_bonus) if phrase_start_bias_applied else 0.0,
        "phrase_start_after_d_t_bonus_applied": phrase_start_after_d_t_bonus_applied,
        "phrase_start_extra_sd_bonus_applied": phrase_start_extra_sd_bonus_applied,
        "phrase_start_other_penalty_applied": phrase_start_other_penalty_applied,
        "subdominant_bonus": float(subdominant_bonus),
        "other_penalty": float(other_penalty),
        "selected_function": int(next_function),
    }

    return next_function, info

@torch.no_grad()
def choose_chord_for_function(
    model,
    pitches: List[int],
    ticks: List[int],
    function_id: int,
    key_pc: int,
    device: torch.device,
    seq_len: int,
    temperature: float,
    prev_chord_root: int | None,
    prev_chord_quality: str | None,
    same_chord_penalty: float,
    same_root_penalty: float,
) -> Tuple[int, str, Dict]:
    candidates = function_candidate_chords(function_id, key_pc)

    x = build_context_tensor(pitches, ticks, seq_len=seq_len).to(device)
    padding_mask = torch.zeros(x.shape[:2], dtype=torch.bool, device=device)

    outputs = model(x, padding_mask=padding_mask)

    root_logits = outputs["root_logits"][0, -1].detach().cpu()
    template_logits = outputs["template_logits"][0, -1].detach().cpu()

    scores = []
    penalty_infos = []

    for root_pc, quality in candidates:
        template_id = quality_to_template(quality)
        s = root_logits[root_pc]

        if 0 <= template_id < template_logits.numel():
            s = s + 0.7 * template_logits[template_id]

        penalty = 0.0

        # Penalize same root, e.g. C -> C, G -> G.
        if prev_chord_root is not None and int(root_pc) == int(prev_chord_root):
            penalty += same_root_penalty

        # Penalize exactly same chord, e.g. C -> C, Em -> Em.
        if (
            prev_chord_root is not None
            and prev_chord_quality is not None
            and int(root_pc) == int(prev_chord_root)
            and str(quality) == str(prev_chord_quality)
        ):
            penalty += same_chord_penalty

        s = s - penalty

        scores.append(s)
        penalty_infos.append(float(penalty))

    scores = torch.stack(scores)
    idx = sample_from_logits(scores, temperature=temperature)

    root_pc, quality = candidates[idx]

    info = {
        "candidate_chords": [
            {
                "root_pc": int(r),
                "quality": q,
                "name": chord_name(r, q),
                "score_after_penalty": float(scores[j].item()),
                "penalty": float(penalty_infos[j]),
            }
            for j, (r, q) in enumerate(candidates)
        ],
        "selected_index": int(idx),
        "prev_chord_root": None if prev_chord_root is None else int(prev_chord_root),
        "prev_chord_quality": prev_chord_quality,
    }

    return root_pc, quality, info

@torch.no_grad()
def choose_next_pitch(
    model,
    pitches: List[int],
    ticks: List[int],
    root_pc: int,
    quality: str,
    device: torch.device,
    seq_len: int,
    onset_tick: int,
    pitch_min: int,
    pitch_max: int,
    temperature: float,
    agreement_weight: float,
    chord_weight: float,
    repeat_penalty: float,
    leap_penalty: float,
    same_dir_penalty: float,
) -> int:
    x = build_context_tensor(pitches, ticks, seq_len=seq_len).to(device)
    padding_mask = torch.zeros(x.shape[:2], dtype=torch.bool, device=device)

    outputs = model(x, padding_mask=padding_mask)

    mu = outputs["mu"][0, -1].detach().cpu()
    mu = mu / (torch.linalg.norm(mu) + 1e-8)

    prev_pitch = pitches[-1] if len(pitches) > 0 else 60
    prev_interval = 0
    if len(pitches) >= 2:
        prev_interval = pitches[-1] - pitches[-2]

    tones = set(chord_tones(root_pc, quality))

    cand_pitches = list(range(pitch_min, pitch_max + 1))
    cand_prev = [prev_pitch for _ in cand_pitches]
    cand_ticks = [onset_tick for _ in cand_pitches]

    e = make_note_direction(
        midi_pitch=torch.tensor(cand_pitches, dtype=torch.long),
        prev_midi_pitch=torch.tensor(cand_prev, dtype=torch.long),
        onset_tick=torch.tensor(cand_ticks, dtype=torch.long),
        ticks_per_beat=480,
        beats_per_bar=4,
        include_register=True,
    ).float()

    e = e / (torch.linalg.norm(e, dim=-1, keepdim=True) + 1e-8)
    agreement = e @ mu

    scores = agreement_weight * agreement

    for i, p in enumerate(cand_pitches):
        pc = p % 12
        interval = p - prev_pitch

        if pc in tones:
            scores[i] += chord_weight
        else:
            scores[i] -= 0.5 * chord_weight

        if p == prev_pitch:
            scores[i] -= repeat_penalty

        if abs(interval) > 7:
            scores[i] -= leap_penalty * (abs(interval) - 7)

        if len(pitches) >= 2:
            if prev_interval > 0 and interval > 0:
                scores[i] -= same_dir_penalty
            elif prev_interval < 0 and interval < 0:
                scores[i] -= same_dir_penalty

    idx = sample_from_logits(scores, temperature=temperature)
    return int(cand_pitches[idx])


def velocity_from_pitch_context(
    pitch: int,
    root_pc: int,
    quality: str,
    base: int = 76,
) -> int:
    tones = set(chord_tones(root_pc, quality))
    if pitch % 12 in tones:
        return min(105, base + 8)
    return base


def render_midi(
    notes: List[Dict],
    out_midi: str,
    tempo: float,
    add_block_chords: bool = True,
):
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)

    melody = pretty_midi.Instrument(program=0, name="vMF Self-Form Melody")
    accomp = pretty_midi.Instrument(program=0, name="vMF Self-Form Chords")

    step_sec = 60.0 / tempo / 2.0  # eighth-note grid

    for n in notes:
        start = n["step"] * step_sec
        end = start + step_sec * 0.9

        melody.notes.append(
            pretty_midi.Note(
                velocity=int(n["velocity"]),
                pitch=int(n["pitch"]),
                start=float(start),
                end=float(end),
            )
        )

        if add_block_chords and n.get("is_block_start", False):
            root_pc = n["root_pc"]
            quality = n["quality"]
            tones = chord_tones(root_pc, quality)

            for pc in tones:
                # place chord around lower-mid register
                pitch = 48 + pc
                while pitch < 48:
                    pitch += 12
                while pitch > 64:
                    pitch -= 12

                accomp.notes.append(
                    pretty_midi.Note(
                        velocity=58,
                        pitch=int(pitch),
                        start=float(start),
                        end=float(start + step_sec * n["block_steps"] * 0.95),
                    )
                )

    pm.instruments.append(melody)
    pm.instruments.append(accomp)

    out_path = Path(out_midi)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(out_path))


def compute_stats(notes: List[Dict]) -> Dict:
    pitches = [n["pitch"] for n in notes]

    if len(pitches) <= 1:
        intervals = []
    else:
        intervals = [pitches[i] - pitches[i - 1] for i in range(1, len(pitches))]

    repeat_rate = 0.0
    leap_rate = 0.0
    avg_abs_interval = 0.0

    if intervals:
        repeat_rate = sum(1 for d in intervals if d == 0) / len(intervals)
        leap_rate = sum(1 for d in intervals if abs(d) > 7) / len(intervals)
        avg_abs_interval = sum(abs(d) for d in intervals) / len(intervals)

    func_seq = [n["function"] for n in notes if n.get("is_block_start", False)]
    chord_seq = [n["chord"] for n in notes if n.get("is_block_start", False)]

    return {
        "n_notes": len(notes),
        "unique_pitch": len(set(pitches)),
        "pitch_min": min(pitches) if pitches else None,
        "pitch_max": max(pitches) if pitches else None,
        "repeat_rate": repeat_rate,
        "avg_abs_interval": avg_abs_interval,
        "leap_rate_gt_7": leap_rate,
        "function_sequence": func_seq,
        "function_name_sequence": [FUNCTION_NAMES[f] for f in func_seq],
        "chord_sequence": chord_seq,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out_midi", type=str, required=True)
    parser.add_argument("--out_json", type=str, required=True)

    parser.add_argument("--key", type=str, default="C")
    parser.add_argument("--blocks", type=int, default=16)
    parser.add_argument("--steps_per_block", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=128)

    parser.add_argument("--tempo", type=float, default=120.0)
    parser.add_argument("--temperature_function", type=float, default=0.75)
    parser.add_argument("--temperature_chord", type=float, default=0.65)
    parser.add_argument("--temperature_pitch", type=float, default=0.50)

    parser.add_argument("--transition_weight", type=float, default=1.0)
    parser.add_argument("--function_weight", type=float, default=0.5)
    parser.add_argument("--same_function_penalty", type=float, default=1.2)
    parser.add_argument("--function_run_penalty", type=float, default=0.8)
    parser.add_argument("--max_same_function", type=int, default=2)
    parser.add_argument("--dominant_to_tonic_bonus", type=float, default=0.8)
    parser.add_argument("--phrase_period", type=int, default=4)
    parser.add_argument("--phrase_start_d_penalty", type=float, default=1.2)
    parser.add_argument("--phrase_start_t_bonus", type=float, default=0.8)
    parser.add_argument("--phrase_start_sd_bonus", type=float, default=0.5)
    parser.add_argument("--subdominant_bonus", type=float, default=0.5)
    parser.add_argument("--tonic_to_subdominant_bonus", type=float, default=0.6)
    parser.add_argument("--other_penalty", type=float, default=0.2)
    parser.add_argument("--phrase_start_after_d_t_bonus", type=float, default=1.3)
    parser.add_argument("--phrase_start_extra_sd_bonus", type=float, default=0.8)
    parser.add_argument("--phrase_start_other_penalty", type=float, default=0.4)

    parser.add_argument("--same_chord_penalty", type=float, default=1.2)
    parser.add_argument("--same_root_penalty", type=float, default=0.4)

    parser.add_argument("--pitch_min", type=int, default=48)
    parser.add_argument("--pitch_max", type=int, default=84)

    parser.add_argument("--agreement_weight", type=float, default=2.5)
    parser.add_argument("--chord_weight", type=float, default=1.4)
    parser.add_argument("--repeat_penalty", type=float, default=0.9)
    parser.add_argument("--leap_penalty", type=float, default=0.20)
    parser.add_argument("--same_dir_penalty", type=float, default=0.25)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    key_pc = KEY_TO_PC.get(args.key, 0)

    model, meta, ckpt_args = load_model(args.checkpoint, device=device)

    # Seed context only. This is not a fixed progression.
    # It gives the model a tonal starting point.
    seed_root = key_pc
    seed_pitches = [60 + seed_root, 64 + seed_root, 67 + seed_root]
    seed_pitches = [p if p <= 84 else p - 12 for p in seed_pitches]
    pitches = seed_pitches[:]
    ticks = [0, 120, 240]

    prev_function = 0  # start from T-like context, then model self-forms.
    function_history = []

    prev_chord_root = None
    prev_chord_quality = None

    notes = []
    block_infos = []

    global_step = 0

    for b in range(args.blocks):
        next_function, f_info = choose_next_function(
            model=model,
            pitches=pitches,
            ticks=ticks,
            prev_function=prev_function,
            device=device,
            seq_len=args.seq_len,
            temperature=args.temperature_function,
            transition_weight=args.transition_weight,
            function_weight=args.function_weight,
            function_history=function_history,
            same_function_penalty=args.same_function_penalty,
            function_run_penalty=args.function_run_penalty,
            max_same_function=args.max_same_function,
            dominant_to_tonic_bonus=args.dominant_to_tonic_bonus,
            block_index=b,
            phrase_period=args.phrase_period,
            phrase_start_d_penalty=args.phrase_start_d_penalty,
            phrase_start_t_bonus=args.phrase_start_t_bonus,
            phrase_start_sd_bonus=args.phrase_start_sd_bonus,
            subdominant_bonus=args.subdominant_bonus,
            tonic_to_subdominant_bonus=args.tonic_to_subdominant_bonus,
            other_penalty=args.other_penalty,
            phrase_start_after_d_t_bonus=args.phrase_start_after_d_t_bonus,
            phrase_start_extra_sd_bonus=args.phrase_start_extra_sd_bonus,
            phrase_start_other_penalty=args.phrase_start_other_penalty,
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

        cname = chord_name(root_pc, quality)

        block_infos.append({
            "block": b,
            "function": int(next_function),
            "function_name": FUNCTION_NAMES[int(next_function)],
            "root_pc": int(root_pc),
            "quality": quality,
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

            vel = velocity_from_pitch_context(pitch, root_pc, quality)

            pitches.append(pitch)
            ticks.append(onset_tick)

            notes.append({
                "step": global_step,
                "pitch": int(pitch),
                "velocity": int(vel),
                "function": int(next_function),
                "function_name": FUNCTION_NAMES[int(next_function)],
                "root_pc": int(root_pc),
                "quality": quality,
                "chord": cname,
                "is_block_start": s == 0,
                "block_steps": args.steps_per_block,
            })

            global_step += 1

        prev_function = next_function
        function_history.append(int(next_function))
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
        "temperature_function": args.temperature_function,
        "temperature_chord": args.temperature_chord,
        "temperature_pitch": args.temperature_pitch,
        "same_function_penalty": args.same_function_penalty,
        "function_run_penalty": args.function_run_penalty,
        "max_same_function": args.max_same_function,
        "dominant_to_tonic_bonus": args.dominant_to_tonic_bonus,
        "phrase_period": args.phrase_period,
        "phrase_start_d_penalty": args.phrase_start_d_penalty,
        "phrase_start_t_bonus": args.phrase_start_t_bonus,
        "phrase_start_sd_bonus": args.phrase_start_sd_bonus,
        "subdominant_bonus": args.subdominant_bonus,
        "tonic_to_subdominant_bonus": args.tonic_to_subdominant_bonus,
        "other_penalty": args.other_penalty,
        "phrase_start_after_d_t_bonus": args.phrase_start_after_d_t_bonus,
        "phrase_start_extra_sd_bonus": args.phrase_start_extra_sd_bonus,
        "phrase_start_other_penalty": args.phrase_start_other_penalty,
        "same_chord_penalty": args.same_chord_penalty,
        "same_root_penalty": args.same_root_penalty,
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
