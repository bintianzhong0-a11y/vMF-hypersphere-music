# -*- coding: utf-8 -*-
"""
Full arrangement generation with Conformer block transition + quota_empirical decoding.

Main preset:
    quota_empirical

Target distribution:
    T     = 0.4615
    D     = 0.1378
    SD    = 0.1928
    OTHER = 0.2079

Arrangement tracks:
    1. melody
    2. chord comping
    3. bass
    4. arpeggio
    5. pad
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import pretty_midi

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from scripts.generate_vmf_self_form_conformer_block import (
    FUNCTION_NAMES,
    KEY_TO_PC,
    load_conformer_block_model,
    choose_next_function_conformer_block,
    make_seed_context,
    quality_to_template_id,
)

from scripts.generate_vmf_self_form import (
    choose_chord_for_function,
    choose_next_pitch,
    compute_stats,
)


def pc_to_midi(pc: int, base: int = 60, low: int = 48, high: int = 84) -> int:
    pc = int(pc) % 12
    p = base + ((pc - base) % 12)

    while p > high:
        p -= 12
    while p < low:
        p += 12

    return int(p)


def chord_intervals(quality: str) -> List[int]:
    q = str(quality)

    if q == "maj":
        return [0, 4, 7]
    if q == "min":
        return [0, 3, 7]
    if q == "dim":
        return [0, 3, 6]
    if q == "aug":
        return [0, 4, 8]
    if q == "sus":
        return [0, 5, 7]
    if q == "7":
        return [0, 4, 7, 10]
    if q == "maj7":
        return [0, 4, 7, 11]
    if q == "min7":
        return [0, 3, 7, 10]

    return [0, 4, 7]


def chord_pitches(
    root_pc: int,
    quality: str,
    base: int = 60,
    low: int = 52,
    high: int = 76,
) -> List[int]:
    root = pc_to_midi(root_pc, base=base, low=low, high=high)
    intervals = chord_intervals(quality)

    notes = []
    for iv in intervals:
        p = root + iv
        while p > high:
            p -= 12
        while p < low:
            p += 12
        notes.append(int(p))

    notes = sorted(set(notes))

    # Avoid too low inversion.
    if len(notes) >= 3 and notes[0] < low + 2:
        notes[0] += 12
        notes = sorted(notes)

    return notes


def add_note(inst, pitch: int, start: float, end: float, velocity: int):
    if end <= start:
        end = start + 0.05

    inst.notes.append(
        pretty_midi.Note(
            velocity=int(max(1, min(127, velocity))),
            pitch=int(max(0, min(127, pitch))),
            start=float(start),
            end=float(end),
        )
    )


def render_full_arrangement(
    notes: List[Dict],
    block_infos: List[Dict],
    out_midi: str,
    tempo: float = 120.0,
    steps_per_block: int = 8,
    add_melody: bool = True,
    add_chords: bool = True,
    add_bass: bool = True,
    add_arp: bool = True,
    add_pad: bool = True,
):
    pm = pretty_midi.PrettyMIDI(initial_tempo=float(tempo))

    melody_inst = pretty_midi.Instrument(program=0, name="vMF Melody Piano")
    chord_inst = pretty_midi.Instrument(program=0, name="vMF Chord Comp")
    bass_inst = pretty_midi.Instrument(program=32, name="vMF Bass")
    arp_inst = pretty_midi.Instrument(program=0, name="vMF Arpeggio")
    pad_inst = pretty_midi.Instrument(program=48, name="vMF Pad")

    quarter = 60.0 / float(tempo)
    step_dur = quarter / 2.0
    block_dur = step_dur * steps_per_block

    # 1. Melody
    if add_melody:
        for n in notes:
            step = int(n["step"])
            start = step * step_dur
            end = start + step_dur * 0.90

            is_block_start = bool(n.get("is_block_start", False))
            velocity = int(n.get("velocity", 78))
            if is_block_start:
                velocity += 8

            add_note(
                melody_inst,
                pitch=int(n["pitch"]),
                start=start,
                end=end,
                velocity=velocity,
            )

    # 2. Arrangement per block
    for binfo in block_infos:
        b = int(binfo["block"])
        root_pc = int(binfo["root_pc"])
        quality = str(binfo["quality"])
        function_name = str(binfo["function_name"])

        block_start = b * block_dur
        block_end = block_start + block_dur

        cps_mid = chord_pitches(
            root_pc=root_pc,
            quality=quality,
            base=60,
            low=55,
            high=76,
        )

        cps_high = chord_pitches(
            root_pc=root_pc,
            quality=quality,
            base=67,
            low=60,
            high=84,
        )

        # velocity depends on function
        if function_name == "T":
            chord_vel = 58
            bass_vel = 74
            pad_vel = 38
        elif function_name == "D":
            chord_vel = 68
            bass_vel = 84
            pad_vel = 42
        elif function_name == "SD":
            chord_vel = 64
            bass_vel = 78
            pad_vel = 44
        else:
            chord_vel = 55
            bass_vel = 70
            pad_vel = 36

        # 2a. Pad, sustained
        if add_pad:
            pad_notes = chord_pitches(
                root_pc=root_pc,
                quality=quality,
                base=60,
                low=60,
                high=79,
            )

            for p in pad_notes:
                add_note(
                    pad_inst,
                    pitch=p,
                    start=block_start,
                    end=block_end * 0.98 + block_start * 0.02,
                    velocity=pad_vel,
                )

        # 2b. Chord comping
        if add_chords:
            # 8-step bar: on 0, 3, 6 gives light pop syncopation.
            comp_steps = [0, 3, 6]
            comp_lengths = [1.6, 1.2, 1.4]

            for cs, clen in zip(comp_steps, comp_lengths):
                st = block_start + cs * step_dur
                en = min(block_end, st + clen * step_dur)

                for p in cps_mid:
                    add_note(
                        chord_inst,
                        pitch=p,
                        start=st,
                        end=en,
                        velocity=chord_vel,
                    )

        # 2c. Bass
        if add_bass:
            bass_root = pc_to_midi(
                root_pc,
                base=36,
                low=36,
                high=52,
            )

            fifth_pc = (root_pc + 7) % 12
            bass_fifth = pc_to_midi(
                fifth_pc,
                base=36,
                low=36,
                high=52,
            )

            # root on beat 1, fifth/root on beat 3
            add_note(
                bass_inst,
                pitch=bass_root,
                start=block_start,
                end=block_start + 3.8 * step_dur,
                velocity=bass_vel,
            )

            add_note(
                bass_inst,
                pitch=bass_fifth,
                start=block_start + 4 * step_dur,
                end=block_end,
                velocity=max(55, bass_vel - 6),
            )

            # dominant blocks get an extra approach tone.
            if function_name == "D":
                approach = bass_root - 1
                if approach < 35:
                    approach += 12

                add_note(
                    bass_inst,
                    pitch=approach,
                    start=block_start + 7 * step_dur,
                    end=block_end,
                    velocity=max(50, bass_vel - 10),
                )

        # 2d. Arpeggio
        if add_arp:
            arp_seq = cps_high[:]
            if len(arp_seq) >= 3:
                arp_pattern = [
                    arp_seq[0],
                    arp_seq[1],
                    arp_seq[2],
                    arp_seq[1],
                    arp_seq[0],
                    arp_seq[-1],
                    arp_seq[1],
                    arp_seq[2],
                ]
            else:
                arp_pattern = arp_seq * 8

            arp_pattern = arp_pattern[:steps_per_block]

            for s, p in enumerate(arp_pattern):
                st = block_start + s * step_dur
                en = st + 0.75 * step_dur

                # make arpeggio softer than melody
                add_note(
                    arp_inst,
                    pitch=p,
                    start=st,
                    end=en,
                    velocity=max(35, chord_vel - 18),
                )

    if add_melody:
        pm.instruments.append(melody_inst)
    if add_chords:
        pm.instruments.append(chord_inst)
    if add_bass:
        pm.instruments.append(bass_inst)
    if add_arp:
        pm.instruments.append(arp_inst)
    if add_pad:
        pm.instruments.append(pad_inst)

    out_path = Path(out_midi)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(out_path))


def function_name_to_id(name: str) -> int:
    name = str(name).upper()
    if name == "T":
        return 0
    if name == "D":
        return 1
    if name == "SD":
        return 2
    if name == "OTHER":
        return 3
    return 0


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

    # quota_empirical preset defaults
    parser.add_argument("--temperature_function", type=float, default=1.02)
    parser.add_argument("--temperature_chord", type=float, default=0.82)
    parser.add_argument("--temperature_pitch", type=float, default=0.52)
    parser.add_argument("--max_same_function", type=int, default=2)

    parser.add_argument("--target_t_ratio", type=float, default=0.4615)
    parser.add_argument("--target_d_ratio", type=float, default=0.1378)
    parser.add_argument("--target_sd_ratio", type=float, default=0.1928)
    parser.add_argument("--target_other_ratio", type=float, default=0.2079)

    parser.add_argument("--target_distribution_weight", type=float, default=3.0)
    parser.add_argument("--quota_distribution_weight", type=float, default=4.0)
    parser.add_argument("--quota_overuse_penalty", type=float, default=3.0)

    parser.add_argument("--sd_coverage_weight", type=float, default=2.5)
    parser.add_argument("--t_overuse_penalty", type=float, default=1.8)
    parser.add_argument("--d_sd_balance_weight", type=float, default=1.5)
    parser.add_argument("--d_over_sd_penalty", type=float, default=0.8)

    parser.add_argument("--phrase_start_d_penalty", type=float, default=0.8)
    parser.add_argument("--phrase_start_sd_bonus", type=float, default=0.5)

    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--tempo", type=float, default=120.0)

    parser.add_argument("--pitch_min", type=int, default=48)
    parser.add_argument("--pitch_max", type=int, default=84)

    parser.add_argument("--agreement_weight", type=float, default=2.5)
    parser.add_argument("--chord_weight", type=float, default=1.4)
    parser.add_argument("--repeat_penalty", type=float, default=0.9)
    parser.add_argument("--leap_penalty", type=float, default=0.25)
    parser.add_argument("--same_dir_penalty", type=float, default=0.25)

    parser.add_argument("--same_chord_penalty", type=float, default=1.3)
    parser.add_argument("--same_root_penalty", type=float, default=0.5)

    parser.add_argument("--no_melody", action="store_true")
    parser.add_argument("--no_chords", action="store_true")
    parser.add_argument("--no_bass", action="store_true")
    parser.add_argument("--no_arp", action="store_true")
    parser.add_argument("--no_pad", action="store_true")

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

    model, ckpt = load_conformer_block_model(
        args.checkpoint,
        device=device,
    )

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

    render_full_arrangement(
        notes=notes,
        block_infos=block_infos,
        out_midi=args.out_midi,
        tempo=args.tempo,
        steps_per_block=args.steps_per_block,
        add_melody=not args.no_melody,
        add_chords=not args.no_chords,
        add_bass=not args.no_bass,
        add_arp=not args.no_arp,
        add_pad=not args.no_pad,
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
        "preset": "quota_empirical_full_arrangement",
        "temperature_function": args.temperature_function,
        "temperature_chord": args.temperature_chord,
        "temperature_pitch": args.temperature_pitch,
        "max_same_function": args.max_same_function,
        "target_t_ratio": args.target_t_ratio,
        "target_d_ratio": args.target_d_ratio,
        "target_sd_ratio": args.target_sd_ratio,
        "target_other_ratio": args.target_other_ratio,
        "target_distribution_weight": args.target_distribution_weight,
        "quota_distribution_weight": args.quota_distribution_weight,
        "quota_overuse_penalty": args.quota_overuse_penalty,
        "sd_coverage_weight": args.sd_coverage_weight,
        "t_overuse_penalty": args.t_overuse_penalty,
        "d_sd_balance_weight": args.d_sd_balance_weight,
        "d_over_sd_penalty": args.d_over_sd_penalty,
        "phrase_start_d_penalty": args.phrase_start_d_penalty,
        "phrase_start_sd_bonus": args.phrase_start_sd_bonus,
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
    print("[preset] quota_empirical_full_arrangement")
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
