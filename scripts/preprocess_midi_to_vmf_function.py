# -*- coding: utf-8 -*-
"""
MIDI -> vMF .pt dataset with harmonic function labels.

Output keys:
    x
    mu
    root
    chord_like
    template
    triad
    seventh
    beat
    bar
    onset
    velocity
    timing
    duration
    function
    function_transition
    midi_path
    estimated_key_pc

Function labels:
    0: T
    1: D
    2: SD
    3: OTHER

Transition:
    prev_function * 4 + current_function
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))


import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import pretty_midi

from models.vmf_coordinates import make_note_direction


TICKS_PER_BEAT = 480
BEATS_PER_BAR = 4

FUNCTION_T = 0
FUNCTION_D = 1
FUNCTION_SD = 2
FUNCTION_OTHER = 3


def estimate_tick(pm: pretty_midi.PrettyMIDI, time_sec: float) -> int:
    try:
        return int(pm.time_to_tick(time_sec))
    except Exception:
        return int(time_sec * 2.0 * TICKS_PER_BEAT)


def get_all_notes(pm: pretty_midi.PrettyMIDI):
    notes = []

    for inst in pm.instruments:
        if inst.is_drum:
            continue

        for n in inst.notes:
            if n.end <= n.start:
                continue
            notes.append(n)

    notes.sort(key=lambda n: (n.start, n.pitch, n.end))
    return notes


def estimate_key_pc(notes) -> int:
    """
    Simple key/tonic estimation by duration-weighted pitch-class histogram.

    This is intentionally lightweight.
    Later, this can be replaced by a more serious key estimator.
    """
    hist = torch.zeros(12)

    for n in notes:
        pc = n.pitch % 12
        dur = max(0.01, n.end - n.start)
        hist[pc] += float(dur)

    return int(hist.argmax().item())


def function_from_root(root_pc: int, tonic_pc: int) -> int:
    """
    Map root relative to estimated tonic into T / D / SD / OTHER.

    Major-key style heuristic:
        T  : I, iii, vi        -> {0, 4, 9}
        D  : V, vii            -> {7, 11}
        SD : ii, IV            -> {2, 5}
    """
    rel = (int(root_pc) - int(tonic_pc)) % 12

    if rel in {0, 4, 9}:
        return FUNCTION_T
    if rel in {7, 11}:
        return FUNCTION_D
    if rel in {2, 5}:
        return FUNCTION_SD
    return FUNCTION_OTHER


def classify_chord_template(pitches: List[int], root_pc: int) -> Tuple[int, int, int]:
    """
    template:
        0 other/none
        1 maj
        2 min
        3 dim
        4 aug
        5 sus
        6 dom7
        7 maj7
        8 min7

    triad:
        0 none/other
        1 maj
        2 min
        3 dim
        4 aug
        5 sus

    seventh:
        0 none
        1 dom7
        2 maj7
        3 min7
        4 m7b5
        5 other7
    """
    if len(pitches) == 0:
        return 0, 0, 0

    pcs = sorted(set([p % 12 for p in pitches]))
    rel = set([(pc - root_pc) % 12 for pc in pcs])

    has_maj = {0, 4, 7}.issubset(rel)
    has_min = {0, 3, 7}.issubset(rel)
    has_dim = {0, 3, 6}.issubset(rel)
    has_aug = {0, 4, 8}.issubset(rel)
    has_sus = {0, 5, 7}.issubset(rel) or {0, 2, 7}.issubset(rel)

    has_b7 = 10 in rel
    has_M7 = 11 in rel

    triad = 0
    template = 0

    if has_maj:
        triad = 1
        template = 1
    elif has_min:
        triad = 2
        template = 2
    elif has_dim:
        triad = 3
        template = 3
    elif has_aug:
        triad = 4
        template = 4
    elif has_sus:
        triad = 5
        template = 5

    seventh = 0

    if has_b7:
        if has_maj:
            template = 6
            seventh = 1
        elif has_min:
            template = 8
            seventh = 3
        elif has_dim:
            seventh = 4
        else:
            seventh = 5

    elif has_M7:
        if has_maj:
            template = 7
            seventh = 2
        else:
            seventh = 5

    return template, triad, seventh


def build_onset_groups(pm: pretty_midi.PrettyMIDI, notes, grid_ticks: int = 120):
    groups: Dict[int, List[int]] = {}
    note_ticks = []

    for n in notes:
        tick = estimate_tick(pm, n.start)
        qtick = int(round(tick / grid_ticks) * grid_ticks)
        note_ticks.append((tick, qtick))
        groups.setdefault(qtick, []).append(n.pitch)

    return groups, note_ticks


def local_mu(x: torch.Tensor, window: int = 8) -> torch.Tensor:
    T, D = x.shape
    mus = []

    for t in range(T):
        a = max(0, t - window)
        b = min(T, t + window + 1)
        chunk = x[a:b]
        mu = chunk.mean(dim=0)
        mu = mu / (torch.linalg.norm(mu) + 1e-8)
        mus.append(mu)

    return torch.stack(mus, dim=0)


def process_midi_file(path: Path, min_notes: int = 8):
    try:
        pm = pretty_midi.PrettyMIDI(str(path))
    except Exception as e:
        print(f"[warn] failed to read {path}: {e}")
        return None

    notes = get_all_notes(pm)

    if len(notes) < min_notes:
        return None

    estimated_key_pc = estimate_key_pc(notes)

    groups, note_ticks = build_onset_groups(pm, notes, grid_ticks=120)

    midi_pitch = []
    prev_pitch = []
    onset_tick = []
    velocity = []
    duration = []
    timing = []

    root = []
    chord_like = []
    template = []
    triad = []
    seventh = []
    beat = []
    bar = []
    onset = []

    function = []
    function_transition = []

    last_pitch = notes[0].pitch
    last_function = FUNCTION_T

    for n, (tick, qtick) in zip(notes, note_ticks):
        pitches_same_onset = groups.get(qtick, [n.pitch])

        # Very lightweight root heuristic:
        # use the lowest note at the onset.
        root_pc = min(pitches_same_onset) % 12

        temp_label, triad_label, seventh_label = classify_chord_template(
            pitches_same_onset,
            root_pc=root_pc,
        )

        f_label = function_from_root(root_pc, estimated_key_pc)
        ft_label = last_function * 4 + f_label

        beat_idx = int((qtick // TICKS_PER_BEAT) % BEATS_PER_BAR)
        bar_idx = beat_idx

        midi_pitch.append(n.pitch)
        prev_pitch.append(last_pitch)
        onset_tick.append(qtick)

        velocity.append(n.velocity / 127.0)
        duration.append(max(0.0, n.end - n.start))
        timing.append((tick - qtick) / float(TICKS_PER_BEAT))

        root.append(root_pc)
        chord_like.append(1.0 if len(pitches_same_onset) >= 3 else 0.0)
        template.append(temp_label)
        triad.append(triad_label)
        seventh.append(seventh_label)
        beat.append(beat_idx)
        bar.append(bar_idx)
        onset.append(1.0)

        function.append(f_label)
        function_transition.append(ft_label)

        last_pitch = n.pitch
        last_function = f_label

    midi_pitch_t = torch.tensor(midi_pitch, dtype=torch.long)
    prev_pitch_t = torch.tensor(prev_pitch, dtype=torch.long)
    onset_tick_t = torch.tensor(onset_tick, dtype=torch.long)

    x = make_note_direction(
        midi_pitch=midi_pitch_t,
        prev_midi_pitch=prev_pitch_t,
        onset_tick=onset_tick_t,
        ticks_per_beat=TICKS_PER_BEAT,
        beats_per_bar=BEATS_PER_BAR,
        include_register=True,
    ).float()

    mu = local_mu(x, window=8).float()

    dur = torch.tensor(duration, dtype=torch.float32).unsqueeze(-1)
    dur = torch.clamp(dur / 2.0, 0.0, 4.0)

    out = {
        "x": x,
        "mu": mu,

        "root": torch.tensor(root, dtype=torch.long),
        "chord_like": torch.tensor(chord_like, dtype=torch.float32),
        "template": torch.tensor(template, dtype=torch.long),
        "triad": torch.tensor(triad, dtype=torch.long),
        "seventh": torch.tensor(seventh, dtype=torch.long),

        "beat": torch.tensor(beat, dtype=torch.long),
        "bar": torch.tensor(bar, dtype=torch.long),
        "onset": torch.tensor(onset, dtype=torch.float32),

        "velocity": torch.tensor(velocity, dtype=torch.float32).unsqueeze(-1),
        "timing": torch.tensor(timing, dtype=torch.float32).unsqueeze(-1),
        "duration": dur,

        "function": torch.tensor(function, dtype=torch.long),
        "function_transition": torch.tensor(function_transition, dtype=torch.long),

        "estimated_key_pc": int(estimated_key_pc),
        "midi_path": str(path),
    }

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--midi_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--min_notes", type=int, default=8)
    args = parser.parse_args()

    midi_dir = Path(args.midi_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    midi_files = []
    for ext in ["*.mid", "*.midi", "*.MID", "*.MIDI"]:
        midi_files.extend(midi_dir.rglob(ext))

    midi_files = sorted(set(midi_files))

    if args.max_files is not None:
        midi_files = midi_files[: args.max_files]

    print("[midi_dir]", midi_dir)
    print("[out_dir]", out_dir)
    print("[num midi]", len(midi_files))

    if len(midi_files) == 0:
        raise FileNotFoundError(f"No MIDI files found in {midi_dir}")

    ok = 0
    skip = 0

    for i, path in enumerate(midi_files):
        obj = process_midi_file(path, min_notes=args.min_notes)

        if obj is None:
            skip += 1
            continue

        stem = path.stem.replace(" ", "_").replace("/", "_")
        out_path = out_dir / f"{i:06d}_{stem}.pt"

        torch.save(obj, out_path)
        ok += 1

        if ok <= 5 or ok % 100 == 0:
            T, D = obj["x"].shape
            print(
                f"[saved] {out_path} "
                f"T={T} D={D} key={obj['estimated_key_pc']}"
            )

    print("[done]")
    print("saved:", ok)
    print("skipped:", skip)
    print("out_dir:", out_dir)


if __name__ == "__main__":
    main()
