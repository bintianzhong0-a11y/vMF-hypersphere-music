# -*- coding: utf-8 -*-
"""
vMF Hypersphere Music Representation
====================================

Utilities for mapping symbolic music information to hyperspherical
vMF-style direction vectors.

Main components:
- MIDI pitch -> pitch class
- pitch class -> circle-of-fifths index
- pitch / fifth / transition / beat positions -> circular coordinates
- note feature vector -> L2-normalized hypersphere direction
- harmonic center direction mu_t
- directional agreement A_{t,i} = e_{t,i}^T mu_t

This file is intentionally lightweight and PyTorch-first.
"""

from __future__ import annotations

from typing import Optional, Dict, Tuple

import math
import torch
import torch.nn.functional as F


EPS: float = 1.0e-8
TWO_PI: float = 2.0 * math.pi


# ---------------------------------------------------------------------
# Basic pitch / circle-of-fifths utilities
# ---------------------------------------------------------------------

def pitch_class(midi_pitch: torch.Tensor) -> torch.Tensor:
    """
    MIDI pitch m -> pitch class p = m mod 12.

    Args:
        midi_pitch: Tensor of MIDI pitches. Any shape.

    Returns:
        Tensor with same shape, values in {0, ..., 11}.
    """
    return torch.remainder(midi_pitch.long(), 12)


def circle_of_fifths_index(pitch_cls: torch.Tensor) -> torch.Tensor:
    """
    Pitch class p -> circle-of-fifths index q = 7p mod 12.

    In this convention:
        C=0 -> 0
        G=7 -> 1
        D=2 -> 2
        ...
    """
    return torch.remainder(7 * pitch_cls.long(), 12)


def circular_angle(index: torch.Tensor, period: int = 12) -> torch.Tensor:
    """
    Discrete circular index -> angle in radians.
    """
    return TWO_PI * index.float() / float(period)


def circular_unit(angle: torch.Tensor) -> torch.Tensor:
    """
    angle -> [cos(angle), sin(angle)]

    Args:
        angle: Tensor of shape (...)

    Returns:
        Tensor of shape (..., 2)
    """
    return torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)


def pitch_class_unit(midi_pitch: torch.Tensor) -> torch.Tensor:
    """
    MIDI pitch -> pitch class unit vector on the pitch-class circle.
    """
    pc = pitch_class(midi_pitch)
    theta = circular_angle(pc, period=12)
    return circular_unit(theta)


def fifth_unit(midi_pitch: torch.Tensor) -> torch.Tensor:
    """
    MIDI pitch -> circle-of-fifths unit vector.
    """
    pc = pitch_class(midi_pitch)
    q = circle_of_fifths_index(pc)
    theta5 = circular_angle(q, period=12)
    return circular_unit(theta5)


def fifth_distance(pc_a: torch.Tensor, pc_b: torch.Tensor) -> torch.Tensor:
    """
    Circle-of-fifths distance between two pitch classes.

    d5(pa, pb) = min(|qa - qb|, 12 - |qa - qb|)
    """
    qa = circle_of_fifths_index(pc_a)
    qb = circle_of_fifths_index(pc_b)
    diff = torch.abs(qa.long() - qb.long())
    return torch.minimum(diff, 12 - diff)


# ---------------------------------------------------------------------
# Transition / beat-position utilities
# ---------------------------------------------------------------------

def pitch_transition_class(
    midi_pitch: torch.Tensor,
    prev_midi_pitch: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Pitch-class transition:
        Δp_t = (p_t - p_{t-1}) mod 12

    If prev_midi_pitch is None, transition is set to zero.
    """
    pc = pitch_class(midi_pitch)

    if prev_midi_pitch is None:
        return torch.zeros_like(pc)

    prev_pc = pitch_class(prev_midi_pitch)
    return torch.remainder(pc - prev_pc, 12)


def transition_unit(
    midi_pitch: torch.Tensor,
    prev_midi_pitch: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    MIDI pitch transition -> circular transition vector.
    """
    dp = pitch_transition_class(midi_pitch, prev_midi_pitch)
    theta_delta = circular_angle(dp, period=12)
    return circular_unit(theta_delta)


def normalized_pitch_delta(
    midi_pitch: torch.Tensor,
    prev_midi_pitch: Optional[torch.Tensor] = None,
    scale: float = 12.0,
) -> torch.Tensor:
    """
    Actual pitch interval Δm compressed by tanh:

        d_t = tanh((m_t - m_{t-1}) / scale)

    Returns shape (..., 1).
    """
    if prev_midi_pitch is None:
        delta = torch.zeros_like(midi_pitch).float()
    else:
        delta = midi_pitch.float() - prev_midi_pitch.float()

    return torch.tanh(delta / float(scale)).unsqueeze(-1)


def normalized_pitch_height(
    midi_pitch: torch.Tensor,
    center: float = 60.0,
    scale: float = 24.0,
) -> torch.Tensor:
    """
    Smooth absolute pitch-height feature.

    Middle C = 60 is treated as center by default.
    This is not a circular feature; it gives rough register information.

    Returns shape (..., 1).
    """
    return torch.tanh((midi_pitch.float() - center) / float(scale)).unsqueeze(-1)


def bar_position_angle(
    onset_tick: torch.Tensor,
    bar_start_tick: torch.Tensor | int | float = 0,
    ticks_per_beat: int = 480,
    beats_per_bar: int = 4,
) -> torch.Tensor:
    """
    Onset tick -> angle inside a bar.

        theta_bar = 2π * ((onset - bar_start) mod (beats_per_bar*r)) / (beats_per_bar*r)
    """
    bar_len = int(ticks_per_beat * beats_per_bar)

    if not torch.is_tensor(bar_start_tick):
        bar_start_tick = torch.tensor(
            bar_start_tick,
            device=onset_tick.device,
            dtype=onset_tick.dtype,
        )

    rel = torch.remainder(onset_tick.long() - bar_start_tick.long(), bar_len)
    return TWO_PI * rel.float() / float(bar_len)


def bar_position_unit(
    onset_tick: torch.Tensor,
    bar_start_tick: torch.Tensor | int | float = 0,
    ticks_per_beat: int = 480,
    beats_per_bar: int = 4,
) -> torch.Tensor:
    """
    Onset tick -> [cos(theta_bar), sin(theta_bar)]
    """
    theta = bar_position_angle(
        onset_tick=onset_tick,
        bar_start_tick=bar_start_tick,
        ticks_per_beat=ticks_per_beat,
        beats_per_bar=beats_per_bar,
    )
    return circular_unit(theta)


def beat_position_index(
    onset_tick: torch.Tensor,
    bar_start_tick: torch.Tensor | int | float = 0,
    ticks_per_beat: int = 480,
    beats_per_bar: int = 4,
) -> torch.Tensor:
    """
    Onset tick -> coarse beat index in a bar.

    Returns values in {0, ..., beats_per_bar - 1}.
    """
    bar_len = int(ticks_per_beat * beats_per_bar)

    if not torch.is_tensor(bar_start_tick):
        bar_start_tick = torch.tensor(
            bar_start_tick,
            device=onset_tick.device,
            dtype=onset_tick.dtype,
        )

    rel = torch.remainder(onset_tick.long() - bar_start_tick.long(), bar_len)
    return torch.div(rel, ticks_per_beat, rounding_mode="floor").long()


# ---------------------------------------------------------------------
# Hyperspherical directions
# ---------------------------------------------------------------------

def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = EPS) -> torch.Tensor:
    """
    Stable L2 normalization.
    """
    return x / (torch.linalg.norm(x, dim=dim, keepdim=True) + eps)


def make_note_feature(
    midi_pitch: torch.Tensor,
    prev_midi_pitch: Optional[torch.Tensor] = None,
    onset_tick: Optional[torch.Tensor] = None,
    bar_start_tick: torch.Tensor | int | float = 0,
    ticks_per_beat: int = 480,
    beats_per_bar: int = 4,
    include_register: bool = True,
) -> torch.Tensor:
    """
    Construct a raw note feature vector x_{t,i}.

    Feature layout:
        pitch class unit          : 2 dims
        circle-of-fifths unit     : 2 dims
        transition unit           : 2 dims
        normalized pitch delta    : 1 dim
        register                  : 1 dim, optional
        bar position unit         : 2 dims, optional if onset_tick is given

    Minimum dimension:
        7 dims without register and onset
        8 dims with register but without onset
        10 dims with register and onset

    Args:
        midi_pitch:
            Tensor of MIDI pitches. Shape (...).
        prev_midi_pitch:
            Previous pitch tensor, same shape as midi_pitch.
            If None, transition is zero.
        onset_tick:
            Onset tick tensor, same shape as midi_pitch.
            If None, bar position features are omitted.
        bar_start_tick:
            Bar start tick. Scalar or tensor broadcastable to onset_tick.
        ticks_per_beat:
            MIDI resolution.
        beats_per_bar:
            Time signature numerator.
        include_register:
            Whether to include absolute pitch height.

    Returns:
        Raw feature tensor of shape (..., D).
    """
    feats = [
        pitch_class_unit(midi_pitch),
        fifth_unit(midi_pitch),
        transition_unit(midi_pitch, prev_midi_pitch),
        normalized_pitch_delta(midi_pitch, prev_midi_pitch),
    ]

    if include_register:
        feats.append(normalized_pitch_height(midi_pitch))

    if onset_tick is not None:
        feats.append(
            bar_position_unit(
                onset_tick=onset_tick,
                bar_start_tick=bar_start_tick,
                ticks_per_beat=ticks_per_beat,
                beats_per_bar=beats_per_bar,
            )
        )

    return torch.cat(feats, dim=-1)


def make_note_direction(
    midi_pitch: torch.Tensor,
    prev_midi_pitch: Optional[torch.Tensor] = None,
    onset_tick: Optional[torch.Tensor] = None,
    bar_start_tick: torch.Tensor | int | float = 0,
    ticks_per_beat: int = 480,
    beats_per_bar: int = 4,
    include_register: bool = True,
    eps: float = EPS,
) -> torch.Tensor:
    """
    Construct hyperspherical note direction e_{t,i}.

        e_{t,i} = x_{t,i} / ||x_{t,i}||_2

    Returns:
        Tensor of shape (..., D), L2-normalized.
    """
    x = make_note_feature(
        midi_pitch=midi_pitch,
        prev_midi_pitch=prev_midi_pitch,
        onset_tick=onset_tick,
        bar_start_tick=bar_start_tick,
        ticks_per_beat=ticks_per_beat,
        beats_per_bar=beats_per_bar,
        include_register=include_register,
    )
    return l2_normalize(x, dim=-1, eps=eps)


def harmonic_center_direction(
    note_directions: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    note_dim: int = -2,
    eps: float = EPS,
) -> torch.Tensor:
    """
    Compute harmonic center direction mu_t from note directions.

        mu_t = normalize(sum_i w_{t,i} e_{t,i})

    Expected common shapes:
        note_directions: [T, N, D] or [B, T, N, D]
        weights:         [T, N]    or [B, T, N]
        mask:            [T, N]    or [B, T, N]

    Args:
        note_directions:
            L2-normalized note direction tensor.
        weights:
            Optional nonnegative weights.
        mask:
            Optional boolean or 0/1 mask for valid notes.
        note_dim:
            Dimension corresponding to note index N.
            Default -2 assumes last two dims are [N, D].

    Returns:
        mu: Tensor with note_dim removed. Example:
            [T, D] or [B, T, D]
    """
    e = note_directions

    if weights is None:
        weights = torch.ones(e.shape[:-1], device=e.device, dtype=e.dtype)
    else:
        weights = weights.to(device=e.device, dtype=e.dtype)

    if mask is not None:
        weights = weights * mask.to(device=e.device, dtype=e.dtype)

    # Add feature dimension
    weighted = e * weights.unsqueeze(-1)

    mu = weighted.sum(dim=note_dim)
    return l2_normalize(mu, dim=-1, eps=eps)


def directional_agreement(
    note_directions: torch.Tensor,
    center_direction: torch.Tensor,
) -> torch.Tensor:
    """
    Directional agreement:

        A_{t,i} = e_{t,i}^T mu_t

    Common shapes:
        note_directions:  [T, N, D]
        center_direction: [T, D]
        return:           [T, N]

        note_directions:  [B, T, N, D]
        center_direction: [B, T, D]
        return:           [B, T, N]
    """
    mu = center_direction.unsqueeze(-2)
    return (note_directions * mu).sum(dim=-1)


def candidate_pitch_directions(
    pitch_min: int = 36,
    pitch_max: int = 84,
    prev_pitch: Optional[int] = None,
    onset_tick: Optional[int] = None,
    bar_start_tick: int = 0,
    ticks_per_beat: int = 480,
    beats_per_bar: int = 4,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create candidate MIDI pitches and their vMF directions.

    Useful for generation-time scoring.

    Returns:
        pitches: [M]
        dirs:    [M, D]
    """
    pitches = torch.arange(pitch_min, pitch_max + 1, device=device).long()

    if prev_pitch is None:
        prev = None
    else:
        prev = torch.full_like(pitches, int(prev_pitch))

    if onset_tick is None:
        onset = None
    else:
        onset = torch.full_like(pitches, int(onset_tick))

    dirs = make_note_direction(
        midi_pitch=pitches,
        prev_midi_pitch=prev,
        onset_tick=onset,
        bar_start_tick=bar_start_tick,
        ticks_per_beat=ticks_per_beat,
        beats_per_bar=beats_per_bar,
        include_register=True,
    )
    return pitches, dirs


def chord_tone_mask(
    midi_pitches: torch.Tensor,
    root: torch.Tensor | int,
    intervals: Tuple[int, ...] = (0, 4, 7),
) -> torch.Tensor:
    """
    Return whether each pitch belongs to a root-normalized chord template.

    Example:
        root = 0, intervals=(0,4,7) -> C major chord tones.

    Args:
        midi_pitches:
            Tensor of candidate MIDI pitches.
        root:
            Root pitch class.
        intervals:
            Root-normalized chord intervals.

    Returns:
        Boolean tensor with same shape as midi_pitches.
    """
    pc = pitch_class(midi_pitches)

    if not torch.is_tensor(root):
        root = torch.tensor(root, device=midi_pitches.device, dtype=torch.long)

    root = root.to(device=midi_pitches.device).long()
    rel = torch.remainder(pc - root, 12)

    allowed = torch.tensor(intervals, device=midi_pitches.device, dtype=torch.long)
    return (rel.unsqueeze(-1) == allowed).any(dim=-1)


def describe_feature_dim(include_register: bool = True, include_onset: bool = True) -> Dict[str, int]:
    """
    Return feature dimension breakdown.
    """
    dims = {
        "pitch_class_unit": 2,
        "fifth_unit": 2,
        "transition_unit": 2,
        "normalized_pitch_delta": 1,
    }
    if include_register:
        dims["register"] = 1
    if include_onset:
        dims["bar_position_unit"] = 2
    dims["total"] = sum(dims.values())
    return dims
