# Method Summary: Conformer-vMF Music Representation and Generation

This document summarizes the main mathematical formulation used in the Conformer-vMF prototype.
The method maps pitch, circle-of-fifths, harmonic context, and temporal position onto a hyperspherical representation, and uses a Conformer model to learn harmony-related and generation-related targets.

> GitHub-safe note: equations are written with fenced `math` blocks instead of raw `$$ ... $$` blocks. This avoids Markdown parsing issues on mobile browsers and under automatic translation.

## 1. Symbolic Input and vMF Coordinates

For a MIDI note at time index `t` and note index `i`, let the MIDI pitch be

```math
m_{t,i} \in \{0,1,\dots,127\}.
```

The pitch class is defined by

```math
p_{t,i} = m_{t,i} \bmod 12.
```

To represent the circle-of-fifths position, we use

```math
q_{t,i} = 7p_{t,i} \bmod 12.
```

The pitch-class angle and circle-of-fifths angle are

```math
\theta^{\mathrm{pc}}_{t,i} = \frac{2\pi}{12}p_{t,i},
\qquad
\theta^{5}_{t,i} = \frac{2\pi}{12}q_{t,i}.
```

The corresponding circular coordinates are

```math
u^{\mathrm{pc}}_{t,i}
=
\left(\cos\theta^{\mathrm{pc}}_{t,i},\ \sin\theta^{\mathrm{pc}}_{t,i}\right).
```

```math
u^{5}_{t,i}
=
\left(\cos\theta^{5}_{t,i},\ \sin\theta^{5}_{t,i}\right).
```

Pitch transition is represented by

```math
\Delta m_{t,i}=m_{t,i}-m_{t-1,i'}.
```

```math
\Delta p_{t,i}=(p_{t,i}-p_{t-1,i'})\bmod 12.
```

```math
\theta^{\Delta}_{t,i}=\frac{2\pi}{12}\Delta p_{t,i},
\qquad
u^{\Delta}_{t,i}
=
\left(\cos\theta^{\Delta}_{t,i},\ \sin\theta^{\Delta}_{t,i}\right).
```

A normalized register feature can be written as

```math
h(m_{t,i}) = \tanh\left(\frac{m_{t,i}-m_{\mathrm{ref}}}{s_m}\right),
```

and a normalized pitch-jump feature as

```math
d_{t,i}=\tanh\left(\frac{\Delta m_{t,i}}{12}\right).
```

For beat and bar position, with ticks-per-beat `r`, beats-per-bar `k`, onset tick `o_{t,i}`, and bar-start tick `b_t`, the beat/bar phase is

```math
\theta^{\mathrm{onset}}_{t,i}
=
2\pi\frac{(o_{t,i}-b_t)\bmod kr}{kr}.
```

The onset coordinate is

```math
u^{\mathrm{onset}}_{t,i}
=
\left(\cos\theta^{\mathrm{onset}}_{t,i},\ \sin\theta^{\mathrm{onset}}_{t,i}\right).
```

The raw note feature vector is formed by concatenating pitch, fifth, transition, register, and temporal features:

```math
x_{t,i}
=
\left[
\nu^{\mathrm{pc}}_{t,i},
\nu^{5}_{t,i},
\nu^{\Delta}_{t,i},
h(m_{t,i}),
d_{t,i},
\nu^{\mathrm{onset}}_{t,i},
\cdots
\right].
```

The note direction on the hypersphere is obtained by L2 normalization:

```math
e_{t,i}
=
\frac{x_{t,i}}{\|x_{t,i}\|_2+\varepsilon}.
```

## 2. Harmonic Direction and vMF Distribution

For a group of simultaneous or near-simultaneous notes `C_t`, the harmonic center direction is defined by a weighted spherical mean:

```math
\mu_t
=
\frac{\sum_{i\in C_t}w_{t,i}e_{t,i}}
{\left\|\sum_{i\in C_t}w_{t,i}e_{t,i}\right\|_2+\varepsilon}.
```

The alignment between each note and the current harmonic center is

```math
A_{t,i}=e_{t,i}^{\top}\mu_t.
```

This value is used as a directional harmony score and can also be used to control velocity during generation.

A vMF distribution over a unit vector `e` is written as

```math
p(e\mid \mu,\kappa)
=
C_D(\kappa)\exp\left(\kappa\mu^{\top}e\right),
```

where `mu` is the mean direction and `kappa >= 0` is the concentration parameter.
Large `kappa` means that directions are tightly concentrated around `mu`.

One possible concentration estimate from the mean resultant length is

```math
\bar R_t
=
\left\|\frac{1}{|C_t|}\sum_{i\in C_t}e_{t,i}\right\|_2,
```

```math
\kappa_t
\approx
\frac{\bar R_t(D-\bar R_t^2)}{1-\bar R_t^2+\varepsilon}.
```

## 3. Conformer-vMF Training Formulation

Let the sequence of vMF-based note/group features be

```math
E = (e_1,e_2,\dots,e_T).
```

The Conformer encoder produces contextual hidden states:

```math
H=(h_1,h_2,\dots,h_T)
=
\mathrm{Conformer}(E).
```

The model has multiple prediction heads:

```math
\hat\mu_t = \mathrm{norm}(W_\mu h_t+b_\mu).
```

```math
\hat\kappa_t = \mathrm{softplus}(W_\kappa h_t+b_\kappa).
```

```math
\hat r_t = \mathrm{softmax}(W_rh_t+b_r).
```

```math
\hat y^{\mathrm{cl}}_t = \sigma(W_{\mathrm{cl}}h_t+b_{\mathrm{cl}}).
```

```math
\hat c_t = \mathrm{softmax}(W_ch_t+b_c).
```

```math
\hat\chi_{t,l}=\sigma(W_{\chi,l}h_t+b_{\chi,l}),
\qquad l=0,1,\dots,11.
```

Here, `r_hat` is the root distribution, `y_cl_hat` is the chord-like probability, `c_hat` is the chord/template distribution, and `chi_hat` is the root-normalized interval prediction.

The direction loss is

```math
\mathcal L_{\mathrm{dir}}
=
\frac{1}{T}\sum_{t=1}^{T}
\left(1-\hat\mu_t^{\top}\mu_t\right).
```

The optional vMF negative log-likelihood form is

```math
\mathcal L_{\mathrm{vMF}}
=
-\frac{1}{T}\sum_{t=1}^{T}
\left[
\log C_D(\hat\kappa_t)
+
\hat\kappa_t\hat\mu_t^{\top}\mu_t
\right].
```

The concentration loss is

```math
\mathcal L_{\kappa}
=
\frac{1}{T}\sum_{t=1}^{T}|\hat\kappa_t-\kappa_t|.
```

The root and template losses are cross-entropy losses:

```math
\mathcal L_{\mathrm{root}}
=
\mathrm{CE}(r_t,\hat r_t),
\qquad
\mathcal L_{\mathrm{template}}
=
\mathrm{CE}(c_t,\hat c_t).
```

The chord-like and interval losses are binary cross-entropy losses:

```math
\mathcal L_{\mathrm{cl}}
=
\mathrm{BCE}(y^{\mathrm{cl}}_t,\hat y^{\mathrm{cl}}_t).
```

```math
\mathcal L_{\mathrm{int}}
=
\sum_{l=0}^{11}
\mathrm{BCE}(y^{\mathrm{int}}_{t,l},\hat\chi_{t,l}).
```

For expressive-performance-related quantities, such as velocity and timing residual, regression losses can be used:

```math
\mathcal L_{\mathrm{vel}}
=
\mathrm{SmoothL1}(v_{t,i},\hat v_{t,i}).
```

```math
\mathcal L_{\mathrm{time}}
=
\mathrm{SmoothL1}(\delta^{\mathrm{time}}_{t,i},\hat\delta^{\mathrm{time}}_{t,i}).
```

The total training objective is a weighted multi-task loss:

```math
\mathcal L
=
\lambda_{\mathrm{dir}}\mathcal L_{\mathrm{dir}}
+
\lambda_{\kappa}\mathcal L_{\kappa}
+
\lambda_{\mathrm{root}}\mathcal L_{\mathrm{root}}
+
\lambda_{\mathrm{cl}}\mathcal L_{\mathrm{cl}}
+
\lambda_{\mathrm{template}}\mathcal L_{\mathrm{template}}
+
\lambda_{\mathrm{int}}\mathcal L_{\mathrm{int}}
+
\lambda_{\mathrm{vel}}\mathcal L_{\mathrm{vel}}
+
\lambda_{\mathrm{time}}\mathcal L_{\mathrm{time}}
+
\cdots.
```

## 4. Optional P2OT Prototype Lens

P2OT is treated as an optional structural lens rather than a required generation module.
The Conformer representation can be read as a distribution over prototypes:

```math
\gamma_t \in \Delta^{K-1},
\qquad
\gamma_{t,k}\geq0,
\qquad
\sum_{k=1}^{K}\gamma_{t,k}=1.
```

A prototype-regularized representation can be written as

```math
z_t
=
\sum_{k=1}^{K}\gamma_{t,k}p_k,
```

where `p_k` is the `k`-th prototype.
Entropy regularization controls whether the prototype distribution is sharp or diffuse:

```math
\mathcal L_{\mathrm{proto}}
=
\mathcal L_{\mathrm{task}}
+
\lambda_H
\left[-\sum_{k=1}^{K}\gamma_{t,k}\log(\gamma_{t,k}+\varepsilon)\right].
```

## 5. Generation-Time Formulation

At generation time, the model predicts harmony-related context:

```math
\hat r_t,
\quad
\hat c_t,
\quad
\hat\mu_t,
\quad
\hat\chi_{t,l},
\quad
\hat y^{\mathrm{cl}}_t.
```

For a candidate MIDI pitch `m`, define the root-normalized interval as

```math
l(m,t)=(m-\hat r_t)\bmod 12.
```

The candidate direction is

```math
e(m)=\frac{x(m)}{\|x(m)\|_2+\varepsilon}.
```

A candidate score can be defined as

```math
S_t(m)
=
\alpha_{\mathrm{int}}\log(\hat\chi_{t,l(m,t)}+\varepsilon)
+
\alpha_{\mu}e(m)^{\top}\hat\mu_t
+
\alpha_{\mathrm{range}}R(m)
-
\alpha_{\mathrm{rep}}\mathbf 1[m=m_{t-1}]
-
\alpha_{\mathrm{leap}}\mathbf 1[|m-m_{t-1}|\geq d_{\mathrm{leap}}]
-
\alpha_{\mathrm{same}}\mathbf 1[\mathrm{sameDirection}(m)].
```

The melody pitch is sampled by temperature-controlled softmax:

```math
P_t(m)
=
\frac{\exp(S_t(m)/\tau_{\mathrm{mel}})}
{\sum_{m'\in\mathcal M}\exp(S_t(m')/\tau_{\mathrm{mel}})}.
```

A start-token mask can be applied to avoid repeated start events:

```math
P_t(m)
\leftarrow
\frac{P_t(m)M_{\mathrm{start}}(m,t)}
{\sum_{m'}P_t(m')M_{\mathrm{start}}(m',t)+\varepsilon}.
```

The generated velocity is controlled by harmonic alignment, melody/bass roles, and random variation:

```math
v_{t,i}
=
\mathrm{clip}
\left(
 v_0
 +\beta_A A_{t,i}
 +\beta_{\mathrm{mel}}M_{t,i}
 +\beta_{\mathrm{bass}}B_{t,i}
 +\epsilon_v,
1,127
\right).
```

Here,

```math
A_{t,i}=e_{t,i}^{\top}\hat\mu_t.
```

Timing can be generated as

```math
\tilde o_{t,i}
=
\mathrm{quantize}_g(o_{t,i})
+
\hat\delta^{\mathrm{time}}_{t,i}
+
\epsilon^{\mathrm{time}}_{t,i},
```

and duration can be written as

```math
\tilde l_{t,i}
=
\mathrm{clip}
\left(
\hat l_{t,i},
 l_{\min},
 l_{\max}
\right).
```

The generated MIDI note is therefore represented by

```math
\tilde n_{t,i}
=
\left(
\tilde o_{t,i},
\tilde o_{t,i}+\tilde l_{t,i},
\tilde m_{t,i},
 v_{t,i}
\right).
```

## 6. Summary

The Conformer-vMF method consists of three stages:

1. Convert MIDI notes into hyperspherical directions using pitch, circle-of-fifths, transition, and beat/bar features.
2. Train a Conformer with multi-task heads for vMF direction, root, chord-like probability, template, interval, velocity, and timing.
3. Generate notes by combining predicted harmony, vMF alignment, interval likelihood, and melodic constraints such as repeat/leap suppression.
