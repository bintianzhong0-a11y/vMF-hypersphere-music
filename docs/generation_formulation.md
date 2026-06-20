# Generation Formulation / 生成部の定式化

This document summarizes the generation stage of the Conformer-vMF prototype.  
本資料では、Conformer-vMF プロトタイプにおける生成部の数式と処理手順を整理する。

The generation stage uses learned harmonic directions, chord-related predictions, and rule-based controls to produce pitch, velocity, timing, and duration.  
生成時には、学習済みの和声方向、コード関連 head の予測、ルールベースの制御を組み合わせて、音高・velocity・timing・duration を決定する。

> Note: This document describes the generation formulation. Model weights and copyrighted MIDI/audio files are not included.  
> 注記：本資料は生成部の定式化を説明するものであり、学習済み重みや著作権のある MIDI / 音源は含めない。

---

## 1. Generation context / 生成時の文脈

At each time step, the model receives or predicts harmonic context such as root, chord-like probability, template, beat context, and bar context.  
各時刻では、root、chord-like 確率、template、beat 文脈、bar 文脈などの和声文脈を取得または予測する。

```math
\hat{r}_t \in \{0,\ldots,11\},
\qquad
p^{cl}_t = \sigma(z^{cl}_t),
\qquad
\hat{c}_t = \operatorname{softmax}(z^{temp}_t).
```

The harmonic center direction is represented by a unit vector on the vMF hypersphere.  
和声中心方向は、vMF 超球面上の単位ベクトルとして表現する。

```math
\hat{\mu}_t
=
\frac{W_{\mu}h_t+b_{\mu}}
{\|W_{\mu}h_t+b_{\mu}\|_2+\epsilon}.
```

The concentration parameter represents how strongly the current notes gather around the harmonic center.  
集中度は、その時刻の音が和声中心方向にどれだけまとまっているかを表す。

```math
\hat{\kappa}_t
=
\operatorname{softplus}(W_{\kappa}h_t+b_{\kappa}).
```

---

## 2. Candidate pitch representation / 候補音高の表現

For a candidate MIDI pitch `m`, the pitch class is defined as follows.  
候補 MIDI 音高 `m` に対して、pitch class を次のように定義する。

```math
p(m)=m\bmod 12.
```

The circle-of-fifths index is given by the perfect-fifth mapping.  
完全五度写像により、五度圏インデックスを次で定義する。

```math
q(m)=7p(m)\bmod 12.
```

The candidate pitch is converted into a vMF note direction.  
候補音高を vMF 音符方向ベクトルに変換する。

```math
e(m)
=
\frac{x(m)}
{\|x(m)\|_2+\epsilon}.
```

Here, `x(m)` can contain pitch-circle coordinates, circle-of-fifths coordinates, transition features, register features, and beat/bar position features.  
ここで `x(m)` には、音高円座標、五度圏座標、遷移特徴、音域特徴、拍・小節位置特徴などを含める。

---

## 3. Harmonic alignment / 和声一致度

The alignment between a candidate note and the current harmonic center is measured by an inner product.  
候補音と現在の和声中心方向の一致度は、内積で測る。

```math
A_t(m)
=
e(m)^{\top}\hat{\mu}_t.
```

A larger value means that the candidate note is closer to the current harmonic direction.  
値が大きいほど、候補音は現在の和声方向に近い。

This alignment is used to prefer harmonically stable notes during generation.  
この一致度を用いて、生成時に和声的に安定した音を優先する。

---

## 4. Chord-tone score / コード構成音スコア

A chord-tone preference can be computed from the predicted root and template.  
予測された root と template から、コード構成音らしさを計算する。

Let `T_t(m)` be a chord-tone score for candidate pitch `m`.  
候補音 `m` のコード構成音スコアを `T_t(m)` とする。

```math
T_t(m)
=
\mathbf{1}
\left[
(p(m)-\hat{r}_t)\bmod 12
\in
\mathcal{I}(\hat{c}_t)
\right].
```

Here, `I(c)` is the interval set of the predicted chord template.  
ここで `I(c)` は、予測されたコード template に対応する root-normalized interval 集合である。

For example, a major triad can be represented as follows.  
例えば、major triad は次の interval 集合として表せる。

```math
\mathcal{I}(\mathrm{maj})
=
\{0,4,7\}.
```

---

## 5. Pitch score / 音高スコア

The final pitch score combines harmonic alignment, chord-tone preference, and several penalties.  
最終的な音高スコアは、和声一致度、コード構成音らしさ、複数の抑制項を組み合わせて定義する。

```math
S_t(m)
=
\beta_A A_t(m)
+
\beta_{tone}T_t(m)
-
\alpha_{rep}R_t(m)
-
\alpha_{leap}L_t(m)
-
\alpha_{same}D_t(m)
-
\alpha_{range}Q(m).
```

| Term | English | 日本語 |
|---|---|---|
| `A_t(m)` | Harmonic alignment | 和声中心との一致度 |
| `T_t(m)` | Chord-tone preference | コード構成音らしさ |
| `R_t(m)` | Repeated-note penalty | 同音反復ペナルティ |
| `L_t(m)` | Large-leap penalty | 大跳躍ペナルティ |
| `D_t(m)` | Same-direction penalty | 同方向連続ペナルティ |
| `Q(m)` | Out-of-range penalty | 音域外ペナルティ |

---

## 6. Repetition, leap, and direction penalties / 反復・跳躍・方向ペナルティ

The repeated-note penalty suppresses excessive pitch sticking.  
同音反復ペナルティは、同じ音への張り付きを抑制する。

```math
R_t(m)
=
\mathbf{1}[m=m_{t-1}].
```

The leap penalty suppresses unnaturally large jumps.  
大跳躍ペナルティは、不自然に大きな跳躍を抑制する。

```math
L_t(m)
=
\max
\left(
0,
|m-m_{t-1}|-d_{leap}
\right).
```

The same-direction penalty suppresses monotonous motion in the same direction.  
同方向連続ペナルティは、上行または下行に進み続ける単調な動きを抑える。

```math
D_t(m)
=
\mathbf{1}
\left[
\operatorname{sign}(m-m_{t-1})
=
\operatorname{sign}(m_{t-1}-m_{t-2})
\right].
```

---

## 7. Sampling distribution / サンプリング分布

Candidate pitches are sampled using a temperature-controlled softmax.  
候補音は、温度付き softmax によってサンプリングする。

```math
P_t(m)
=
\frac{
\exp(S_t(m)/\tau_{mel})
}{
\sum_{m'\in \mathcal{M}}
\exp(S_t(m')/\tau_{mel})
}.
```

Here, `tau_mel` controls the diversity of the generated melody.  
ここで `tau_mel` は、生成旋律の多様性を制御する。

A smaller temperature makes generation more deterministic.  
温度が小さいほど、生成はより決定的になる。

A larger temperature increases diversity but may reduce stability.  
温度が大きいほど多様性は増すが、安定性は下がる可能性がある。

---

## 8. Velocity generation / velocity 生成

Velocity is generated from a base value, harmonic alignment, melody emphasis, bass emphasis, and random fluctuation.  
velocity は、基本値、和声一致度、メロディ補正、ベース補正、ランダム揺らぎから生成する。

```math
v_{t,i}
=
\operatorname{clip}
\left(
v_0
+
\beta_A A_{t,i}
+
\beta_{mel}M_{t,i}
+
\beta_{bass}B_{t,i}
+
\epsilon^v_{t,i},
1,
127
\right).
```

Here, `M_{t,i}` is the melody flag and `B_{t,i}` is the bass flag.  
ここで `M_{t,i}` はメロディ音フラグ、`B_{t,i}` はベース音フラグである。

The random fluctuation can be sampled as follows.  
ランダム揺らぎは、例えば次のようにサンプリングできる。

```math
\epsilon^v_{t,i}
\sim
\mathcal{N}(0,\sigma_v^2).
```

---

## 9. Timing generation / timing 生成

The onset time is generated by adding a timing residual to the grid position.  
onset 時刻は、グリッド位置に timing residual を加えることで生成する。

```math
\tilde{o}_{t,i}
=
o^{grid}_{t,i}
+
\hat{\delta}^{time}_{t,i}
+
\epsilon^{time}_{t,i}.
```

The timing noise can be sampled as follows.  
timing の揺らぎは、例えば次のようにサンプリングできる。

```math
\epsilon^{time}_{t,i}
\sim
\mathcal{N}(0,\sigma_{time}^2).
```

The generated onset can be quantized if a strict rhythmic grid is needed.  
厳密なリズムグリッドが必要な場合は、生成された onset を量子化する。

```math
o_{t,i}^{out}
=
g
\left\lfloor
\frac{\tilde{o}_{t,i}}{g}
+
\frac{1}{2}
\right\rfloor.
```

Here, `g` is the quantization width.  
ここで `g` は量子化幅である。

---

## 10. Duration generation / duration 生成

Duration can be predicted directly or selected from a rhythmic template.  
duration は、直接予測するか、リズム template から選択する。

```math
\ell_{t,i}
=
\operatorname{clip}
\left(
\hat{\ell}_{t,i},
\ell_{min},
\ell_{max}
\right).
```

The offset time is then computed as follows.  
offset 時刻は次で与える。

```math
f_{t,i}
=
o_{t,i}^{out}
+
\ell_{t,i}.
```

---

## 11. Chord and accompaniment generation / コード・伴奏生成

For accompaniment generation, the predicted root and template define a set of chord tones.  
伴奏生成では、予測された root と template からコード構成音集合を定義する。

```math
\mathcal{C}_t
=
\left\{
\hat{r}_t + l \pmod{12}
\mid
l\in \mathcal{I}(\hat{c}_t)
\right\}.
```

Chord notes are placed in a suitable register.  
コード音は、自然な伴奏音域に配置する。

```math
m^{chord}_{t,j}
=
12O_{chord}
+
c_{t,j},
\qquad
c_{t,j}\in \mathcal{C}_t.
```

Bass notes are placed in a lower register.  
ベース音は、低音域に配置する。

```math
m^{bass}_{t}
=
12O_{bass}
+
\hat{r}_t.
```

---

## 12. MIDI rendering / MIDI 書き出し

The generated notes are finally rendered as MIDI events.  
最後に、生成された音符を MIDI event として書き出す。

```math
\mathrm{MIDI}
=
\operatorname{Render}
\left(
m_{t,i},
o_{t,i}^{out},
f_{t,i},
v_{t,i}
\right).
```

Each MIDI note consists of pitch, onset, offset, and velocity.  
各 MIDI note は、音高、onset、offset、velocity から構成される。

---

## 13. Generation algorithm / 生成アルゴリズム

The overall generation process is summarized as follows.  
生成全体の流れは次のように要約できる。

```text
Input:
  trained Conformer-vMF model
  generation length
  tempo, beat, bar settings
  optional chord progression or prompt

For each time step t:
  1. Predict or set root, chord-like probability, template, beat context, and bar context.
  2. Estimate harmonic direction mu_t and concentration kappa_t.
  3. Enumerate candidate pitches in the allowed pitch range.
  4. Convert each candidate pitch into a vMF direction.
  5. Compute harmonic alignment A_t(m).
  6. Compute pitch score S_t(m).
  7. Sample or select pitch m from P_t(m).
  8. Generate velocity from alignment and part flags.
  9. Generate timing and duration.
  10. Render MIDI note events.

Output:
  generated MIDI performance
```

---

## 14. v6.1 generation controls / v6.1 での生成制御

In the v6.1 generation setting, several constraints were introduced to reduce common failure modes.  
v6.1 の生成設定では、よくある失敗を抑えるために複数の制御を導入した。

| Control | English | 日本語 |
|---|---|---|
| start mask | Prevents repeated start tokens after the first event | 最初以外の start 連続を禁止 |
| repeat penalty | Reduces pitch sticking | 同音反復を抑制 |
| leap penalty | Reduces excessive jumps | 大跳躍を抑制 |
| same-direction penalty | Reduces monotonous motion | 同方向連続を抑制 |
| chord-tone preference | Encourages harmonically stable notes | コード構成音を優先 |

These controls are not separate neural heads.  
これらの制御は、独立した neural head ではない。

They are generation-time constraints applied to the sampling score.  
これらは、生成時の sampling score に加える制約である。

---

## 15. Example generation metrics / 生成結果の例

A representative v6.1 generation produced 100 events over a 16-bar setting.  
代表的な v6.1 生成では、16 小節設定で 100 event を生成した。

| Metric | Value | Interpretation / 解釈 |
|---|---:|---|
| `n_events` | 100 | Number of generated note events / 生成音符イベント数 |
| `unique_pitch` | 21 | Pitch diversity / 音高の多様性 |
| `repeat_rate` | 0.46 | Repeated-note ratio / 同音反復率 |
| `step_rate` | 0.26 | Stepwise motion ratio / 順次進行率 |
| `skip_rate` | 0.17 | Moderate skip ratio / 中程度跳躍率 |
| `leap_rate` | 0.10 | Large-leap ratio / 大跳躍率 |
| `same_direction_rate` | 0.47 | Same-direction continuation ratio / 同方向連続率 |

These metrics are not a subjective listening evaluation.  
これらの指標は、主観的な聴取評価ではない。

They are objective descriptors for checking whether the generated melody is overly repetitive, unstable, or excessively jumpy.  
生成旋律が過度に反復的、不安定、または大跳躍に偏っていないかを確認するための客観的な記述指標である。

---

## 16. Summary / まとめ

The generation part of Conformer-vMF uses the learned harmonic direction to evaluate candidate notes on the vMF hypersphere.  
Conformer-vMF の生成部では、学習された和声方向を用いて、vMF 超球面上で候補音を評価する。

Pitch generation is controlled by harmonic alignment, chord-tone preference, and penalties for repetition, large leaps, and monotonic motion.  
音高生成は、和声一致度、コード構成音らしさ、同音反復・大跳躍・単調な同方向進行へのペナルティによって制御される。

Velocity, timing, and duration are generated using alignment-based formulas and residual corrections.  
velocity、timing、duration は、方向一致度に基づく式と residual 補正によって生成される。

This makes the generation process interpretable: each generated note can be related back to harmonic direction, chord context, and melodic motion constraints.  
これにより、生成された各音を、和声方向、コード文脈、旋律運動制約と対応づけて解釈できる。
