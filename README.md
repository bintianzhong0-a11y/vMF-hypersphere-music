# vMF Hypersphere Music Representation

This repository contains experimental artifacts for visualizing musical harmony and performance-related representations on a von Mises–Fisher (vMF) hypersphere.

本リポジトリは、音楽に含まれるコード種別・root・5度圏距離・和声中心を、vMF 分布の方向ベクトルとして扱い、学習された超球面上の構造を可視化・分析・演奏生成するための研究成果物です。

## Overview

In this project, musical events are represented as directional vectors on a vMF hypersphere.
The learned direction parameter $\mu$ is used to analyze and model musically meaningful relationships such as chord-type similarity, inferred root structure, pitch-class organization, and circle-of-fifths proximity.

The model is trained in a multi-task setting.
It predicts not only the vMF direction, but also several musical attributes, including root class, chord-like probability, chord template, triad type, seventh type, beat position, bar position, onset-related information, and performance-related parameters.

In addition to these note- and event-level representations, this repository introduces a Conformer-based block-level harmonic function transition module.
This module estimates higher-level harmonic functions such as Tonic (T), Dominant (D), Subdominant (SD), and OTHER, and uses them as a structural control signal during generation.

During generation, the system applies quota-aware empirical decoding based on the training target distribution.
This prevents autoregressive generation from collapsing into overly repetitive tonic/dominant patterns while preserving the learned Conformer transition logits.

The final generation pipeline produces full-arrangement MIDI files containing melody, chord comping, bass, arpeggio, and pad tracks.
The aim of this project is to examine whether vMF hypersphere representations can serve both as an interpretable musical structure space and as a practical control mechanism for symbolic music generation.

## 概要

本研究では、音楽的なイベントを vMF 超球面上の方向ベクトルとして表現します。
学習された方向パラメータ $\mu$ を用いて、コード種別の類似性、推定 root 構造、音高クラスの配置、5度圏上の近接性といった音楽的関係を分析・モデル化します。

$$
\qquad
p(e \mid \mu, \kappa)=
C_D(\kappa)\exp\left(\kappa \mu^{\top} e\right)
\qquad (\lVert \mu_t \rVert_2 = 1,\quad \mu_t \in ℝ^{10})
\qquad
$$


| 記号           | 数学的意味       | 音楽的意味                 |
| ------------ | ----------- | --------------------- |
| $\mu$        | vMF 分布の中心方向 | 現在の音楽状態・和声中心の方向       |
| $\kappa$    | 中心方向への集中度   | その音楽状態の確信度・まとまりの強さ    |
| $e$          | 音符イベント方向    | 個々の音符やイベントの方向         |
| $\mu^\top e$ | 方向一致度       | 音符が現在の和声中心にどれだけ合っているか |


本モデルは、単一のラベルだけを学習するのではなく、multi-task 学習として構成されています。
具体的には、vMF 方向に加えて、root class、chord-like probability、chord template、triad type、seventh type、beat position、bar position、onset 関連情報、velocity・timing・duration などの演奏関連パラメータを同時に扱います。

さらに本リポジトリでは、これらの note-level / event-level 表現に加えて、Conformer による block-level harmonic function transition module を導入しています。
このモジュールでは、Tonic（T）、Dominant（D）、Subdominant（SD）、OTHER といった大域的な和声機能を推定し、生成時の構造制御信号として利用します。

生成時には、学習時の target function 分布に基づく quota-aware empirical decoding を適用します。
これにより、Conformer が学習した遷移 logits を主軸に保ちながら、自己回帰生成が T / D に過度に偏ることを抑制します。

### vMF方向表現による和声的近接性の利用

本研究では、音高・5度圏・音高遷移・拍節位置などの音楽的特徴を、vMF超球面上の方向ベクトルとして統一的に扱った。  
これにより、生成時には現在の和声中心方向 $\mu$ に近い候補音を優先しやすくなる。

特に、5度圏上で近い音や和声は、vMF空間上でも近い方向として扱えるため、単なる離散的なコードラベルではなく、角度に基づく連続的な和声近接性を生成に反映できる。  
その結果、現在の和声・音高・拍節文脈に対して自然で親和性の高い響きを生成できる可能性がある。
<details>

$$
\[
\mathrm{score}_t(p) =
w_{\mathrm{vMF}} e_p^{\top}\mu_t
+
w_{\mathrm{chord}}
\mathbf{1}
\left[
p \bmod 12 \in \mathcal{C}_t
\right]
+
w_{\mathrm{motion}}
\mathbf{1}
\left[
|p-p_{t-1}| \leq s
\right]
+
w_{\mathrm{range}}
\mathbf{1}
\left[
p_{\min} \leq p \leq p_{\max}
\right]-
P_t(p)
\]
$$

</details>

最終的な生成パイプラインでは、melody、chord comping、bass、arpeggio、pad の各トラックを含む full-arrangement MIDI を出力します。
本研究の目的は、vMF 超球面表現が、解釈可能な音楽構造空間として機能するだけでなく、記号音楽生成における実用的な制御表現としても利用できるかを検証することです。

## Music Theory Formulation Summary

This repository includes a PDF summary of the music-theory-based formulations used in this project.

The document explains how pitch class, circle-of-fifths coordinates, fifth-distance, chord templates, harmonic functions, vMF direction parameters, and quota-aware empirical decoding are connected to the Conformer-vMF generation pipeline.

- [View PDF](docs/music_theory_formulation_summary_vmf.pdf)
- [Download PDF](https://github.com/bintianzhong0-a11y/vMF-hypersphere-music/raw/main/docs/music_theory_formulation_summary_vmf.pdf)

### 日本語説明

本リポジトリには、本研究で用いる音楽理論由来の立式をまとめた PDF も含めています。

この資料では、pitch class、5度圏座標、5度圏距離、コードテンプレート、和声機能、vMF の方向パラメータ、quota-aware empirical decoding が、Conformer-vMF による生成パイプラインとどのように接続されるかを整理しています。

- [PDFを表示](docs/music_theory_formulation_summary_vmf.pdf)
- [PDFをダウンロード](https://github.com/bintianzhong0-a11y/vMF-hypersphere-music/raw/main/docs/music_theory_formulation_summary_vmf.pdf)

## Generated Audio / 生成音源

A generated audio sample from the Conformer-vMF prototype is available for listening, together with spectrogram and chromagram visualizations.  
Conformer-vMF プロトタイプによる生成音源を、スペクトログラムおよびクロマ図とともに掲載しています。

## Differentiation from Other Generation Systems / 他の生成システムとの差別化

This project does not primarily aim to compete with large-scale music generation systems in final audio quality.  
本研究は、大規模音楽生成AIと最終的な音質で直接競争することを主目的としていません。

Instead, it focuses on interpretable and controllable symbolic music generation using a vMF hyperspherical representation.  
代わりに、vMF 超球面表現を用いた、解釈可能で制御しやすい symbolic music generation を目指しています。

The main distinction is that pitch, circle-of-fifths structure, metrical position, and harmonic context are represented as directions, and generation is controlled using alignment with the learned harmonic direction.  
主な差別化点は、音高・5度圏構造・拍節位置・和声文脈を方向として表現し、学習された和声方向との一致度に基づいて生成を制御する点です。

- [Differentiation from other music generation systems / 他の生成システムとの差別化](docs/generation_system_differentiation.md)

- [Pop1K7 1000 Conformer-vMF block transition experiment](docs/pop1k7_block_transition_experiment.md)
ading README_git_summary_combined_snippet.md…]()


<audio controls src="audio/vMF_generated_sample.mp3"></audio>

- [Listen to generated audio / 生成音源を開く](audio/vMF_generated_sample.mp3)
- [Generated audio showcase / 生成音源の試聴と可視化](docs/generated_audio_showcase.md)

### Spectrogram / スペクトログラム

![Generated audio spectrogram](figures/vMF_generated_spectrogram.png)

### Chromagram / クロマ図

![Generated audio chromagram](figures/vMF_generated_chromagram.png)

---
## Key Findings / 主な発見

Learned `μ` vectors form interpretable structures on the hypersphere.

学習された `μ` ベクトルは、超球面上で解釈可能な構造を形成している。

Chord-type centers show meaningful angular relationships.

コード種別ごとの中心方向には、意味のある角度関係が見られた。

Inferred root centers show a strong correspondence with circle-of-fifths distance.

推定された root 中心は、5度圏距離と強い対応関係を示した。

The circle-of-fifths distance and vMF angular distance showed a high correlation.

### Block Transition 実験条件
<details>

| 英語項目                                  | 日本語項目          | 内容                                   |
| ------------------------------------- | -------------- | ------------------------------------ |
| Data                                  | 使用データ          | Pop1K7由来 `vmf_processed_pop1k7_1000` |
| Input dimension                       | 入力次元           | `input_dim=10`                       |
| vMF direction dimension               | vMF方向ベクトル次元    | `mu_dim=10`                          |
| Number of root classes                | ルート音分類数        | 12                                   |
| Number of template classes            | コードテンプレート分類数   | 9                                    |
| Number of triad classes               | 三和音分類数         | 6                                    |
| Number of seventh classes             | セブンス分類数        | 6                                    |
| Number of function classes            | 和声機能分類数        | 4                                    |
| Number of function transition classes | 和声機能遷移分類数      | 16                                   |
| Target labels                         | 推定対象ラベル        | T / D / SD / OTHER                   |
| Events per block                      | 1ブロックあたりのイベント数 | 8                                    |
| Number of blocks                      | 使用ブロック数        | 16                                   |
| Block stride                          | ブロックの移動幅       | 4                                    |
| Phrase period                         | フレーズ周期         | 4                                    |
| Minimum blocks                        | 最小ブロック数        | 8                                    |
| Batch size                            | バッチサイズ         | 16                                   |
| Validation ratio                      | 検証データ割合        | 0.1                                  |
| Seed                                  | 乱数シード          | 42                                   |

</details>

### 5度圏距離と vMF 空間上の角距離は、高い相関

* Pearson correlation = `0.954`
　Pearson 相関係数 = `0.954`

* Spearman correlation = `0.953`
  Spearman 相関係数 = `0.953`

These results suggest that the learned vMF `μ`-space may partially encode music-theoretical proximity.

これらの結果は、学習された vMF `μ` 空間が、音楽理論上の近接性を部分的に符号化している可能性を示している。

### Block Transition Experiment / 和声機能遷移実験

This experiment evaluates whether Conformer-vMF can predict block-level harmonic function transitions from Pop1K7-derived MIDI data.  
この実験では、Pop1K7 由来の MIDI データから、Conformer-vMF が block 単位の和声機能遷移を予測できるかを検証しています。

The processed dataset contains 1000 distinct MIDI files and approximately 1.43 million note events.  
処理済みデータセットは、1000個の異なる MIDI ファイルと約143万イベントから構成されています。

#### validation set の正解クラス分布
| クラス   |     件数 |      割合 |
| ----- | -----: | ------: |
| T     | 28,612 |  46.29% |
| D     |  8,509 |  13.77% |
| SD    | 11,968 |  19.36% |
| OTHER | 12,726 |  20.59% |
| 合計    | 61,815 | 100.00% |


After head training and fine-tuning, the validation performance reached
head 学習と fine-tuning の後、検証性能は
```text
val_acc = 0.832
val_macro_f1 = 0.808
```
に到達しました。
| 学習段階        | epoch | train Acc | train Macro-F1 | val Acc | val Macro-F1 | val loss |
| ----------- | ----: | --------: | -------------: | ------: | -----------: | -------: |
| conformer凍結      |    15 |    0.5812 |         0.4535 |  0.5874 |       0.4596 |   0.9893 |
| 四層conformer |    10 |    0.8084 |         0.7804 |  0.8320 |       0.8077 |   0.4359 |


### その他の補助的成果
<details>

#### vMF Conformer：root・コード・拍節などの推定結果

| 実験                 | 対象            |       指標 |         結果 | 解釈                   |
| ------------------ | ------------- | -------: | ---------: | -------------------- |
| vMF Conformer v4.7 | Root推定        |      Acc | **0.8536** | root 12分類はかなり高水準     |
| vMF Conformer v4.7 | Root推定        | Macro-F1 | **0.8522** | クラス偏り込みでも安定          |
| vMF Conformer v4.7 | Chord Quality |      Acc | **0.7248** | コード種別は中程度以上          |
| vMF Conformer v4.7 | Chord Quality | Macro-F1 | **0.3591** | 少数コード種別で難しさあり        |
| vMF Conformer v4.7 | chord-like 2値 |      Acc | **0.9416** | コードらしさ判定はかなり強い       |
| vMF Conformer v4.7 | chord-like 2値 | Macro-F1 | **0.8556** | 非コード/コードの分離も良好       |
| vMF Conformer v4.7 | Onset         |      Acc | **0.5725** | onsetはまだ改善余地あり       |
| vMF Conformer v4.7 | Chroma        |      Acc | **0.9199** | 音高集合の再構成は強い          |
| vMF Conformer v4.7 | Velocity      |      MAE |  **8.942** | velocity生成誤差は実用範囲に近い |
| vMF Conformer v4.7 | vMF方向一致       |  cos sim |  **0.772** | 方向表現はかなり学習できている      |
| vMF Conformer v4.7 | Template 12分類 |      Acc | **0.3599** | 細かいコードテンプレート分類は難しい   |

</details>

## Main Visualizations / 主な可視化結果

### Circle-of-fifths distance vs vMF angular distance / 5度圏距離と vMF 角距離

![Circle-of-fifths distance vs vMF angular distance](figures/vMF_fifth_map.png)

This figure compares the distance between inferred root classes on the circle of fifths with their angular distance on the learned vMF hypersphere.
この図は、推定された root クラス間の 5度圏上の距離と、学習された vMF 超球面上での角距離を比較したものです。

The strong positive correlation suggests that the learned `μ`-space captures part of the tonal structure.
強い正の相関が見られることから、学習された `μ` 空間が、調性構造の一部を捉えている可能性が示唆されます。

### Chord-type angular distance / コード種別間の角距離

![Angular distance between chord-type centers](figures/vMF_heat2_map.png)

This heatmap shows angular distances between chord-type centers on the learned vMF `μ` hypersphere.
このヒートマップは、学習された vMF `μ` 超球面上におけるコード種別中心間の角距離を示しています。

A smaller angular distance means that two chord types are represented as closer directions in the learned hyperspherical space.
角距離が小さいほど、2つのコード種別が学習空間内で近い方向として表現されていることを意味します。

### Chord-type cosine similarity / コード種別間の cos 類似度

![Cosine similarity between chord-type centers](figures/vMF_heat_map.png)

This heatmap shows cosine similarities between chord-type centers. Higher values indicate closer directional alignment in the learned hyperspherical space.
このヒートマップは、コード種別中心間の cos 類似度を示しています。値が大きいほど、学習された超球面空間内で方向が近く、類似した表現になっていることを表します。

### PCA Variable Analysis / PCA変数分析

This section analyzes which original vMF variables explain the chord-center arrangements shown in the sphere and circle visualizations.  
この節では、球面・円上に表示されたコード中心の配置が、元の vMF 変数のどの要素によって説明されているかを分析します。

- [PCA variable analysis / PCA変数分析](docs/pca_variable_analysis.md)
- [Chord center visualization analysis / コード中心可視化まとめ](docs/chord_center_visualization_analysis.md)

The PCA input is the 10-dimensional vMF center vector averaged for each chord category.  
PCA の入力は、コードカテゴリごとに平均化した 10 次元 vMF 中心ベクトルです。

For `template` and `triad`, the first principal component is strongly related to circle-of-fifths components.  
`template` と `triad` では、第1主成分が 5度圏成分と強く関係しています。

For `seventh`, the first principal component is more strongly related to pitch-transition components, suggesting a connection with progression and resolution context.  
`seventh` では、第1主成分が音高遷移成分とより強く関係しており、進行・解決文脈との関連が示唆されます。

### Representative figures / 代表図

#### Template chord centers / template コード中心

![Template sphere with music labels](figures/chord_center_visualizations/pop1k7_1000_template_sphere_music_labels.png)

![Template sphere with music labels](figures/chord_center_visualizations/pop1k7_1000_template_circle_music_labels.png)
#### Template angular distance / template 角距離

![Template angular distance heatmap](figures/chord_center_visualizations/pop1k7_1000_template_angular_distance_heatmap_music_labels.png)

#### Template PC1 loadings / template PC1 loading

![Template PC1 loadings](figures/pca_variable_analysis/template_PC1_loadings.png)

#### Seventh PC1 loadings / seventh PC1 loading

![Seventh PC1 loadings](figures/pca_variable_analysis/seventh_PC1_loadings.png)

### Interpretation note / 解釈上の注意

These visualizations should not be interpreted as proof that the model discovered music theory entirely from scratch.  
これらの可視化は、モデルが音楽理論を完全にゼロから発見したことを示すものではありません。

Rather, they show that the designed vMF representation preserves and organizes chord-related information in a geometrically interpretable way.  
むしろ、設計した vMF 表現が、コード関連情報を幾何的に解釈しやすい形で保持・整理していることを示すものです。
loading README_pca_results_snippet.md…]()


### Inferred root angular distance / 推定 root 間の角距離

![vMF angular distance between inferred root centers](figures/vMF_root_inf_dist.png)

This heatmap visualizes angular distances between inferred root centers.
このヒートマップは、推定された root 中心同士の角距離を可視化したものです。

It helps examine whether the learned root representations reflect harmonic relationships such as the circle of fifths.
これにより、学習された root 表現が、5度圏のような和声的関係を反映しているかを確認できます。

### Root classes on the learned hypersphere / 学習された超球面上の root クラス

![Root 12 classes on learned mu hypersphere](figures/vMF_root_map.png)

The 12 inferred root classes are projected onto a 3D PCA sphere for visualization.
12 種類の推定 root クラスを、可視化のために 3 次元 PCA 空間へ射影しています。

This figure provides an intuitive view of how root categories are arranged in the learned vMF representation space.
この図により、学習された vMF 表現空間内で root カテゴリがどのように配置されているかを直感的に確認できます。

### Learned `μ` hypersphere / 学習された `μ` 超球面

![Learned mu hypersphere visualization](figures/vMF_mu_map.png)

This figure visualizes the learned `μ` directions. Points are colored by concentration parameter `κ`.
この図は、学習された `μ` 方向を可視化したものです。各点の色は集中度パラメータ `κ` を表しています。

The visualization shows how musical states are distributed on the learned hyperspherical representation.
この可視化により、音楽的状態が学習済みの超球面表現上にどのように分布しているかを確認できます。

### Chord types on the learned hypersphere / 学習された超球面上のコード種別

![Chord types on learned mu hypersphere](figures/vMF_chord1_map.png)

This figure shows chord-type distributions on the learned `μ` hypersphere.
この図は、学習された `μ` 超球面上におけるコード種別の分布を示しています。

It helps examine whether different chord types form separable or structured regions in the learned space.
これにより、異なるコード種別が学習空間内で分離可能な領域や構造を形成しているかを確認できます。

### Filtered chord-type visualization / フィルタ後のコード種別可視化

![Filtered chord types on learned mu hypersphere](figures/vMF_chord2_map.png)

This figure removes `other` and `no_chord` classes to make the chord-type structure easier to observe.
この図では、コード種別の構造を観察しやすくするために、`other` と `no_chord` クラスを除外しています。

By filtering out less informative classes, the relationships among musically meaningful chord types become clearer.
情報量の少ないクラスを除外することで、音楽的に意味のあるコード種別同士の関係がより見やすくなります。


## Additional Documents

- [Evaluation criteria and comparison with prior work](docs/vmf_evaluation_comparison.pdf)  
  Evaluation metrics, beginner-friendly explanations, and reference comparisons with related benchmarks and prior studies.

- [Conformer-vMF variables, losses, and generation parameters](docs/conformer_vmf_tables.pdf)  
  Input variables, prediction heads, loss functions, and generation-time control parameters used in the Conformer-vMF prototype.

## Generation Formulation / 生成部

The generation stage uses learned harmonic directions and chord-related predictions to control pitch, velocity, timing, and duration.
生成時には、学習された和声方向とコード関連予測を用いて、音高・velocity・timing・duration を制御します。

* [Generation formulation / 生成部の定式化](docs/generation_formulation.md)

The document includes pitch scoring, sampling, velocity generation, timing residuals, duration control, accompaniment rendering, and v6.1 generation constraints.
この資料には、音高スコア、サンプリング、velocity 生成、timing residual、duration 制御、伴奏生成、v6.1 の生成制約を掲載しています。



## Method Summary

The workflow is as follows:

1. Extract note-level or group-level musical information from symbolic music data.
2. Encode musical states into directional representations.
3. Learn vMF parameters, especially the direction vector \(\mu\) and concentration \(\kappa\).
4. Aggregate \(\mu\) vectors by chord type or inferred root.
5. Compute cosine similarity and angular distance between centers.
6. Compare learned angular distances with music-theoretical distances, especially the circle-of-fifths distance.

The angular distance between two normalized vMF centers \(\mu_i\) and \(\mu_j\) is computed as:

```math
d_{\mathrm{angle}}(\mu_i, \mu_j)
= \arccos(\mu_i^\top \mu_j)
```

Cosine similarity is computed as:

```math
\mathrm{cos}(\mu_i, \mu_j)
= \mu_i^\top \mu_j
```

A mathematical summary of the Conformer-vMF training and generation formulation is available here:

- [Method summary: Conformer-vMF music representation and generation](docs/method_summary_github_safe.md)

This document describes the vMF coordinate construction, harmonic center direction, multi-task training losses, optional P²OT prototype lens, and generation-time pitch/velocity/timing equations.

- [Method summary: Conformer-vMF music representation and generation](docs/method_summary.md)  
  Mathematical summary of the Conformer-vMF training and generation formulation.

- [Method summary Japanese version](docs/method_summary_ja.md)  
  日本語版の手法概要です。vMF 座標、和声中心方向、学習時の損失関数、生成時の pitch / velocity / timing 数式を整理しています。

## Method Details / 手法詳細

Detailed pages from the black-and-white LaTeX Method Summary are provided below as PNG images.  
白黒LaTeX版の Method Summary を、README で見やすい PNG 画像として掲載しています。

- [Full PDF version / PDF版](docs/conformer_vmf_method_summary_bw.pdf)

<details>
<summary>Open Method Details pages / Method Details の各ページを開く</summary>

### Page 1
![Method Details page 1](figures/method_details_page_1.png)

### Page 2
![Method Details page 2](figures/method_details_page_2.png)

### Page 3
![Method Details page 3](figures/method_details_page_3.png)

### Page 4
![Method Details page 4](figures/method_details_page_4.png)

### Page 5
![Method Details page 5](figures/method_details_page_5.png)

### Page 6
![Method Details page 6](figures/method_details_page_6.png)

### Page 7
![Method Details page 7](figures/method_details_page_7.png)

### Page 8
![Method Details page 8](figures/method_details_page_8.png)

### Page 9
![Method Details page 9](figures/method_details_page_9.png)

</details>

## Repository Structure

```text
.
├── README.md
├── figures/
│   ├── vMF_heat2_map.png
│   ├── vMF_heat_map.png
│   ├── vMF_root_inf_dist.png
│   ├── vMF_fifth_map.png
│   ├── vMF_root_map.png
│   ├── vMF_mu_map.png
│   ├── vMF_chord1_map.png
│   └── vMF_chord2_map.png
├── notebooks/
│   └── example_colab_notebooks.ipynb
├── scripts/
│   ├── visualize_chord_type_centers.py
│   ├── visualize_root_centers.py
│   ├── circle_of_fifths_vs_vmf_distance.py
│   └── single_song_mu_trajectory.py
└── results/
    └── example_outputs.csv
```

## Requirements

Typical dependencies:

```text
numpy
pandas
matplotlib
scikit-learn
plotly
torch
scipy
```

Install them with:

```bash
pip install numpy pandas matplotlib scikit-learn plotly torch scipy
```

## Usage

Example:

```bash
python scripts/circle_of_fifths_vs_vmf_distance.py
```

For Google Colab usage, some paths may need to be modified, especially paths under:

```text
/content/drive/MyDrive/
```

## Notes

- The root labels in these visualizations are inferred from pitch-class sets.
- They are not necessarily ground-truth chord annotations.
- The 3D sphere plots are PCA projections of higher-dimensional \(\mu\) vectors.
- The original learned \(\mu\) dimension used in the visualization is `mu_dim = 8`.
- Copyrighted MIDI files, audio files, and private datasets are not included in this repository.

## Research Context

This project is part of an ongoing study on music representation learning and expressive piano performance generation using vMF hyperspherical representations.

The broader motivation is to construct a representation space where musical relationships such as harmony, tonal proximity, chord quality, and performance expression can be handled through direction, angle, and concentration.

## Citation

If you use or refer to this repository, please cite it as:

```bibtex
@misc{tanaka_vmf_hypersphere_music,
  author = {Tanaka, Akira},
  title = {vMF Hypersphere Music Representation},
  year = {2026},
  note = {GitHub repository}
}
```

## 再現方法：Conformer-vMF によるフルアレンジ生成

本リポジトリでは、vMF 超球面音楽表現と Conformer による block-level harmonic function transition model を用いて、フルアレンジ MIDI を生成する再現用パイプラインを公開しています。

本命設定は quota_empirical preset です。

1. リポジトリを clone する

```bash
git clone https://github.com/bintianzhong0-a11y/vMF-hypersphere-music.git
cd vMF-hypersphere-music
```

2. 依存ライブラリをインストールする

```bash
pip install -r requirements.txt
```

3. 学習済み checkpoint をダウンロードする

GitHub の Release ページから、以下の checkpoint ZIP をダウンロードしてください。

```text
Release v0.1.0
Conformer-vMF quota empirical checkpoint v0.1.0
```

ダウンロードした ZIP ファイルを展開します。

```text
vmf_conformer_block_transition_pop1k7_1000_finetune_from_head_best.zip
```

展開後の .pt ファイルを、以下の場所に配置してください。

```text
checkpoints/vmf_conformer_block_transition_pop1k7_1000_finetune_from_head_best.pt
```

4. フルアレンジ MIDI を生成する

以下のコマンドを実行します。

```bash
python scripts/generate_vmf_full_arrangement_conformer_block.py \
  --checkpoint checkpoints/vmf_conformer_block_transition_pop1k7_1000_finetune_from_head_best.pt \
  --out_midi results/generated/vmf_full_arrangement_quota_empirical_seed5.mid \
  --out_json results/generated/vmf_full_arrangement_quota_empirical_seed5_stats.json \
  --key C \
  --blocks 16 \
  --steps_per_block 8 \
  --tempo 120 \
  --seed 5
```

このコマンドにより、以下のトラックを含むフルアレンジ MIDI が生成されます。

1. Melody
2. Chord comping
3. Bass
4. Arpeggio
5. Pad

5. 複数 seed で生成する

```bash
for seed in 1 2 3 4 5
do
  python scripts/generate_vmf_full_arrangement_conformer_block.py \
    --checkpoint checkpoints/vmf_conformer_block_transition_pop1k7_1000_finetune_from_head_best.pt \
    --out_midi results/generated/vmf_full_arrangement_quota_empirical_seed${seed}.mid \
    --out_json results/generated/vmf_full_arrangement_quota_empirical_seed${seed}_stats.json \
    --key C \
    --blocks 16 \
    --steps_per_block 8 \
    --tempo 120 \
    --seed ${seed}
done
```

生成時の target function 分布

quota_empirical preset では、学習時の block-level target function 分布を参照します。

```text
T     = 0.4615
D     = 0.1378
SD    = 0.1928
OTHER = 0.2079
```

16 block の生成では、おおよそ以下の配分を目標にします。

```text
T     : 7〜8 blocks
D     : 2〜3 blocks
SD    : 約3 blocks
OTHER : 2〜3 blocks
```

この decoding 方針により、Conformer が学習した block-level transition logits を主軸にしながら、自己回帰生成で T / D に偏りすぎることを抑え、学習データに近い harmonic function 分布を保った生成を行います。

詳細

詳細な再現手順は、以下のファイルにも記載しています。

REPRODUCE_QUOTA_EMPIRICAL.md


## License

This repository is released for research and educational purposes.  
Please check the license file before reuse.

## Reproducible full-arrangement generation

This repository includes a Conformer-vMF full-arrangement generation pipeline.

The main reproducible script is:

```bash
python scripts/generate_vmf_full_arrangement_conformer_block.py
```

The recommended preset is `quota_empirical`, which uses the training target function distribution:

```text
T     = 0.4615
D     = 0.1378
SD    = 0.1928
OTHER = 0.2079
```

## Model Dimensions and Parameter Counts

In the current implementation, the generation pipeline can be summarized as follows:

```text
MIDI
→ 10-dimensional vMF input features
→ Conformer encoder
→ vMF / multi-task prediction heads
→ block-level harmonic function transition head
→ full-arrangement MIDI
```

The raw MIDI sequence is first converted into a 10-dimensional event representation.
This representation includes pitch-class coordinates, circle-of-fifths coordinates, pitch-transition information, pitch height, pitch delta, and beat/bar position features.

The Conformer encoder maps this 10-dimensional input into a 128-dimensional hidden representation:

```math
X \in \mathbb{R}^{T \times 10}
\quad \longrightarrow \quad
H \in \mathbb{R}^{T \times 128}
```

The model predicts a 10-dimensional vMF mean direction and a scalar concentration parameter:

```math
\mu_t \in \mathbb{R}^{10},
\qquad
\kappa_t \in \mathbb{R}^{1}
```

In addition to the vMF outputs, the model also predicts musical attributes such as root class, chord template, triad type, seventh type, beat position, bar position, onset, velocity, timing, duration, and harmonic function labels.

For block-level generation, the model pools the Conformer hidden states into block-level representations and predicts the next harmonic function:

```math
H_b \in \mathbb{R}^{128}
\quad \longrightarrow \quad
f_b \in \{T, D, SD, OTHER\}
```

The current Conformer-vMF block transition model has the following dimensions and parameter counts:

```text
Component	Dimension	Parameters
vMF input feature	10	0
Conformer hidden dimension	128	—
Conformer encoder body	128 hidden, 4 layers, 4 heads	1,532,288
vMF and event-level heads	(\mu:10), (\kappa:1), plus multi-task outputs	225,337
Function and transition heads	4 classes + 16 transition classes	36,116
Block transition head	256 → 128 → 4	51,076
Total model	Conformer-vMF block transition	1,844,817
```

The final MIDI generation stage uses the predicted harmonic function sequence and decoded chord structure to generate melody, chord comping, bass, arpeggio, and pad tracks.
This MIDI decoding stage is rule- and score-based in the current implementation, so it does not add additional trainable parameters.


See:

```text
REPRODUCE_QUOTA_EMPIRICAL.md
```

for the full reproduction command.
