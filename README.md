# vMF Hypersphere Music Representation

This repository contains experimental artifacts for visualizing musical harmony and performance-related representations on a von Mises–Fisher (vMF) hypersphere.

本リポジトリは、音楽に含まれるコード種別・root・5度圏距離・和声中心を、vMF 分布の方向ベクトルとして扱い、学習された超球面上の構造を可視化・分析するための研究成果物です。

## Overview

In this project, musical events are represented as directional vectors on a learned vMF hypersphere.  
The goal is to examine whether the learned direction parameter \(\mu\) reflects musically meaningful relationships such as chord-type similarity, inferred root structure, and circle-of-fifths proximity.

本研究では、音楽的な状態を vMF 分布の方向パラメータ \(\mu\) として表現し、コード種別・推定 root・5度圏上の近接性が、学習された超球面上でどのように現れるかを分析します。

## Key Findings

- Learned \(\mu\) vectors form interpretable structures on the hypersphere.
- Chord-type centers show meaningful angular relationships.
- Inferred root centers show a strong correspondence with circle-of-fifths distance.
- The circle-of-fifths distance and vMF angular distance showed a high correlation:

```text
Pearson correlation  = 0.954
Spearman correlation = 0.953
```

These results suggest that the learned vMF \(\mu\)-space may partially encode music-theoretical proximity.

## Main Visualizations

### Circle-of-fifths distance vs vMF angular distance

![Circle-of-fifths distance vs vMF angular distance](figures/vMF_fifth_map.png)

This figure compares the distance between inferred root classes on the circle of fifths with their angular distance on the learned vMF hypersphere.  
The strong positive correlation suggests that the learned \(\mu\)-space captures part of the tonal structure.

### Chord-type angular distance

![Angular distance between chord-type centers](figures/vMF_heat2_map.png)

This heatmap shows angular distances between chord-type centers on the learned vMF \(\mu\) hypersphere.

### Chord-type cosine similarity

![Cosine similarity between chord-type centers](figures/vMF_heat_map.png)

This heatmap shows cosine similarities between chord-type centers. Higher values indicate closer directional alignment in the learned hyperspherical space.

### Inferred root angular distance

![vMF angular distance between inferred root centers](figures/vMF_root_inf_dist.png)

This heatmap visualizes angular distances between inferred root centers.

### Root classes on the learned hypersphere

![Root 12 classes on learned mu hypersphere](figures/vMF_root_map.png)

The 12 inferred root classes are projected onto a 3D PCA sphere for visualization.

### Learned \(\mu\) hypersphere

![Learned mu hypersphere visualization](figures/vMF_mu_map.png)

This figure visualizes the learned \(\mu\) directions. Points are colored by concentration parameter \(\kappa\).

### Chord types on the learned hypersphere

![Chord types on learned mu hypersphere](figures/vMF_chord1_map.png)

This figure shows chord-type distributions on the learned \(\mu\) hypersphere.

### Filtered chord-type visualization

![Filtered chord types on learned mu hypersphere](figures/vMF_chord2_map.png)

This figure removes `other` and `no_chord` classes to make the chord-type structure easier to observe.

## Additional Documents

- [Evaluation criteria and comparison with prior work](docs/vmf_evaluation_comparison.pdf)  
  Evaluation metrics, beginner-friendly explanations, and reference comparisons with related benchmarks and prior studies.

- [Conformer-vMF variables, losses, and generation parameters](docs/conformer_vmf_tables.pdf)  
  Input variables, prediction heads, loss functions, and generation-time control parameters used in the Conformer-vMF prototype.

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

## Method Summary

A mathematical summary of the Conformer-vMF training and generation formulation is available here:

- [Method summary: Conformer-vMF music representation and generation](docs/method_summary.md)

This document describes the vMF coordinate construction, harmonic center direction, multi-task training losses, optional P²OT prototype lens, and generation-time pitch/velocity/timing equations.

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

## License

This repository is released for research and educational purposes.  
Please check the license file before reuse.

