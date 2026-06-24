# GitHub upload instructions for PCA and chord-center results

## 1. Upload folders

Upload the following folders into the root of your GitHub repository.

```text
docs/
figures/
results/
```

The package is already organized so that files go to the correct locations.

## 2. README update

Open `README.md` and paste the contents of:

```text
README_pca_results_snippet.md
```

Recommended location:

```text
Key Findings / 主な発見
↓
PCA Variable Analysis / PCA変数分析
↓
Main Visualizations / 主な可視化結果
```

## 3. Recommended commit message

```text
Add PCA variable analysis and chord-center visualizations
```

## 4. Final repository structure

```text
vMF-hypersphere-music/
├── docs/
│   ├── pca_variable_analysis.md
│   ├── chord_center_visualization_analysis.md
│   └── interactive_chord_center_visualizations/
├── figures/
│   ├── pca_variable_analysis/
│   └── chord_center_visualizations/
├── results/
│   ├── pca_variable_analysis/
│   └── chord_center_visualizations/
└── README.md
```

## 5. Notes

The HTML files are optional. They are included under:

```text
docs/interactive_chord_center_visualizations/
```

GitHub may display them as source code rather than interactive pages unless GitHub Pages is enabled.
