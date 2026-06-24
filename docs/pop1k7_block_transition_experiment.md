# Pop1K7_1000 Conformer-vMF Block Transition 実験まとめ

## 1. 実験の位置付け

本実験は、Pop1K7 由来の MIDI データを vMF 超球面表現へ変換し、和声機能の遷移、すなわち `function_transition` を block 単位で予測するための Conformer-vMF block transition 実験である。

確認済みの成果物は以下である。

```text
results/vmf_conformer_block_transition_pop1k7_1000_head_train_log.csv
results/vmf_conformer_block_transition_pop1k7_1000_finetune_from_head_train_log.csv

checkpoints/vmf_conformer_block_transition_pop1k7_1000_head_best.pt
checkpoints/vmf_conformer_block_transition_pop1k7_1000_finetune_from_head_best.pt
```

---

## 2. データセット統計

使用データは以下の processed dataset である。

```text
/content/drive/MyDrive/vMF-hypersphere-music/data/vmf_processed_pop1k7_1000
```

集計結果は以下の通りである。

| 項目 | 値 |
|---|---:|
| processed `.pt` ファイル数 | 1000 |
| unique MIDI path | 1000 |
| unique MIDI name | 1000 |
| duplicated MIDI path | 0 |
| 総イベント数 | 1,429,964 |
| 平均イベント数 / MIDI | 1,429.964 |
| 入力特徴次元 `x` | 10 |
| vMF方向 `mu` 次元 | 10 |

この結果から、本実験は **Pop1K7由来の1000個の異なるMIDIファイルを processed 化したデータセット**に基づくと説明できる。

---

## 3. processed `.pt` ファイルの内容

各 `.pt` ファイルには、以下のような時系列データが保存されている。

| キー | 形状の例 | 内容 |
|---|---:|---|
| `x` | `(T, 10)` | 入力特徴量 |
| `mu` | `(T, 10)` | vMF方向ベクトル |
| `root` | `(T,)` | root ラベル |
| `chord_like` | `(T,)` | chord-like 度合い |
| `template` | `(T,)` | コードテンプレート |
| `triad` | `(T,)` | 三和音カテゴリ |
| `seventh` | `(T,)` | seventh 関連カテゴリ |
| `beat` | `(T,)` | 拍位置 |
| `bar` | `(T,)` | 小節内位置 |
| `onset` | `(T,)` | onset 情報 |
| `velocity` | `(T, 1)` | velocity |
| `timing` | `(T, 1)` | timing |
| `duration` | `(T, 1)` | duration |
| `function` | `(T,)` | 和声機能ラベル |
| `function_transition` | `(T,)` | 和声機能遷移ラベル |
| `estimated_key_pc` | `int` | 推定キー |
| `midi_path` | `str` | 元 MIDI ファイルのパス |

例として、`000000_0.pt` では `x` と `mu` の形状は `(899, 10)` であり、元MIDIは以下である。

```text
/content/drive/MyDrive/vmf_raw_midi_dataset/Pop1K7/extracted/Pop1K7/midi_synchronized/src_001/0.mid
```

---

## 4. モデル・学習設定

checkpoint 内の `args` および `meta` から確認できた主要設定は以下である。

| 項目 | 値 |
|---|---:|
| `data_dir` | `vmf_processed_pop1k7_1000` |
| `pretrained_checkpoint` | `vmf_conformer_function_pop1k7_1000_best.pt` |
| `input_dim` | 10 |
| `mu_dim` | 10 |
| `d_model` | 128 |
| Conformer layers | 4 |
| attention heads | 4 |
| dropout | 0.1 |
| convolution kernel size | 15 |
| `events_per_block` | 8 |
| `num_blocks` | 16 |
| `block_stride` | 4 |
| `phrase_period` | 4 |
| `min_blocks` | 8 |
| `block_emb_dim` | 32 |
| `block_hidden_dim` | 128 |
| batch size | 16 |
| validation ratio | 0.1 |
| learning rate | 0.0001 |
| weight decay | 0.0001 |
| grad clip | 1.0 |
| seed | 42 |

分類対象は和声機能4クラスである。

| ID | class |
|---:|---|
| 0 | T |
| 1 | D |
| 2 | SD |
| 3 | OTHER |

また、`function_transition` は4機能間の遷移であり、`num_function_transition = 16` として保存されている。

---

## 5. 学習段階の整理

本実験は大きく2段階で構成される。

### 5.1 head_train 段階

`vmf_conformer_block_transition_pop1k7_1000_head_best.pt` に対応する段階である。

checkpoint の `args` では以下が確認された。

```text
freeze_base: True
epochs: 15
save_name: vmf_conformer_block_transition_pop1k7_1000_head
```

したがって、この段階では **Conformer本体を凍結し、block transition head を中心に学習した**と解釈できる。

### 5.2 finetune_from_head 段階

`vmf_conformer_block_transition_pop1k7_1000_finetune_from_head_best.pt` に対応する段階である。

`head_best.pt` と `finetune_from_head_best.pt` の state_dict 差分比較により、以下が確認された。

| 項目 | 値 |
|---|---:|
| head checkpoint parameter tensors | 238 |
| finetune checkpoint parameter tensors | 238 |
| changed block params | 132 |

差分が確認された代表的なパラメータは以下である。

```text
input_proj.0.weight
blocks.0.self_attn.in_proj_weight
blocks.0.self_attn.out_proj.weight
blocks.0.conv.depthwise_conv.weight
blocks.1.self_attn.in_proj_weight
blocks.2.self_attn.out_proj.weight
blocks.3.conv.depthwise_conv.weight
blocks.3.ffn2.net.1.weight
block_transition_head.net.3.weight
block_transition_head.net.6.weight
```

このため、fine-tuning 段階では、block transition head だけでなく、**input projection および Conformer blocks.0〜3 の重みも更新された**と説明できる。

---

## 6. head_train の性能

`head_train_log.csv` の最終 epoch 15 における主要指標は以下である。

| 指標 | 値 |
|---|---:|
| epoch | 15 |
| train_loss | 1.006934 |
| train_acc | 0.581213 |
| train_macro_f1 | 0.453454 |
| val_loss | 0.989265 |
| val_acc | 0.587373 |
| val_macro_f1 | 0.459589 |

### 6.1 head_train のクラス別 F1

| class | val F1 |
|---|---:|
| T | 0.696876 |
| D | 0.241776 |
| SD | 0.261795 |
| OTHER | 0.637909 |

head_train 段階では、T と OTHER は比較的検出されているが、D と SD の F1 が低く、遷移分類の細部はまだ十分ではない。

---

## 7. finetune_from_head の性能

`finetune_from_head_train_log.csv` の最終 epoch 10 における主要指標は以下である。

| 指標 | 値 |
|---|---:|
| epoch | 10 |
| train_loss | 0.490557 |
| train_acc | 0.808449 |
| train_macro_f1 | 0.780354 |
| val_loss | 0.435884 |
| val_acc | 0.831970 |
| val_macro_f1 | 0.807746 |

### 7.1 finetune_from_head のクラス別 F1

| class | val F1 |
|---|---:|
| T | 0.875350 |
| D | 0.737720 |
| SD | 0.772450 |
| OTHER | 0.845465 |

fine-tuning 後は、全クラスで F1 が改善しており、特に head_train で弱かった D と SD が大きく改善している。

---

## 8. head_train から finetune への改善量

| 指標 | head_train 最終 | finetune 最終 | 改善量 |
|---|---:|---:|---:|
| train_acc | 0.581213 | 0.808449 | +0.227236 |
| train_macro_f1 | 0.453454 | 0.780354 | +0.326900 |
| val_acc | 0.587373 | 0.831970 | +0.244597 |
| val_macro_f1 | 0.459589 | 0.807746 | +0.348157 |
| val_loss | 0.989265 | 0.435884 | -0.553381 |

最も重要な改善は、**val_macro_f1 が 0.459589 から 0.807746 へ改善した点**である。

---

## 9. 研究上の解釈

本実験から、以下のことが確認できる。

1. Pop1K7 由来の1000個の異なるMIDIファイルを processed 化できている。
2. 各 MIDI は、音高・root・コードテンプレート・拍節・velocity・timing・duration・和声機能・機能遷移を含む時系列データとして保存されている。
3. head_train 段階では、Conformer本体を凍結して block transition head を学習している。
4. fine-tuning 段階では、差分比較により Conformer blocks も更新されている。
5. fine-tuning により、検証 Macro-F1 は約 0.808、検証 Accuracy は約 0.832 に到達している。
6. 特に D と SD の F1 が大きく改善し、和声機能遷移の分類性能が向上している。

---
