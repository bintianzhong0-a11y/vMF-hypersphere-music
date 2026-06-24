## Differentiation from Other Generation Systems / 他の生成システムとの差別化

This project does not primarily aim to compete with large-scale music generation systems in final audio quality.  
本研究は、大規模音楽生成AIと最終的な音質で直接競争することを主目的としていません。

Instead, it focuses on interpretable and controllable symbolic music generation using a vMF hyperspherical representation.  
代わりに、vMF 超球面表現を用いた、解釈可能で制御しやすい symbolic music generation を目指しています。

The main distinction is that pitch, circle-of-fifths structure, metrical position, and harmonic context are represented as directions, and generation is controlled using alignment with the learned harmonic direction.  
主な差別化点は、音高・5度圏構造・拍節位置・和声文脈を方向として表現し、学習された和声方向との一致度に基づいて生成を制御する点です。

- [Differentiation from other music generation systems / 他の生成システムとの差別化](docs/generation_system_differentiation.md)

## Block Transition Experiment / 和声機能遷移実験

This experiment evaluates whether Conformer-vMF can predict block-level harmonic function transitions from Pop1K7-derived MIDI data.  
この実験では、Pop1K7 由来の MIDI データから、Conformer-vMF が block 単位の和声機能遷移を予測できるかを検証しています。

The processed dataset contains 1000 distinct MIDI files and approximately 1.43 million note events.  
処理済みデータセットは、1000個の異なる MIDI ファイルと約143万イベントから構成されています。

After head training and fine-tuning, the validation performance reached `val_acc = 0.832` and `val_macro_f1 = 0.808`.  
head 学習と fine-tuning の後、検証性能は `val_acc = 0.832`、`val_macro_f1 = 0.808` に到達しました。

- [Pop1K7 1000 Conformer-vMF block transition experiment](docs/pop1k7_block_transition_experiment.md)
