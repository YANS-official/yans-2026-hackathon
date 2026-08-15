# 引用妥当性判定タスク（YANS 2026 ハッカソン）

論文中の引用文（引用文脈）と引用先の論文のペアを見て、その引用が**妥当（1）か不適切（0）か**を
判定するタスクです。

## 進め方（3ステップ）

1. **ベースライン notebook を実行する** — データの読み込みから提出ファイルの生成までを
   通しで体験できます（下の表の Colab バッジから開けます）
2. **手元で検証する** — ラベル付きの `dev_labeled.jsonl`（200問）で、自分の工夫が
   効いているかを確認します
3. **リーダーボードに提出する** — `dev_leaderboard.jsonl`（100問・ラベルなし）への
   予測（Task 1）または事例割当（Task 2）を提出してスコアを確認します。
   最終評価は当日 17:30 公開の `test.jsonl` で行います

## ベースライン notebook（Google Colab）

| Task | Notebook | 実行環境 |
|------|----------|------|
| Task 1（独自モデル部門）: 特徴量＋ロジスティック回帰 | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YANS-official/yans-2026-hackathon/blob/main/notebooks/task1_baseline.ipynb) | CPU で数分 |
| Task 1（独自モデル部門）: Fine-tuning | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YANS-official/yans-2026-hackathon/blob/main/notebooks/task1_finetune.ipynb) | **GPU ランタイム必須**（下記） |
| Task 2（事例設計部門）: 事例選択 | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YANS-official/yans-2026-hackathon/blob/main/notebooks/task2_baseline.ipynb) | CPU で可（kNN の埋め込み計算のみ GPU 推奨） |

- **GPU が必要なのは `task1_finetune.ipynb` のみ**です。Colab のメニュー
  ［ランタイム］→［ランタイムのタイプを変更］→ **T4 GPU** を選んでから実行してください。
  ほかの2冊は標準（CPU）ランタイムで動きます（Task 2 の kNN 節の埋め込み計算は
  CPU では数十分かかるため、GPU ランタイムでの実行を推奨します）
- データは notebook 内のセルが本リポジトリをクローンして自動取得します。
  ローカルで実行する場合は、本リポジトリをクローンして `notebooks/` の notebook を開いてください

## データ

データファイルはすべて `data/` ディレクトリにあります。

| ファイル | 内容 |
|----------|------|
| `data/papers.jsonl` | 論文メタデータ（タイトル・著者・年・概要）＋ LaTeX 本文。1行1論文 |
| `data/train.jsonl` | 学習用。`label` 付き（1000問） |
| `data/dev_labeled.jsonl` | 手元検証用。`label` 付き（200問） |
| `data/dev_leaderboard.jsonl` | リーダーボード対象。**入力のみ**（`label` なし、100問） |
| `data/test.jsonl` | 最終評価用。**入力のみ**（`label` なし、100問）。**当日 17:30 に公開**（それまでは同梱されません） |

### レコード形式

`train.jsonl` / `dev_labeled.jsonl`:
```json
{"id": "train-001", "citation_context": "... [CITE] ...", "cited_paper_id": "Vol21No05_02", "label": 1}
```
`dev_leaderboard.jsonl` / `test.jsonl` は `label` を含みません（`id` / `citation_context` / `cited_paper_id` のみ）。

論文（`papers.jsonl`）:
```json
{"paper_id": "Vol21No05_02", "title": "...", "author": "...", "year": "2014", "abstract": "...", "tex_content": "..."}
```
`cited_paper_id` は `papers.jsonl` の `paper_id` に対応します。`[CITE]` は引用文脈中の引用位置を示すプレースホルダです。

## 提出ファイルの形式（JSONL・1行1問）

**Task 1（独自モデル部門）**: 各問に予測ラベル（0 か 1）を付けます。
```json
{"id": "dev-001", "prediction": 1}
```

**Task 2（事例設計部門）**: 各問に few-shot の事例（最大5件）を割り当てます。
```json
{"id": "dev-001", "exemplars": [{"citation_context": "... [CITE] ...", "cited_paper_id": "Vol21No05_02", "label": 1}]}
```

いずれもベースライン notebook の最終セルが生成します。形式は提出サーバが自動検査します。

## レギュレーション

使えるデータとモデル（Task 1: モデル1つあたりパラメータ数 100M 以下 等）、提出ファイルの
詳細な形式規定（文字数上限・件数上限）、提出回数は **[REGULATIONS.md](REGULATIONS.md)** に
まとまっています。取り組みを始める前に一読してください。

## 評価

指標は Accuracy（参考: F1 / Precision / Recall）。`dev_leaderboard` / `test` の正解は非公開です。

Task 1 の予測ラベルは、`dev_labeled.jsonl` に対して手元で採点できます:
```bash
python -m src.evaluate.metrics --gold data/dev_labeled.jsonl --pred your_predictions.jsonl
```
Task 2 の手元検証のやり方（自前の API キーで判定を再現する方法）は
`notebooks/task2_baseline.ipynb` に含まれています。
