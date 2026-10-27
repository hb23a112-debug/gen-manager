# gen-manager — 生成結果の管理ツール

画像・動画・音声などの生成結果を、**設定とセットで記録 → 自動整理 → 一覧 → 条件ごとに比較**する単一ファイルのツールです。
Python 3.9 以上・標準ライブラリのみ(`ingest` だけ Pillow を使用)。記録は管理フォルダ内の SQLite(`gen_manager.db`)に保存します。

## 要件との対応

| 要件 | 機能 |
|---|---|
| 生成設定と出力ファイルを記録 | `run` / `record` / `ingest`、Python API の `mgr.run(...)` |
| ファイル名や保存先を自動整理 | `日付/実験名/連番_条件名_xx.拡張子` に自動配置(`--layout` で変更可) |
| 生成時間や失敗の有無を一覧化 | `list`(状態・時間・出力数・成功率・失敗理由) |
| 複数条件の検証結果を比較 | `compare`(条件別の成功率・平均/中央値/最短/最長時間・評価値・**設定の差分**) |

## 使い方

```bash
# 1. 任意のコマンドを包んで実行(時間計測・失敗判定・出力の回収を自動化)
python gen_manager.py run -e LoRA検証 -c "lora=0.6" --set steps=20 --set cfg=7 \
      --watch ComfyUI/output -- python generate.py --lora 0.6

# 2. 後から手動で記録
python gen_manager.py record -e LoRA検証 -c "lora=0.9" --set steps=30 \
      --duration 41.5 out1.png out2.png
python gen_manager.py record -e LoRA検証 -c "lora=0.9" --status failed --error "CUDA OOM"

# 3. ComfyUI が PNG に埋め込んだ設定を読んで既存の output を一括登録(同じ画像は二重登録しない)
python gen_manager.py ingest ComfyUI/output -e comfy

# 4. 見る・比べる
python gen_manager.py list                      # 一覧(--status failed / --since 2026-10-01 / --keys steps,cfg)
python gen_manager.py show 3                    # 1件の詳細
python gen_manager.py compare -e LoRA検証         # 条件ごとに比較
python gen_manager.py compare --runs 3 5 8      # 特定の実行同士を比較
python gen_manager.py compare -e LoRA検証 --csv compare.csv
python gen_manager.py export --csv all.csv      # 全記録をCSVへ(Excelで文字化けしない)
python gen_manager.py check                     # 記録したファイルが残っているか確認(--hash で内容も照合)
```

`--root` で管理フォルダを指定します(既定は `./gen_library`、環境変数 `GEN_MANAGER_ROOT` でも可)。サブコマンドの前後どちらに書いても動きます。

### Python から使う

```python
from gen_manager import Manager

mgr = Manager("gen_library")
with mgr.run(experiment="LoRA検証", condition="lora=0.6", settings={"steps": 20}) as r:
    path = generate(...)        # 例外が出れば自動で「失敗」+エラー内容を記録(例外は再送出)
    r.add_output(path)
    r.set_metric("score", 0.82)  # 任意の評価値。compare で平均が出る
```

## 挙動のポイント

- **元ファイルは既定で触りません(コピー)。** `--move` で移動、`--in-place` で場所だけ記録。
- **失敗の判定:** 例外 / コマンドの終了コードが 0 以外 / コマンドが見つからない / 出力ファイルが消えている・読めない。
  `run --watch DIR --require-output` なら「出力が1つも増えなかった」も失敗にします。
  `run` は子コマンドの終了コードをそのまま返すので、シェルスクリプトやCIに組み込めます。
- **所要時間の集計は成功した実行のみ。** 失敗は途中で止まって短く出るため、平均を歪めるからです。
- **`compare` の設定差分:** グループ間で値が違う項目だけを表示します。`{a, b}` はグループ内で値がばらついている印です。
  `compare --by run` / `--by experiment` で比較の単位を変えられます。
- **安全面:** ファイル名の禁止文字・Windows予約名(CON など)を置換。`--layout` の結果が絶対パス・`..` なら拒否。保存先が管理フォルダの外になる場合も拒否。同名は `-2`, `-3` を付けて上書きしません。
- 状態 `RUN?` は、実行中か、途中でプロセスが強制終了された記録です。

### `--layout` のフィールド

`date`(2026-10-27) / `time`(HHMMSS) / `experiment` / `condition` / `run`(連番ID) / `seq`(その実行内の連番) / `stem`(元のファイル名) / `ext`(.png など)

```bash
--layout "{experiment}/{condition}/{stem}{ext}"
--layout "{date}/{run:04d}/{seq:02d}_{stem}{ext}"
```

## テスト

```bash
python test_gen_manager.py
```

記録・整理(コピー/移動/衝突/不正文字/不正レイアウト)、例外と Ctrl+C の失敗記録、比較の集計と設定差分、ComfyUI PNG の読み取り(Pillow があれば)、CLI の終了コードを検証しています。

**テストしていない範囲:** 実際の ComfyUI 環境での `ingest`(標準的な KSampler 構成を想定。独自ノード構成だと設定の一部が取れないことがあります)、Windows 上での動作(パス処理は考慮済みですが、実機では未確認)、複数プロセスから同時に書き込む運用。

## ライセンス

MIT License(`LICENSE` を参照)
