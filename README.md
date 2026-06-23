# Attention Monitor

Web カメラ映像から在席・注視ステータス（FOCUSED / DISTRACTED / AWAY）を
リアルタイム判定し、オーバーレイ表示とセッション統計を出すツール。

- 在席 / 主被写体：YOLO-pose（COCO 学習済み）＋ Supervision
- 頭部姿勢＋視線：MediaPipe FaceLandmarker
- 表示：OpenCV / PySide6（Qt）

学習・アノテーション不要。

## セットアップ

```bash
uv sync
```

## 実行

```bash
uv run python -m attention_monitor
```

ウィンドウの ✕、メニュー「ファイル」→「終了」、または `Ctrl+Q` で終了し、
ターミナルにセッション統計を表示します。

## キャリブレーション

頭の向きと視線を「画面を見た状態」を基準にしたズレで判定します。正面を向いて
メニュー「調整」→「基準を設定」（`Ctrl+R`）を実行すると、頭部姿勢と視線が
同時に基準化されます（起動時にも自動で一度）。

「調整」メニューの「骨格を表示」で、ポーズ骨格オーバーレイの表示 / 非表示を
切り替えられます。

## テスト

```bash
uv run pytest
```

## 設定

`src/attention_monitor/config.py` の `Config` で調整できます
（`yaw_center_threshold` / `pitch_center_threshold` / `gaze_threshold` /
`ema_alpha` / `commit_seconds` など）。
