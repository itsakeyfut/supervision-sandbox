# Attention Monitor

Web カメラ映像から在席・注視ステータス（FOCUSED / DISTRACTED / AWAY）を
リアルタイム判定し、オーバーレイ表示とセッション統計を出すツール。

YOLO-pose（COCO 学習済み）＋ Supervision ＋ OpenCV。学習・アノテーション不要。

## セットアップ

```bash
uv sync
```

## 実行

```bash
uv run python -m attention_monitor
```

`q` キーで終了し、ターミナルにセッション統計を表示します。

## キャリブレーション

頭の向き判定は「画面を見た状態」を基準にしたズレで行います。正しく座った状態で
メニュー「調整」→「基準を設定」（`Ctrl+R`）を実行すると、その向きが基準になります
（起動時にも自動で一度基準化されます）。閾値は `src/attention_monitor/config.py`
（`yaw_center_threshold` / `pitch_center_threshold` / `ema_alpha`）で調整できます。

## テスト

```bash
uv run pytest
```

## 設定

`src/attention_monitor/config.py` の `Config` で閾値を調整できます
（`yaw_center_threshold` / `pitch_center_threshold` / `commit_seconds` など）。
