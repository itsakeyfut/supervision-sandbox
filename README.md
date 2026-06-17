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

## テスト

```bash
uv run pytest
```

## 設定

`src/attention_monitor/config.py` の `Config` で閾値を調整できます
（`yaw_center_threshold` / `pitch_center_threshold` / `commit_seconds` など）。
