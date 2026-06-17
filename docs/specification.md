# Attention Monitor — 仕様書・設計書

最終更新: 2026-06-18

## 1. 概要

Web カメラの映像をリアルタイムに解析し、配信者の **在席・注視ステータス** を自動判定するツール。
ライブ映像にステータスのオーバーレイを重ね、セッション中の統計（各ステータスの累計時間・集中率）を集計・表示する。

本ツールは、将来構築する「配信向けリアルタイムシステム」の中核技術（検出 → 状態判定 → 描画のループ）を最小構成で体験・検証する位置づけ。判定ロジックを I/O から分離することで、本命システムへの移植性を確保する。

### 技術スタック

- **Python** 3.13
- **uv**（プロジェクト・依存・仮想環境の管理）
- **Ultralytics YOLO**（`yolo11n-pose.pt`、COCO 学習済み pose モデル）
- **Supervision**（検出結果のデータ構造・アノテーション）
- **OpenCV**（カメラ入力・描画・ウィンドウ表示）

### 非対象（YAGNI / 今回やらないこと）

- 録画・動画ファイル保存
- 外部連携（OBS 連携・HTTP/ファイルでのステータス出力）
- 厳密な視線推定（eye gaze）— 本ツールは頭の向き（head pose）ベースの近似
- 独自データの学習・アノテーション（Roboflow 等は不要。後述）
- ゾーン/ライン越えカウント

## 2. 学習・アノテーションについて（補足）

本ツールに **アノテーション・データセット・学習は不要**。

- `yolo11n-pose.pt` は COCO 学習済みで、人および顔・体のキーポイント（鼻・両目・両耳・肩 等）を最初から検出できる。
- 注視判定は、この既存キーポイント座標に対する **幾何計算のみ**で行う。新クラスの学習が発生しない。
- Roboflow / アノテーションが必要になるのは「COCO に無い独自物体の検出」「ファインチューニング」「独自モデルの学習」を行う場合のみ。本ツールは該当しない（将来の本命システムで独自物体検出が必要になった時の出番）。

## 3. ステータス定義

主被写体（最大の顔 = 配信者）1 名について、以下の 3 ステータスを判定する。

| ステータス | 意味 | 判定条件 |
|---|---|---|
| `AWAY` | 離席 | 主被写体の顔キーポイントが検出されない |
| `FOCUSED` | 画面を見ている | 顔あり かつ 頭の向きが中央しきい値内 |
| `DISTRACTED` | 画面を見ていない | 顔あり かつ 頭の向きが中央から外れている |

## 4. アーキテクチャ

### 4.1 設計方針

- **パイプライン**: フレームを各段（取得 → 検出 → 判定 → 状態確定 → 統計 → 描画）に流す。
- **純粋ロジックの分離**: 注視判定（`attention`）と状態機械（`state`）は OpenCV / YOLO に依存しない純粋関数群とし、数値入力 → 列挙ステータス出力に限定する。単体テスト可能・移植可能にする。

### 4.2 モジュール構成

```
src/attention_monitor/
├─ config.py        # 閾値・モデルパス等を dataclass で一元管理
├─ capture.py       # フレーム供給源。web カメラを抽象化（後で動画/RTSP に差し替え可）
├─ detector.py      # YOLO-pose 推論ラッパ → sv.Detections + キーポイントを返す
├─ attention.py     # 【純粋ロジック】キーポイント → 頭の向き(yaw/pitch) → 注視クラス
├─ state.py         # 【純粋ロジック】状態機械。ヒステリシスでちらつきを抑える
├─ stats.py         # セッション統計（各ステータス累計時間・集中率%）
├─ render.py        # アノテーター＋HUD（ステータスバナー/統計パネル）描画
├─ pipeline.py      # 上記を毎フレーム結線するオーケストレータ
└─ app.py           # エントリポイント（ループ・キー操作・終了時サマリ）
```

### 4.3 各モジュールの責務・インタフェース

- **config.py** — `Config` データクラス。`model_path`, `camera_index`, `yaw_center_threshold`(度), `pitch_center_threshold`(度), `commit_seconds`(確定までの継続秒数), `keypoint_confidence_min` 等を保持。
  - 依存: なし
- **capture.py** — `FrameSource` 抽象。`read() -> frame | None`, `release()`。実装 `WebcamSource(camera_index)`。
  - 依存: OpenCV
- **detector.py** — `PoseDetector(model_path)`。`detect(frame) -> Detections`（`sv.Detections` ＋ 各検出のキーポイント配列）。
  - 依存: Ultralytics, Supervision
- **attention.py** — 純粋関数。`select_primary(detections) -> keypoints | None`（最大の顔を選定）、`estimate_head_pose(keypoints) -> HeadPose(yaw, pitch)`、`classify(head_pose | None, config) -> Status`（raw ステータス）。
  - 依存: なし（NumPy のみ）
- **state.py** — `StateMachine(config)`。`update(raw_status, dt) -> Status`（確定ステータス）。新 raw が `commit_seconds` 継続して初めて確定値を切り替えるヒステリシスを持つ。
  - 依存: なし
- **stats.py** — `SessionStats`。`update(committed_status, dt)`、`summary() -> 各ステータス累計時間・集中率%`。
  - 依存: なし
- **render.py** — `Renderer(config)`。`draw(frame, detections, raw_status, committed_status, stats) -> frame`。Supervision のアノテーター（キーポイント/ボックス）＋ ステータスバナー ＋ 統計 HUD を重ねる。
  - 依存: OpenCV, Supervision
- **pipeline.py** — `Pipeline`。上記を保持し、`process(frame, dt) -> frame`（1 フレーム分の結線）。
  - 依存: 上記モジュール
- **app.py** — エントリポイント。ループ・FPS 計測・`q` で終了・終了時に統計サマリを標準出力へ。
  - 依存: OpenCV, Pipeline

### 4.4 データフロー（毎フレーム）

```
WebcamSource.read()
  → PoseDetector.detect()                → Detections(+keypoints)
  → attention.select_primary()           → 主被写体 keypoints
  → attention.estimate_head_pose()       → HeadPose(yaw, pitch)
  → attention.classify()                 → raw Status
  → StateMachine.update(raw, dt)         → committed Status
  → SessionStats.update(committed, dt)
  → Renderer.draw(...)                   → 注釈付きフレーム
  → cv2.imshow / q で終了
```

## 5. 注視判定の詳細（attention.py）

### 5.1 主被写体の選定

複数人が映る場合、顔キーポイント（鼻・両目・両耳）のバウンディング面積が最大の人物を主被写体とする。

### 5.2 頭の向き（head pose）の近似

COCO キーポイント（0:鼻, 1:左目, 2:右目, 3:左耳, 4:右耳）を用いる。

- **yaw（左右）**: 両耳の中点に対する鼻の水平オフセットを、両耳間距離で正規化した値から近似角度を求める。横を向くと鼻が片寄り、片耳の信頼度が低下するため、その非対称性も補助に用いる。
- **pitch（上下）**: 鼻と両目の縦位置関係から簡易に近似する。

信頼度が `keypoint_confidence_min` 未満のキーポイントは欠損として扱う。顔キーポイントが不足して頭の向きを推定できない場合は `AWAY` 相当とする。

### 5.3 ステータス分類（raw）

- 顔なし / 推定不能 → `AWAY`
- `|yaw| <= yaw_center_threshold` かつ `|pitch| <= pitch_center_threshold` → `FOCUSED`
- それ以外（中央から外れている） → `DISTRACTED`

## 6. 状態機械とヒステリシス（state.py）

ちらつき防止が実用性の肝。生の判定（raw）と確定ステータス（committed）を分離する。

- 現在の確定ステータスと異なる raw が現れたら候補として計時を開始。
- その候補が `commit_seconds`（既定 0.7 秒）連続して観測されて初めて確定ステータスを切り替える。
- 候補が途切れたら計時をリセット。
- 統計・オーバーレイのメインバナーには **確定ステータスのみ**を反映する（raw は補助表示に留める）。

## 7. 統計（stats.py）

- 各ステータス（`AWAY` / `FOCUSED` / `DISTRACTED`）の累計時間を `dt` 加算で集計。
- 集中率 = `FOCUSED` 時間 / （在席時間 = `FOCUSED` + `DISTRACTED`）× 100%。
- セッション総時間。
- 終了時に標準出力へサマリを表示。

## 8. 描画（render.py）

- Supervision のアノテーターで主被写体のキーポイント／ボックスを描画。
- 画面上部に確定ステータスのバナー（色分け: FOCUSED=緑 / DISTRACTED=橙 / AWAY=赤）。
- 画面隅に統計 HUD（各ステータス累計時間・集中率・FPS）。
- 補助表示として raw ステータスや yaw/pitch を小さく表示（デバッグ用、任意）。

## 9. プロジェクト構成（uv）

```
attention-monitor/
├─ pyproject.toml          # uv 管理。deps: ultralytics, supervision, opencv-python
├─ src/attention_monitor/  # §4.2 のモジュール群
├─ tests/                  # attention / state の純粋ロジックを単体テスト
├─ models/                 # yolo11n-pose.pt（初回 DL 先）
├─ docs/specification.md   # 本書
└─ README.md
```

- 流れ: `uv init` → `uv add ultralytics supervision opencv-python` → `uv run python -m attention_monitor`
- venv は uv が自動管理。
- モデル `yolo11n-pose.pt` は初回実行時に自動ダウンロード（`models/` 配下）。

## 10. テスト方針

純粋ロジック 2 モジュールを pytest で担保する。

- **attention**: 既知のキーポイント座標（正面 / 左右に振る / 上下に振る / 欠損）を入力し、期待する yaw/pitch の符号・ステータスが出るか。
- **state**: raw ステータス列と `dt` を与え、ヒステリシス（`commit_seconds` 未満では切り替わらない、超えたら切り替わる、途切れでリセット）が正しく働くか。

I/O 系（capture / detector / render）は単体テスト対象外（手動確認）。

## 11. 操作

- 起動: `uv run python -m attention_monitor`
- 終了: ウィンドウ上で `q` キー → 統計サマリを標準出力に表示して終了。

## 12. 設定の初期値（暫定）

| 項目 | 既定値 |
|---|---|
| `camera_index` | 0 |
| `model_path` | `models/yolo11n-pose.pt` |
| `yaw_center_threshold` | 20°（要キャリブレーション） |
| `pitch_center_threshold` | 15°（要キャリブレーション） |
| `commit_seconds` | 0.7 |
| `keypoint_confidence_min` | 0.5 |

しきい値は実環境で調整する前提で config から可変とする。
