# Attention Monitor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Web カメラ映像から配信者の在席・注視ステータス（AWAY / FOCUSED / DISTRACTED）をリアルタイム判定し、オーバーレイ表示＋セッション統計を出すツールを作る。

**Architecture:** フレームを流すパイプライン構成。注視判定（`attention`）と状態確定（`state`）を OpenCV/YOLO から切り離した純粋ロジックにして単体テスト可能にする。検出は Ultralytics YOLO-pose（COCO 学習済み）＋ Supervision、入出力は OpenCV。

**Tech Stack:** Python 3.13 / uv / ultralytics (`yolo11n-pose.pt`) / supervision / opencv-python / numpy / pytest

## Global Constraints

- Python `>=3.13`、依存管理は **uv** のみ（pip 直叩き禁止）。
- パッケージは src レイアウト：`src/attention_monitor/`。
- 純粋ロジック（`attention.py` / `state.py` / `stats.py` / `config.py` / `status.py`）は OpenCV・Ultralytics・Supervision を import しない（numpy と stdlib のみ）。
- COCO キーポイント index 固定：`NOSE=0, L_EYE=1, R_EYE=2, L_EAR=3, R_EAR=4`。
- 学習・アノテーションは行わない（学習済みモデルをそのまま使用）。
- ステータスは `status.Status` Enum に一元化し、文字列リテラルで分岐しない。
- モデル重み（`*.pt`）は git にコミットしない（`.gitignore`）。
- 各タスクの最後に commit する。

---

## File Structure

| ファイル | 責務 | 依存 |
|---|---|---|
| `pyproject.toml` | uv プロジェクト定義・依存 | - |
| `.gitignore` | 重み・venv・キャッシュ除外 | - |
| `src/attention_monitor/__init__.py` | パッケージマーカー | - |
| `src/attention_monitor/__main__.py` | `python -m attention_monitor` 入口 | app |
| `src/attention_monitor/status.py` | `Status` Enum | stdlib |
| `src/attention_monitor/config.py` | `Config` dataclass | stdlib |
| `src/attention_monitor/attention.py` | 主被写体選定・頭の向き推定・分類（純粋） | numpy, status |
| `src/attention_monitor/state.py` | ヒステリシス付き状態機械（純粋） | status |
| `src/attention_monitor/stats.py` | セッション統計（純粋） | status |
| `src/attention_monitor/capture.py` | Web カメラ入力 | opencv |
| `src/attention_monitor/detector.py` | YOLO-pose 推論ラッパ | ultralytics, supervision |
| `src/attention_monitor/render.py` | アノテーション・HUD 描画 | opencv, supervision, status |
| `src/attention_monitor/pipeline.py` | 毎フレームの結線 | 上記すべて |
| `src/attention_monitor/app.py` | ループ・キー操作・終了サマリ | opencv, pipeline, capture, config |
| `tests/test_attention.py` | attention の単体テスト | pytest, numpy |
| `tests/test_state.py` | state の単体テスト | pytest |
| `tests/test_stats.py` | stats の単体テスト | pytest |
| `README.md` | 使い方 | - |

> 注（仕様書からの細部調整）：(1) モデル重みはルート直下に置かず ultralytics 既定のキャッシュ／ダウンロードに任せ、`model_path` 既定は `"yolo11n-pose.pt"`（bare name）。`models/` ディレクトリは作らず `*.pt` を gitignore。(2) ステータス Enum を `status.py` に独立させる（純粋ロジック間の結合を減らす）。(3) 主被写体は「最大の顔」ではなく「キーポイント bbox が最大の人物」（＝最も手前＝配信者）として選ぶ。体だけ見えて顔が見えない（後ろ向き）状態を「在席だが DISTRACTED」と区別できるようにするため。

---

## Task 1: プロジェクト雛形（uv + src レイアウト）

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/attention_monitor/__init__.py`
- Create: `tests/__init__.py`

**Interfaces:**
- Consumes: なし
- Produces: import 可能な `attention_monitor` パッケージ、`uv run` 環境。

- [ ] **Step 1: `pyproject.toml` を作成**

```toml
[project]
name = "attention-monitor"
version = "0.1.0"
description = "Realtime presence & attention status monitor (YOLO-pose + Supervision)"
requires-python = ">=3.13"
dependencies = [
    "ultralytics>=8.3.0",
    "supervision>=0.26.0",
    "opencv-python>=4.10.0",
    "numpy>=1.26",
]

[project.scripts]
attention-monitor = "attention_monitor.app:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/attention_monitor"]

[dependency-groups]
dev = ["pytest>=8.0"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

- [ ] **Step 2: `.gitignore` を作成**

```gitignore
# Python
__pycache__/
*.py[cod]
.pytest_cache/

# uv / venv
.venv/

# model weights
*.pt

# OS
.DS_Store
```

- [ ] **Step 3: パッケージ／テストのマーカーを作成**

`src/attention_monitor/__init__.py`:

```python
"""Realtime presence & attention status monitor."""

__version__ = "0.1.0"
```

`tests/__init__.py`:

```python
```

- [ ] **Step 4: 依存を解決して環境を作る**

Run: `uv sync`
Expected: `.venv` が作られ、ultralytics / supervision / opencv-python / pytest が解決される（成功で終了）。

- [ ] **Step 5: パッケージが import できることを確認**

Run: `uv run python -c "import attention_monitor; print(attention_monitor.__version__)"`
Expected: `0.1.0` と表示される。

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore src/attention_monitor/__init__.py tests/__init__.py uv.lock
git commit -m "chore: scaffold uv project (src layout)"
```

---

## Task 2: Status Enum と Config

**Files:**
- Create: `src/attention_monitor/status.py`
- Create: `src/attention_monitor/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: なし
- Produces:
  - `Status(Enum)`：メンバ `AWAY`, `FOCUSED`, `DISTRACTED`（値はそれぞれ `"away"`, `"focused"`, `"distracted"`）
  - `Config` dataclass（フィールドと既定値は下記）

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_config.py`:

```python
from attention_monitor.status import Status
from attention_monitor.config import Config


def test_status_members():
    assert {s.value for s in Status} == {"away", "focused", "distracted"}


def test_config_defaults():
    cfg = Config()
    assert cfg.camera_index == 0
    assert cfg.model_path == "yolo11n-pose.pt"
    assert cfg.yaw_center_threshold == 20.0
    assert cfg.pitch_center_threshold == 20.0
    assert cfg.pitch_neutral_ratio == 0.5
    assert cfg.commit_seconds == 0.7
    assert cfg.keypoint_confidence_min == 0.5
    assert cfg.window_name == "Attention Monitor"
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL（`ModuleNotFoundError: attention_monitor.status`）

- [ ] **Step 3: `status.py` を実装**

```python
from enum import Enum


class Status(Enum):
    AWAY = "away"
    FOCUSED = "focused"
    DISTRACTED = "distracted"
```

- [ ] **Step 4: `config.py` を実装**

```python
from dataclasses import dataclass


@dataclass
class Config:
    camera_index: int = 0
    model_path: str = "yolo11n-pose.pt"
    yaw_center_threshold: float = 20.0      # |yaw| <= これなら正面とみなす（度）
    pitch_center_threshold: float = 20.0    # |pitch| <= これなら水平とみなす（度）
    pitch_neutral_ratio: float = 0.5        # 正面時の「鼻が目より下にある量 / 顔幅」の基準値
    commit_seconds: float = 0.7             # ステータス確定までの継続秒数
    keypoint_confidence_min: float = 0.5    # これ未満のキーポイントは欠損扱い
    window_name: str = "Attention Monitor"
```

- [ ] **Step 5: テストが通ることを確認**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS（2 件）

- [ ] **Step 6: Commit**

```bash
git add src/attention_monitor/status.py src/attention_monitor/config.py tests/test_config.py
git commit -m "feat: add Status enum and Config dataclass"
```

---

## Task 3: attention.select_primary（主被写体の選定）

**Files:**
- Create: `src/attention_monitor/attention.py`
- Test: `tests/test_attention.py`

**Interfaces:**
- Consumes: `Status`（後続ステップで使用）
- Produces:
  - 定数 `NOSE=0, L_EYE=1, R_EYE=2, L_EAR=3, R_EAR=4`
  - `@dataclass(frozen=True) PrimarySubject(index: int, xy: np.ndarray, confidence: np.ndarray)`（`xy` 形状 `(K,2)`、`confidence` 形状 `(K,)`）
  - `select_primary(xy: np.ndarray | None, confidence: np.ndarray | None, min_confidence: float) -> PrimarySubject | None`
    - `xy` 形状 `(N,K,2)`、`confidence` 形状 `(N,K)`
    - 有効キーポイント（`confidence >= min_confidence`）が 3 点未満の人物は除外
    - 残った人物のうち、有効キーポイントの外接矩形（幅×高さ）が最大の人物を返す
    - 候補がなければ `None`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_attention.py`:

```python
import numpy as np

from attention_monitor.attention import select_primary, PrimarySubject


def _person(points):
    """points: list of (x, y, conf) 長さ K=17。足りない分は conf=0 で埋める。"""
    arr = np.zeros((17, 3), dtype=float)
    for i, (x, y, c) in enumerate(points):
        arr[i] = (x, y, c)
    return arr[:, :2], arr[:, 2]


def test_returns_none_when_empty():
    xy = np.zeros((0, 17, 2))
    conf = np.zeros((0, 17))
    assert select_primary(xy, conf, 0.5) is None


def test_skips_person_with_too_few_valid_keypoints():
    # 有効点が 2 点だけ（>=3 が必要）
    xy0, c0 = _person([(0, 0, 0.9), (10, 0, 0.9)])
    xy = xy0[None, ...]
    conf = c0[None, ...]
    assert select_primary(xy, conf, 0.5) is None


def test_picks_largest_person():
    small_xy, small_c = _person([(0, 0, 0.9), (10, 0, 0.9), (5, 10, 0.9)])
    big_xy, big_c = _person([(0, 0, 0.9), (100, 0, 0.9), (50, 100, 0.9)])
    xy = np.stack([small_xy, big_xy])
    conf = np.stack([small_c, big_c])
    result = select_primary(xy, conf, 0.5)
    assert isinstance(result, PrimarySubject)
    assert result.index == 1  # big
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `uv run pytest tests/test_attention.py -v`
Expected: FAIL（`ImportError: cannot import name 'select_primary'`）

- [ ] **Step 3: `attention.py` を実装（この時点では select_primary まで）**

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# COCO keypoint indices
NOSE = 0
L_EYE = 1
R_EYE = 2
L_EAR = 3
R_EAR = 4


@dataclass(frozen=True)
class PrimarySubject:
    index: int
    xy: np.ndarray          # shape (K, 2)
    confidence: np.ndarray  # shape (K,)


def select_primary(xy, confidence, min_confidence):
    """最も手前（キーポイント外接矩形が最大）の人物を返す。なければ None。"""
    if xy is None or confidence is None or len(xy) == 0:
        return None

    best_index = -1
    best_area = -1.0
    for i in range(len(xy)):
        valid = confidence[i] >= min_confidence
        if int(valid.sum()) < 3:
            continue
        pts = xy[i][valid]
        width = float(pts[:, 0].max() - pts[:, 0].min())
        height = float(pts[:, 1].max() - pts[:, 1].min())
        area = max(width, 1.0) * max(height, 1.0)
        if area > best_area:
            best_area = area
            best_index = i

    if best_index < 0:
        return None
    return PrimarySubject(best_index, xy[best_index], confidence[best_index])
```

- [ ] **Step 4: テストが通ることを確認**

Run: `uv run pytest tests/test_attention.py -v`
Expected: PASS（3 件）

- [ ] **Step 5: Commit**

```bash
git add src/attention_monitor/attention.py tests/test_attention.py
git commit -m "feat: add primary subject selection (attention.select_primary)"
```

---

## Task 4: attention.estimate_head_pose（頭の向き推定）

**Files:**
- Modify: `src/attention_monitor/attention.py`
- Test: `tests/test_attention.py`（追記）

**Interfaces:**
- Consumes: `PrimarySubject`, 定数 `NOSE..R_EAR`
- Produces:
  - `@dataclass(frozen=True) HeadPose(yaw: float, pitch: float)`（度。yaw=0 正面・pitch=0 水平、pitch>0 が下向き）
  - `estimate_head_pose(subject: PrimarySubject, min_confidence: float, pitch_neutral_ratio: float) -> HeadPose | None`
    - yaw：両耳が有効ならその x、無効なら両目の x を左右基準にする。鼻が無効 or 左右基準が取れなければ `None`
    - `span = abs(right_x - left_x)`、`ratio = (nose_x - (left_x+right_x)/2) / (span/2)` を `[-1,1]` にクリップし `*90`
    - pitch：両目が有効なら `(nose_y - eye_mid_y)/span - pitch_neutral_ratio` を `*90`、両目無効なら `0.0`

- [ ] **Step 1: 失敗するテストを書く（追記）**

`tests/test_attention.py` に追記：

```python
from attention_monitor.attention import estimate_head_pose, HeadPose


def _subject_with(nose, l_eye, r_eye, l_ear, r_ear):
    """各引数は (x, y, conf)。"""
    from attention_monitor.attention import PrimarySubject

    arr = np.zeros((17, 3), dtype=float)
    arr[0], arr[1], arr[2], arr[3], arr[4] = nose, l_eye, r_eye, l_ear, r_ear
    return PrimarySubject(0, arr[:, :2], arr[:, 2])


def test_head_pose_forward_is_centered():
    # 鼻が両耳の中点、目は鼻の 0.5*span 上 → yaw=0, pitch=0
    subj = _subject_with(
        nose=(0, 50, 0.9),
        l_eye=(-20, 0, 0.9), r_eye=(20, 0, 0.9),
        l_ear=(-50, 0, 0.9), r_ear=(50, 0, 0.9),
    )
    pose = estimate_head_pose(subj, 0.5, pitch_neutral_ratio=0.5)
    assert isinstance(pose, HeadPose)
    assert abs(pose.yaw) < 1e-6
    assert abs(pose.pitch) < 1e-6


def test_head_pose_turned_gives_large_yaw():
    # 鼻が右耳に寄る → |yaw| 大
    subj = _subject_with(
        nose=(40, 50, 0.9),
        l_eye=(-20, 0, 0.9), r_eye=(20, 0, 0.9),
        l_ear=(-50, 0, 0.9), r_ear=(50, 0, 0.9),
    )
    pose = estimate_head_pose(subj, 0.5, pitch_neutral_ratio=0.5)
    assert abs(pose.yaw) > 30.0


def test_head_pose_returns_none_without_nose():
    subj = _subject_with(
        nose=(0, 50, 0.0),  # 無効
        l_eye=(-20, 0, 0.9), r_eye=(20, 0, 0.9),
        l_ear=(-50, 0, 0.9), r_ear=(50, 0, 0.9),
    )
    assert estimate_head_pose(subj, 0.5, pitch_neutral_ratio=0.5) is None


def test_head_pose_pitch_down():
    # 鼻が目より 0.8*span 下 → pitch=(0.8-0.5)*90=27 度
    subj = _subject_with(
        nose=(0, 80, 0.9),
        l_eye=(-20, 0, 0.9), r_eye=(20, 0, 0.9),
        l_ear=(-50, 0, 0.9), r_ear=(50, 0, 0.9),
    )
    pose = estimate_head_pose(subj, 0.5, pitch_neutral_ratio=0.5)
    assert pose.pitch == 27.0
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `uv run pytest tests/test_attention.py -v`
Expected: FAIL（`ImportError: cannot import name 'estimate_head_pose'`）

- [ ] **Step 3: `attention.py` に実装を追記**

`HeadPose` dataclass を `PrimarySubject` の下に追加：

```python
@dataclass(frozen=True)
class HeadPose:
    yaw: float    # degrees, 0 = facing camera
    pitch: float  # degrees, 0 = level, + = looking down
```

`estimate_head_pose` を追加：

```python
def estimate_head_pose(subject, min_confidence, pitch_neutral_ratio):
    """主被写体の顔キーポイントから頭の向き(yaw/pitch)を近似。推定不能なら None。"""
    xy = subject.xy
    conf = subject.confidence

    def ok(i):
        return conf[i] >= min_confidence

    if not ok(NOSE):
        return None

    # yaw: 両耳優先、なければ両目
    if ok(L_EAR) and ok(R_EAR):
        left_x, right_x = float(xy[L_EAR, 0]), float(xy[R_EAR, 0])
    elif ok(L_EYE) and ok(R_EYE):
        left_x, right_x = float(xy[L_EYE, 0]), float(xy[R_EYE, 0])
    else:
        return None

    span = abs(right_x - left_x)
    if span < 1e-6:
        return None

    mid_x = (left_x + right_x) / 2.0
    ratio = (float(xy[NOSE, 0]) - mid_x) / (span / 2.0)
    yaw = float(np.clip(ratio, -1.0, 1.0)) * 90.0

    # pitch: 両目があれば鼻の縦オフセットから近似
    if ok(L_EYE) and ok(R_EYE):
        eye_mid_y = (float(xy[L_EYE, 1]) + float(xy[R_EYE, 1])) / 2.0
        pitch_ratio = (float(xy[NOSE, 1]) - eye_mid_y) / span
        pitch = (pitch_ratio - pitch_neutral_ratio) * 90.0
    else:
        pitch = 0.0

    return HeadPose(yaw=yaw, pitch=float(pitch))
```

- [ ] **Step 4: テストが通ることを確認**

Run: `uv run pytest tests/test_attention.py -v`
Expected: PASS（既存 3 + 新規 4 = 7 件）

- [ ] **Step 5: Commit**

```bash
git add src/attention_monitor/attention.py tests/test_attention.py
git commit -m "feat: add head pose estimation (attention.estimate_head_pose)"
```

---

## Task 5: attention.classify（ステータス分類）

**Files:**
- Modify: `src/attention_monitor/attention.py`
- Test: `tests/test_attention.py`（追記）

**Interfaces:**
- Consumes: `Status`, `HeadPose`
- Produces:
  - `classify(subject_present: bool, head_pose: HeadPose | None, yaw_threshold: float, pitch_threshold: float) -> Status`
    - `subject_present is False` → `Status.AWAY`
    - `head_pose is None`（在席だが顔が読めない＝後ろ向き等） → `Status.DISTRACTED`
    - `abs(yaw) <= yaw_threshold and abs(pitch) <= pitch_threshold` → `Status.FOCUSED`
    - それ以外 → `Status.DISTRACTED`

- [ ] **Step 1: 失敗するテストを書く（追記）**

`tests/test_attention.py` に追記：

```python
from attention_monitor.attention import classify
from attention_monitor.status import Status


def test_classify_away_when_absent():
    assert classify(False, None, 20.0, 20.0) is Status.AWAY


def test_classify_distracted_when_present_but_no_pose():
    assert classify(True, None, 20.0, 20.0) is Status.DISTRACTED


def test_classify_focused_when_centered():
    pose = HeadPose(yaw=5.0, pitch=-3.0)
    assert classify(True, pose, 20.0, 20.0) is Status.FOCUSED


def test_classify_distracted_when_yaw_exceeds():
    pose = HeadPose(yaw=45.0, pitch=0.0)
    assert classify(True, pose, 20.0, 20.0) is Status.DISTRACTED


def test_classify_distracted_when_pitch_exceeds():
    pose = HeadPose(yaw=0.0, pitch=27.0)
    assert classify(True, pose, 20.0, 20.0) is Status.DISTRACTED
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `uv run pytest tests/test_attention.py -v`
Expected: FAIL（`ImportError: cannot import name 'classify'`）

- [ ] **Step 3: `attention.py` に実装を追記**

ファイル先頭の import に追加：

```python
from attention_monitor.status import Status
```

末尾に追加：

```python
def classify(subject_present, head_pose, yaw_threshold, pitch_threshold):
    """在席フラグと頭の向きからステータスを決める（生判定）。"""
    if not subject_present:
        return Status.AWAY
    if head_pose is None:
        return Status.DISTRACTED
    if abs(head_pose.yaw) <= yaw_threshold and abs(head_pose.pitch) <= pitch_threshold:
        return Status.FOCUSED
    return Status.DISTRACTED
```

- [ ] **Step 4: テストが通ることを確認**

Run: `uv run pytest tests/test_attention.py -v`
Expected: PASS（既存 7 + 新規 5 = 12 件）

- [ ] **Step 5: Commit**

```bash
git add src/attention_monitor/attention.py tests/test_attention.py
git commit -m "feat: add status classification (attention.classify)"
```

---

## Task 6: state.StateMachine（ヒステリシス）

**Files:**
- Create: `src/attention_monitor/state.py`
- Test: `tests/test_state.py`

**Interfaces:**
- Consumes: `Status`
- Produces:
  - `StateMachine(commit_seconds: float)`
    - 属性 `committed: Status`（初期値 `Status.AWAY`）
    - `update(raw: Status, dt: float) -> Status`：
      - `raw == committed` なら候補をリセットして `committed` を返す
      - `raw` が現候補と同じなら経過時間に `dt` を加算、違えば候補を `raw` に切替え経過を `dt` で開始
      - 経過 `>= commit_seconds` で `committed = raw` に確定・候補リセット
      - 常に最新の `committed` を返す

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_state.py`:

```python
from attention_monitor.state import StateMachine
from attention_monitor.status import Status


def test_starts_away():
    sm = StateMachine(commit_seconds=1.0)
    assert sm.committed is Status.AWAY


def test_no_switch_before_commit_seconds():
    sm = StateMachine(commit_seconds=1.0)
    assert sm.update(Status.FOCUSED, 0.5) is Status.AWAY
    assert sm.update(Status.FOCUSED, 0.4) is Status.AWAY  # 累計 0.9 < 1.0


def test_switch_after_commit_seconds():
    sm = StateMachine(commit_seconds=1.0)
    sm.update(Status.FOCUSED, 0.5)
    sm.update(Status.FOCUSED, 0.4)
    assert sm.update(Status.FOCUSED, 0.2) is Status.FOCUSED  # 累計 1.1 >= 1.0


def test_candidate_change_resets_timer():
    sm = StateMachine(commit_seconds=1.0)
    sm.update(Status.FOCUSED, 0.6)
    # 候補が変わると計時リセット（DISTRACTED を 0.6 から数え直し）
    assert sm.update(Status.DISTRACTED, 0.6) is Status.AWAY
    assert sm.update(Status.DISTRACTED, 0.5) is Status.DISTRACTED  # 0.6+0.5=1.1


def test_raw_equal_committed_clears_candidate():
    sm = StateMachine(commit_seconds=1.0)
    sm.update(Status.FOCUSED, 0.9)        # 候補 FOCUSED, 経過 0.9
    sm.update(Status.AWAY, 0.5)           # raw == committed(AWAY) → 候補クリア
    assert sm.update(Status.FOCUSED, 0.5) is Status.AWAY  # 0.5 のみ → まだ確定しない
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `uv run pytest tests/test_state.py -v`
Expected: FAIL（`ModuleNotFoundError: attention_monitor.state`）

- [ ] **Step 3: `state.py` を実装**

```python
from attention_monitor.status import Status


class StateMachine:
    """生判定(raw)が commit_seconds 継続して初めて確定ステータスを切り替える。"""

    def __init__(self, commit_seconds):
        self.commit_seconds = commit_seconds
        self.committed = Status.AWAY
        self._candidate = None
        self._elapsed = 0.0

    def update(self, raw, dt):
        if raw == self.committed:
            self._candidate = None
            self._elapsed = 0.0
            return self.committed

        if raw == self._candidate:
            self._elapsed += dt
        else:
            self._candidate = raw
            self._elapsed = dt

        if self._elapsed >= self.commit_seconds:
            self.committed = raw
            self._candidate = None
            self._elapsed = 0.0

        return self.committed
```

- [ ] **Step 4: テストが通ることを確認**

Run: `uv run pytest tests/test_state.py -v`
Expected: PASS（5 件）

- [ ] **Step 5: Commit**

```bash
git add src/attention_monitor/state.py tests/test_state.py
git commit -m "feat: add hysteresis state machine (state.StateMachine)"
```

---

## Task 7: stats.SessionStats（セッション統計）

**Files:**
- Create: `src/attention_monitor/stats.py`
- Test: `tests/test_stats.py`

**Interfaces:**
- Consumes: `Status`
- Produces:
  - `SessionStats()`
    - 属性 `durations: dict[Status, float]`（3 ステータスを 0.0 初期化）
    - `update(status: Status, dt: float) -> None`
    - プロパティ `total -> float`（全合計）
    - プロパティ `present -> float`（FOCUSED+DISTRACTED）
    - プロパティ `focus_rate -> float`（present>0 のとき `100*FOCUSED/present`、それ以外 `0.0`）
    - `summary() -> str`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_stats.py`:

```python
from attention_monitor.stats import SessionStats
from attention_monitor.status import Status


def test_accumulates_durations():
    s = SessionStats()
    s.update(Status.FOCUSED, 1.0)
    s.update(Status.FOCUSED, 0.5)
    s.update(Status.AWAY, 2.0)
    assert s.durations[Status.FOCUSED] == 1.5
    assert s.durations[Status.AWAY] == 2.0
    assert s.total == 3.5


def test_focus_rate():
    s = SessionStats()
    s.update(Status.FOCUSED, 3.0)
    s.update(Status.DISTRACTED, 1.0)
    s.update(Status.AWAY, 10.0)  # present に含めない
    assert s.present == 4.0
    assert s.focus_rate == 75.0


def test_focus_rate_zero_when_no_presence():
    s = SessionStats()
    s.update(Status.AWAY, 5.0)
    assert s.focus_rate == 0.0


def test_summary_is_string_with_values():
    s = SessionStats()
    s.update(Status.FOCUSED, 2.0)
    text = s.summary()
    assert isinstance(text, str)
    assert "focused" in text
    assert "%" in text
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `uv run pytest tests/test_stats.py -v`
Expected: FAIL（`ModuleNotFoundError: attention_monitor.stats`）

- [ ] **Step 3: `stats.py` を実装**

```python
from attention_monitor.status import Status


class SessionStats:
    def __init__(self):
        self.durations = {
            Status.AWAY: 0.0,
            Status.FOCUSED: 0.0,
            Status.DISTRACTED: 0.0,
        }

    def update(self, status, dt):
        self.durations[status] += dt

    @property
    def total(self):
        return sum(self.durations.values())

    @property
    def present(self):
        return self.durations[Status.FOCUSED] + self.durations[Status.DISTRACTED]

    @property
    def focus_rate(self):
        present = self.present
        if present <= 0.0:
            return 0.0
        return 100.0 * self.durations[Status.FOCUSED] / present

    def summary(self):
        lines = ["=== Session Summary ==="]
        for status in (Status.FOCUSED, Status.DISTRACTED, Status.AWAY):
            lines.append(f"{status.value}: {self.durations[status]:.1f}s")
        lines.append(f"focus rate: {self.focus_rate:.1f}%")
        lines.append(f"total: {self.total:.1f}s")
        return "\n".join(lines)
```

- [ ] **Step 4: テストが通ることを確認**

Run: `uv run pytest tests/test_stats.py -v`
Expected: PASS（4 件）

- [ ] **Step 5: Commit**

```bash
git add src/attention_monitor/stats.py tests/test_stats.py
git commit -m "feat: add session statistics (stats.SessionStats)"
```

---

## Task 8: capture.WebcamSource（カメラ入力）

**Files:**
- Create: `src/attention_monitor/capture.py`

**Interfaces:**
- Consumes: なし
- Produces:
  - `WebcamSource(camera_index: int)`：オープン失敗時に `RuntimeError`
  - `read() -> np.ndarray | None`（読めなければ `None`）
  - `release() -> None`

これは I/O モジュールのため単体テストは行わず、スモーク確認のみ。

- [ ] **Step 1: `capture.py` を実装**

```python
import cv2


class WebcamSource:
    """OpenCV VideoCapture を薄くラップしたフレーム供給源。"""

    def __init__(self, camera_index):
        self._cap = cv2.VideoCapture(camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"camera {camera_index} を開けませんでした")

    def read(self):
        ok, frame = self._cap.read()
        if not ok:
            return None
        return frame

    def release(self):
        self._cap.release()
```

- [ ] **Step 2: import スモーク確認**

Run: `uv run python -c "from attention_monitor.capture import WebcamSource; print('ok')"`
Expected: `ok`（カメラは開かないので import のみ確認）

- [ ] **Step 3: Commit**

```bash
git add src/attention_monitor/capture.py
git commit -m "feat: add webcam frame source (capture.WebcamSource)"
```

---

## Task 9: detector.PoseDetector（YOLO-pose 推論）

**Files:**
- Create: `src/attention_monitor/detector.py`

**Interfaces:**
- Consumes: `model_path: str`
- Produces:
  - `PoseDetector(model_path: str)`
  - `detect(frame: np.ndarray) -> sv.KeyPoints`（`.xy` 形状 `(N,K,2)`、`.confidence` 形状 `(N,K)`。検出なしなら N=0）

スモーク確認は合成フレーム（黒画像）で行う。初回はモデル重みのダウンロードが走る。

- [ ] **Step 1: `detector.py` を実装**

```python
import supervision as sv
from ultralytics import YOLO


class PoseDetector:
    """Ultralytics YOLO-pose を Supervision の KeyPoints に変換して返す。"""

    def __init__(self, model_path):
        self._model = YOLO(model_path)

    def detect(self, frame):
        results = self._model(frame, verbose=False)[0]
        return sv.KeyPoints.from_ultralytics(results)
```

- [ ] **Step 2: 合成フレームでスモーク確認（重みDLが走る）**

Run:
```bash
uv run python -c "import numpy as np; from attention_monitor.detector import PoseDetector; d=PoseDetector('yolo11n-pose.pt'); kp=d.detect(np.zeros((480,640,3),dtype='uint8')); print(type(kp).__name__, kp.xy.shape)"
```
Expected: `KeyPoints (0, 17, 2)`（黒画像なので検出 0。初回はモデルDLログが出る）

- [ ] **Step 3: Commit**

```bash
git add src/attention_monitor/detector.py
git commit -m "feat: add YOLO-pose detector (detector.PoseDetector)"
```

---

## Task 10: render.Renderer（描画・HUD）

**Files:**
- Create: `src/attention_monitor/render.py`

**Interfaces:**
- Consumes: `Config`, `Status`, `PrimarySubject`, `SessionStats`, `sv.KeyPoints`
- Produces:
  - `Renderer(config: Config)`
  - `draw(frame: np.ndarray, keypoints: sv.KeyPoints, primary: PrimarySubject | None, raw: Status, committed: Status, stats: SessionStats, fps: float) -> np.ndarray`

合成フレームでスモーク確認（同形状の ndarray が返る）。

- [ ] **Step 1: `render.py` を実装**

```python
import cv2
import numpy as np
import supervision as sv

from attention_monitor.status import Status

_STATUS_COLOR = {
    Status.FOCUSED: (0, 180, 0),       # 緑 (BGR)
    Status.DISTRACTED: (0, 165, 255),  # 橙
    Status.AWAY: (0, 0, 220),          # 赤
}
_STATUS_LABEL = {
    Status.FOCUSED: "FOCUSED",
    Status.DISTRACTED: "DISTRACTED",
    Status.AWAY: "AWAY",
}


class Renderer:
    def __init__(self, config):
        self._config = config
        self._vertex = sv.VertexAnnotator(radius=4)
        self._edge = sv.EdgeAnnotator(thickness=2)

    def draw(self, frame, keypoints, primary, raw, committed, stats, fps):
        out = frame.copy()

        # スケルトン（検出があれば）
        if keypoints is not None and len(keypoints) > 0:
            out = self._edge.annotate(out, keypoints)
            out = self._vertex.annotate(out, keypoints)

        h, w = out.shape[:2]
        color = _STATUS_COLOR[committed]

        # 上部ステータスバナー
        cv2.rectangle(out, (0, 0), (w, 48), color, thickness=-1)
        cv2.putText(
            out, _STATUS_LABEL[committed], (12, 34),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA,
        )

        # 統計 HUD（左下）
        lines = [
            f"FPS: {fps:4.1f}   raw: {_STATUS_LABEL[raw]}",
            f"focused:    {stats.durations[Status.FOCUSED]:6.1f}s",
            f"distracted: {stats.durations[Status.DISTRACTED]:6.1f}s",
            f"away:       {stats.durations[Status.AWAY]:6.1f}s",
            f"focus rate: {stats.focus_rate:5.1f}%",
        ]
        y0 = h - 16 - 22 * (len(lines) - 1)
        overlay = out.copy()
        cv2.rectangle(overlay, (0, y0 - 22), (320, h), (0, 0, 0), thickness=-1)
        out = cv2.addWeighted(overlay, 0.45, out, 0.55, 0)
        for i, line in enumerate(lines):
            cv2.putText(
                out, line, (12, y0 + 22 * i),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
            )

        return out
```

- [ ] **Step 2: 合成フレームでスモーク確認**

Run:
```bash
uv run python -c "import numpy as np; import supervision as sv; from attention_monitor.config import Config; from attention_monitor.render import Renderer; from attention_monitor.status import Status; from attention_monitor.stats import SessionStats; r=Renderer(Config()); kp=sv.KeyPoints(xy=np.zeros((0,17,2),dtype='float32')); out=r.draw(np.zeros((480,640,3),dtype='uint8'),kp,None,Status.AWAY,Status.AWAY,SessionStats(),30.0); print(out.shape)"
```
Expected: `(480, 640, 3)`

- [ ] **Step 3: Commit**

```bash
git add src/attention_monitor/render.py
git commit -m "feat: add overlay renderer with status banner and stats HUD"
```

---

## Task 11: pipeline.Pipeline（結線）

**Files:**
- Create: `src/attention_monitor/pipeline.py`

**Interfaces:**
- Consumes: `Config`, `PoseDetector`, `select_primary`, `estimate_head_pose`, `classify`, `StateMachine`, `SessionStats`, `Renderer`
- Produces:
  - `Pipeline(config: Config)`：属性 `stats: SessionStats`, `state: StateMachine`
  - `process(frame: np.ndarray, dt: float, fps: float) -> np.ndarray`

合成フレーム（黒）でスモーク確認：例外なく動き、AWAY 累積が増える。

- [ ] **Step 1: `pipeline.py` を実装**

```python
from attention_monitor.attention import classify, estimate_head_pose, select_primary
from attention_monitor.detector import PoseDetector
from attention_monitor.render import Renderer
from attention_monitor.state import StateMachine
from attention_monitor.stats import SessionStats


class Pipeline:
    def __init__(self, config):
        self._config = config
        self._detector = PoseDetector(config.model_path)
        self._renderer = Renderer(config)
        self.state = StateMachine(config.commit_seconds)
        self.stats = SessionStats()

    def process(self, frame, dt, fps):
        cfg = self._config
        keypoints = self._detector.detect(frame)

        primary = select_primary(
            keypoints.xy, keypoints.confidence, cfg.keypoint_confidence_min
        )
        if primary is not None:
            head_pose = estimate_head_pose(
                primary, cfg.keypoint_confidence_min, cfg.pitch_neutral_ratio
            )
        else:
            head_pose = None

        raw = classify(
            primary is not None, head_pose,
            cfg.yaw_center_threshold, cfg.pitch_center_threshold,
        )
        committed = self.state.update(raw, dt)
        self.stats.update(committed, dt)

        return self._renderer.draw(
            frame, keypoints, primary, raw, committed, self.stats, fps
        )
```

> 注：`sv.KeyPoints` が検出ゼロのとき `confidence` が `None` になる実装差に備え、`select_primary` は `confidence is None` を `None` 返しで処理済み（Task 3）。

- [ ] **Step 2: 合成フレームでスモーク確認**

Run:
```bash
uv run python -c "import numpy as np; from attention_monitor.config import Config; from attention_monitor.pipeline import Pipeline; from attention_monitor.status import Status; p=Pipeline(Config()); out=p.process(np.zeros((480,640,3),dtype='uint8'),0.1,30.0); print(out.shape, round(p.stats.durations[Status.AWAY],2))"
```
Expected: `(480, 640, 3) 0.1`

- [ ] **Step 3: Commit**

```bash
git add src/attention_monitor/pipeline.py
git commit -m "feat: wire per-frame pipeline (pipeline.Pipeline)"
```

---

## Task 12: app.main と __main__（エントリポイント）

**Files:**
- Create: `src/attention_monitor/app.py`
- Create: `src/attention_monitor/__main__.py`

**Interfaces:**
- Consumes: `Config`, `WebcamSource`, `Pipeline`
- Produces: `main() -> None`

カメラを使う実機確認。

- [ ] **Step 1: `app.py` を実装**

```python
import time

import cv2

from attention_monitor.capture import WebcamSource
from attention_monitor.config import Config
from attention_monitor.pipeline import Pipeline


def main():
    config = Config()
    source = WebcamSource(config.camera_index)
    pipeline = Pipeline(config)

    prev = time.perf_counter()
    fps = 0.0
    try:
        while True:
            frame = source.read()
            if frame is None:
                break

            now = time.perf_counter()
            dt = now - prev
            prev = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else 1.0 / dt

            out = pipeline.process(frame, dt, fps)
            cv2.imshow(config.window_name, out)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        source.release()
        cv2.destroyAllWindows()
        print(pipeline.stats.summary())
```

- [ ] **Step 2: `__main__.py` を実装**

```python
from attention_monitor.app import main

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 実機確認（カメラ必要）**

Run: `uv run python -m attention_monitor`
Expected:
- ウィンドウが開き、自分の映像にスケルトンが重なる。
- 正面を見る → 上部バナーが緑 `FOCUSED`。
- 横／下を向く → 約 0.7 秒後に橙 `DISTRACTED`。
- フレームアウト → 約 0.7 秒後に赤 `AWAY`。
- `q` で終了し、ターミナルに Session Summary が出る。

> しきい値が体感と合わなければ `config.py` の `yaw_center_threshold` / `pitch_center_threshold` / `pitch_neutral_ratio` / `commit_seconds` を調整する。

- [ ] **Step 4: Commit**

```bash
git add src/attention_monitor/app.py src/attention_monitor/__main__.py
git commit -m "feat: add application entrypoint (app.main / __main__)"
```

---

## Task 13: README と最終確認

**Files:**
- Create: `README.md`

- [ ] **Step 1: `README.md` を作成**

````markdown
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
````

- [ ] **Step 2: 全テストを通す**

Run: `uv run pytest -v`
Expected: PASS（test_config 2 + test_attention 12 + test_state 5 + test_stats 4 = 23 件）

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add README"
```

---

## Self-Review チェック結果

- **Spec coverage:** §3 ステータス→Task5、§4 アーキ/モジュール→Task1–12、§5 注視判定→Task3/4、§6 状態機械→Task6、§7 統計→Task7、§8 描画→Task10、§9 uv→Task1、§10 テスト→Task3–7、§11 操作→Task12。全節にタスク対応あり。
- **Placeholder scan:** TBD/TODO・抽象的指示なし。各コードステップに実コードを記載。
- **Type consistency:** `Status` / `PrimarySubject(index,xy,confidence)` / `HeadPose(yaw,pitch)` / `select_primary(xy,confidence,min_confidence)` / `estimate_head_pose(subject,min_confidence,pitch_neutral_ratio)` / `classify(subject_present,head_pose,yaw_threshold,pitch_threshold)` / `StateMachine.update(raw,dt)` / `SessionStats.update(status,dt)` / `Renderer.draw(...)` / `Pipeline.process(frame,dt,fps)` をタスク間で一貫使用。
