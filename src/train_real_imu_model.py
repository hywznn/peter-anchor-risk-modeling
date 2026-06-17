"""실제 공개 IMU 기록으로 fall-risk 분류 모델을 학습한다.

이 스크립트는 `fetch_real_data.py`가 만든 window-level 데이터셋을 입력으로 사용한다.
현재 공개 데이터는 규모가 작고 feature engineering 이후 tabular 형태가 되므로,
복잡한 deep-learning sequence model보다 설명 가능한 baseline 모델을 사용한다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold


# 모든 입력/출력 경로는 repo root 기준으로 잡는다.
# 이렇게 하면 커맨드라인에서 직접 실행해도 되고, 다른 모듈에서 import해도 경로가 안정적이다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "real_imu_fall_detection_windows.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
METRICS_PATH = OUTPUT_DIR / "real_imu_model_metrics.json"
MODEL_PATH = OUTPUT_DIR / "real_imu_fall_model.joblib"
RANDOM_STATE = 42

# 모델에 실제로 들어가는 feature만 명시한다.
# 라벨, activity 이름, recording_id는 모델 입력에서 제외해 target leakage를 막는다.
FEATURE_COLUMNS = [
    "imu_acceleration_peak_ms2",
    "imu_acceleration_mean_ms2",
    "imu_acceleration_std_ms2",
    "gyro_peak_dps",
    "gyro_mean_dps",
    "body_angle_peak_deg",
    "body_angle_mean_deg",
    "worker_sway_proxy_ms2",
]
LABEL_NAMES = ["normal_activity", "fall_risk"]


def _to_builtin(value: Any) -> Any:
    """NumPy 타입을 JSON으로 저장 가능한 Python 기본 타입으로 변환한다."""
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_to_builtin(item) for item in value.tolist()]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _build_model() -> RandomForestClassifier:
    """동일한 hyperparameter를 가진 baseline classifier를 만든다.

    현재 입력은 작은 tabular feature table이므로 RandomForest가 첫 모델로 적합하다.
    또한 feature importance를 제공해 포트폴리오나 면접에서 모델 판단 근거를
    설명하기 쉽다.
    """
    return RandomForestClassifier(
        # tree 수를 충분히 두면 variance를 줄일 수 있고, 현재 데이터 규모에서는 속도 부담도 작다.
        n_estimators=240,
        # window가 117개뿐이므로 tree가 너무 깊어져 데이터를 외우지 않도록 제한한다.
        max_depth=8,
        # leaf에 최소 2개 sample을 두어 극소수 window만 설명하는 rule 생성을 줄인다.
        min_samples_leaf=2,
        # normal/fall class 수가 같지 않으므로 class_weight로 큰 class 쏠림을 완화한다.
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def _cross_validate(df: pd.DataFrame, n_splits: int = 3) -> dict[str, Any]:
    """Group-aware 교차검증을 수행하고 out-of-fold 예측값을 모은다.

    핵심은 `groups` 인자다. 같은 원본 recording에서 나온 window들은 서로 유사하므로,
    같은 recording이 train fold와 test fold에 동시에 들어가면 모델 성능이 과대평가될 수 있다.
    """
    x = df[FEATURE_COLUMNS]
    y = df["fall_event"].astype(int)
    groups = df["recording_id"]

    # StratifiedGroupKFold는 class 비율을 가능한 한 유지하면서도 같은 group이
    # train/test에 동시에 들어가지 않게 해준다.
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    # 각 row는 test fold에 들어갈 때 정확히 한 번 예측된다.
    # 이렇게 모은 out-of-fold prediction으로 전체 데이터 기준 confusion matrix를 만든다.
    out_of_fold_pred = np.full(len(df), fill_value=-1, dtype=int)
    fold_summaries: list[dict[str, Any]] = []

    for fold, (train_index, test_index) in enumerate(splitter.split(x, y, groups), start=1):
        model = _build_model()
        model.fit(x.iloc[train_index], y.iloc[train_index])
        fold_pred = model.predict(x.iloc[test_index])
        out_of_fold_pred[test_index] = fold_pred

        y_test = y.iloc[test_index]
        fold_summaries.append(
            {
                "fold": fold,
                "test_windows": int(len(test_index)),
                # 어떤 recording이 test로 빠졌는지 저장해 검증 설계를 나중에 확인할 수 있게 한다.
                "test_recordings": sorted(df.iloc[test_index]["recording_id"].unique().tolist()),
                "accuracy": accuracy_score(y_test, fold_pred),
                "fall_recall": recall_score(y_test, fold_pred, pos_label=1, zero_division=0),
                "confusion_matrix": confusion_matrix(y_test, fold_pred, labels=[0, 1]).tolist(),
            }
        )

    # -1이 남아 있으면 어떤 row가 한 번도 test fold에서 예측되지 않았다는 뜻이다.
    # 이 경우 out-of-fold 검증 결과가 불완전하므로 오류로 처리한다.
    if np.any(out_of_fold_pred < 0):
        raise ValueError("Cross-validation did not produce predictions for every row.")

    return {
        "validation_method": "StratifiedGroupKFold out-of-fold validation",
        "n_splits": n_splits,
        "out_of_fold_predictions": out_of_fold_pred,
        "fold_summaries": fold_summaries,
    }


def train_model(df: pd.DataFrame) -> tuple[RandomForestClassifier, dict[str, Any]]:
    """최종 모델을 학습하고 검증 지표를 반환한다.

    검증 지표는 out-of-fold prediction으로 계산한다. 검증이 끝난 뒤 저장용 모델은
    사용 가능한 전체 공개 데이터로 다시 학습한다. 작은 baseline 프로젝트에서
    흔히 쓰는 방식으로, 검증은 정직하게 하고 demo model은 전체 데이터를 활용한다.
    """
    missing = [column for column in FEATURE_COLUMNS + ["fall_event", "recording_id"] if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    x = df[FEATURE_COLUMNS]
    y = df["fall_event"].astype(int)

    # 검증 예측값은 같은 recording이 train/test에 동시에 들어가지 않도록 만든다.
    validation = _cross_validate(df)
    y_pred = validation["out_of_fold_predictions"]

    # 검증 후에는 저장용 demo model을 전체 window 데이터로 학습한다.
    model = _build_model()
    model.fit(x, y)

    # Feature importance는 인과 설명은 아니지만, RandomForest가 어떤 engineered signal에
    # 많이 의존했는지 보여주는 참고 자료로 쓸 수 있다.
    feature_importance = sorted(
        [
            {"feature": feature, "importance": float(importance)}
            for feature, importance in zip(FEATURE_COLUMNS, model.feature_importances_)
        ],
        key=lambda item: item["importance"],
        reverse=True,
    )

    metrics = {
        "primary_model": "RandomForestClassifier",
        "data_source": str(DATA_PATH.relative_to(PROJECT_ROOT)),
        "features": FEATURE_COLUMNS,
        "labels": LABEL_NAMES,
        "validation_method": validation["validation_method"],
        "n_splits": validation["n_splits"],
        "evaluated_windows": int(len(df)),
        "evaluated_recordings": int(df["recording_id"].nunique()),
        "class_counts": {
            "normal_activity": int((y == 0).sum()),
            "fall_risk": int((y == 1).sum()),
        },
        "accuracy": accuracy_score(y, y_pred),
        "macro_f1": f1_score(y, y_pred, average="macro", zero_division=0),
        "fall_recall": recall_score(y, y_pred, pos_label=1, zero_division=0),
        "classification_report": classification_report(
            y,
            y_pred,
            labels=[0, 1],
            target_names=LABEL_NAMES,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y, y_pred, labels=[0, 1]).tolist(),
        "fold_summaries": validation["fold_summaries"],
        "random_state": RANDOM_STATE,
        "validation_note": (
            "Metrics use out-of-fold predictions from real public IMU fall / "
            "daily-activity recordings with group-aware folds by recording. They "
            "do not validate Peter Anchor hardware, rope tension, harness pressure, "
            "or field safety performance."
        ),
        "feature_importance": feature_importance,
    }
    return model, _to_builtin(metrics)


def main() -> dict[str, Any]:
    """Window 데이터셋을 읽고 모델 학습 결과물을 저장한다."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} does not exist. Run python src/fetch_real_data.py first.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    model, metrics = train_model(df)

    # 학습된 classifier와 검증 지표는 분리해서 저장한다.
    # joblib 파일은 모델 재사용용이고, JSON 파일은 검토/포트폴리오 설명용이다.
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Primary model: RandomForestClassifier")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Fall recall: {metrics['fall_recall']:.4f}")
    print(f"Saved metrics to {METRICS_PATH}")
    return metrics


if __name__ == "__main__":
    main()
