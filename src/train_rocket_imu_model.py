"""ROCKET 논문 아이디어를 IMU fall-risk 분류에 적용한다.

이 파일은 ROCKET 공식 구현체가 아니라, 논문의 핵심 아이디어를 현재 작은
IMU 데이터셋에 맞게 직접 재현한 `ROCKET-inspired` 파이프라인이다.

핵심 아이디어:

1. 100시점 x 9채널 IMU window를 그대로 사용한다.
2. 여러 개의 무작위 1D convolution kernel을 만든다.
3. 각 kernel 반응에서 max activation과 PPV(proportion of positive values)를 뽑는다.
4. 이렇게 만든 feature에 regularized LogisticRegression을 학습한다.

왜 이 방식을 추가하는가:

- 기존 RandomForest는 요약 feature 기반 baseline이다.
- 1D CNN/GRU는 직접 학습 parameter가 있어 현재 117개 window에서는 과적합 위험이 컸다.
- ROCKET 계열은 convolution feature를 무작위로 고정하고 선형 분류기만 학습하므로,
  작은 데이터에서도 논문 기반 시계열 feature extraction을 시도해볼 수 있다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from compare_sequence_models import LABEL_NAMES, RANDOM_STATE, load_sequence_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
METRICS_PATH = OUTPUT_DIR / "rocket_imu_model_metrics.json"
CONFUSION_PLOT_PATH = OUTPUT_DIR / "rocket_confusion_matrix_oof.png"
PCA_PLOT_PATH = OUTPUT_DIR / "rocket_feature_space_pca.png"
SEQUENCE_COMPARISON_PATH = OUTPUT_DIR / "sequence_model_comparison_metrics.json"
ALL_MODEL_COMPARISON_PLOT_PATH = OUTPUT_DIR / "all_model_comparison.png"


@dataclass(frozen=True)
class RandomKernel:
    """ROCKET-inspired transform에 사용할 무작위 convolution kernel."""

    length: int
    dilation: int
    padding: bool
    channels: np.ndarray
    weights: np.ndarray
    bias: float


def _to_builtin(value: Any) -> Any:
    """NumPy 타입을 JSON 저장 가능한 Python 기본 타입으로 바꾼다."""
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [_to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_to_builtin(item) for item in value.tolist()]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _generate_kernels(
    *,
    n_kernels: int,
    n_time_steps: int,
    n_channels: int,
    random_state: int,
) -> list[RandomKernel]:
    """데이터와 독립적으로 무작위 convolution kernel 목록을 만든다.

    Kernel은 train/test label을 보지 않고 생성한다. 이렇게 해야 kernel 자체가
    특정 fold의 검증 데이터 정보를 학습하지 않는다.
    """
    rng = np.random.default_rng(random_state)
    candidate_lengths = np.array([7, 9, 11, 15], dtype=int)
    kernels: list[RandomKernel] = []

    for _ in range(n_kernels):
        length = int(rng.choice(candidate_lengths[candidate_lengths <= n_time_steps]))
        max_dilation = max(1, (n_time_steps - 1) // (length - 1))
        dilation = int(rng.integers(1, max_dilation + 1))
        padding = bool(rng.integers(0, 2))

        # 다변량 IMU에서는 모든 축을 항상 함께 보지 않고, 일부 채널 조합도 탐색한다.
        # 예를 들어 accel_x/z만 보는 kernel, gyro 전체를 보는 kernel 등이 생길 수 있다.
        n_selected_channels = int(rng.integers(1, n_channels + 1))
        channels = np.sort(rng.choice(n_channels, size=n_selected_channels, replace=False))

        weights = rng.normal(0.0, 1.0, size=(length, n_selected_channels)).astype(np.float32)
        weights -= weights.mean()
        norm = np.linalg.norm(weights)
        if norm > 0:
            weights /= norm
        bias = float(rng.uniform(-1.0, 1.0))

        kernels.append(
            RandomKernel(
                length=length,
                dilation=dilation,
                padding=padding,
                channels=channels.astype(np.int64),
                weights=weights,
                bias=bias,
            )
        )
    return kernels


def _apply_kernel(x_sequence: np.ndarray, kernel: RandomKernel) -> np.ndarray:
    """하나의 kernel에서 max activation과 PPV feature를 계산한다."""
    receptive_field = (kernel.length - 1) * kernel.dilation + 1
    selected = x_sequence[:, :, kernel.channels]

    if kernel.padding:
        pad_total = receptive_field - 1
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        selected = np.pad(selected, ((0, 0), (pad_left, pad_right), (0, 0)), mode="constant")

    n_positions = selected.shape[1] - receptive_field + 1
    activations = np.empty((selected.shape[0], n_positions), dtype=np.float32)
    offsets = np.arange(kernel.length) * kernel.dilation

    for position in range(n_positions):
        window = selected[:, position + offsets, :]
        activations[:, position] = np.tensordot(
            window,
            kernel.weights,
            axes=([1, 2], [0, 1]),
        ) + kernel.bias

    max_activation = activations.max(axis=1)
    positive_proportion = (activations > 0).mean(axis=1)
    return np.column_stack([max_activation, positive_proportion]).astype(np.float32)


def transform_with_random_kernels(x_sequence: np.ndarray, kernels: list[RandomKernel]) -> np.ndarray:
    """여러 random kernel 반응을 하나의 tabular feature matrix로 변환한다."""
    feature_blocks = [_apply_kernel(x_sequence, kernel) for kernel in kernels]
    return np.concatenate(feature_blocks, axis=1).astype(np.float32)


def _scale_sequence_train_only(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """train fold 기준으로만 9채널 센서값을 표준화한다."""
    scaler = StandardScaler()
    n_channels = x_train.shape[2]
    scaler.fit(x_train.reshape(-1, n_channels))
    train_scaled = scaler.transform(x_train.reshape(-1, n_channels)).reshape(x_train.shape)
    test_scaled = scaler.transform(x_test.reshape(-1, n_channels)).reshape(x_test.shape)
    return train_scaled.astype(np.float32), test_scaled.astype(np.float32)


def _evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """공통 평가 지표를 계산한다."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "fall_recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def _fit_linear_classifier(x_train_features: np.ndarray, y_train: np.ndarray) -> tuple[LogisticRegression, StandardScaler]:
    """ROCKET feature에 regularized linear classifier를 학습한다."""
    feature_scaler = StandardScaler()
    x_train_scaled = feature_scaler.fit_transform(x_train_features)
    classifier = LogisticRegression(
        penalty="l2",
        C=0.35,
        class_weight="balanced",
        solver="liblinear",
        max_iter=5000,
        random_state=RANDOM_STATE,
    )
    classifier.fit(x_train_scaled, y_train)
    return classifier, feature_scaler


def evaluate_rocket_pipeline(n_kernels: int = 512) -> dict[str, Any]:
    """ROCKET-inspired 파이프라인을 group-aware out-of-fold 방식으로 검증한다."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    windows, x_sequence, y, groups = load_sequence_dataset()
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    out_of_fold_pred = np.full(len(y), fill_value=-1, dtype=np.int64)
    out_of_fold_probability = np.full(len(y), fill_value=np.nan, dtype=np.float32)
    fold_summaries: list[dict[str, Any]] = []

    for fold, (train_index, test_index) in enumerate(splitter.split(x_sequence, y, groups), start=1):
        x_train_scaled, x_test_scaled = _scale_sequence_train_only(
            x_sequence[train_index],
            x_sequence[test_index],
        )
        kernels = _generate_kernels(
            n_kernels=n_kernels,
            n_time_steps=x_sequence.shape[1],
            n_channels=x_sequence.shape[2],
            random_state=RANDOM_STATE + fold,
        )
        train_features = transform_with_random_kernels(x_train_scaled, kernels)
        test_features = transform_with_random_kernels(x_test_scaled, kernels)

        classifier, feature_scaler = _fit_linear_classifier(train_features, y[train_index])
        test_features_scaled = feature_scaler.transform(test_features)
        fold_probability = classifier.predict_proba(test_features_scaled)[:, 1]
        fold_pred = (fold_probability >= 0.5).astype(np.int64)

        out_of_fold_pred[test_index] = fold_pred
        out_of_fold_probability[test_index] = fold_probability

        fold_summaries.append(
            {
                "fold": fold,
                "test_windows": int(len(test_index)),
                "test_recordings": sorted(windows.iloc[test_index]["recording_id"].unique().tolist()),
                "rocket_features": int(train_features.shape[1]),
                **_evaluate_predictions(y[test_index], fold_pred),
            }
        )

    if np.any(out_of_fold_pred < 0) or np.isnan(out_of_fold_probability).any():
        raise ValueError("일부 window의 out-of-fold 예측이 만들어지지 않았습니다.")

    metrics = {
        "model": "ROCKET-inspired random convolution transform + LogisticRegression",
        "paper_basis": [
            {
                "title": "ROCKET: Exceptionally fast and accurate time series classification using random convolutional kernels",
                "url": "https://arxiv.org/abs/1910.13051",
            },
            {
                "title": "MiniRocket: A Very Fast (Almost) Deterministic Transform for Time Series Classification",
                "url": "https://arxiv.org/abs/2012.08791",
            },
        ],
        "implementation_note": (
            "This is a small ROCKET-inspired implementation for this repo, not the official ROCKET or MiniRocket package."
        ),
        "input_shape": {
            "windows": int(x_sequence.shape[0]),
            "time_steps": int(x_sequence.shape[1]),
            "channels": int(x_sequence.shape[2]),
        },
        "n_kernels": n_kernels,
        "rocket_features": int(n_kernels * 2),
        "validation_method": "3-fold StratifiedGroupKFold out-of-fold validation by recording_id",
        "overfitting_controls": [
            "random kernels are fixed and not learned from labels",
            "only a regularized linear classifier is trained",
            "class_weight=balanced",
            "train-fold-only sensor scaling",
            "train-fold-only ROCKET feature scaling",
            "group-aware out-of-fold validation",
        ],
        "fold_summaries": fold_summaries,
        "out_of_fold_probabilities": out_of_fold_probability,
        "out_of_fold_predictions": out_of_fold_pred,
        **_evaluate_predictions(y, out_of_fold_pred),
    }
    return _to_builtin(metrics)


def _plot_confusion_matrix(metrics: dict[str, Any]) -> None:
    """ROCKET-inspired 모델의 confusion matrix를 저장한다."""
    matrix = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Purples")
    ax.set_title("ROCKET-inspired IMU model")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(LABEL_NAMES, rotation=20, ha="right")
    ax.set_yticklabels(LABEL_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(CONFUSION_PLOT_PATH, dpi=160)
    plt.close(fig)


def _plot_feature_space_preview(n_kernels: int = 512) -> None:
    """전체 데이터 기준 ROCKET feature PCA preview를 만든다.

    이 그림은 학습 성능 평가용이 아니라, random convolution feature가 fall/normal
    window를 어떤 feature space로 펼치는지 확인하기 위한 설명용 시각화다.
    """
    windows, x_sequence, y, _ = load_sequence_dataset()
    sequence_scaler = StandardScaler()
    n_channels = x_sequence.shape[2]
    x_scaled = sequence_scaler.fit_transform(x_sequence.reshape(-1, n_channels)).reshape(x_sequence.shape)
    kernels = _generate_kernels(
        n_kernels=n_kernels,
        n_time_steps=x_sequence.shape[1],
        n_channels=x_sequence.shape[2],
        random_state=RANDOM_STATE,
    )
    features = transform_with_random_kernels(x_scaled.astype(np.float32), kernels)
    features = StandardScaler().fit_transform(features)
    embedding = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(features)

    fig, ax = plt.subplots(figsize=(7, 5))
    for label_value, label_name, color in [(0, "normal_activity", "#2f80ed"), (1, "fall_risk", "#c0392b")]:
        mask = y == label_value
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            label=label_name,
            alpha=0.78,
            s=48,
            c=color,
        )
    ax.set_title("ROCKET-inspired feature space preview")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PCA_PLOT_PATH, dpi=160)
    plt.close(fig)


def _plot_all_model_comparison(rocket_metrics: dict[str, Any]) -> None:
    """기존 모델 비교 결과와 ROCKET-inspired 결과를 함께 시각화한다."""
    if SEQUENCE_COMPARISON_PATH.exists():
        sequence_results = json.loads(SEQUENCE_COMPARISON_PATH.read_text(encoding="utf-8"))["results"]
        model_results = {
            "RandomForest": sequence_results["RandomForest"],
            "1D CNN": sequence_results["1D CNN"],
            "GRU": sequence_results["GRU"],
            "ROCKET-inspired": rocket_metrics,
        }
    else:
        model_results = {"ROCKET-inspired": rocket_metrics}

    model_names = list(model_results)
    metrics = ["accuracy", "macro_f1", "fall_recall"]
    x = np.arange(len(model_names))
    width = 0.24

    fig, ax = plt.subplots(figsize=(10, 5))
    for offset, metric in enumerate(metrics):
        values = [model_results[name][metric] for name in model_names]
        ax.bar(x + (offset - 1) * width, values, width=width, label=metric)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("IMU model comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(ALL_MODEL_COMPARISON_PLOT_PATH, dpi=160)
    plt.close(fig)


def main() -> dict[str, Any]:
    """ROCKET-inspired 파이프라인을 실행하고 결과물을 저장한다."""
    metrics = evaluate_rocket_pipeline()
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    _plot_confusion_matrix(metrics)
    _plot_feature_space_preview()
    _plot_all_model_comparison(metrics)

    print("Model: ROCKET-inspired random convolution transform + LogisticRegression")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Fall recall: {metrics['fall_recall']:.4f}")
    print(f"Saved metrics to {METRICS_PATH}")
    print(f"Saved confusion matrix to {CONFUSION_PLOT_PATH}")
    print(f"Saved feature preview to {PCA_PLOT_PATH}")
    print(f"Saved all-model comparison to {ALL_MODEL_COMPARISON_PLOT_PATH}")
    return metrics


if __name__ == "__main__":
    main()
