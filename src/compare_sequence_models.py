"""IMU 9축 시계열 모델을 비교 실험한다.

기존 `train_real_imu_model.py`는 센서 시계열을 window-level 요약 feature로
바꾼 뒤 RandomForest를 학습한다. 이 파일은 사용자의 고도화 요구에 맞춰
가속도 x/y/z, 자이로 x/y/z, 기울기 x/y/z 값을 100시점 x 9채널 시계열로
구성하고, 작은 딥러닝 모델인 1D CNN과 GRU를 비교한다.

현재 공개 IMU window는 117개뿐이므로 Transformer나 3D CNN처럼 parameter가
큰 모델은 비교 대상에서 제외한다. 대신 작은 모델, group-aware 검증, early
stopping, dropout, weight decay를 사용해 과적합 위험을 낮춘다.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# 경로는 repo root 기준으로 고정한다. 이렇게 하면 어느 디렉터리에서 실행해도
# 입력 CSV와 출력 파일 위치가 흔들리지 않는다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
SAMPLES_PATH = DATA_DIR / "real_imu_fall_detection_samples.csv"
WINDOWS_PATH = DATA_DIR / "real_imu_fall_detection_windows.csv"
RESULTS_PATH = OUTPUT_DIR / "sequence_model_comparison_metrics.json"
PLOT_PATH = OUTPUT_DIR / "sequence_model_comparison.png"
CONFUSION_PLOT_PATH = OUTPUT_DIR / "sequence_model_confusion_matrices.png"
RANDOM_STATE = 42

# 사용자가 질문한 3축 센서값을 그대로 9채널 시계열 입력으로 사용한다.
# acceleration_svm이나 angular_velocity_svm은 축별 값을 요약한 값이므로,
# 이번 실험에서는 “x/y/z를 입체적으로 본다”는 목적에 맞춰 제외한다.
SEQUENCE_COLUMNS = [
    "accel_x_ms2",
    "accel_y_ms2",
    "accel_z_ms2",
    "gyro_x_dps",
    "gyro_y_dps",
    "gyro_z_dps",
    "inclination_x_deg",
    "inclination_y_deg",
    "inclination_z_deg",
]

# 기존 baseline과 비교하기 위해 engineered feature도 같은 fold에서 다시 평가한다.
TABULAR_FEATURE_COLUMNS = [
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


@dataclass(frozen=True)
class TrainingConfig:
    """작은 데이터용 딥러닝 학습 설정을 모아둔다."""

    max_epochs: int = 140
    patience: int = 16
    min_delta: float = 0.0005
    batch_size: int = 16
    learning_rate: float = 0.001
    weight_decay: float = 0.001


class TinyCnn1d(nn.Module):
    """100시점 x 9채널 IMU window를 분류하는 작은 1D CNN 모델."""

    def __init__(self, n_channels: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            # Conv1d는 시간축을 따라 인접한 센서 패턴을 훑는다.
            # kernel_size=5는 약 5개 sample 안의 짧은 충격/회전 변화를 본다는 의미다.
            nn.Conv1d(n_channels, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=0.25),
            nn.Conv1d(16, 24, kernel_size=5, padding=2),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            # window 길이가 조금 달라져도 마지막에는 채널별 대표값 하나로 압축한다.
            nn.AdaptiveAvgPool1d(output_size=1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.35),
            nn.Linear(24, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """입력 shape를 (batch, time, channels)에서 Conv1d 형식으로 바꿔 예측한다."""
        x = x.transpose(1, 2)
        return self.classifier(self.network(x)).squeeze(1)


class TinyGru(nn.Module):
    """IMU window의 시간 순서를 직접 따라가는 작은 GRU 모델."""

    def __init__(self, n_channels: int) -> None:
        super().__init__()
        # hidden_size를 작게 둬서 117개 window를 외우는 방향으로 커지지 않게 한다.
        self.gru = nn.GRU(
            input_size=n_channels,
            hidden_size=16,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(p=0.40)
        self.classifier = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """마지막 시점의 hidden state를 window 전체 요약 표현으로 사용한다."""
        _, hidden = self.gru(x)
        last_hidden = hidden[-1]
        return self.classifier(self.dropout(last_hidden)).squeeze(1)


def _to_builtin(value: Any) -> Any:
    """NumPy/PyTorch 타입을 JSON 저장 가능한 Python 기본 타입으로 바꾼다."""
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
    if isinstance(value, torch.Tensor):
        return _to_builtin(value.detach().cpu().numpy())
    return value


def _set_reproducible_seed(seed: int) -> None:
    """fold마다 가능한 한 같은 조건에서 학습되도록 난수를 고정한다."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def _count_parameters(model: nn.Module) -> int:
    """학습 가능한 parameter 수를 세어 모델 복잡도를 비교한다."""
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def load_sequence_dataset() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """window CSV와 sample CSV를 연결해 100시점 x 9채널 입력을 만든다.

    `real_imu_fall_detection_windows.csv`에는 각 window의 시작/끝 sample 번호와
    recording_id가 있고, `real_imu_fall_detection_samples.csv`에는 실제 x/y/z
    센서 시계열이 들어 있다. 두 파일을 연결하면 딥러닝 모델에 넣을 수 있는
    3차원 배열 `(window 수, 시간 길이, 센서 채널 수)`을 만들 수 있다.
    """
    if not SAMPLES_PATH.exists() or not WINDOWS_PATH.exists():
        raise FileNotFoundError("먼저 python src/fetch_real_data.py 를 실행해 실제 IMU CSV를 만들어야 합니다.")

    samples = pd.read_csv(SAMPLES_PATH)
    windows = pd.read_csv(WINDOWS_PATH)

    missing_sample_columns = [column for column in ["source_file", "sample_index"] + SEQUENCE_COLUMNS if column not in samples.columns]
    missing_window_columns = [column for column in ["recording_id", "start_sample", "end_sample", "fall_event"] if column not in windows.columns]
    if missing_sample_columns or missing_window_columns:
        raise ValueError(
            f"필수 컬럼이 없습니다. samples={missing_sample_columns}, windows={missing_window_columns}"
        )

    samples_by_recording = {
        source_file: frame.sort_values("sample_index").reset_index(drop=True)
        for source_file, frame in samples.groupby("source_file", sort=False)
    }

    sequences: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[str] = []

    for _, window in windows.iterrows():
        recording_id = str(window["recording_id"])
        source_samples = samples_by_recording[recording_id]

        # window 생성 때 사용한 시작/끝 sample 번호로 원본 시계열을 다시 잘라낸다.
        # 길이가 100이 아니면 원본 CSV와 window CSV가 서로 맞지 않는 것이므로 오류로 본다.
        start_sample = int(window["start_sample"])
        end_sample = int(window["end_sample"])
        sequence_frame = source_samples[
            (source_samples["sample_index"] >= start_sample)
            & (source_samples["sample_index"] <= end_sample)
        ]
        if len(sequence_frame) != int(window["n_samples"]):
            raise ValueError(
                f"{recording_id} window {start_sample}-{end_sample} 길이가 맞지 않습니다: "
                f"expected={int(window['n_samples'])}, actual={len(sequence_frame)}"
            )

        sequences.append(sequence_frame[SEQUENCE_COLUMNS].to_numpy(dtype=np.float32))
        labels.append(int(window["fall_event"]))
        groups.append(recording_id)

    x_sequence = np.stack(sequences).astype(np.float32)
    y = np.array(labels, dtype=np.int64)
    group_array = np.array(groups)
    return windows, x_sequence, y, group_array


def _make_inner_validation_split(
    y_train_outer: np.ndarray,
    groups_train_outer: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """outer train fold 안에서 early stopping용 validation fold를 만든다.

    여기서도 group을 유지해야 한다. 같은 recording의 window 일부가 train에 있고
    일부가 validation에 있으면 early stopping 기준도 새 데이터 성능을 제대로
    반영하지 못한다.
    """
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    local_index = np.arange(len(y_train_outer))
    inner_train_index, inner_val_index = next(splitter.split(local_index, y_train_outer, groups_train_outer))
    return inner_train_index, inner_val_index


def _build_random_forest() -> RandomForestClassifier:
    """기존 baseline과 같은 성격의 RandomForest 모델을 만든다."""
    return RandomForestClassifier(
        n_estimators=240,
        max_depth=8,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def _evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """공통 성능 지표를 계산한다."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "fall_recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def evaluate_random_forest(windows: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> dict[str, Any]:
    """engineered feature RandomForest를 같은 outer fold 기준으로 평가한다."""
    x_tabular = windows[TABULAR_FEATURE_COLUMNS]
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    out_of_fold_pred = np.full(len(windows), fill_value=-1, dtype=np.int64)
    fold_summaries: list[dict[str, Any]] = []

    for fold, (train_index, test_index) in enumerate(splitter.split(x_tabular, y, groups), start=1):
        model = _build_random_forest()
        model.fit(x_tabular.iloc[train_index], y[train_index])
        fold_pred = model.predict(x_tabular.iloc[test_index])
        out_of_fold_pred[test_index] = fold_pred
        fold_summaries.append(
            {
                "fold": fold,
                "test_windows": int(len(test_index)),
                "test_recordings": sorted(windows.iloc[test_index]["recording_id"].unique().tolist()),
                **_evaluate_predictions(y[test_index], fold_pred),
            }
        )

    return {
        "model": "RandomForest engineered features",
        "input_shape": "window-level tabular feature",
        "trainable_parameters": None,
        "overfitting_controls": [
            "max_depth=8",
            "min_samples_leaf=2",
            "class_weight=balanced",
            "StratifiedGroupKFold by recording_id",
        ],
        "fold_summaries": fold_summaries,
        "out_of_fold_predictions": out_of_fold_pred,
        **_evaluate_predictions(y, out_of_fold_pred),
    }


def _make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    """NumPy 배열을 PyTorch DataLoader로 변환한다."""
    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _loss_for_logits(logits: torch.Tensor, targets: torch.Tensor, loss_fn: nn.Module) -> torch.Tensor:
    """binary classification용 loss를 계산한다."""
    return loss_fn(logits, targets)


def _predict_probabilities(model: nn.Module, x: np.ndarray, batch_size: int) -> np.ndarray:
    """학습된 딥러닝 모델의 fall 확률을 반환한다."""
    model.eval()
    probabilities: list[np.ndarray] = []
    loader = _make_loader(x, np.zeros(len(x), dtype=np.float32), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch_x, _ in loader:
            logits = model(batch_x)
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probabilities)


def _run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    """학습 또는 평가 epoch 하나를 수행하고 평균 loss를 반환한다."""
    is_training = optimizer is not None
    model.train(mode=is_training)
    losses: list[float] = []
    for batch_x, batch_y in loader:
        if is_training:
            optimizer.zero_grad()
        logits = model(batch_x)
        loss = _loss_for_logits(logits, batch_y, loss_fn)
        if is_training:
            loss.backward()
            # 작은 데이터에서 gradient가 튀면 빠르게 과적합될 수 있어 clipping을 둔다.
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses))


def train_deep_model(
    model_factory: Callable[[int], nn.Module],
    x_train_outer: np.ndarray,
    y_train_outer: np.ndarray,
    groups_train_outer: np.ndarray,
    x_test_outer: np.ndarray,
    config: TrainingConfig,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """한 outer fold에서 딥러닝 모델을 학습하고 test fold 예측을 만든다."""
    _set_reproducible_seed(seed)
    inner_train_local, inner_val_local = _make_inner_validation_split(y_train_outer, groups_train_outer)

    # sequence scaler는 inner train fold로만 fit한다. validation과 outer test는 학습 과정에서
    # 보지 않은 데이터이므로, train scaler로 transform만 한다.
    scaler = StandardScaler()
    n_channels = x_train_outer.shape[2]
    scaler.fit(x_train_outer[inner_train_local].reshape(-1, n_channels))
    x_train = scaler.transform(x_train_outer[inner_train_local].reshape(-1, n_channels)).reshape(
        x_train_outer[inner_train_local].shape
    ).astype(np.float32)
    x_val = scaler.transform(x_train_outer[inner_val_local].reshape(-1, n_channels)).reshape(
        x_train_outer[inner_val_local].shape
    ).astype(np.float32)
    x_test = scaler.transform(x_test_outer.reshape(-1, n_channels)).reshape(x_test_outer.shape).astype(np.float32)

    y_train = y_train_outer[inner_train_local].astype(np.float32)
    y_val = y_train_outer[inner_val_local].astype(np.float32)

    model = model_factory(x_train_outer.shape[2])
    trainable_parameters = _count_parameters(model)

    positive_count = max(float((y_train == 1).sum()), 1.0)
    negative_count = max(float((y_train == 0).sum()), 1.0)
    pos_weight = torch.tensor([negative_count / positive_count], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    train_loader = _make_loader(x_train, y_train, batch_size=config.batch_size, shuffle=True)
    val_loader = _make_loader(x_val, y_val, batch_size=config.batch_size, shuffle=False)

    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_val_loss = float("inf")
    best_train_loss = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.max_epochs + 1):
        train_loss = _run_one_epoch(model, train_loader, loss_fn, optimizer=optimizer)
        val_loss = _run_one_epoch(model, val_loader, loss_fn, optimizer=None)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss - config.min_delta:
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_val_loss = val_loss
            best_train_loss = train_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.patience:
            break

    model.load_state_dict(best_state)
    test_probabilities = _predict_probabilities(model, x_test, batch_size=config.batch_size)

    diagnostics = {
        "best_epoch": int(best_epoch),
        "epochs_trained": int(len(history)),
        "best_train_loss": float(best_train_loss),
        "best_val_loss": float(best_val_loss),
        "val_minus_train_loss_gap": float(best_val_loss - best_train_loss),
        "trainable_parameters": trainable_parameters,
    }
    return test_probabilities, diagnostics


def evaluate_deep_sequence_model(
    name: str,
    model_factory: Callable[[int], nn.Module],
    windows: pd.DataFrame,
    x_sequence: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    config: TrainingConfig,
) -> dict[str, Any]:
    """1D CNN 또는 GRU를 outer out-of-fold 방식으로 평가한다."""
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    out_of_fold_probabilities = np.full(len(windows), fill_value=np.nan, dtype=np.float32)
    fold_summaries: list[dict[str, Any]] = []
    trainable_parameters: int | None = None

    for fold, (train_index, test_index) in enumerate(splitter.split(x_sequence, y, groups), start=1):
        probabilities, diagnostics = train_deep_model(
            model_factory=model_factory,
            x_train_outer=x_sequence[train_index],
            y_train_outer=y[train_index],
            groups_train_outer=groups[train_index],
            x_test_outer=x_sequence[test_index],
            config=config,
            seed=RANDOM_STATE + fold,
        )
        out_of_fold_probabilities[test_index] = probabilities
        trainable_parameters = diagnostics["trainable_parameters"]

        fold_pred = (probabilities >= 0.5).astype(np.int64)
        fold_summaries.append(
            {
                "fold": fold,
                "test_windows": int(len(test_index)),
                "test_recordings": sorted(windows.iloc[test_index]["recording_id"].unique().tolist()),
                **diagnostics,
                **_evaluate_predictions(y[test_index], fold_pred),
            }
        )

    if np.isnan(out_of_fold_probabilities).any():
        raise ValueError(f"{name}에서 일부 window의 out-of-fold 예측이 만들어지지 않았습니다.")

    out_of_fold_pred = (out_of_fold_probabilities >= 0.5).astype(np.int64)
    loss_gaps = [summary["val_minus_train_loss_gap"] for summary in fold_summaries]

    return {
        "model": name,
        "input_shape": f"{x_sequence.shape[1]} time steps x {x_sequence.shape[2]} IMU channels",
        "sequence_columns": SEQUENCE_COLUMNS,
        "trainable_parameters": trainable_parameters,
        "overfitting_controls": [
            "StratifiedGroupKFold by recording_id",
            "inner group-aware validation fold for early stopping",
            f"early stopping patience={config.patience}",
            f"dropout in model",
            f"AdamW weight_decay={config.weight_decay}",
            "train-fold-only StandardScaler",
            "gradient clipping max_norm=1.0",
            "small hidden/channel sizes",
        ],
        "mean_val_minus_train_loss_gap": float(np.mean(loss_gaps)),
        "fold_summaries": fold_summaries,
        "out_of_fold_probabilities": out_of_fold_probabilities,
        "out_of_fold_predictions": out_of_fold_pred,
        **_evaluate_predictions(y, out_of_fold_pred),
    }


def _plot_metric_comparison(results: dict[str, dict[str, Any]]) -> None:
    """모델별 핵심 지표를 막대그래프로 저장한다."""
    model_names = list(results)
    metrics = ["accuracy", "macro_f1", "fall_recall"]
    x = np.arange(len(model_names))
    width = 0.24

    fig, ax = plt.subplots(figsize=(10, 5))
    for offset, metric in enumerate(metrics):
        values = [results[name][metric] for name in model_names]
        ax.bar(x + (offset - 1) * width, values, width=width, label=metric)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Sequence model comparison with group-aware out-of-fold validation")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=160)
    plt.close(fig)


def _plot_confusion_matrices(results: dict[str, dict[str, Any]]) -> None:
    """모델별 confusion matrix를 하나의 이미지로 저장한다."""
    model_names = list(results)
    fig, axes = plt.subplots(1, len(model_names), figsize=(5 * len(model_names), 4))
    if len(model_names) == 1:
        axes = [axes]

    for ax, name in zip(axes, model_names):
        matrix = np.array(results[name]["confusion_matrix"])
        image = ax.imshow(matrix, cmap="Blues")
        ax.set_title(name)
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


def compare_models() -> dict[str, Any]:
    """RandomForest, 1D CNN, GRU를 같은 기준으로 비교하고 결과를 저장한다."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    windows, x_sequence, y, groups = load_sequence_dataset()
    config = TrainingConfig()

    results = {
        "RandomForest": evaluate_random_forest(windows, y, groups),
        "1D CNN": evaluate_deep_sequence_model(
            name="1D CNN",
            model_factory=TinyCnn1d,
            windows=windows,
            x_sequence=x_sequence,
            y=y,
            groups=groups,
            config=config,
        ),
        "GRU": evaluate_deep_sequence_model(
            name="GRU",
            model_factory=TinyGru,
            windows=windows,
            x_sequence=x_sequence,
            y=y,
            groups=groups,
            config=config,
        ),
    }

    # 작은 공개 데이터에서는 정확도 하나보다 macro F1과 fall recall을 같이 봐야 한다.
    # 그래도 단일 추천이 필요하면 macro F1을 우선 기준으로 둔다.
    recommended_model = max(results, key=lambda name: results[name]["macro_f1"])
    experiment = {
        "experiment": "IMU sequence model comparison",
        "data_source": {
            "samples": str(SAMPLES_PATH.relative_to(PROJECT_ROOT)),
            "windows": str(WINDOWS_PATH.relative_to(PROJECT_ROOT)),
        },
        "dataset_shape": {
            "windows": int(len(windows)),
            "recordings": int(windows["recording_id"].nunique()),
            "sequence_tensor": list(x_sequence.shape),
            "class_counts": {
                "normal_activity": int((y == 0).sum()),
                "fall_risk": int((y == 1).sum()),
            },
        },
        "validation": (
            "All models use 3-fold StratifiedGroupKFold by recording_id. "
            "Deep models additionally use an inner group-aware validation fold for early stopping."
        ),
        "excluded_candidates": {
            "Transformer": "현재 window 117개 규모에서는 parameter 수가 커 과적합 위험이 높아 제외했다.",
            "3D CNN": "IMU는 영상/voxel 같은 3차원 격자가 아니라 9채널 시계열이므로 현재 목적에는 과하다.",
        },
        "recommended_model_by_macro_f1": recommended_model,
        "results": results,
        "output_plots": {
            "metric_comparison": str(PLOT_PATH.relative_to(PROJECT_ROOT)),
            "confusion_matrices": str(CONFUSION_PLOT_PATH.relative_to(PROJECT_ROOT)),
        },
    }

    _plot_metric_comparison(results)
    _plot_confusion_matrices(results)

    json_ready = _to_builtin(experiment)
    RESULTS_PATH.write_text(json.dumps(json_ready, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Model comparison complete")
    for name, metrics in results.items():
        print(
            f"{name}: accuracy={metrics['accuracy']:.4f}, "
            f"macro_f1={metrics['macro_f1']:.4f}, fall_recall={metrics['fall_recall']:.4f}"
        )
    print(f"Recommended by macro F1: {recommended_model}")
    print(f"Saved metrics to {RESULTS_PATH}")
    print(f"Saved plot to {PLOT_PATH}")
    return json_ready


if __name__ == "__main__":
    compare_models()
