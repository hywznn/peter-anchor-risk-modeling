"""데이터 확인과 모델 검증 결과를 시각화한다.

이 차트들은 모델링 근거 자료다. 단순히 accuracy 숫자 하나만 제시하지 않고,
class balance, feature 분포, 풍속 맥락, 모델 오류, feature importance를 함께 확인한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# 재현 가능한 로컬 실행을 위해 repo root 기준 경로를 사용한다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
WINDOWS_PATH = DATA_DIR / "real_imu_fall_detection_windows.csv"
WIND_PATH = DATA_DIR / "real_weather_wind_seoul_2025.csv"
METRICS_PATH = OUTPUT_DIR / "real_imu_model_metrics.json"

# value_counts() 결과 순서가 달라져도 차트 순서가 안정적으로 유지되도록 고정한다.
RISK_ORDER = ["normal_activity", "fall_risk"]
COLORS = {
    "normal_activity": "#3f6f8f",
    "fall_risk": "#b5534a",
}


def _style_axes(ax, title: str, xlabel: str | None = None, ylabel: str | None = None) -> None:
    """모든 matplotlib 축에 공통 스타일을 적용한다."""
    ax.set_title(title, fontsize=12, weight="bold", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(axis="y", color="#d9dee7", linewidth=0.8, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_class_distribution(windows: pd.DataFrame) -> Path:
    """각 risk class별 window 개수를 막대그래프로 표시한다.

    기본 데이터 확인 질문에 답하기 위한 차트다. 모델이 심하게 불균형한 target으로
    학습되는지 확인할 수 있다. 현재 데이터셋은 작기 때문에 비율만 보여주는 것보다
    실제 개수를 함께 보여주는 편이 더 정직하다.
    """
    counts = windows["risk_type"].value_counts().reindex(RISK_ORDER, fill_value=0)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(counts.index, counts.values, color=[COLORS[label] for label in counts.index])
    _style_axes(ax, "Window Class Distribution", ylabel="Window Count")
    ax.set_ylim(0, max(counts.values) * 1.2)
    for bar in bars:
        # CSV나 metrics JSON을 열지 않아도 이미지 안에서 정확한 개수를 확인할 수 있게 한다.
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts.values) * 0.03,
            f"{int(bar.get_height())}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    fig.tight_layout()
    path = OUTPUT_DIR / "data_check_class_distribution.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_feature_distributions(windows: pd.DataFrame) -> Path:
    """normal window와 fall window의 주요 engineered feature 분포를 비교한다.

    Boxplot은 중앙값, 분산, outlier를 빠르게 보여준다. 모델 성능을 보기 전에
    애초에 설계한 feature가 두 class를 어느 정도 구분하는지 설명하는 데 유용하다.
    """
    features = [
        ("imu_acceleration_peak_ms2", "Peak Acceleration (m/s^2)"),
        ("gyro_peak_dps", "Peak Angular Velocity (deg/s)"),
        ("body_angle_peak_deg", "Peak Body Angle Proxy (deg)"),
        ("worker_sway_proxy_ms2", "Worker Sway Proxy (m/s^2)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2))
    for ax, (feature, label) in zip(axes.ravel(), features):
        # 고정된 RISK_ORDER에 맞춰 class별 숫자 배열을 만든다.
        data = [
            windows.loc[windows["risk_type"] == risk_type, feature].values
            for risk_type in RISK_ORDER
        ]
        box = ax.boxplot(data, tick_labels=RISK_ORDER, patch_artist=True, showfliers=True)
        # class distribution 차트와 동일한 색을 사용해 시각적 일관성을 맞춘다.
        for patch, risk_type in zip(box["boxes"], RISK_ORDER):
            patch.set_facecolor(COLORS[risk_type])
            patch.set_alpha(0.85)
        for median in box["medians"]:
            median.set_color("#1f2933")
            median.set_linewidth(1.4)
        _style_axes(ax, label, ylabel="Value")
        ax.tick_params(axis="x", rotation=15)

    fig.tight_layout()
    path = OUTPUT_DIR / "data_check_feature_distributions.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_wind_summary(wind: pd.DataFrame) -> Path:
    """Open-Meteo 시간별 풍속 데이터를 월별 요약 차트로 표시한다.

    풍속 데이터는 공개 IMU recording과 시간/장소가 동기화되지 않았으므로 현재 모델
    입력에는 사용하지 않는다. 이 차트는 환경 맥락과 향후 wind-collision risk 확장
    가능성을 보여주기 위한 자료다.
    """
    wind = wind.copy()
    wind["timestamp"] = pd.to_datetime(wind["timestamp"])

    # 월별 집계는 시간별 데이터 전체를 그대로 그리는 것보다 읽기 쉽다.
    # 평균 풍속은 일반 노출 수준을, 최대 돌풍은 짧은 극단 상황을 설명한다.
    monthly = (
        wind.set_index("timestamp")
        .resample("ME")[["wind_speed_10m_kmh", "wind_gusts_10m_kmh"]]
        .agg({"wind_speed_10m_kmh": "mean", "wind_gusts_10m_kmh": "max"})
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(monthly["timestamp"], monthly["wind_speed_10m_kmh"], marker="o", label="Mean wind speed")
    ax.plot(monthly["timestamp"], monthly["wind_gusts_10m_kmh"], marker="o", label="Max wind gust")
    _style_axes(ax, "Monthly Wind Summary - Seoul 2025", ylabel="km/h")
    ax.legend(frameon=False)
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    path = OUTPUT_DIR / "data_check_wind_summary.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_confusion_matrix(metrics: dict) -> Path:
    """저장된 metrics에서 out-of-fold confusion matrix를 시각화한다."""
    matrix = np.array(metrics["confusion_matrix"])
    labels = metrics["labels"]

    fig, ax = plt.subplots(figsize=(5.8, 5))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Out-of-fold Confusion Matrix", fontsize=12, weight="bold", pad=10)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels=labels)

    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            # 셀 색이 진하면 흰 글씨, 옅으면 어두운 글씨를 사용해 숫자 가독성을 유지한다.
            color = "white" if matrix[row, col] > threshold else "#1f2933"
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", color=color, fontsize=12)

    fig.tight_layout()
    path = OUTPUT_DIR / "model_confusion_matrix_oof.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_feature_importance(metrics: dict) -> Path:
    """전체 데이터로 학습한 RandomForest의 feature importance를 시각화한다."""
    importance = pd.DataFrame(metrics["feature_importance"]).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(importance["feature"], importance["importance"], color="#3f6f8f")
    _style_axes(ax, "RandomForest Feature Importance", xlabel="Importance")
    fig.tight_layout()
    path = OUTPUT_DIR / "model_feature_importance_real_imu.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main() -> list[Path]:
    """README와 문서에서 사용하는 모든 차트를 생성한다."""
    if not WINDOWS_PATH.exists():
        raise FileNotFoundError(f"{WINDOWS_PATH} does not exist. Run python src/fetch_real_data.py first.")
    if not WIND_PATH.exists():
        raise FileNotFoundError(f"{WIND_PATH} does not exist. Run python src/fetch_real_data.py first.")
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"{METRICS_PATH} does not exist. Run python src/train_real_imu_model.py first.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 각 입력은 이전 pipeline 단계에서 생성된다.
    # - windows: fetch_real_data.py가 만든 모델링 데이터셋
    # - wind: fetch_real_data.py가 만든 환경 맥락 데이터
    # - metrics: train_real_imu_model.py가 만든 검증/모델 결과
    windows = pd.read_csv(WINDOWS_PATH)
    wind = pd.read_csv(WIND_PATH)
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

    paths = [
        plot_class_distribution(windows),
        plot_feature_distributions(windows),
        plot_wind_summary(wind),
        plot_confusion_matrix(metrics),
        plot_feature_importance(metrics),
    ]
    for path in paths:
        print(f"Saved {path}")
    return paths


if __name__ == "__main__":
    main()
