"""Peter Anchor 모델링에 사용할 실제 공개 데이터를 수집하고 변환한다.

이 프로젝트에 필요한 로프 장력, 하네스 압력 같은 Peter Anchor 전용
센서 데이터는 공개 현장 데이터로 확인되지 않았다. 따라서 이 스크립트는
공개 출처에서 실제로 가져올 수 있는 데이터만 사용한다.

- 공개 GitHub 데이터셋의 웨어러블 IMU 낙상/일상동작 기록
- Open-Meteo의 과거 풍속 및 돌풍 데이터

데이터 수집 단계는 모델 학습 코드와 의도적으로 분리했다. 이렇게 나누면
프로젝트 흐름을 다음처럼 명확하게 설명할 수 있다.

1. 공개 원본 데이터를 가져온다.
2. 단위를 정규화하고 해석 가능한 센서 feature를 만든다.
3. 모델링, 시각화, 검토에 재사용할 CSV 파일로 저장한다.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


# 프로젝트 루트 기준 경로를 사용하면 어느 위치에서 실행해도 파일 경로가
# 안정적으로 잡힌다. 예를 들어 repo root에서 `python src/fetch_real_data.py`
# 로 실행해도 되고, 다른 스크립트에서 import해도 동일하게 동작한다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_IMU_DIR = DATA_DIR / "raw" / "fall_detection_imu_jasonkau"
IMU_SAMPLES_PATH = DATA_DIR / "real_imu_fall_detection_samples.csv"
IMU_WINDOWS_PATH = DATA_DIR / "real_imu_fall_detection_windows.csv"
WIND_PATH = DATA_DIR / "real_weather_wind_seoul_2025.csv"
MANIFEST_PATH = DATA_DIR / "real_data_manifest.json"

# GitHub 저장소 안의 원본 Excel 파일 목록을 찾기 위한 API 주소다.
# 파일명을 직접 하드코딩하지 않고 tree API를 사용하면, 어떤 공개 파일을
# 가져왔는지 재현 가능하게 남길 수 있다.
GITHUB_TREE_API = (
    "https://api.github.com/repos/jasonkau/fall-detection-dataset-IMU/git/trees/main?recursive=1"
)
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/jasonkau/fall-detection-dataset-IMU/main/"
GITHUB_REPO_URL = "https://github.com/jasonkau/fall-detection-dataset-IMU"

# Open-Meteo는 별도 비공개 API key 없이 재현 가능한 공개 API라 선택했다.
# 단, 이 풍속 데이터는 IMU 데이터와 같은 시간/장소에서 수집된 것이 아니므로
# 현재 IMU 모델 입력으로 합치지 않고 환경 맥락 데이터로만 저장한다.
OPEN_METEO_ARCHIVE_API = "https://archive-api.open-meteo.com/v1/archive"
SEOUL_LATITUDE = 37.5665
SEOUL_LONGITUDE = 126.9780
WIND_START_DATE = "2025-01-01"
WIND_END_DATE = "2025-12-31"

# 원본 Excel 파일에는 안정적으로 사용할 수 있는 header row가 없다.
# 따라서 데이터셋 설명을 기준으로 첫 12개 센서 컬럼에 명시적인 이름을 붙인다.
# 이 컬럼명은 이후 전처리, 피처 엔지니어링, 모델링에서 계속 사용된다.
IMU_COLUMNS = [
    "sample_index",
    "acceleration_svm_raw",
    "accel_x_raw",
    "accel_y_raw",
    "accel_z_raw",
    "gyro_x_dps",
    "gyro_y_dps",
    "gyro_z_dps",
    "angular_velocity_svm_dps",
    "inclination_x_deg",
    "inclination_y_deg",
    "inclination_z_deg",
]


def _read_json(url: str) -> dict[str, Any]:
    """지정한 HTTP JSON endpoint를 읽는다.

    GitHub나 공개 API는 기본 urllib 요청을 일부 환경에서 거부할 수 있다.
    그래서 프로젝트 이름이 들어간 User-Agent를 명시해 요청의 출처를 분명히 한다.
    """
    request = Request(url, headers={"User-Agent": "peter-anchor-risk-modeling"})
    with urlopen(request, timeout=60) as response:
        return json.load(response)


def _download(url: str, destination: Path) -> None:
    """원격 파일 하나를 로컬 경로에 다운로드한다.

    원본 IMU 파일은 `Daily Activities/`, `Fall Events/`처럼 중첩 폴더를
    포함하므로, 저장 전에 부모 디렉터리를 먼저 만든다.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "peter-anchor-risk-modeling"})
    with urlopen(request, timeout=120) as response:
        destination.write_bytes(response.read())


def discover_imu_files() -> list[str]:
    """GitHub 저장소에서 공개 IMU Excel 파일 경로를 모두 찾는다.

    수동으로 일부 파일만 고르는 대신 공개 데이터셋 전체를 사용하기 위한 단계다.
    정렬된 목록을 반환하면 CSV 생성 순서가 항상 같아져 재현성과 diff 안정성이 좋아진다.
    """
    tree = _read_json(GITHUB_TREE_API)
    return sorted(
        item["path"]
        for item in tree["tree"]
        if item["type"] == "blob" and item["path"].endswith(".xlsx")
    )


def download_imu_files(paths: list[str], refresh: bool = False) -> list[Path]:
    """IMU Excel 파일을 다운로드하고 로컬 파일 경로 목록을 반환한다.

    `refresh=False`이면 이미 받은 파일은 다시 받지 않는다. 반복 실행이 빨라지고,
    필요할 때만 강제로 원본을 다시 받을 수 있다.
    """
    local_paths: list[Path] = []
    for source_path in paths:
        local_path = RAW_IMU_DIR / source_path
        if refresh or not local_path.exists():
            # GitHub의 하위 폴더 구조는 유지하되, URL 안의 공백은 인코딩한다.
            raw_url = GITHUB_RAW_BASE + quote(source_path, safe="/")
            _download(raw_url, local_path)
        local_paths.append(local_path)
    return local_paths


def _activity_label(path: Path) -> str:
    """Excel 파일명을 모델링에 쓰기 쉬운 activity label로 바꾼다."""
    return path.stem.split(".", 1)[-1].replace(" ", "_").replace("-", "_").lower()


def _activity_group(path: Path) -> str:
    """원본 폴더명을 이진 분류용 그룹으로 매핑한다."""
    return "fall" if "Fall Events" in path.parts else "daily_activity"


def load_imu_file(path: Path) -> pd.DataFrame:
    """원본 IMU Excel 파일 하나를 읽고 sample-level feature를 만든다.

    여기서 반환되는 표는 여전히 시계열 데이터다. 한 row가 하나의 센서 sample을
    의미하며, 이후 `build_imu_windows`에서 모델 입력용 window feature로 집계한다.
    """
    # 원본 데이터에는 사용할 만한 header가 없으므로 header 없이 읽고,
    # 문서화된 센서 컬럼만 유지한다. 스프레드시트 export 과정에서 생긴 빈 컬럼이
    # feature로 섞이지 않도록 첫 12개 컬럼만 사용한다.
    df = pd.read_excel(path, header=None).iloc[:, : len(IMU_COLUMNS)]
    df.columns = IMU_COLUMNS

    # 선택한 컬럼을 모두 숫자로 강제 변환한다. 숫자로 변환할 수 없는 값은 NaN이 되고,
    # sample index가 없는 row는 잘못된 row로 보고 제거한다.
    df = df.apply(pd.to_numeric, errors="coerce").dropna(subset=["sample_index"])

    group = _activity_group(path)
    label = _activity_label(path)
    relative_source = path.relative_to(RAW_IMU_DIR).as_posix()

    # 원본 가속도 값은 centi-m/s^2 스케일로 저장되어 있다.
    # 예를 들어 966은 9.66 m/s^2로 해석하는 것이 물리적으로 자연스럽다.
    for column in [
        "acceleration_svm_raw",
        "accel_x_raw",
        "accel_y_raw",
        "accel_z_raw",
    ]:
        df[column.replace("_raw", "_ms2")] = df[column] / 100.0

    # inclination 값은 자세 변화의 proxy로 사용한다. Y축은 직립 상태에서 +/-90도
    # 근처에 머무는 경우가 많으므로, 90도에서 얼마나 벗어났는지를 계산하고
    # X/Z축의 절댓값과 함께 가장 큰 값을 body angle proxy로 사용한다.
    y_axis_deviation = (90.0 - df["inclination_y_deg"].abs()).abs()
    df["body_angle_deg"] = np.maximum.reduce(
        [
            df["inclination_x_deg"].abs(),
            y_axis_deviation,
            df["inclination_z_deg"].abs(),
        ]
    ).clip(0, 90)

    # 작업자 흔들림은 공개 데이터에 직접 측정값이 없으므로 X/Z축 가속도 크기로
    # lateral movement proxy를 만든다. 없는 센서를 가상으로 만드는 대신,
    # 실제 IMU 값에서 투명하게 유도 가능한 proxy만 사용한다.
    df["lateral_acceleration_ms2"] = np.sqrt(
        df["accel_x_ms2"].pow(2) + df["accel_z_ms2"].pow(2)
    )

    # 라벨은 공개 데이터셋의 폴더 구조에서 가져온다. 이렇게 해야 target label이
    # 임의 수작업이 아니라 원본 데이터 출처와 연결된다.
    df["activity_group"] = group
    df["activity_label"] = label
    df["fall_event"] = int(group == "fall")
    df["source_file"] = relative_source
    return df


def build_imu_samples(local_paths: list[Path]) -> pd.DataFrame:
    """모든 원본 recording을 하나의 sample-level CSV 테이블로 합친다."""
    frames = [load_imu_file(path) for path in local_paths]
    samples = pd.concat(frames, ignore_index=True)

    # 사람이 보기 쉬운 순서로 컬럼을 정렬한다. 출처와 라벨을 먼저 두고,
    # 그 뒤에 실제 센서 feature와 파생 proxy feature를 둔다.
    ordered_columns = [
        "source_file",
        "activity_group",
        "activity_label",
        "fall_event",
        "sample_index",
        "acceleration_svm_ms2",
        "accel_x_ms2",
        "accel_y_ms2",
        "accel_z_ms2",
        "gyro_x_dps",
        "gyro_y_dps",
        "gyro_z_dps",
        "angular_velocity_svm_dps",
        "inclination_x_deg",
        "inclination_y_deg",
        "inclination_z_deg",
        "body_angle_deg",
        "lateral_acceleration_ms2",
    ]
    return samples[ordered_columns]


def build_imu_windows(samples: pd.DataFrame, window_size: int = 100, stride: int = 50) -> pd.DataFrame:
    """sample-level IMU 데이터를 모델 입력용 sliding window feature로 집계한다.

    추락 유사 동작은 센서 row 하나로 판단하기보다 짧은 시간 동안의 움직임 패턴으로
    보는 편이 자연스럽다. Windowing을 사용하면 짧은 구간 안의 peak acceleration,
    변동성, 자세 변화, 흔들림을 함께 볼 수 있다.

    공개 데이터 설명 기준으로 sampling rate는 100 Hz다. 기본 `window_size=100`은
    약 1초 구간을 의미하고, `stride=50`은 50% overlap을 만들어 데이터 수를 늘리면서도
    시간적 연속성을 보존한다.
    """
    records: list[dict[str, Any]] = []
    for source_file, group in samples.groupby("source_file", sort=True):
        # Window는 반드시 같은 recording 안에서만 만든다. 서로 다른 파일은 서로 다른
        # activity나 fall scenario이므로 경계를 넘어 window를 만들면 라벨이 섞일 수 있다.
        group = group.sort_values("sample_index").reset_index(drop=True)
        for start in range(0, max(len(group) - window_size + 1, 0), stride):
            window = group.iloc[start : start + window_size]
            if len(window) < window_size:
                continue

            # worker sway proxy는 window 내부 lateral acceleration의 범위로 표현한다.
            # 범위가 클수록 좌우/측면 움직임 변화가 크다고 해석할 수 있다.
            lateral_range = (
                window["lateral_acceleration_ms2"].max()
                - window["lateral_acceleration_ms2"].min()
            )
            records.append(
                {
                    # 같은 recording에서 나온 인접 window들은 서로 매우 비슷하다.
                    # 나중에 group-aware cross-validation을 하기 위해 recording_id를 보존한다.
                    "recording_id": source_file,
                    "activity_group": window["activity_group"].iloc[0],
                    "activity_label": window["activity_label"].iloc[0],
                    "fall_event": int(window["fall_event"].iloc[0]),
                    "risk_type": "fall_risk" if int(window["fall_event"].iloc[0]) else "normal_activity",
                    "start_sample": int(window["sample_index"].iloc[0]),
                    "end_sample": int(window["sample_index"].iloc[-1]),
                    "n_samples": int(len(window)),
                    # peak/mean/std는 짧은 구간의 움직임 강도와 변동성을 설명한다.
                    "imu_acceleration_peak_ms2": window["acceleration_svm_ms2"].max(),
                    "imu_acceleration_mean_ms2": window["acceleration_svm_ms2"].mean(),
                    "imu_acceleration_std_ms2": window["acceleration_svm_ms2"].std(ddof=0),
                    # gyro feature는 회전 강도를 포착한다. 급격한 자세 변화나 fall event에서
                    # 회전 성분이 커질 수 있어 중요한 후보 feature다.
                    "gyro_peak_dps": window["angular_velocity_svm_dps"].max(),
                    "gyro_mean_dps": window["angular_velocity_svm_dps"].mean(),
                    # body angle은 inclination에서 만든 자세 proxy다.
                    "body_angle_peak_deg": window["body_angle_deg"].max(),
                    "body_angle_mean_deg": window["body_angle_deg"].mean(),
                    "worker_sway_proxy_ms2": lateral_range,
                }
            )
    return pd.DataFrame.from_records(records)


def fetch_wind_data() -> pd.DataFrame:
    """Open-Meteo에서 시간별 과거 풍속과 돌풍 데이터를 가져온다.

    이 데이터는 Peter Anchor 콘셉트의 환경 맥락을 설명하기 위해 저장한다.
    공개 IMU recording이 2025년 서울에서 수집된 것이 아니므로, 현재 학습 테이블에
    직접 join하지 않는다.
    """
    params = {
        "latitude": SEOUL_LATITUDE,
        "longitude": SEOUL_LONGITUDE,
        "start_date": WIND_START_DATE,
        "end_date": WIND_END_DATE,
        "hourly": "wind_speed_10m,wind_gusts_10m",
        "timezone": "Asia/Seoul",
    }
    url = OPEN_METEO_ARCHIVE_API + "?" + urlencode(params)
    data = _read_json(url)
    hourly = data["hourly"]

    # Open-Meteo는 각 시간별 field를 병렬 배열 형태로 반환한다.
    # 분석과 시각화를 쉽게 하기 위해 일반적인 tabular DataFrame으로 변환한다.
    wind = pd.DataFrame(
        {
            "timestamp": hourly["time"],
            "wind_speed_10m_kmh": hourly["wind_speed_10m"],
            "wind_gusts_10m_kmh": hourly["wind_gusts_10m"],
        }
    )
    wind["latitude"] = SEOUL_LATITUDE
    wind["longitude"] = SEOUL_LONGITUDE
    wind["source"] = "Open-Meteo Historical Weather API"
    return wind


def main(refresh: bool = False) -> dict[str, Any]:
    """실제 공개 데이터 수집 파이프라인 전체를 실행한다.

    반환값은 manifest dictionary이며, 동일한 내용이 파일로도 저장된다.
    Manifest는 데이터 출처와 공개 데이터로 확보하지 못한 항목을 남기기 때문에
    포트폴리오 신뢰도 측면에서 중요하다.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 공개 원본 파일 목록을 찾고 다운로드한다.
    imu_source_paths = discover_imu_files()
    local_imu_paths = download_imu_files(imu_source_paths, refresh=refresh)

    # 2. 원본 센서 파일을 sample-level 및 window-level 데이터셋으로 변환한다.
    samples = build_imu_samples(local_imu_paths)
    windows = build_imu_windows(samples)

    # 3. 별도 환경 맥락 데이터인 풍속 데이터를 가져온다.
    wind = fetch_wind_data()

    # CSV를 사람이 직접 열어 봤을 때 읽기 쉽도록 숫자값을 반올림한다.
    # 모델링에는 과도한 소수점 정밀도가 필요하지 않고, GitHub나 spreadsheet에서
    # 확인할 때도 반올림된 값이 더 다루기 쉽다.
    sample_numeric_columns = samples.select_dtypes(include=["number"]).columns
    window_numeric_columns = windows.select_dtypes(include=["number"]).columns
    wind_numeric_columns = wind.select_dtypes(include=["number"]).columns
    samples[sample_numeric_columns] = samples[sample_numeric_columns].round(4)
    windows[window_numeric_columns] = windows[window_numeric_columns].round(4)
    wind[wind_numeric_columns] = wind[wind_numeric_columns].round(4)

    samples.to_csv(IMU_SAMPLES_PATH, index=False)
    windows.to_csv(IMU_WINDOWS_PATH, index=False)
    wind.to_csv(WIND_PATH, index=False)

    # Manifest는 데이터 lineage와 scope limitation을 기록한다.
    # 특히 Peter Anchor 하드웨어에서 직접 측정해야 하는데 공개 데이터로 없는 항목을
    # 명시해, 없는 데이터를 측정한 것처럼 보이지 않게 한다.
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": {
            "imu": {
                "name": "fall-detection-dataset-IMU",
                "url": GITHUB_REPO_URL,
                "license": "MIT",
                "source_files": imu_source_paths,
                "sample_rows": int(len(samples)),
                "window_rows": int(len(windows)),
                "notes": (
                    "Public wearable IMU recordings for fall and daily activity detection. "
                    "Acceleration values are normalized from source centi-m/s^2 scale."
                ),
            },
            "wind": {
                "name": "Open-Meteo Historical Weather API",
                "url": OPEN_METEO_ARCHIVE_API,
                "location": "Seoul, South Korea",
                "latitude": SEOUL_LATITUDE,
                "longitude": SEOUL_LONGITUDE,
                "start_date": WIND_START_DATE,
                "end_date": WIND_END_DATE,
                "rows": int(len(wind)),
                "notes": "Hourly historical wind speed and gusts at 10 m.",
            },
        },
        "not_publicly_available": [
            "Peter Anchor field rope tension time series",
            "Peter Anchor harness pressure time series",
            "suction readiness logs from the proposed hardware",
        ],
        "outputs": {
            "imu_samples": str(IMU_SAMPLES_PATH.relative_to(PROJECT_ROOT)),
            "imu_windows": str(IMU_WINDOWS_PATH.relative_to(PROJECT_ROOT)),
            "wind": str(WIND_PATH.relative_to(PROJECT_ROOT)),
        },
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved IMU samples: {IMU_SAMPLES_PATH} ({len(samples):,} rows)")
    print(f"Saved IMU windows: {IMU_WINDOWS_PATH} ({len(windows):,} rows)")
    print(f"Saved wind data: {WIND_PATH} ({len(wind):,} rows)")
    print(f"Saved manifest: {MANIFEST_PATH}")
    return manifest


if __name__ == "__main__":
    main()
