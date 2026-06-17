# AI Modeling Process

## 1. 모델링 목적

Peter Anchor의 AI 모델링 목적은 제품 안전성을 증명하는 것이 아니라, 공개 실제 센서 데이터로 추락 위험 감지 구조가 성립할 수 있는지 검토하는 것이다.

구체적으로는 IMU 기반 움직임 데이터에서 `normal_activity`와 `fall_risk`를 분류할 수 있는지 확인한다. 로프 장력, 하네스 압력, 흡착 준비 상태는 공개 데이터가 없어 모델 입력에서 제외하고, 향후 직접 수집해야 할 센서 데이터로 분리한다.

문제 정의, 데이터 선택, 모델 선택, 특허/선행기술 근거는 `docs/evidence_sources.md`에 별도로 정리했다.

## 2. 가설 설정 및 근거

### Hypothesis 1

IMU의 가속도, 각속도, 기울기 정보는 fall-risk와 normal activity를 구분하는 데 유효할 것이다.

근거:

- `fall-detection-dataset-IMU`는 LSM6DSO IMU 센서로 수집한 fall event와 daily activity 데이터를 제공한다.
- 원본 데이터에는 acceleration SVM, 3축 가속도, 3축 gyro, angular velocity SVM, 3축 inclination이 포함된다.
- Source: https://github.com/jasonkau/fall-detection-dataset-IMU
- Supporting evidence: IMU 기반 fall risk assessment 연구와 LSM6DSO 센서 근거는 `docs/evidence_sources.md`에 정리했다.

### Hypothesis 2

추락 위험은 단일 sample 하나보다 짧은 시간 window의 움직임 패턴으로 보는 것이 더 적합할 것이다.

근거:

- 낙상/추락 움직임은 순간 peak acceleration, 각속도 변화, 자세 변화가 일정 시간 안에서 함께 나타나는 이벤트다.
- 따라서 raw sample 단위가 아니라 100-sample sliding window로 feature를 집계했다.

### Hypothesis 3

풍속 데이터는 Peter Anchor의 `wind_collision_risk` 확장에는 필요하지만, 현재 IMU 모델에는 직접 결합하지 않는 것이 타당하다.

근거:

- Open-Meteo Historical Weather API는 시간별 `wind_speed_10m`, `wind_gusts_10m`를 제공한다.
- 그러나 현재 IMU 데이터와 풍속 데이터는 같은 장소와 같은 시간에 동기화된 데이터가 아니다.
- 따라서 현재 모델에는 넣지 않고, 향후 현장 동기화 데이터 수집 후 `wind_collision_risk` 모델로 확장한다.
- Source: https://open-meteo.com/en/docs/historical-weather-api

### Hypothesis 4

가속도 x/y/z, 자이로 x/y/z, 기울기 x/y/z를 요약하지 않고 9축 시계열로 직접 학습하면 낙상 패턴을 더 세밀하게 볼 수 있을 것이다.

근거:

- IMU 원본에는 각 sample마다 9개 축의 센서값이 시간 순서대로 들어 있다.
- 낙상은 특정 순간의 숫자 하나보다 짧은 시간 동안의 충격, 회전, 자세 변화 흐름으로 나타날 수 있다.
- 따라서 `100 time steps x 9 IMU channels` 형태로 1D CNN과 GRU를 비교 실험했다.
- 단, 현재 공개 데이터는 window 117개로 작기 때문에 딥러닝 모델이 과적합될 가능성이 높다는 한계도 함께 검증했다.
- Supporting evidence: 1D CNN/GRU 기반 IMU 시계열 모델링 연구는 `docs/evidence_sources.md`에 정리했다.

### Hypothesis 5

ROCKET 계열 random convolution kernel transform은 작은 IMU 데이터에서도 9축 시계열 패턴을 효과적으로 feature화할 수 있을 것이다.

근거:

- ROCKET 논문은 시계열에 무작위 convolution kernel을 적용해 feature를 만들고, 선형 분류기로 학습하는 방식을 제안한다.
- 이 방식은 1D CNN/GRU처럼 convolution/recurrent weight를 직접 학습하지 않기 때문에, 작은 데이터에서 과적합 부담이 상대적으로 낮다.
- 현재 프로젝트에서는 공식 ROCKET 패키지가 아니라, random kernel, max activation, PPV feature, L2 LogisticRegression 구조를 직접 구현한 `ROCKET-inspired` 파이프라인을 추가했다.
- Supporting evidence: ROCKET/MiniRocket 논문 근거는 `docs/evidence_sources.md`와 `docs/rocket_paper_pipeline.md`에 정리했다.

## 3. 데이터 수집

### IMU 데이터

- Source: `jasonkau/fall-detection-dataset-IMU`
- File type: `.xlsx`
- Raw files: 13 files
- Activity types: daily activity 8종, fall event 5종
- Output: `data/real_imu_fall_detection_samples.csv`
- Rows: 6,526

### 풍속 데이터

- Source: Open-Meteo Historical Weather API
- Location: Seoul, South Korea
- Period: 2025-01-01 to 2025-12-31
- Output: `data/real_weather_wind_seoul_2025.csv`
- Rows: 8,760

### 한국 공식 데이터 대안

- Source: 기상청 ASOS 시간자료 조회서비스
- URL: https://www.data.go.kr/data/15057210/openapi.do
- Note: 공식 관측 데이터로 적합하지만 API 활용 신청과 인증키가 필요하다.

## 4. 데이터 확인

IMU 데이터 확인 결과:

| Item | Value |
| --- | --- |
| Sample rows | 6,526 |
| Window rows | 117 |
| Normal activity windows | 72 |
| Fall-risk windows | 45 |

데이터 확인 시각화:

- `outputs/data_check_class_distribution.png`
- `outputs/data_check_feature_distributions.png`

풍속 데이터 확인 결과:

| Item | Value |
| --- | --- |
| Weather rows | 8,760 |
| Mean wind speed | 7.597 km/h |
| Max wind speed | 29.200 km/h |
| Max wind gust | 69.800 km/h |

풍속 확인 시각화:

- `outputs/data_check_wind_summary.png`

## 5. 데이터 전처리

전처리는 `src/fetch_real_data.py`에서 수행한다.

- 원본 `.xlsx` 파일에서 첫 12개 센서 컬럼을 읽는다.
- 결측 또는 숫자로 변환할 수 없는 row를 제거한다.
- 원본 acceleration 값은 centi-m/s² 스케일로 저장되어 있어 `/100`으로 단위 정규화한다.
- activity 폴더명을 기준으로 `fall_event` 라벨을 만든다.
- `Fall Events`는 `fall_event = 1`, `Daily Activities`는 `fall_event = 0`으로 설정한다.

## 6. 피처 엔지니어링 및 스케일링

### Sample-level feature

- `acceleration_svm_ms2`
- `accel_x_ms2`
- `accel_y_ms2`
- `accel_z_ms2`
- `angular_velocity_svm_dps`
- `inclination_x_deg`
- `inclination_y_deg`
- `inclination_z_deg`

### Derived feature

- `body_angle_deg`: 3축 inclination 기반 자세 변화 proxy
- `lateral_acceleration_ms2`: X/Z축 lateral movement proxy

### Window-level feature

100-sample window, 50-sample stride로 다음 feature를 생성했다.

- `imu_acceleration_peak_ms2`
- `imu_acceleration_mean_ms2`
- `imu_acceleration_std_ms2`
- `gyro_peak_dps`
- `gyro_mean_dps`
- `body_angle_peak_deg`
- `body_angle_mean_deg`
- `worker_sway_proxy_ms2`

별도 StandardScaler는 적용하지 않았다. 사용 모델이 RandomForest이기 때문에 feature scale에 민감하지 않고, 단위 변환과 window-level 집계가 핵심 전처리다.

### Sequence-level tensor

시계열 모델 비교를 위해 sample-level 데이터를 다시 window와 연결해 다음 형태의 입력도 만들었다.

```text
X shape = (117, 100, 9)
```

의미:

- `117`: window 개수
- `100`: window 하나에 포함된 sample 수
- `9`: 가속도 x/y/z, 자이로 x/y/z, 기울기 x/y/z

시계열 모델에서는 StandardScaler를 사용했다. 단, 검증 데이터 통계가 학습에 새지 않도록 scaler는 각 fold의 train 데이터로만 fit하고 validation/test 데이터에는 transform만 적용했다.

### ROCKET-inspired random convolution feature

논문 기반 고도화 파이프라인에서는 100시점 x 9채널 IMU window에 무작위 convolution kernel을 적용했다.

각 kernel에서 다음 두 feature를 만든다.

- `max activation`: 해당 kernel이 가장 강하게 반응한 정도
- `PPV`: positive activation 비율

현재 설정은 512개 kernel을 사용하므로 다음 feature matrix를 만든다.

```text
512 kernels x 2 features = 1024 ROCKET-inspired features
```

이 feature는 train fold에서만 scaling한 뒤 L2 LogisticRegression에 입력한다.

## 7. 모델 선정

초기 baseline 모델은 `RandomForestClassifier`를 사용했다.

초기 선정 이유:

- window-level tabular feature에 적합하다.
- acceleration, gyro, body angle 사이의 비선형 관계를 다룰 수 있다.
- 데이터 수가 많지 않은 초기 검증 단계에서 baseline 모델로 안정적이다.
- feature importance를 통해 어떤 센서 feature가 판단에 영향을 줬는지 설명할 수 있다.
- 딥러닝 모델보다 작은 데이터셋에서 과적합 위험과 설명 비용이 낮다.

추가 비교 모델:

- `1D CNN`: 시간축을 따라 짧은 충격, 회전, 자세 변화 패턴을 찾기 위한 모델
- `GRU`: 센서값의 시간 순서를 따라가며 낙상 전후 흐름을 보기 위한 모델
- `ROCKET-inspired`: random convolution kernel로 시계열을 feature화하고 regularized linear classifier를 학습하는 논문 기반 모델

제외한 모델:

- `Transformer`: 현재 window 117개 규모에서는 parameter 수가 커 과적합 위험이 높다.
- `3D CNN`: IMU는 영상이나 voxel 같은 3차원 격자가 아니라 9채널 시계열이므로 현재 목적에는 과하다.

비교 결과, 현재 공개 데이터 규모에서는 1D CNN과 GRU보다 RandomForest가 안정적이었고, 이후 추가한 ROCKET-inspired 모델이 가장 높은 Macro F1을 보였다. 따라서 현재 best model은 ROCKET-inspired 모델이며, RandomForest는 설명 가능한 baseline으로 유지한다.

## 8. 모델링

RandomForest baseline 입력:

- 8개 window-level IMU feature

ROCKET-inspired model 입력:

- 100 time steps x 9 IMU channels
- 512 random kernels에서 만든 1024개 feature

모델 출력:

- `0`: `normal_activity`
- `1`: `fall_risk`

모델 설정:

- `n_estimators=240`
- `max_depth=8`
- `min_samples_leaf=2`
- `class_weight="balanced"`
- `random_state=42`

구현 파일:

- `src/train_real_imu_model.py`
- `src/compare_sequence_models.py`
- `src/train_rocket_imu_model.py`

## 9. 추가 학습 및 튜닝

현재는 baseline 모델링 단계이므로 대규모 hyperparameter tuning은 수행하지 않았다.

다만 사용자의 고도화 요구에 따라 9축 IMU 시계열 모델 비교를 수행했다.

딥러닝 과적합 방지 방법:

- `StratifiedGroupKFold`로 같은 recording이 train/test에 동시에 들어가지 않게 함
- outer train fold 내부에 group-aware validation fold를 만들어 early stopping 적용
- dropout 적용
- AdamW weight decay 적용
- train fold로만 StandardScaler fit
- gradient clipping 적용
- 작은 hidden size와 channel 수 사용

실험 결과:

| Model | Accuracy | Macro F1 | Fall recall |
| --- | ---: | ---: | ---: |
| RandomForest | 0.7521 | 0.7485 | 0.8222 |
| 1D CNN | 0.5470 | 0.5469 | 0.6889 |
| GRU | 0.5556 | 0.5408 | 0.4889 |
| ROCKET-inspired | 0.8547 | 0.8472 | 0.8222 |

결론:

> 1D CNN과 GRU는 현재 데이터 수에서 충분히 일반화되지 못했다. ROCKET-inspired 모델은 시계열 패턴을 random convolution feature로 변환하고 선형 분류기만 학습해 작은 데이터에서 더 안정적인 성능을 보였다. 따라서 현재 best model은 ROCKET-inspired 모델이다.

추가 학습 방향:

- 더 많은 subject와 작업 환경 데이터 확보 후 cross-validation 적용
- `window_size`, `stride` 비교 실험
- ROCKET kernel 수, kernel 길이, regularization strength 비교
- recall 우선 objective 기준 threshold tuning
- 충분한 현장 데이터 확보 후 1D CNN/GRU 재학습
- 실제 현장 데이터 확보 후 `wind_collision_risk`, `harness_error`, `emergency_stabilization` 다중 분류 확장

## 10. 모델 검증

검증은 `StratifiedGroupKFold` out-of-fold 방식으로 수행했다.

이유:

- window를 무작위로 나누면 같은 recording에서 나온 비슷한 window가 train/test에 동시에 들어갈 수 있다.
- 그러면 실제보다 성능이 높게 나올 수 있다.
- 따라서 `recording_id` 기준으로 train/test를 분리했다.
- 전체 117개 window가 한 번씩 test fold에 들어가도록 out-of-fold prediction을 모아 최종 confusion matrix를 계산했다.

평가 지표:

- Accuracy
- Macro F1
- Fall recall
- Classification report
- Confusion matrix

검증 결과:

| Model | Accuracy | Macro F1 | Fall recall | Confusion matrix |
| --- | ---: | ---: | ---: | --- |
| RandomForest | 0.7521 | 0.7485 | 0.8222 | `[[51, 21], [8, 37]]` |
| 1D CNN | 0.5470 | 0.5469 | 0.6889 | `[[33, 39], [14, 31]]` |
| GRU | 0.5556 | 0.5408 | 0.4889 | `[[43, 29], [23, 22]]` |
| ROCKET-inspired | 0.8547 | 0.8472 | 0.8222 | `[[63, 9], [8, 37]]` |

ROCKET-inspired confusion matrix:

| True \ Pred | Normal | Fall |
| --- | ---: | ---: |
| Normal | 63 | 9 |
| Fall | 8 | 37 |

모델 검증 시각화:

- `outputs/model_confusion_matrix_oof.png`
- `outputs/model_feature_importance_real_imu.png`
- `outputs/sequence_model_comparison.png`
- `outputs/sequence_model_confusion_matrices.png`
- `outputs/all_model_comparison.png`
- `outputs/rocket_confusion_matrix_oof.png`
- `outputs/rocket_feature_space_pca.png`

## 11. 기대효과 및 검증 결과 해석

### 기대효과

- 공개 실제 IMU 데이터만으로도 fall-like motion과 normal activity를 구분하는 baseline 모델을 만들 수 있다.
- ROCKET 논문 기반 random convolution feature 파이프라인을 직접 구현해, 단순 baseline보다 높은 Macro F1을 얻었다.
- 포트폴리오에서 합성 데이터가 아니라 실제 공개 데이터 기반의 데이터 수집, 전처리, feature engineering, 모델링, 검증 흐름을 설명할 수 있다.
- 향후 Peter Anchor 실증 데이터가 수집되면 같은 파이프라인을 확장할 수 있다.

### 검증 결과 해석

ROCKET-inspired 모델은 fall-risk recall 0.8222로, 전체 out-of-fold fall window 45개 중 37개를 탐지했다. RandomForest와 fall recall은 같지만, normal activity를 fall로 잘못 판단한 false alarm이 21개에서 9개로 줄어 Macro F1이 개선되었다.

다만 이 결과는 공개 IMU 데이터셋에 대한 검증이지 Peter Anchor 제품 성능 검증이 아니다. 실제 제품화 단계에서는 rope tension, harness pressure, worker IMU, building-face wind, suction readiness를 같은 시간축으로 수집해 다시 학습하고 검증해야 한다.
