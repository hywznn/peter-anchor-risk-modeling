# IMU Sequence Model Comparison

논문/센서/검증 방식에 대한 근거 자료는 `docs/evidence_sources.md`에 함께 정리했다.

## 1. 왜 시계열 모델을 추가했는가

기존 모델은 100개 sample window를 하나의 표 형태 feature로 요약했다.

예를 들면 다음과 같은 값이다.

- 최대 가속도
- 평균 가속도
- 가속도 표준편차
- 최대 자이로 값
- 평균 자이로 값
- 최대 자세 기울기
- 평균 자세 기울기
- 좌우 흔들림 proxy

이 방식은 작은 데이터에서 안정적이고 설명하기 쉽다. 하지만 원본 IMU에는 다음 9개 축의 시간 흐름이 그대로 들어 있다.

- 가속도 x/y/z
- 자이로 x/y/z
- 기울기 x/y/z

따라서 “x/y/z 축을 더 입체적으로 보고 싶다”는 관점에서는 100개 시점 전체를 모델에 넣는 시계열 모델도 비교할 필요가 있다.

## 2. 시계열 입력 구조

시계열 모델의 입력은 다음 형태로 만들었다.

```text
X shape = (117, 100, 9)
```

의미는 다음과 같다.

| 축 | 의미 |
| --- | --- |
| 117 | window 개수 |
| 100 | window 하나에 들어가는 시간 sample 수 |
| 9 | IMU 센서 채널 수 |

9개 채널은 다음과 같다.

```text
accel_x_ms2
accel_y_ms2
accel_z_ms2
gyro_x_dps
gyro_y_dps
gyro_z_dps
inclination_x_deg
inclination_y_deg
inclination_z_deg
```

즉, 기존 모델은 window 하나를 8개 요약 feature로 봤고, 시계열 모델은 window 하나를 `100시점 x 9채널` 움직임 패턴으로 본다.

## 3. 비교한 모델

### RandomForest

기존 baseline 모델이다. window-level 요약 feature를 입력으로 받는다.

장점:

- 작은 데이터에서 비교적 안정적이다.
- feature importance로 판단 근거를 설명하기 쉽다.
- scale에 덜 민감하다.

### 1D CNN

1D CNN은 시간축을 따라 짧은 센서 패턴을 훑는다.

예를 들어 낙상 상황에서 짧은 시간 안에 가속도가 튀거나, 회전값이 급격히 변하는 패턴을 찾는 데 적합하다.

사용 이유:

- IMU는 영상이 아니라 시간에 따라 변하는 센서 데이터다.
- 따라서 2D/3D CNN보다 시간축 기반 1D CNN이 더 자연스럽다.

### GRU

GRU는 시계열의 순서를 따라가며 앞뒤 흐름을 반영한다.

사용 이유:

- 낙상은 단일 시점보다 “넘어지기 전 움직임 -> 기울어짐 -> 충격” 같은 흐름이 중요할 수 있다.
- LSTM보다 parameter가 적어 작은 데이터에서 먼저 비교하기 적합하다.

## 4. 제외한 모델

### Transformer

Transformer는 시간 구간 사이의 관계를 잘 학습할 수 있지만, 현재 window 수가 117개뿐이라 parameter 수에 비해 데이터가 너무 작다. 따라서 과적합 위험이 높아 이번 비교에서는 제외했다.

### 3D CNN

3D CNN은 영상, voxel, 3차원 격자 데이터에 적합하다. 하지만 IMU의 x/y/z는 공간 격자라기보다 9개 센서 채널의 시간 흐름이다. 그래서 현재 목적에는 3D CNN보다 1D CNN/GRU가 더 적합하다.

## 5. 과적합 방지 방법

딥러닝 모델은 작은 데이터에서 쉽게 외울 수 있다. 그래서 다음 장치를 넣었다.

| 방법 | 이유 |
| --- | --- |
| `StratifiedGroupKFold` | 같은 원본 recording이 학습/검증에 동시에 들어가지 않게 한다 |
| inner validation fold | early stopping 기준을 만들기 위해 outer train 내부를 다시 나눈다 |
| early stopping | validation loss가 좋아지지 않으면 학습을 멈춘다 |
| dropout | 일부 뉴런을 확률적으로 꺼서 외우기를 줄인다 |
| weight decay | 가중치가 과도하게 커지는 것을 막는다 |
| 작은 hidden/channel 수 | 모델 규모를 데이터 크기에 맞춘다 |
| train-fold-only scaling | 검증/test 데이터 통계가 학습에 새지 않게 한다 |
| gradient clipping | 작은 데이터에서 학습이 불안정하게 튀는 것을 줄인다 |

## 6. 비교 결과

검증은 모든 모델에서 동일하게 3-fold `StratifiedGroupKFold` out-of-fold 방식으로 수행했다.

| Model | Accuracy | Macro F1 | Fall recall | Confusion matrix |
| --- | ---: | ---: | ---: | --- |
| RandomForest | 0.7521 | 0.7485 | 0.8222 | `[[51, 21], [8, 37]]` |
| 1D CNN | 0.5470 | 0.5469 | 0.6889 | `[[33, 39], [14, 31]]` |
| GRU | 0.5556 | 0.5408 | 0.4889 | `[[43, 29], [23, 22]]` |
| ROCKET-inspired | 0.8547 | 0.8472 | 0.8222 | `[[63, 9], [8, 37]]` |

결과 파일:

- `outputs/sequence_model_comparison_metrics.json`
- `outputs/sequence_model_comparison.png`
- `outputs/sequence_model_confusion_matrices.png`
- `outputs/rocket_imu_model_metrics.json`
- `outputs/rocket_confusion_matrix_oof.png`
- `outputs/rocket_feature_space_pca.png`

## 7. 해석

초기 비교에서는 RandomForest가 1D CNN과 GRU보다 좋은 성능을 보였다.

이 결과는 “딥러닝이 항상 나쁘다”는 뜻이 아니다. 현재 공개 데이터가 window 117개로 작기 때문에, 1D CNN과 GRU가 시계열 패턴을 충분히 학습하기에는 데이터가 부족하다는 뜻에 가깝다.

이후 ROCKET 논문을 기반으로 random convolution kernel transform + LogisticRegression 파이프라인을 추가했다. 이 방식은 9축 IMU 시계열을 직접 사용하지만, kernel을 직접 학습하지 않고 무작위로 고정한다. 그래서 작은 데이터에서 deep learning보다 과적합 부담이 낮다.

따라서 현재 포트폴리오의 정직한 결론은 다음과 같다.

> 1D CNN과 GRU는 현재 데이터 규모에서 충분히 일반화되지 못했다. 반면 ROCKET-inspired random convolution pipeline은 시계열 구조를 활용하면서 학습 parameter를 줄여 현재 데이터에서 가장 좋은 Macro F1을 보였다. 따라서 현재 best model은 ROCKET-inspired 모델이고, RandomForest는 설명 가능한 baseline으로 유지한다.
