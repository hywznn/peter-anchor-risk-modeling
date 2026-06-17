# ROCKET-inspired IMU Pipeline

이 문서는 Peter Anchor Risk Modeling에 새로 추가한 논문 기반 시계열 분류 파이프라인을 설명한다.

## 1. 왜 이 파이프라인을 추가했는가

기존 repo에는 다음 모델이 있었다.

- engineered feature 기반 RandomForest
- 9축 IMU 시계열 기반 1D CNN
- 9축 IMU 시계열 기반 GRU

하지만 아쉬운 점이 있었다.

- RandomForest는 안정적이지만 전통적인 baseline에 가깝다.
- 1D CNN과 GRU는 시계열 모델이지만 데이터가 117 window뿐이라 과적합 위험이 크다.
- 논문을 직접 파고, 그 아이디어를 현재 데이터에 맞춰 구현한 흔적이 약했다.

그래서 ROCKET 계열 시계열 분류 논문을 기반으로 새로운 파이프라인을 추가했다.

## 2. 참고 논문

### ROCKET

Source: [ROCKET: Exceptionally fast and accurate time series classification using random convolutional kernels](https://arxiv.org/abs/1910.13051)

핵심 아이디어:

- 시계열에 많은 무작위 convolution kernel을 적용한다.
- 각 kernel 반응을 feature로 변환한다.
- 변환된 feature에 단순한 선형 분류기를 학습한다.

짧은 발췌:

> “random convolutional kernels”

### MiniRocket

Source: [MiniRocket: A Very Fast (Almost) Deterministic Transform for Time Series Classification](https://arxiv.org/abs/2012.08791)

핵심 아이디어:

- ROCKET을 더 빠르고 거의 deterministic하게 바꾼 방식이다.
- kernel transform 후 선형 분류기를 사용하는 큰 방향은 유지한다.

짧은 발췌:

> “almost deterministic transform”

## 3. 현재 repo에서의 구현 방식

구현 파일:

- `src/train_rocket_imu_model.py`

이 구현은 공식 ROCKET/MiniRocket 패키지가 아니다. 현재 프로젝트의 작은 IMU 데이터셋에 맞춰 직접 작성한 `ROCKET-inspired` 구현이다.

입력:

```text
117 windows x 100 time steps x 9 IMU channels
```

9개 채널:

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

파이프라인:

```text
100 x 9 IMU window
-> train-fold-only sensor scaling
-> random convolution kernels
-> max activation + PPV feature extraction
-> train-fold-only feature scaling
-> L2 LogisticRegression
-> StratifiedGroupKFold out-of-fold validation
```

## 4. ROCKET feature란 무엇인가

각 random kernel은 IMU window의 일부 시간 패턴을 훑는다.

예를 들어 어떤 kernel은 다음 패턴에 반응할 수 있다.

- 특정 축에서 짧게 튀는 가속도
- 회전 속도가 급격히 증가하는 구간
- 기울기 변화와 가속도가 같이 나타나는 구간
- 여러 IMU 채널의 조합 변화

각 kernel에서 두 가지 feature를 만든다.

| Feature | 의미 |
| --- | --- |
| max activation | 해당 kernel이 가장 강하게 반응한 정도 |
| PPV | positive activation 비율, 즉 해당 패턴이 window 안에서 얼마나 자주 나타났는지 |

현재 구현은 512개 kernel을 사용하므로 최종 feature 수는 다음과 같다.

```text
512 kernels x 2 features = 1024 features
```

## 5. 왜 작은 데이터에 적합한가

1D CNN이나 GRU는 convolution weight나 recurrent weight를 데이터에서 직접 학습한다. 현재 데이터가 117 window뿐이라 이런 모델은 쉽게 과적합될 수 있다.

반면 ROCKET-inspired 방식은 kernel을 무작위로 만들고 고정한다. 직접 학습되는 부분은 마지막 선형 분류기뿐이다.

그래서 다음 장점이 있다.

- raw 9축 시계열을 그대로 활용한다.
- 딥러닝보다 학습 parameter가 적다.
- 작은 데이터에서도 시계열 패턴 feature를 많이 만들 수 있다.
- RandomForest보다 논문 기반 시계열 분류 파이프라인이라는 기술적 깊이가 있다.

## 6. 과적합 방지

| 방법 | 이유 |
| --- | --- |
| fixed random kernels | kernel 자체는 label을 보고 학습하지 않는다 |
| LogisticRegression L2 penalty | 선형 분류기 가중치가 과도하게 커지는 것을 막는다 |
| class_weight balanced | 정상/낙상 class 불균형을 완화한다 |
| train-fold-only sensor scaling | 검증 데이터 통계가 학습에 새지 않게 한다 |
| train-fold-only feature scaling | ROCKET feature scaling도 train fold에서만 학습한다 |
| StratifiedGroupKFold | 같은 recording에서 나온 window가 train/test에 동시에 들어가지 않게 한다 |

## 7. 결과

검증 방식:

```text
3-fold StratifiedGroupKFold out-of-fold validation by recording_id
```

| Model | Accuracy | Macro F1 | Fall recall | Confusion matrix |
| --- | ---: | ---: | ---: | --- |
| RandomForest | 0.7521 | 0.7485 | 0.8222 | `[[51, 21], [8, 37]]` |
| 1D CNN | 0.5470 | 0.5469 | 0.6889 | `[[33, 39], [14, 31]]` |
| GRU | 0.5556 | 0.5408 | 0.4889 | `[[43, 29], [23, 22]]` |
| ROCKET-inspired | 0.8547 | 0.8472 | 0.8222 | `[[63, 9], [8, 37]]` |

해석:

- ROCKET-inspired 모델은 fall recall은 RandomForest와 동일하게 0.8222다.
- 대신 normal activity를 fall로 잘못 판단하는 false alarm이 21개에서 9개로 줄었다.
- 그 결과 Accuracy와 Macro F1이 가장 높아졌다.

## 8. 결과 파일

- `outputs/rocket_imu_model_metrics.json`
- `outputs/rocket_confusion_matrix_oof.png`
- `outputs/rocket_feature_space_pca.png`
- `outputs/all_model_comparison.png`

## 9. 포트폴리오에서의 표현

추천 표현:

> 기존 baseline 모델 이후, ROCKET 논문을 기반으로 9축 IMU 시계열을 random convolution kernel feature로 변환하는 파이프라인을 직접 구현했습니다. 공식 패키지를 단순 사용한 것이 아니라, 논문의 핵심 아이디어인 random kernel transform, max/PPV feature, regularized linear classifier 구조를 현재 데이터에 맞게 재현했습니다. 그 결과 기존 RandomForest 대비 Macro F1이 0.7485에서 0.8472로 개선되었습니다.

주의할 표현:

> 이 구현은 공식 ROCKET/MiniRocket 재현체가 아니라, 현재 프로젝트에 맞춘 ROCKET-inspired implementation입니다.
