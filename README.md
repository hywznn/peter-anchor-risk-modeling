# Peter Anchor Risk Modeling

고층 외벽 청소 작업자의 추락 위험을 줄이기 위한 **AI 안전 웨어러블 위험 감지 모델링 프로젝트**다.  
현재 구현 범위는 웨어러블 로봇 전체가 아니라, Peter Anchor 시스템에 들어갈 수 있는 **Risk Detection Layer**다.

이 repo는 팀 경진대회 산출물인 **“피터 앵커: 거미줄형 비상 그물 흡착망 AI 안전 웨어러블 로봇”**을 배경으로 삼되, 그중 실제 공개 데이터로 검증 가능한 AI 위험 감지 파트를 별도로 후속 고도화한 것이다. 원본 팀 산출물과 현재 개인 데이터 모델링 repo의 연결 관계는 [docs/original_concept_context.md](docs/original_concept_context.md)에 정리했다.

## 0. Contribution Note

원본 보고서와 포스터는 팀 프로젝트 산출물이므로, 이 repo에서 해당 내용을 개인 단독 성과처럼 주장하지 않는다.

원본 PPT, PDF, 포스터, 상장 스캔본은 팀 산출물 및 개인정보/권리 이슈가 있을 수 있어 이 repo에 공개하지 않는다. 이 repo에는 공개 가능한 코드, 데이터 처리 과정, 모델링 결과, 후속 분석 문서만 포함한다.

이 repo에서 개인 작업으로 정리하는 범위는 다음이다.

- 원본 팀 기획의 AI 위험 감지 파트를 실제 공개 데이터 기반 문제로 재정의
- 공개 IMU 데이터와 풍속 데이터 수집 파이프라인 구성
- 가상 데이터 제거 및 실제 데이터 기반 전처리, feature engineering, 모델링
- RandomForest, 1D CNN, GRU 비교 실험
- ROCKET 논문 기반 random convolution kernel 시계열 파이프라인 직접 구현
- 과적합 방지 검증 구조와 결과 해석
- 원본 가방형 그물망 아이디어를 emergency protection layer로 재정리

## 1. 핵심 문제

고층 외벽 청소는 로프, 하네스, 외부 기상 조건에 크게 의존하는 고위험 작업이다. KOSHA의 건물 외벽 청소 작업 지침도 외벽·유리창 청소가 높은 곳에서 수행되며 추락 사고가 치명적일 수 있음을 전제로 한다.

이 프로젝트는 다음 질문에서 출발했다.

> 작업자의 움직임 센서 데이터만으로 fall-like motion과 normal activity를 구분할 수 있을까?

## 2. 프로젝트 범위

현재 모델이 하는 일:

- 실제 공개 IMU 데이터 수집
- 낙상 이벤트와 일상 동작 분류
- 9축 IMU 시계열 모델과 baseline 모델 비교
- 위험 감지 이후의 안전 대응 정책 설계

현재 모델이 하지 않는 일:

- Peter Anchor 제품 안전성 인증
- 로프 장력, 하네스 압력, 흡착 상태 예측
- 근력 보조 로봇 제어
- 현장 고층 외벽 작업 성능 검증

로프 장력, 하네스 압력, 흡착 준비 상태는 공개 데이터로 확인되지 않았으므로 임의 생성하지 않고 향후 직접 수집해야 할 센서로 분리했다.

## 3. 근거 자료

주요 근거는 [docs/evidence_sources.md](docs/evidence_sources.md)에 정리했다.

| 구분 | 근거 |
| --- | --- |
| 작업 위험성 | KOSHA 건물 외벽 청소 작업 기술지침, OSHA fall protection 기준 |
| 센서 타당성 | LSM6DSO 6축 IMU, IMU 기반 fall detection 연구 |
| 데이터 출처 | `jasonkau/fall-detection-dataset-IMU`, Open-Meteo Historical Weather API |
| 모델 선택 | RandomForest baseline, 1D CNN, GRU, ROCKET-inspired pipeline 비교 실험 |
| 안전 대응 연계 | fall detection 특허, smart harness 특허, wearable airbag/device 사례 |

## 4. 실제 데이터

| 데이터 | 출처 | 사용 목적 |
| --- | --- | --- |
| IMU fall/daily activity | [jasonkau/fall-detection-dataset-IMU](https://github.com/jasonkau/fall-detection-dataset-IMU) | fall-risk 분류 모델 학습 |
| 과거 풍속 | [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) | 향후 wind risk 확장용 환경 맥락 |

생성 데이터:

| 파일 | 설명 |
| --- | --- |
| `data/real_imu_fall_detection_samples.csv` | 원본 IMU sample-level 데이터 |
| `data/real_imu_fall_detection_windows.csv` | 모델 입력용 100-sample window 데이터 |
| `data/real_weather_wind_seoul_2025.csv` | 서울 2025년 시간별 풍속 데이터 |
| `data/real_data_manifest.json` | 데이터 출처와 한계 기록 |

## 5. 모델링 흐름

```text
실제 공개 데이터 수집
-> 원본 Excel header 보정
-> 센서 단위 변환 및 라벨링
-> 100-sample sliding window 생성
-> feature engineering
-> RandomForest baseline 학습
-> 1D CNN / GRU 시계열 모델 비교
-> ROCKET-inspired random convolution kernel pipeline
-> StratifiedGroupKFold out-of-fold 검증
-> 안전 대응 정책과 연결
```

## 6. 피처 설계

원본 IMU는 다음 축을 포함한다.

```text
가속도 x/y/z
자이로 x/y/z
기울기 x/y/z
```

기존 baseline은 100개 sample 구간을 다음 feature로 요약한다.

- 최대/평균/표준편차 가속도
- 최대/평균 자이로
- 최대/평균 몸 기울기
- 좌우 흔들림 proxy

시계열 비교 실험에서는 원본 9축 흐름을 그대로 사용했다.

```text
117 windows x 100 time steps x 9 IMU channels
```

## 7. 모델 비교 결과

모든 모델은 같은 기준의 3-fold `StratifiedGroupKFold` out-of-fold 검증으로 비교했다. 같은 원본 recording에서 나온 window가 train/test에 동시에 들어가지 않도록 해 데이터 누수를 줄였다.

| Model | Input | Accuracy | Macro F1 | Fall recall |
| --- | --- | ---: | ---: | ---: |
| RandomForest | engineered window features | 0.7521 | 0.7485 | 0.8222 |
| 1D CNN | 100 x 9 sequence | 0.5470 | 0.5469 | 0.6889 |
| GRU | 100 x 9 sequence | 0.5556 | 0.5408 | 0.4889 |
| ROCKET-inspired | random convolution features | 0.8547 | 0.8472 | 0.8222 |

현재 공개 데이터는 window 117개로 작다. 1D CNN과 GRU는 데이터 규모에 비해 직접 학습 parameter가 있어 성능이 낮았다. 반면 ROCKET-inspired 방식은 random convolution kernel을 고정하고 regularized linear classifier만 학습해, 작은 데이터에서도 시계열 패턴을 더 안정적으로 활용했다.

최종 선택:

> 현재 단계의 best model은 ROCKET-inspired random convolution kernel pipeline이다. 다만 이 구현은 공식 ROCKET/MiniRocket 패키지가 아니라, 논문의 핵심 아이디어를 현재 IMU 데이터에 맞춰 직접 재현한 것이다.

## 8. 결과물

| 파일 | 설명 |
| --- | --- |
| `outputs/real_imu_model_metrics.json` | RandomForest 검증 결과 |
| `outputs/sequence_model_comparison_metrics.json` | RandomForest, 1D CNN, GRU 비교 결과 |
| `outputs/rocket_imu_model_metrics.json` | ROCKET-inspired 논문 기반 파이프라인 결과 |
| `outputs/model_confusion_matrix_oof.png` | baseline confusion matrix |
| `outputs/model_feature_importance_real_imu.png` | feature importance |
| `outputs/sequence_model_comparison.png` | 모델별 지표 비교 |
| `outputs/sequence_model_confusion_matrices.png` | 모델별 confusion matrix |
| `outputs/all_model_comparison.png` | ROCKET-inspired까지 포함한 전체 모델 비교 |
| `outputs/rocket_confusion_matrix_oof.png` | ROCKET-inspired confusion matrix |
| `outputs/rocket_feature_space_pca.png` | ROCKET feature space preview |

## 9. 문서 구조

| 문서 | 내용 |
| --- | --- |
| [docs/original_concept_context.md](docs/original_concept_context.md) | 경진대회 원본 기획과 현재 AI 모델링 repo의 연결 |
| [docs/evidence_sources.md](docs/evidence_sources.md) | 논문, 특허, 공식 자료 근거 정리 |
| [docs/ai_modeling_process.md](docs/ai_modeling_process.md) | 목적, 가설, 수집, 전처리, 모델링, 검증 |
| [docs/sequence_model_comparison.md](docs/sequence_model_comparison.md) | 1D CNN/GRU 비교와 과적합 방지 |
| [docs/rocket_paper_pipeline.md](docs/rocket_paper_pipeline.md) | ROCKET 논문 기반 random convolution pipeline |
| [docs/system_architecture.md](docs/system_architecture.md) | 현재/미래 시스템 구조 |
| [docs/backpack_safety_net_integration.md](docs/backpack_safety_net_integration.md) | 가방형 그물망 emergency layer 연계 |
| [docs/wearable_robot_extension_plan.md](docs/wearable_robot_extension_plan.md) | 근력 보조 웨어러블 로봇 확장 조건 |

## 10. 실행 방법

```bash
pip install -r requirements.txt
python src/fetch_real_data.py
python src/train_real_imu_model.py
python src/compare_sequence_models.py
python src/train_rocket_imu_model.py
python src/visualize_real_data.py
```

## 11. 한계

- 공개 IMU 데이터는 실제 센서 기록이지만 고층 외벽 청소 현장 데이터는 아니다.
- 풍속 데이터는 IMU와 같은 시간/장소에서 수집된 데이터가 아니므로 현재 모델 입력에 결합하지 않았다.
- 현재 모델 성능은 Peter Anchor 제품 성능이나 안전 인증 결과로 해석할 수 없다.
- 실제 적용에는 로프 장력, 하네스 압력, 현장 IMU, 외벽 근접 풍속, 흡착 준비 상태를 같은 시간축으로 수집해야 한다.
