# Peter Anchor

AI 센서 기반 하네스 결합형 웨어러블 안전 시스템 기획 및 위험 모델링

## 프로젝트 개요

Peter Anchor는 팀 경진대회에서 제안된 하네스 결합형 AI 안전 웨어러블 시스템 콘셉트다. 이 포트폴리오 프로젝트는 해당 팀 기획 보고서 중 AI 위험 감지 파트를 실제 공개 센서 데이터 기반 분석으로 확장한 후속 개인 작업이다.

초기에는 합성 센서 데이터로 위험 판단 구조를 실험했지만, 현재 포트폴리오 버전에서는 공개 IMU fall-detection 데이터셋과 과거 풍속 API를 사용하도록 전환했다.

현재 구현은 근력 보조 로봇 모델이 아니라 fall-risk 감지 모델이다. 웨어러블 로봇으로 주장하려면 EMG/관절토크/assistive torque 데이터를 기반으로 보조 토크 추정 또는 근력 보조 제어 모델을 추가해야 한다.

## 문제 정의

고층 외벽 청소 작업은 로프와 하네스에 의존하는 고위험 작업이다. 핵심 문제는 청소 자동화 자체가 아니라, 자세 불안정, 급격한 가속도 변화, 돌풍에 의한 흔들림 같은 위험 전조를 작업자가 즉시 감지하기 어렵다는 점이다.

## 내가 수행한 후속 작업

- 실제 공개 IMU fall-detection 데이터셋을 조사하고 수집 파이프라인을 구현했다.
- Open-Meteo Historical Weather API로 서울 지역 시간별 풍속 데이터를 수집했다.
- 원본 IMU 엑셀 파일을 sample-level 및 window-level CSV로 변환했다.
- fall-risk 판단에 사용할 acceleration, angular velocity, body-angle proxy, sway proxy feature를 설계했다.
- RandomForest 기반 fall-risk 분류 모델을 학습하고, recording 단위 group-aware split으로 평가했다.
- 1D CNN, GRU, ROCKET-inspired random convolution kernel pipeline을 비교해 시계열 모델링 가능성을 검토했다.
- ROCKET 논문의 핵심 아이디어를 현재 IMU 데이터에 맞춰 직접 구현하고, 기존 baseline보다 높은 Macro F1을 얻었다.
- 공개 데이터로 확보되지 않는 로프 장력, 하네스 압력, 흡착 준비 상태는 직접 수집 필요 데이터로 분리했다.

## 기술적 접근

초기 baseline 모델 입력값은 실제 IMU 기록에서 추출한 다음 feature로 구성했다.

- peak/mean/std acceleration
- peak/mean angular velocity
- peak/mean body-angle proxy
- worker-sway proxy

위험 유형은 공개 데이터 라벨 범위에 맞춰 `normal_activity`와 `fall_risk`로 제한했다. 기존 기획서에서 정의한 `harness_error`, `wind_collision_risk`, `emergency_stabilization`은 실제 센서·현장 데이터가 확보된 뒤 확장할 항목으로 남겼다.

추가로 9축 IMU 시계열을 그대로 사용하는 논문 기반 파이프라인을 구현했다.

- 입력: `117 windows x 100 time steps x 9 IMU channels`
- 방식: random convolution kernel transform
- Feature: max activation, PPV
- Classifier: L2 LogisticRegression
- 검증: recording 단위 `StratifiedGroupKFold`

## 정량 지표

실제 공개 IMU 데이터 기반 모델링 결과는 다음과 같다.

| 항목 | 결과 |
| --- | --- |
| IMU sample rows | 6,526 |
| IMU window rows | 117 |
| Weather rows | 8,760 |
| Validation | 3-fold StratifiedGroupKFold out-of-fold |
| Best model | ROCKET-inspired |
| Accuracy | 0.8547 |
| Macro F1 | 0.8472 |
| Fall recall | 0.8222 |
| Confusion matrix | `[[63, 9], [8, 37]]` |

모델 비교:

| Model | Accuracy | Macro F1 | Fall recall |
| --- | ---: | ---: | ---: |
| RandomForest | 0.7521 | 0.7485 | 0.8222 |
| 1D CNN | 0.5470 | 0.5469 | 0.6889 |
| GRU | 0.5556 | 0.5408 | 0.4889 |
| ROCKET-inspired | 0.8547 | 0.8472 | 0.8222 |

다음 지표는 사업 기획 단계의 목표/가정이며, 실제 제품 검증 결과가 아니다.

| 항목 | 목표/가정 |
| --- | --- |
| 위험 전조 감지율 | 80% 목표 |
| 충돌·부상 피해 완화 | 30% 기대 |
| 렌탈 모델 | 월 60만 원 |
| 연간 렌탈비 | 720만 원 |
| 운영 규모 | 200대 가정 |
| BEP | 18개월 가정 |

## 한계와 개선 방향

공개 IMU 데이터는 실제 센서 기록이지만, 고층 외벽 청소 현장 데이터는 아니다. 또한 공개 풍속 API는 지역 기상 이력이며, 건물 외벽 근처의 실제 풍속을 대체할 수 없다.

제품화 단계에서는 로프 장력 센서, 하네스 압력 센서, 현장 IMU 로그, 근접 풍속계, 비상 안정화 모듈 테스트 데이터를 직접 수집해야 한다. 이후 안전 인증과 현장 파일럿 테스트를 거쳐야 실제 성능을 주장할 수 있다.

웨어러블 로봇으로 확장하려면 추가로 EMG, 관절각, 관절속도, 관절토크, 작업 부하, actuator torque, 보조 전후 피로도 또는 대사비용 감소 지표가 필요하다.

## 이 프로젝트를 통해 배운 점

안전 관련 AI 프로젝트에서는 “그럴듯한 데이터”보다 데이터 출처와 검증 범위를 명확히 밝히는 것이 더 중요하다는 점을 정리했다. 또한 데이터가 작을 때는 무작정 깊은 모델을 쓰기보다, 논문 기반 시계열 feature transform처럼 데이터 규모에 맞는 모델링 전략을 선택하는 것이 더 설득력 있다는 점을 배웠다.
