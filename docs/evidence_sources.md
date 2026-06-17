# Evidence Sources

이 문서는 Peter Anchor Risk Modeling의 문제 정의, 데이터 선택, 모델링 방식, 안전 대응 레이어 설계를 뒷받침하는 근거 자료를 정리한다.

중요한 주의점:

- 아래 특허는 선행기술 조사와 아이디어 근거를 위한 참고 자료다.
- 특허 인용은 제품 구현 가능성이나 자유실시권을 보장하지 않는다.
- 실제 제품화 전에는 별도의 freedom-to-operate 검토, 안전 인증, 현장 실증이 필요하다.

## 1. 문제 정의 근거

### KOSHA 건물 외벽 청소 작업 기술지침

Source: [건물 외벽 청소 작업에 관한 기술지침, G-67-2011](https://www.aposho.org/kosha/intro/jeonnamBranch_A.do?articleNo=341019&attachNo=187798&mode=download)

짧은 발췌:

> “추락 등 사고 발생 시 치명적인 사고”

프로젝트 반영:

- Peter Anchor의 문제를 단순 편의 장비가 아니라 고위험 작업자의 안전 문제로 설정했다.
- 현재 AI 모델의 목적도 작업자 대체가 아니라 위험 전조 감지와 대응 보조로 정의했다.

### OSHA Personal Fall Protection Systems

Source: [29 CFR 1910.140 Personal fall protection systems](https://www.ecfr.gov/current/title-29/subtitle-B/chapter-XVII/part-1910/subpart-I/section-1910.140)

짧은 발췌:

> “protected from being cut, abraded, melted, or otherwise damaged”

프로젝트 반영:

- 로프/하네스 상태는 실제 안전 시스템에서 중요한 관리 대상이다.
- 현재 공개 데이터에는 로프 장력, 마모, 하네스 압력 정보가 없으므로 이를 가상 생성하지 않고 향후 직접 수집 대상으로 분리했다.

### OSHA Safety Net Systems

Source: [OSHA Construction eTool - Safety Net Systems](https://www.osha.gov/etools/construction/falls/safety-net)

짧은 발췌:

> “as close as practicable under the surface”

프로젝트 반영:

- 사용자가 말한 가방형 그물망은 ML 모델 자체가 아니라 감지 이후 작동하는 Emergency Protection Layer로 두는 것이 타당하다.
- 안전망은 추락을 예측하는 센서가 아니라, 위험 판단 이후 피해를 줄이는 물리적 대응 장치에 가깝다.

## 2. 데이터와 센서 근거

### 공개 IMU Fall Detection Dataset

Source: [jasonkau/fall-detection-dataset-IMU](https://github.com/jasonkau/fall-detection-dataset-IMU)

짧은 발췌:

> “simulated falls and common daily activities”

프로젝트 반영:

- `Fall Events` 폴더는 `fall_event = 1`로, `Daily Activities` 폴더는 `fall_event = 0`으로 라벨링했다.
- 이 라벨은 임의 생성이 아니라 원본 데이터셋의 폴더 구조에 근거한다.
- 단, 해당 데이터는 고층 외벽 청소 현장 데이터가 아니라 공개 낙상/일상동작 IMU 데이터다.

### STMicroelectronics LSM6DSO

Source: [LSM6DSO product page](https://www.st.com/en/mems-and-sensors/lsm6dso.html)

짧은 발췌:

> “3-axis digital accelerometer and a 3-axis digital gyroscope”

프로젝트 반영:

- 원본 데이터셋이 사용하는 LSM6DSO는 3축 가속도와 3축 자이로를 포함하는 IMU다.
- 그래서 프로젝트의 9축 시계열 입력은 가속도 x/y/z, 자이로 x/y/z, 기울기 x/y/z로 구성했다.

### Open-Meteo Historical Weather API

Source: [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)

짧은 발췌:

> “a comprehensive record of past weather conditions”

프로젝트 반영:

- 풍속과 돌풍 데이터는 Peter Anchor의 wind risk 확장 가능성을 설명하기 위해 수집했다.
- 현재 IMU 데이터와 시간/장소가 동기화된 데이터가 아니므로, 현 모델 입력에는 결합하지 않았다.

## 3. 모델링 근거

### IMU 기반 fall risk assessment

Source: [Wearable Sensor Systems for Fall Risk Assessment: A Review](https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2022.921506/full)

짧은 발췌:

> “IMUs can be placed at various locations on the body”

프로젝트 반영:

- IMU는 신체 움직임을 측정해 fall risk 또는 fall status 분석에 활용될 수 있다.
- 따라서 Peter Anchor의 첫 AI 레이어를 카메라가 아닌 착용형 IMU 기반 risk detection으로 설정했다.

### 1D CNN / GRU 시계열 모델 근거

Source: [Kim et al., 2021, PubMed](https://pubmed.ncbi.nlm.nih.gov/34833704/)

짧은 발췌:

> “1D convolutional neural network and a gated recurrent unit”

프로젝트 반영:

- IMU는 시간 흐름을 가진 multivariate time-series 데이터이므로 1D CNN과 GRU를 후보로 비교했다.
- 현재 프로젝트에서는 `117 windows x 100 time steps x 9 channels` 입력으로 1D CNN과 GRU를 실험했다.
- 다만 현재 데이터 수가 작아 최종 baseline은 RandomForest로 유지했다.

### CNN-BiLSTM fall detection 연구

Source: [Li et al., 2022, Applied Sciences](https://www.mdpi.com/2076-3417/12/19/9671)

짧은 발췌:

> “acceleration and angular velocity data”

프로젝트 반영:

- 가속도와 각속도를 이용한 낙상 분류는 기존 연구에서도 반복적으로 쓰이는 입력 구조다.
- 본 프로젝트의 feature engineering도 최대 가속도, 회전 강도, 자세 기울기 변화를 중심으로 설계했다.

### ROCKET random convolution kernel transform

Source: [ROCKET: Exceptionally fast and accurate time series classification using random convolutional kernels](https://arxiv.org/abs/1910.13051)

짧은 발췌:

> “random convolutional kernels”

프로젝트 반영:

- 9축 IMU window를 직접 시계열로 다루기 위해 ROCKET 논문의 random convolution kernel transform 아이디어를 적용했다.
- 현재 구현은 공식 ROCKET 패키지가 아니라, random kernel, max activation, PPV feature, regularized linear classifier 구조를 작은 IMU 데이터에 맞춰 직접 구현한 `ROCKET-inspired` 파이프라인이다.
- 기존 RandomForest보다 Macro F1이 높아져 현재 repo의 best model로 정리했다.

### MiniRocket deterministic transform

Source: [MiniRocket: A Very Fast (Almost) Deterministic Transform for Time Series Classification](https://arxiv.org/abs/2012.08791)

짧은 발췌:

> “almost deterministic transform”

프로젝트 반영:

- ROCKET 계열이 단순한 deep learning 모델이 아니라, time-series transform + linear classifier 계열로 발전했다는 점을 확인했다.
- 현재 데이터 수가 작으므로 parameter가 큰 Transformer나 깊은 CNN보다 ROCKET 계열 접근이 더 설득력 있는 후보라고 판단했다.

### StratifiedGroupKFold 검증 근거

Source: [scikit-learn StratifiedGroupKFold documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html)

짧은 발췌:

> “stratified folds with non-overlapping groups”

프로젝트 반영:

- 같은 원본 recording에서 나온 window들은 서로 유사하므로 train/test에 동시에 들어가면 데이터 누수가 생길 수 있다.
- `recording_id`를 group으로 사용해 같은 원본 파일에서 나온 window들이 한 fold 안에 같이 움직이도록 했다.
- class 비율도 가능한 한 유지해야 하므로 StratifiedGroupKFold를 선택했다.

## 4. 특허와 선행기술 근거

### Wearable fall detection

Source: [US8217795B2 - Method and system for fall detection](https://patents.google.com/patent/US8217795B2/en)

짧은 발췌:

> “wearable monitoring device that monitors the movement”

프로젝트 반영:

- 착용형 장치가 움직임 센서를 통해 fall 여부를 판단하는 구조는 선행기술로 존재한다.
- Peter Anchor의 현재 모델은 이 범주 중 risk detection layer에 해당한다.

### Pre-impact fall prediction

Source: [US8990041B2 - Fall detection](https://patents.google.com/patent/US8990041B2/en)

짧은 발췌:

> “predict whether a fall event is imminent”

프로젝트 반영:

- 단순 사후 감지보다 낙상 전/중간의 kinematic signal을 이용하는 방향이 중요하다.
- 본 프로젝트의 `fall_risk` 분류는 향후 pre-impact 대응으로 확장될 수 있다.

### Smart harness monitoring

Source: [US20170193799A1 - Fall Protection Monitoring System](https://patents.google.com/patent/US20170193799A1/en)

짧은 발췌:

> “motion sensors on safety hooks and on a safety harness”

프로젝트 반영:

- 향후 Peter Anchor는 IMU뿐 아니라 하네스, 후크, 로프 연결 상태 센서를 함께 보는 구조로 확장할 수 있다.
- 현재 공개 데이터에는 이 정보가 없으므로 architecture 문서에서 future sensor로 분리했다.

### Fall risk detection and ambulatory sensor

Source: [US20160100776A1 - Fall detection and fall risk detection systems and methods](https://patents.google.com/patent/US20160100776A1/en)

짧은 발췌:

> “quantifies the subject's gait”

프로젝트 반영:

- 착용형 센서로 활동량, 보행, 낙상 위험을 함께 분석하는 방향은 Peter Anchor의 장기 확장과 연결된다.
- 근력 보조 웨어러블 로봇으로 확장하려면 IMU 외에 EMG, 관절각, 관절토크, actuator 데이터가 필요하다.

### Wearable airbag / cushion deployment

Source: [WO2005110133A1 - A jacket and belt with airbags](https://patents.google.com/patent/WO2005110133A1/en)

짧은 발췌:

> “automatically inflate airbags”

프로젝트 반영:

- 위험 감지 이후 물리적 보호 장치를 전개하는 구조는 선행기술로 확인된다.
- Peter Anchor의 가방형 그물망은 이와 같은 emergency mitigation 계층으로 해석할 수 있다.

### Fall protection system with gear storage

Source: [US8061479B2 - Fall protection system](https://patents.google.com/patent/US8061479B2/en)

짧은 발췌:

> “incorporating gear storage capability”

프로젝트 반영:

- 착용 안전장비와 수납/팩 구조를 결합하는 아이디어는 기존 fall protection 영역에도 존재한다.
- 따라서 가방형 그물망은 “센서 모델”이 아니라 하네스/팩 기반 안전 대응 장치로 연결하는 것이 자연스럽다.

### FDA-reviewed wearable fall injury mitigation device

Source: [FDA DEN240021 decision summary](https://www.accessdata.fda.gov/cdrh_docs/reviews/DEN240021.pdf)

짧은 발췌:

> “wearable belt designed to protect”

프로젝트 반영:

- 웨어러블 장치가 fall injury mitigation을 목표로 설계될 수 있음을 보여주는 실제 규제 문서 사례다.
- 단, 대상은 고령자 낙상 보호 장치이며 Peter Anchor 제품 검증 자료는 아니다.

## 5. 현재 프로젝트로부터의 정직한 결론

근거 자료를 종합하면 다음 결론이 타당하다.

1. 고층 외벽 청소는 추락 시 치명적인 위험을 갖는 작업이다.
2. IMU는 착용자의 움직임, 자세, 회전 변화를 측정할 수 있어 fall-risk detection의 1차 센서로 사용할 수 있다.
3. 공개 데이터로 검증 가능한 범위는 `normal_activity`와 `fall_risk`의 IMU 패턴 분류다.
4. 로프 장력, 하네스 압력, 흡착 준비 상태, 가방형 그물망 전개 성능은 현재 데이터로 검증할 수 없다.
5. 1D CNN/GRU도 비교했지만 현재 데이터 규모에서는 성능이 낮았다.
6. ROCKET-inspired random convolution kernel 파이프라인을 추가한 결과, 현재 공개 데이터 기준 best model은 ROCKET-inspired 모델이다.
7. Peter Anchor의 다음 단계는 실제 현장 센서 동기화 데이터 수집이다.
