# Glossary and Design Rationale

## 핵심 용어

### Wearable Safety Monitoring System

뜻:

- 작업자가 착용한 센서로 위험 상태를 감지하고 경고 또는 의사결정을 보조하는 시스템.

이 프로젝트에서의 의미:

- 현재 구현된 Peter Anchor의 정확한 위치다.
- IMU 기반으로 `fall_risk`를 감지하지만, 실제 근력 보조는 하지 않는다.

왜 이 용어를 쓰는가:

- 현재 모델은 sensing과 risk detection까지만 구현되어 있기 때문이다.
- 로봇이라고 부르려면 actuator와 제어 모델이 필요하다.

### Wearable Robot / Exoskeleton / Exosuit

뜻:

- 사람이 착용하고, 구동부가 힘 또는 토크를 제공해 움직임을 보조하는 로봇 시스템.

필수 요소:

- 센서
- 사용자 의도 추정
- 보조 힘/토크 계산
- actuator
- 안전 제어

왜 현재 프로젝트와 다른가:

- 현재 프로젝트에는 actuator torque 예측이나 실제 힘 보조가 없다.

### IMU

뜻:

- Inertial Measurement Unit.
- 가속도계와 자이로스코프를 통해 움직임과 회전을 측정하는 센서.

이 프로젝트에서 쓰는 이유:

- fall-risk는 급격한 가속도 변화, 자세 변화, 흔들림과 관련이 있기 때문이다.

### EMG

뜻:

- Electromyography.
- 근육이 활성화될 때 발생하는 전기 신호를 측정하는 센서.

웨어러블 로봇에 필요한 이유:

- 근육 활성도와 움직임 의도를 추정할 수 있다.
- 근력 보조가 필요한지 판단하는 데 중요하다.

### Joint Angle

뜻:

- 무릎, 고관절, 발목, 팔꿈치 같은 관절의 각도.

필요한 이유:

- 현재 자세와 움직임 범위를 알 수 있다.
- 보조 토크를 어느 방향으로 줘야 하는지 판단하는 데 필요하다.

### Joint Torque / Moment

뜻:

- 관절을 회전시키는 힘의 크기.

필요한 이유:

- wearable robot이 실제로 도와줘야 하는 target 값이다.
- torque label이 있어야 assistive torque regression을 학습할 수 있다.

### Assistive Torque

뜻:

- 로봇 또는 exosuit가 사용자 관절이나 자세 유지에 제공하는 보조 토크.

왜 중요한가:

- 이 값이 있어야 “근력 보조 로봇”이라고 말할 수 있다.
- 단순 위험 감지와 로봇 제어를 구분하는 핵심 개념이다.

### Actuator

뜻:

- 모터, 공압 장치, 유압 장치처럼 실제 힘을 내는 구동부.

필요한 이유:

- AI가 예측한 보조량을 실제 물리적 힘으로 바꾸는 장치다.

### Backpack Safety Net

뜻:

- 원래 기획에 있던 가방 내장형 그물망 보호 장치.
- 추락 위험이 커졌을 때 전개되어 충격을 줄이는 emergency protection device로 정의한다.

이 프로젝트에서의 의미:

- AI가 `fall_risk`를 감지한 뒤, safety response policy가 준비 또는 전개를 결정하는 대상이다.

왜 risk detection layer가 아닌가:

- 그물망은 위험을 판단하는 모델이 아니라, 판단 이후 작동하는 물리적 보호 장치이기 때문이다.

### Safety Supervisor

뜻:

- AI 모델의 위험 예측값과 장치 준비 상태를 함께 보고 최종 행동을 결정하는 정책 계층.

필요한 이유:

- 모델이 `fall_risk`라고 예측해도 그물망이 준비되지 않았거나 하네스 연결이 확인되지 않으면 바로 전개하면 안 된다.
- 따라서 예측값과 장치 상태를 함께 판단해야 한다.

### Windowing

뜻:

- 시계열 데이터를 일정 길이 구간으로 잘라 분석하는 방법.

이 프로젝트에서 쓰는 이유:

- 추락 위험은 한 순간의 row 하나보다 짧은 시간 동안의 움직임 패턴으로 나타난다.
- 그래서 100-sample window와 50-sample stride를 사용했다.

### Feature Engineering

뜻:

- 원본 데이터에서 모델이 학습하기 좋은 입력값을 만드는 과정.

이 프로젝트의 예:

- peak acceleration
- mean acceleration
- acceleration standard deviation
- peak gyro
- body angle proxy
- worker sway proxy

왜 필요한가:

- 현재 데이터 규모가 작기 때문에 raw sequence를 deep learning에 바로 넣는 것보다, 의미 있는 통계 feature를 만드는 것이 더 안정적이다.

### RandomForest

뜻:

- 여러 decision tree를 조합해 예측하는 ensemble machine learning 모델.

선택 이유:

- 작은 tabular dataset에 강하다.
- 비선형 조건을 다룰 수 있다.
- feature importance로 설명 가능하다.
- 현재 단계의 baseline 모델로 적절하다.

### StratifiedGroupKFold

뜻:

- class 비율을 최대한 유지하면서, 같은 group이 train/test에 동시에 들어가지 않도록 나누는 교차검증 방법.

이 프로젝트에서 group은:

- `recording_id`

선택 이유:

- 같은 recording에서 나온 window들은 서로 매우 비슷하다.
- random split을 쓰면 같은 recording의 유사 window가 train/test에 섞여 성능이 과대평가될 수 있다.

### Out-of-fold Prediction

뜻:

- 교차검증에서 각 fold의 test set 예측값을 모아 전체 데이터에 대한 검증 결과를 만드는 방식.

선택 이유:

- 단일 holdout보다 더 많은 window를 평가에 사용한다.
- 현재처럼 데이터가 작은 경우 confusion matrix가 너무 작아지는 문제를 줄인다.

### Recall

뜻:

- 실제 위험 상황 중 모델이 얼마나 많이 잡아냈는지 보는 지표.

수식:

- recall = true positive / (true positive + false negative)

왜 중요한가:

- 안전 문제에서는 위험을 놓치는 것이 특히 치명적이다.
- 그래서 accuracy보다 fall recall을 중요하게 본다.

### Confusion Matrix

뜻:

- 실제 라벨과 예측 라벨을 표로 비교한 것.

현재 결과:

| True \ Pred | Normal | Fall |
| --- | ---: | ---: |
| Normal | 51 | 21 |
| Fall | 8 | 37 |

해석:

- fall window 45개 중 37개를 잡았다.
- fall window 8개는 놓쳤다.
- normal window 21개를 fall로 과탐지했다.

### Arm vs Deploy

뜻:

- `arm`: 장치를 전개 직전 상태로 준비한다.
- `deploy`: 실제로 그물망을 펼친다.

왜 구분하는가:

- 위험이 조금 높아졌다고 바로 전개하면 오작동과 작업 방해가 생길 수 있다.
- 그래서 중간 위험에서는 준비만 하고, 높은 위험과 장치 준비 조건이 모두 맞을 때 전개한다.

## 방향 선택 이유

### 왜 가상 데이터를 제거했나

안전 관련 프로젝트에서 가상 데이터를 실제 센서 데이터처럼 제시하면 신뢰도가 떨어진다. 그래서 공개 실제 IMU 데이터와 공개 풍속 데이터로 바꿨다.

### 왜 현재는 로봇이 아니라 안전 모니터링 시스템으로 정의했나

현재 모델은 위험 감지까지만 한다. 근력 보조, 관절 보조 토크, actuator 제어가 없기 때문에 wearable robot이라고 주장하지 않는 것이 맞다.

### 왜 바로 assistive torque 모델을 만들지 않았나

현재 fall IMU 데이터에는 torque target이 없다. target 없는 상태에서 보조 토크를 만들면 다시 가정 기반 가짜 데이터가 된다. 따라서 로봇 확장은 별도 데이터셋과 별도 모델 단계로 분리했다.

### 왜 EMG와 joint torque 데이터가 필요한가

근력 보조는 단순히 “넘어질 것 같다”를 감지하는 문제가 아니라 “어느 관절에 얼마만큼의 힘을 도와줄 것인가”를 계산하는 문제다. 이를 위해 근육 활성도, 관절 움직임, 실제 관절 토크가 필요하다.

### 왜 현재 프로젝트를 sensing layer로 남기는가

웨어러블 로봇도 먼저 상태를 감지해야 한다. 현재 프로젝트는 로봇 전체는 아니지만, 로봇으로 확장될 때 필요한 위험 감지 계층으로 사용할 수 있다.

### 왜 가방형 그물망을 emergency protection layer로 둔다

그물망은 근력 보조 장치가 아니라 추락 상황에서 충격을 줄이는 보호 장치다. 따라서 assistive torque control과 같은 근력 보조 계층이 아니라, risk detection 결과를 받아 작동하는 emergency protection layer로 두는 것이 맞다.

### 왜 바로 그물망 전개를 모델이 결정하지 않게 했나

AI 모델은 움직임 패턴만 보고 `fall_risk`를 예측한다. 하지만 실제 전개 여부는 그물망 준비 상태, 하네스 연결, 높이, 수동 override 같은 안전 조건도 함께 봐야 한다. 그래서 모델 다음에 `Safety Supervisor` 정책 계층을 둔다.

### 왜 RandomForest를 선택했나

데이터가 117개 window로 작고, feature가 tabular 형태이기 때문이다. 딥러닝은 데이터가 부족하면 과적합 가능성이 높고 설명이 어렵다. RandomForest는 baseline으로 안정적이고 feature importance도 제공한다.

### 왜 시각화를 추가했나

모델링에서는 성능 숫자만 보여주면 부족하다. 데이터 분포, class imbalance, feature 차이, confusion matrix를 함께 보여줘야 모델이 어떤 데이터를 보고 무엇을 틀렸는지 설명할 수 있다.
