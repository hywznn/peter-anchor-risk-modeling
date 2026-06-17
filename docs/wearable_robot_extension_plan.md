# Wearable Robot Extension Plan

## Why the Current Project Is Not Yet a Wearable Robot

현재 구현은 IMU 기반 `fall_risk` 감지 모델이다. 센서를 착용한다는 점에서 wearable safety system에는 해당하지만, 근력 보조나 관절 보조 토크를 생성하지 않기 때문에 wearable robot 또는 exoskeleton이라고 주장하기에는 부족하다.

현재 모델을 risk detection layer로 두고, 향후 EMG/관절각/관절토크 기반 assistive control layer를 추가해야 한다는 근거는 `docs/evidence_sources.md`와 아래 후보 공개 데이터 출처에 정리했다.

웨어러블 로봇으로 확장하려면 다음 중 하나 이상이 필요하다.

- 사용자의 움직임 의도 추정
- 근육 활성도 또는 피로도 추정
- 필요한 관절 보조 토크 예측
- actuator가 실제로 보조 토크를 출력하는 제어 구조
- 보조 전후 작업 부담 감소 검증

## Revised Product Direction

현재 프로젝트의 정확한 명칭은 다음이 더 적절하다.

> AI sensor-based wearable safety monitoring system

웨어러블 로봇으로 확장한 버전은 다음처럼 정의할 수 있다.

> AI-assisted harness exosuit for fall-risk detection and posture/strength assistance

이 경우 AI 모델은 두 계층으로 나뉜다.

1. Risk detection model
   - 현재 구현된 IMU 기반 fall-risk 감지
2. Emergency protection policy
   - fall-risk 감지 결과를 가방형 그물망 준비/전개 행동으로 연결
3. Assistive control model
   - EMG/IMU/관절각 기반 근력 보조 필요도 또는 assistive torque 예측

가방형 그물망은 근력 보조 장치가 아니라 비상 보호 장치다. 따라서 웨어러블 로봇의 `assistive control`과는 별도로 `emergency protection layer`로 다루는 것이 적절하다.

## Data Required for Robotic Assistance

현재 fall IMU 데이터로는 근력 보조량을 직접 학습할 수 없다. 다음 데이터가 필요하다.

| Data | Why Needed |
| --- | --- |
| EMG | 근육 활성도와 사용자 의도 추정 |
| Joint angle / angular velocity | 관절 움직임 상태 파악 |
| Joint torque / moment | 실제 필요한 보조 토크의 target |
| Ground reaction force or load | 작업 부하와 균형 상태 계산 |
| Actuator torque/current | 로봇이 실제 제공한 보조량 검증 |
| Fatigue/metabolic indicators | 보조 효과 검증 |

가방형 그물망 전개를 검증하려면 다음 데이터도 필요하다.

| Data | Why Needed |
| --- | --- |
| Net deployment time | 위험 감지 후 실제 전개가 충분히 빠른지 확인 |
| Deployment success/failure logs | 전개 신뢰성 검증 |
| Net anchor load | 고정점과 소재 안전성 검증 |
| Dummy fall impact force | 충격 완화 성능 검증 |
| Harness tension during deployment | 작업자에게 전달되는 힘 확인 |

## Candidate Public Data Sources

### Lower-limb biomechanics and wearable sensors

- Source: A human lower-limb biomechanics and wearable sensors dataset
- URL: https://www.nature.com/articles/s41597-023-02840-6
- Fit: diverse activity biomechanics with wearable sensor context.

### Camargo lower-limb biomechanics dataset

- Source: Georgia Tech EPIC Lab open-source biomechanics dataset
- URL: https://www.epic.gatech.edu/opensource-biomechanics-camargo-et-al/
- Fit: includes EMG, IMU, goniometers, motion capture markers, and force plates. Suitable for locomotion intent and joint mechanics modeling.

### ENABL3S neuromechanical dataset

- Source: Benchmark Datasets for Bilateral Lower-Limb Neuromechanical Signals
- URL: https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2018.00014/full
- Fit: bilateral EMG and joint/limb kinematics from wearable sensors during locomotion transitions.

### EMG-to-torque / exoskeleton assistance studies

- Source: Robust Torque Predictions From Electromyography Across Multiple Levels of Active Exoskeleton Assistance
- URL: https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2021.700823/full
- Fit: shows EMG-based joint torque prediction under different exoskeleton assistance levels.

### Upper-limb torque prediction dataset

- Source: Zenodo upper-limb torque prediction dataset
- URL: https://data.niaid.nih.gov/resources?id=zenodo_11209323
- Fit: kinematic, dynamic, and EMG data from 17 participants. Suitable if Peter Anchor is reframed as upper-body or arm support assistance.

## Possible AI Modeling Targets

### Option A. Assistance Need Classification

목표:

- 현재 작업자가 보조가 필요한 상태인지 분류

입력:

- IMU acceleration
- body angle
- EMG activation
- joint velocity
- load proxy

출력:

- `no_assist`
- `posture_support`
- `strength_assist`
- `emergency_stabilization`

장점:

- 현재 fall-risk classification 구조에서 확장하기 쉽다.

한계:

- 실제 assistive torque 값이 없으면 보조량을 정밀하게 제어할 수 없다.

### Option A-2. Emergency Protection Response Policy

목표:

- `fall_risk` 확률이 높을 때 가방형 그물망을 준비하거나 전개할지 결정한다.

입력:

- fall-risk probability
- backpack net readiness
- harness connection status
- altitude
- manual override

출력:

- `monitor`
- `worker_warning`
- `arm_backpack_safety_net`
- `deploy_backpack_safety_net`
- `manager_alert`

장점:

- 현재 fall-risk 감지 모델과 원래 기획의 그물망 장치를 직접 연결할 수 있다.
- 근력 보조 데이터가 없어도 concept-level safety response는 설계할 수 있다.

한계:

- 실제 전개 시간, 충격 흡수, 고정점 하중이 검증되지 않으면 제품 성능으로 주장할 수 없다.

### Option B. Assistive Torque Regression

목표:

- 관절별 필요한 보조 토크를 연속값으로 예측

입력:

- EMG
- joint angle
- joint angular velocity
- IMU
- load or force plate data

출력:

- hip/knee/ankle 또는 shoulder/elbow assistive torque

장점:

- 웨어러블 로봇이라고 주장하기 가장 적합하다.

한계:

- 현재 fall IMU 데이터로는 불가능하고, torque label이 있는 biomechanics dataset이 필요하다.

### Option C. Fatigue-aware Assist Control

목표:

- 피로 누적 또는 근육 부담이 커질 때 보조 강도를 높인다.

입력:

- EMG amplitude/frequency features
- motion repetition count
- posture duration
- heart rate or fatigue survey if available

출력:

- fatigue risk score
- assist level

장점:

- 고층 외벽 작업처럼 반복 자세와 피로가 중요한 도메인에 적합하다.

한계:

- 피로 label 또는 생체신호가 필요하다.

## Recommended Pivot

현재 repo는 다음처럼 정리하는 것이 가장 정직하다.

1. Current project:
   - AI wearable safety monitoring
   - Real IMU-based fall-risk detection
   - Concept-level backpack safety net response policy

2. Next project:
   - Wearable robot / exosuit extension
   - EMG + kinematics 기반 assistive torque prediction

즉, 지금 결과를 억지로 웨어러블 로봇이라고 부르기보다, `sensing and risk detection layer`로 두고 그 위에 `assistive control layer`를 추가하는 방향이 맞다.
