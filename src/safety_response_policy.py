"""fall-risk 모델 출력을 안전 대응 행동으로 변환한다.

이 모듈은 concept-level response policy다. 현재 구현된 fall-risk 감지 계층을
원래 기획의 가방형 그물망 보호 장치와 연결하지만, 실제 하드웨어 검증을 마친
제어 소프트웨어는 아니다.

정책 계층은 의도적으로 machine-learning model과 분리했다.

- 모델은 IMU 움직임 feature로 fall risk를 추정한다.
- 정책은 장치 준비 상태를 함께 보고 어떤 행동이 허용되는지 결정한다.

안전 시스템에서는 이 분리가 중요하다. 모델 점수가 높다고 해서 바로 장치를
전개하면 안 되고, 그물망 준비 상태, 하네스 연결, 작업 높이, 수동 override 조건도
함께 확인해야 한다.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskPrediction:
    """fall-risk 감지 모델의 출력값.

    `fall_probability`는 0과 1 사이 값이라고 가정한다. 그래도 실제 시스템에서는
    upstream 호출자가 항상 완벽하다고 볼 수 없으므로 정책 함수에서 방어적으로
    값을 clamp한다. `risk_label`은 로그나 dashboard에 남길 사람이 읽을 수 있는 라벨이다.
    """

    fall_probability: float
    risk_label: str


@dataclass(frozen=True)
class SafetyDeviceState:
    """안전 장치 준비 상태와 작업 맥락.

    이 값들은 모델이 학습하는 값이 아니다. 가방형 그물망을 준비하거나 전개하기 전에
    safety supervisor가 확인해야 하는 하드웨어/운영 상태를 나타낸다.
    """

    # 제안한 가방형 그물망이 물리적으로 전개 가능한 상태일 때만 True다.
    backpack_net_ready: bool
    # 하네스 연결이 확인된 경우에만 True다. 이 확인 없이 전개하면 안전하다는 착각을 줄 수 있다.
    harness_connected: bool
    # 대략적인 작업 높이다. 최소 높이 조건은 그물망이 펼쳐질 수 없거나 도움이 되지 않는
    # 상황에서 전개 로직이 작동하지 않도록 막는 guard 역할을 한다.
    altitude_m: float
    # 작업자나 관리자가 자동 행동을 막을 수 있는 수동 override 상태다.
    manual_override: bool = False


@dataclass(frozen=True)
class SafetyResponse:
    """안전 대응 정책이 선택한 행동.

    `requires_hardware_validation`은 해당 행동이 실제 물리 전개 검증을 필요로 하는지 표시한다.
    monitoring이나 warning은 소프트웨어 수준에서 먼저 검토할 수 있지만, 그물망 준비/전개는
    반드시 하드웨어 테스트가 필요하다.
    """

    action: str
    reason: str
    requires_hardware_validation: bool = True


def decide_safety_response(
    prediction: RiskPrediction,
    state: SafetyDeviceState,
    warning_threshold: float = 0.45,
    arm_threshold: float = 0.65,
    deploy_threshold: float = 0.82,
    minimum_deploy_altitude_m: float = 2.0,
) -> SafetyResponse:
    """모델 위험도와 장치 준비 상태를 바탕으로 안전 행동을 선택한다.

    여기의 threshold 값은 concept-level 값이지 인증된 안전 기준이 아니다.
    나중에 실제 전개 데이터, false alarm 비용, 안전 요구사항을 바탕으로 조정할 수 있도록
    함수 파라미터로 열어두었다.
    """
    # 확률값을 0~1 범위로 제한한다. upstream 모델 또는 통합 코드 오류로 -0.2, 1.4 같은
    # 불가능한 값이 들어와도 정책 로직이 안정적으로 동작하도록 하기 위함이다.
    probability = max(0.0, min(1.0, prediction.fall_probability))

    # 수동 override는 최우선 조건이다. 실제 안전 시스템에서는 점검, 유지보수, 비정상 상황에서
    # 작업자나 관리자가 자동 전개를 막을 수 있어야 한다.
    if state.manual_override:
        return SafetyResponse(
            action="hold_manual_override",
            reason="Manual override is active, so automatic deployment is blocked.",
        )

    # 가장 높은 위험 구간이다. 단, fall risk가 높더라도 모든 하드웨어 준비 조건이 맞아야
    # 실제 그물망 전개 행동을 선택할 수 있다.
    if probability >= deploy_threshold:
        if not state.harness_connected:
            return SafetyResponse(
                action="manager_alert",
                reason="Fall risk is high, but harness connection is not confirmed.",
            )
        if not state.backpack_net_ready:
            return SafetyResponse(
                action="manager_alert",
                reason="Fall risk is high, but backpack safety net is not ready.",
            )
        if state.altitude_m < minimum_deploy_altitude_m:
            return SafetyResponse(
                action="worker_warning",
                reason="Fall risk is high, but altitude is below the deployment threshold.",
            )
        return SafetyResponse(
            action="deploy_backpack_safety_net",
            reason="Fall risk is high and the backpack safety net is ready.",
        )

    # 중간 이상 위험 구간이다. 실제 전개는 하지 않고 장치를 준비 상태로 올려,
    # 위험이 더 커졌을 때 반응 시간을 줄인다.
    if probability >= arm_threshold:
        if state.backpack_net_ready and state.harness_connected:
            return SafetyResponse(
                action="arm_backpack_safety_net",
                reason="Fall risk is elevated, so prepare the net without deploying it yet.",
            )
        return SafetyResponse(
            action="manager_alert",
            reason="Fall risk is elevated, but safety device readiness is incomplete.",
        )

    # 초기 경고 구간이다. 아직 하드웨어를 준비하거나 전개하지 않고, 작업자/관리자에게
    # 위험 상승 신호를 전달한다.
    if probability >= warning_threshold:
        return SafetyResponse(
            action="worker_warning",
            reason="Fall risk is rising, but it is below the net arming threshold.",
            requires_hardware_validation=False,
        )

    # 낮은 위험 구간이다. 별도 행동 없이 계속 모니터링한다.
    return SafetyResponse(
        action="monitor",
        reason="Fall-risk probability is below the warning threshold.",
        requires_hardware_validation=False,
    )


if __name__ == "__main__":
    # 모델이 높은 fall risk를 예측했을 때 정책 함수가 어떻게 호출되는지 보여주는 최소 예시다.
    # 실제 전개 테스트가 아니라, concept-level 정책이 어떤 response를 선택하는지만 확인한다.
    example_prediction = RiskPrediction(fall_probability=0.87, risk_label="fall_risk")
    example_state = SafetyDeviceState(
        backpack_net_ready=True,
        harness_connected=True,
        altitude_m=12.0,
    )
    print(decide_safety_response(example_prediction, example_state))
