from dataclasses import dataclass

from kinematics.geometry.config.wheel import WheelConfig


@dataclass
class StaticSetupConfig:
    static_camber: float
    static_toe: float
    static_caster: float


@dataclass
class SuspensionConfig:
    steered: bool
    wheel: WheelConfig
    static_setup: StaticSetupConfig
