from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SafetyLimits:
    base_speed_mps: float = 0.15
    base_accel_mps2: float = 0.20
    tcp_speed_mps: float = 0.15
    head_zone_tcp_speed_mps: float = 0.08
    head_zone_radius_m: float = 0.42
    head_force_stop_n: float = 12.0
    command_timeout_s: float = 60.0

    def limited_tcp_speed(self, distance_to_head_m: float) -> float:
        return (
            self.head_zone_tcp_speed_mps
            if distance_to_head_m <= self.head_zone_radius_m
            else self.tcp_speed_mps
        )


def should_stop_for_head_force(force_n: float, limits: SafetyLimits) -> bool:
    return force_n >= limits.head_force_stop_n
