from __future__ import annotations

from dataclasses import dataclass

from .types import Lifecycle, Stage, Substate


STAGE_ORDER = (
    Stage.TABLE_SETUP,
    Stage.FEEDING,
    Stage.BEAN_RECOVERY,
    Stage.CLEANUP,
)

SUBSTATE_ORDER = (
    Substate.NAVIGATE,
    Substate.APPROACH,
    Substate.GRASP,
    Substate.CARRY,
    Substate.RELEASE,
    Substate.VERIFY,
)


@dataclass
class PolicyStateMachine:
    lifecycle: Lifecycle = Lifecycle.READY
    stage_index: int = 0
    substate_index: int = 0
    retries: int = 0
    max_retries: int = 3
    paused_from: Lifecycle | None = None
    failure_reason: str | None = None

    @property
    def stage(self) -> Stage:
        return STAGE_ORDER[min(self.stage_index, len(STAGE_ORDER) - 1)]

    @property
    def substate(self) -> Substate:
        if self.lifecycle is Lifecycle.RECOVERY:
            recovery = (
                Substate.BACKOFF,
                Substate.REOPEN,
                Substate.REAPPROACH,
                Substate.REGRASP,
            )
            return recovery[min(self.substate_index, len(recovery) - 1)]
        if self.lifecycle is Lifecycle.READY:
            return Substate.IDLE
        return SUBSTATE_ORDER[min(self.substate_index, len(SUBSTATE_ORDER) - 1)]

    def start(self) -> None:
        if self.lifecycle not in {Lifecycle.READY, Lifecycle.PAUSED}:
            raise RuntimeError(f"cannot start from {self.lifecycle}")
        self.lifecycle = Lifecycle.RUNNING
        self.substate_index = 0

    def pause(self) -> None:
        if self.lifecycle not in {Lifecycle.RUNNING, Lifecycle.RECOVERY}:
            return
        self.paused_from = self.lifecycle
        self.lifecycle = Lifecycle.PAUSED

    def resume(self) -> None:
        if self.lifecycle is not Lifecycle.PAUSED:
            return
        self.lifecycle = self.paused_from or Lifecycle.RUNNING
        self.paused_from = None

    def step_succeeded(self) -> None:
        if self.lifecycle not in {Lifecycle.RUNNING, Lifecycle.RECOVERY}:
            raise RuntimeError(f"cannot advance from {self.lifecycle}")
        if self.lifecycle is Lifecycle.RECOVERY:
            self.substate_index += 1
            if self.substate_index >= 4:
                self.lifecycle = Lifecycle.RUNNING
                self.substate_index = 1
            return
        self.substate_index += 1
        if self.substate_index < len(SUBSTATE_ORDER):
            return
        self.retries = 0
        self.substate_index = 0
        self.stage_index += 1
        if self.stage_index >= len(STAGE_ORDER):
            self.stage_index = len(STAGE_ORDER) - 1
            self.lifecycle = Lifecycle.COMPLETE

    def recoverable_failure(self, reason: str) -> None:
        if self.retries >= self.max_retries:
            self.fail(reason)
            return
        self.retries += 1
        self.lifecycle = Lifecycle.RECOVERY
        self.substate_index = 0
        self.failure_reason = reason

    def fail(self, reason: str) -> None:
        self.lifecycle = Lifecycle.FAILED
        self.failure_reason = reason

