from airsign_task3.state_machine import PolicyStateMachine
from airsign_task3.types import Lifecycle, Substate


def test_pause_resume_and_recovery() -> None:
    machine = PolicyStateMachine()
    machine.start()
    assert machine.lifecycle is Lifecycle.RUNNING
    machine.pause()
    assert machine.lifecycle is Lifecycle.PAUSED
    machine.resume()
    machine.recoverable_failure("missed grasp")
    assert machine.lifecycle is Lifecycle.RECOVERY
    assert machine.substate is Substate.BACKOFF
    for _ in range(4):
        machine.step_succeeded()
    assert machine.lifecycle is Lifecycle.RUNNING


def test_retry_limit_fails_episode() -> None:
    machine = PolicyStateMachine(max_retries=1)
    machine.start()
    machine.recoverable_failure("first")
    for _ in range(4):
        machine.step_succeeded()
    machine.recoverable_failure("second")
    assert machine.lifecycle is Lifecycle.FAILED

