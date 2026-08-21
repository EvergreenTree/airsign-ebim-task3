# Ground-truth declaration

AirSign declares this as a **Policy Submission using simulator ground truth**
for EBiM Phase I.

The policy reads live transforms, bounds, bean state, and task-region state for
planning, verification, recovery, and scoring. These reads replace perception
only. Once an episode begins, policy code does not write task-object transforms,
toggle task-object kinematic state, set task-object velocities, or otherwise
bypass robot contact. Full scene reset is available only as a new episode.

All manipulation commands pass through the robot actuator boundary: bounded
mobile-base wheel commands, RMPFlow-derived FR3 joint targets, and force-aware
Robotiq gripper targets. A source guard in `tests/test_no_object_mutation.py`
fails when prohibited object-mutation calls appear in policy/planning code.

