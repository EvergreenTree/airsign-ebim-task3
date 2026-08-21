# Provenance of the vendored benchmark slice

Every file under `vendor/benchmark/` is an unmodified copy of official
`EBiM-Benchmark/benchmark` content. Nothing here is authored by AirSign.

- Pinned revision: `e36119cc43e949dc6269bfe5c1e7f613f9f24d0c` (branch `main`)
- Robot asset revision: `c2439d961b652b1eda6122bf530c58cb9559b219` (branch `Robotiq_DEMO`)

`scripts/fetch_benchmark.sh` rebuilds this tree directly from the official
repository. Verified on 2026-08-22: every one of the 42 benchmark files below
comes back byte-for-byte identical from that fetch. The upstream tree is a
superset — a sparse checkout of `scripts/` and `task3_isaacsim/` also
materialises 65 further files from those paths that the policy never loads —
and it keeps its `.git` directory, so the pinned revision can be verified in
place. `BENCHMARK_COMMIT` and this file are the only two entries below that
are not upstream content.

The container builds from the upstream fetch by default; see
`--build-arg BENCHMARK_SOURCE=vendored` in the README for the offline build.

## Files not carried by a plain checkout of the pinned commit

### `assets/robot_room.usd`

Git LFS object of the pinned commit. The tracked blob at `assets/robot_room.usd` is a 133-byte LFS pointer; these are the resolved object bytes, whose LFS OID is also this SHA-256.

- size: 55384382 bytes
- sha256: `bd04da2643bb515ebe311a6a17fd36bf9b32be95ad9e8893a68d44cf2dcc56d3`

### `task1_isaacsim/assets/Robotiq_2f_85_with_d405_mobile_fr3_duo_v0_2.usd`

Branch `Robotiq_DEMO` (`c2439d961b652b1eda6122bf530c58cb9559b219`), path `DEMO/Robotiq_2f_85_with_d405_mobile_fr3_duo_v0_2.usd`. This file is **not** tracked on `main` at the pinned commit; the Task 1 setup obtains it separately, and the Task 3 launchers load it as the robot.

- size: 71246695 bytes
- sha256: `aa1a833de48cc543c73957461dab82fe0979320b7c0b6a0a113d24b500075e5c`

## Complete manifest

| Path | Bytes | SHA-256 |
|---|---:|---|
| `BENCHMARK_COMMIT` | 41 | `7a69b11f8f181d22c09be3eb7efff35631375a419fae3580a73e4abc2e0d99e5` |
| `CONTRIBUTORS.md` | 1299 | `39b80073c8626b01fa736a33762239c1e4ef8e49e7efb1f9372c0698841173ab` |
| `LICENSE` | 11435 | `ab2cd053f20ef36b0aefbd7af45cd31c96f314d586c717f62791d6f636014026` |
| `NOTICE` | 1523 | `2c08fc3d938d6d623897deaf4fc421791e04a002c5ddbc6ff2574ecfb1a9dcf1` |
| `assets/Collected_head/.collect.mapping.json` | 977 | `d0a8e68f3b782cd25b83850a967d8fabb2ab1899d134b4939f5b3a4728a657d6` |
| `assets/Collected_head/head.usd` | 738843 | `853da3938d842d50313811fa20b0fc3fe4c55a3691c26f58cad4452c3c16ee92` |
| `assets/Collected_head/textures/N_normal2.png` | 709708 | `6abbad36969dd0ec1776f36b84e20a4b743dc720cf6bd263cd72803973e90719` |
| `assets/Collected_head/textures/Std_Eye_L_Diffuse.png` | 488794 | `71cbdbbd0eeac4106cfce4e0704bc680e070f8d2cf090b660c31ee54da86c888` |
| `assets/Collected_head/textures/bake_albedo2.png` | 740879 | `6ca2d831fd2c1bcd3e45200aebd327d943cb2c7ddb4159c71065348af54c68ff` |
| `assets/bowl2.usd` | 205439 | `d08ef749e1730a2e2f3dd008686c94fbd99dd94a389796ec08ec516de538123b` |
| `assets/robot_room.usd` | 55384382 | `bd04da2643bb515ebe311a6a17fd36bf9b32be95ad9e8893a68d44cf2dcc56d3` |
| `scripts/common/__init__.py` | 135 | `ae2643de98fb2609abe45653afd7e114a120f3295e7031a7fa8bf2c0a0763273` |
| `scripts/common/path_utils.py` | 841 | `ed9e85571d3753ec202c21964be21b47d29e3d39a51b1406b368d1019ac1f541` |
| `scripts/common/tmr_base_control.py` | 6576 | `6aacec7a4897996e529a2700ec996515fe171e25382e574831717c2d143cb378` |
| `scripts/scenes/scene_robot_room_keyboard.py` | 67074 | `7935d47a95bb36622ddfabc259ce33ed8bc4063378b3009173e8409dcddb583e` |
| `task1_isaacsim/assets/Robotiq_2f_85_with_d405_mobile_fr3_duo_v0_2.usd` | 71246695 | `aa1a833de48cc543c73957461dab82fe0979320b7c0b6a0a113d24b500075e5c` |
| `task3_isaacsim/.env.example` | 616 | `27d8d4dba4d9faeb3ab19eea9486cf935265d42bad5dd5ace1d2ba5fca1b69f8` |
| `task3_isaacsim/README.md` | 11164 | `710d5cfd82c1bf22ccb5c94fbe786cfa3888193ad454ea71cfb13bc711eb7f12` |
| `task3_isaacsim/assets/lula/mobile_fr3_duo/left_arm_description.yaml` | 2067 | `e313f2c3e648d70ef9a45d6157ee492c1d566a8e301871ab21e1f3ab468b6233` |
| `task3_isaacsim/assets/lula/mobile_fr3_duo/left_arm_rmpflow_config.yaml` | 2250 | `3049c0c93c7c3c47f26831980549c70329fcc1d82d62357005bb55355c5eec0d` |
| `task3_isaacsim/assets/lula/mobile_fr3_duo/mobile_fr3_duo_v0_2_franka_hand.urdf` | 48159 | `f279d60e3ae10675c2bab47617f954cce06b1da45e2a1164ef62ffa29b76e0a7` |
| `task3_isaacsim/assets/lula/mobile_fr3_duo/right_arm_description.yaml` | 2067 | `8e06249b6c846ec6512f5084d0b3031346e4372054d30ce47cb79f974ba360f3` |
| `task3_isaacsim/assets/lula/mobile_fr3_duo/right_arm_rmpflow_config.yaml` | 2251 | `777029809bd281c47601e877ad71c517108e8e0cb00cde81721816bd31717568` |
| `task3_isaacsim/deprecated/README.md` | 909 | `28b1ec00f804604c7c837b8c6c71c2af005f89d686094f0dc7531c896dd886bf` |
| `task3_isaacsim/deprecated/scripts/common/dual_arm_lula.py` | 20656 | `2c6e8e3fcf0c3e297b641adfcde6605f2c5a953b5e5bcb0404699e1291922fd7` |
| `task3_isaacsim/deprecated/scripts/common/keyboard_arm_teleop.py` | 5721 | `a3b01bdf5e74ace0cb1a74b9f3f7bfb04f15ae6f44f5b35656dc314b295e47b4` |
| `task3_isaacsim/deprecated/scripts/common/teleop_commands.py` | 2835 | `5217ec18f1f74d82da80367a6dc8fb7237b8875a881fbddd952a1103ef7a49bc` |
| `task3_isaacsim/deprecated/scripts/common/teleop_targets.py` | 20425 | `54b27fb5e10bca4aa2d271708cb3d5b45fb3cda1d173f7ac98e1db95b0cfb842` |
| `task3_isaacsim/deprecated/scripts/scene_robot_room_rmpflow.py` | 36629 | `1fe5d09929c910763071eca848ac0b3f22f802363b6798f74dcf300b46c81f4f` |
| `task3_isaacsim/deprecated/tests/test_dual_arm_lula.py` | 23126 | `e0a58a84e3afcf75fa9b2b6a123a2b8d6a2b49d1bdaf3658a6d429e3898056cd` |
| `task3_isaacsim/deprecated/tests/test_keyboard_arm_teleop.py` | 4197 | `ef744bf27ba8632bf608b9c223867197714759b55eddedaf3e8e88a0b74616a3` |
| `task3_isaacsim/deprecated/tests/test_scene_robot_room_rmpflow.py` | 9148 | `52daab576e68ac4c2b2402aac1a3402d716796bd8c22fb6d9253500ebb02541b` |
| `task3_isaacsim/deprecated/tests/test_teleop_commands.py` | 3490 | `1a93e7e2de0bf2eeac66a9b0cffbd11f37a82560cca8d7657383886dbb8732eb` |
| `task3_isaacsim/deprecated/tests/test_teleop_targets.py` | 17693 | `29e8a920c446aa8984c52163ccaacf2d447abe511f389c8df4b9ec1d4df989a4` |
| `task3_isaacsim/docker-compose.yml` | 3008 | `b635323f2c98d3e4ac4f37a142235cf9c119774a2074dde4bdbebdd6aef9bb91` |
| `task3_isaacsim/scripts/eval_overlay.py` | 12051 | `d2b0545c32a1a6e903388a9fd1b9e1050ad0a1158ce05829a9b3d9de32bc362e` |
| `task3_isaacsim/scripts/gripper_profiles.py` | 2322 | `258162177bf1b34b9e0a1d8f9a800c2bd8f9e0610891b7b0e8558556886a895c` |
| `task3_isaacsim/scripts/run_helper_containers.sh` | 4963 | `9f6f44a487c59f0ae3320b9a6f9b2c7000f6eebcc2be8012ed7163eeb69543f2` |
| `task3_isaacsim/scripts/run_isaacsim_teleop.sh` | 6087 | `568b307328d5f14912baf4d17accedb1f79f7c0006165232edf20d345700aa6c` |
| `task3_isaacsim/scripts/scene_room.py` | 5905 | `7aa7d1a9549ff964cf7a954b80430059fff330661d10b87c9246143dce282728` |
| `task3_isaacsim/tests/test_eval_overlay.py` | 1116 | `443f2a809ce1aaf8bfbcf3caeec89220e2e466e40f1c65dde0aff9d85b3a88a3` |
| `task3_isaacsim/tests/test_gripper_profiles.py` | 2490 | `fb13489c44426cc6c4c2aa4e1b0976b18888bf9ead36d16b0ae8228c0be1c28c` |
| `task3_isaacsim/tests/test_launcher.py` | 4656 | `f0a9b899ba3402392baaa312fc76df5c3ddeff7ffd97b77cc442359f49f865d2` |
