# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Replays the actions of an episode from a dataset on a robot.

Requires: pip install 'lerobot[core_scripts]'  (includes dataset + hardware + viz extras)

Examples:

```shell
lerobot-replay \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.id=black \
    --dataset.repo_id=<USER>/record-test \
    --dataset.episode=0
```

Example replay with bimanual so100:
```shell
lerobot-replay \
  --robot.type=bi_so_follower \
  --robot.left_arm_port=/dev/tty.usbmodem5A460851411 \
  --robot.right_arm_port=/dev/tty.usbmodem5A460812391 \
  --robot.id=bimanual_follower \
  --dataset.repo_id=${HF_USER}/bimanual-so100-handover-cube \
  --dataset.episode=0
```

"""

import logging
import math
from numbers import Real
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat

from lerobot.configs import parser
from lerobot.datasets import LeRobotDataset
from lerobot.processor import (
    make_default_robot_action_processor,
)
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_openarm_follower,
    bi_so_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    openarm_follower,
    reachy2,
    so_follower,
    unitree_g1,
)
from lerobot.utils.constants import ACTION
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import (
    init_logging,
    log_say,
)


@dataclass
class DatasetReplayConfig:
    # Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).
    repo_id: str
    # Episode to replay.
    episode: int
    # Root directory where the dataset will be stored (e.g. 'dataset/path'). If None, defaults to $HF_LEROBOT_HOME/repo_id.
    root: str | Path | None = None
    # Limit the frames per second. By default, uses the policy fps.
    fps: int = 30


@dataclass
class ReplayConfig:
    robot: RobotConfig
    dataset: DatasetReplayConfig
    # Use vocal synthesis to read events.
    play_sounds: bool = True
    # Move the robot gradually from its current pose to the first episode action before replaying.
    # Set to 0 to start replay immediately.
    move_to_start_time_s: float = 2.0


@parser.wrap()
def replay(cfg: ReplayConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    robot_action_processor = make_default_robot_action_processor()

    robot = make_robot_from_config(cfg.robot)
    dataset = LeRobotDataset(cfg.dataset.repo_id, root=cfg.dataset.root, episodes=[cfg.dataset.episode])

    actions = dataset.select_columns(ACTION)

    robot.connect()

    try:
        if cfg.move_to_start_time_s < 0:
            raise ValueError("move_to_start_time_s must be greater than or equal to 0.")

        if cfg.move_to_start_time_s > 0 and dataset.num_frames > 0:
            log_say("Moving to episode start", cfg.play_sounds, blocking=True)
            first_action = _dataset_action_to_robot_action(dataset, actions, 0)
            _move_to_start_pose(
                robot=robot,
                robot_action_processor=robot_action_processor,
                target_action=first_action,
                move_time_s=cfg.move_to_start_time_s,
                fps=dataset.fps,
            )

        log_say("Replaying episode", cfg.play_sounds, blocking=True)
        for idx in range(dataset.num_frames):
            start_episode_t = time.perf_counter()

            action = _dataset_action_to_robot_action(dataset, actions, idx)

            robot_obs = robot.get_observation()

            processed_action = robot_action_processor((action, robot_obs))

            _ = robot.send_action(processed_action)

            dt_s = time.perf_counter() - start_episode_t
            precise_sleep(max(1 / dataset.fps - dt_s, 0.0))
    finally:
        robot.disconnect()


def _dataset_action_to_robot_action(dataset: LeRobotDataset, actions, idx: int) -> dict:
    action_array = actions[idx][ACTION]
    return {name: action_array[i] for i, name in enumerate(dataset.features[ACTION]["names"])}


def _move_to_start_pose(
    robot: Robot,
    robot_action_processor,
    target_action: dict,
    move_time_s: float,
    fps: int,
) -> None:
    start_observation = robot.get_observation()
    start_action = {
        name: start_observation[name]
        for name in target_action
        if name in start_observation and _is_numeric(start_observation[name])
    }

    if not start_action:
        logging.warning(
            "Could not infer the robot start pose from observation keys. Starting replay without pre-positioning."
        )
        return

    num_steps = max(1, math.ceil(move_time_s * fps))
    control_interval = move_time_s / num_steps

    for step in range(1, num_steps + 1):
        start_loop_t = time.perf_counter()
        alpha = step / num_steps
        action = target_action.copy()
        for name, start_value in start_action.items():
            action[name] = float(start_value) + (float(target_action[name]) - float(start_value)) * alpha

        robot_obs = robot.get_observation()
        processed_action = robot_action_processor((action, robot_obs))
        robot.send_action(processed_action)

        dt_s = time.perf_counter() - start_loop_t
        precise_sleep(max(control_interval - dt_s, 0.0))


def _is_numeric(value) -> bool:
    return isinstance(value, Real)


def main():
    register_third_party_plugins()
    replay()


if __name__ == "__main__":
    main()
