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
Simple script to control a robot from teleoperation.

Requires: pip install 'lerobot[hardware]'

Example:

```shell
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --teleop.id=blue \
    --display_data=true
```

Example teleoperation with bimanual so100:

```shell
lerobot-teleoperate \
  --robot.type=bi_so_follower \
  --robot.left_arm_config.port=/dev/tty.usbmodem5A460822851 \
  --robot.right_arm_config.port=/dev/tty.usbmodem5A460814411 \
  --robot.id=bimanual_follower \
  --robot.left_arm_config.cameras='{
    wrist: {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30},
  }' --robot.right_arm_config.cameras='{
    wrist: {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30},
  }' \
  --teleop.type=bi_so_leader \
  --teleop.left_arm_config.port=/dev/tty.usbmodem5A460852721 \
  --teleop.right_arm_config.port=/dev/tty.usbmodem5A460819811 \
  --teleop.id=bimanual_leader \
  --display_data=true
```

"""

import ast
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.orbbec import OrbbecCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq import ZMQCameraConfig  # noqa: F401
from lerobot.common.control_utils import is_headless
from lerobot.configs import parser
from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
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
    unitree_g1 as unitree_g1_robot,
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_openarm_leader,
    bi_so_leader,
    gamepad,
    homunculus,
    keyboard,
    koch_leader,
    make_teleoperator_from_config,
    omx_leader,
    openarm_leader,
    openarm_mini,
    reachy2_teleoperator,
    so_leader,
    unitree_g1,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, move_cursor_up
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data, shutdown_rerun

ROLLOUT_CONFIG_PATH = Path(__file__).parents[1] / "rollout" / "configs.py"
MOVE_PATH_CONSTANT = "DEFAULT_MOVE_PATH_BEFORE_POLICY"
MOVE_PATH_JOINT_KEYS = [f"joint_{idx}.pos" for idx in range(1, 8)]


@dataclass
class TeleoperateConfig:
    # TODO: pepijn, steven: if more robots require multiple teleoperators (like lekiwi) its good to make this possibele in teleop.py and record.py with List[Teleoperator]
    teleop: TeleoperatorConfig
    robot: RobotConfig
    # Limit the maximum frames per second.
    fps: int = 60
    teleop_time_s: float | None = None
    # Display all cameras on screen
    display_data: bool = False
    # Display data on a remote Rerun server
    display_ip: str | None = None
    # Port of the remote Rerun server
    display_port: int | None = None
    # Whether to  display compressed images in Rerun
    display_compressed_images: bool = False
    # When True, pressing space records the current robot joint_1..joint_7
    # positions into rollout/configs.py for pre-policy startup motion.
    move_path_before_policy_record: bool = False


def teleop_loop(
    teleop: Teleoperator,
    robot: Robot,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
    display_data: bool = False,
    duration: float | None = None,
    display_compressed_images: bool = False,
    move_path_before_policy_record: bool = False,
):
    """
    This function continuously reads actions from a teleoperation device, processes them through optional
    pipelines, sends them to a robot, and optionally displays the robot's state. The loop runs at a
    specified frequency until a set duration is reached or it is manually interrupted.

    Args:
        teleop: The teleoperator device instance providing control actions.
        robot: The robot instance being controlled.
        fps: The target frequency for the control loop in frames per second.
        display_data: If True, fetches robot observations and displays them in the console and Rerun.
        display_compressed_images: If True, compresses images before sending them to Rerun for display.
        duration: The maximum duration of the teleoperation loop in seconds. If None, the loop runs indefinitely.
        teleop_action_processor: An optional pipeline to process raw actions from the teleoperator.
        robot_action_processor: An optional pipeline to process actions before they are sent to the robot.
        robot_observation_processor: An optional pipeline to process raw observations from the robot.
    """

    display_len = max(len(key) for key in robot.action_features)
    listener = None
    events = {"record_move_path_waypoint": False, "stop_recording": False}
    recorded_waypoints = _load_move_path_before_policy() if move_path_before_policy_record else []
    if move_path_before_policy_record:
        listener, events = _init_move_path_keyboard_listener()
        logging.info(
            "Pre-policy path recording enabled. Press SPACE to append current joint_1..joint_7 pose; ESC to exit."
        )

    start = time.perf_counter()
    try:
        while True:
            loop_start = time.perf_counter()

            # Get robot observation
            # Not really needed for now other than for visualization
            # teleop_action_processor can take None as an observation
            # given that it is the identity processor as default
            obs = robot.get_observation()

            if move_path_before_policy_record and events.pop("record_move_path_waypoint", False):
                waypoint = _extract_move_path_waypoint(obs)
                recorded_waypoints.append(waypoint)
                _write_move_path_before_policy(recorded_waypoints)
                logging.info("Recorded pre-policy waypoint %d: %s", len(recorded_waypoints), waypoint)

            if move_path_before_policy_record and events.get("stop_recording", False):
                return

            if robot.name == "unitree_g1":
                teleop.send_feedback(obs)

            # Get teleop action
            raw_action = teleop.get_action()

            # Process teleop action through pipeline
            teleop_action = teleop_action_processor((raw_action, obs))

            # Process action for robot through pipeline
            robot_action_to_send = robot_action_processor((teleop_action, obs))

            # Send processed action to robot (robot_action_processor.to_output should return RobotAction)
            _ = robot.send_action(robot_action_to_send)

            if display_data:
                # Process robot observation through pipeline
                obs_transition = robot_observation_processor(obs)

                log_rerun_data(
                    observation=obs_transition,
                    action=teleop_action,
                    compress_images=display_compressed_images,
                )

                print("\n" + "-" * (display_len + 10))
                print(f"{'NAME':<{display_len}} | {'NORM':>7}")
                # Display the final robot action that was sent
                for motor, value in robot_action_to_send.items():
                    print(f"{motor:<{display_len}} | {value:>7.2f}")
                move_cursor_up(len(robot_action_to_send) + 3)

            dt_s = time.perf_counter() - loop_start
            precise_sleep(max(1 / fps - dt_s, 0.0))
            loop_s = time.perf_counter() - loop_start
            print(f"Teleop loop time: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")
            move_cursor_up(1)

            if duration is not None and time.perf_counter() - start >= duration:
                return
    finally:
        if listener is not None:
            listener.stop()


def _init_move_path_keyboard_listener():
    events = {"record_move_path_waypoint": False, "stop_recording": False}
    if is_headless():
        logging.warning("Headless environment detected; pre-policy path keyboard recording disabled")
        return None, events

    try:
        from pynput import keyboard
    except Exception:
        logging.warning("pynput unavailable; pre-policy path keyboard recording disabled")
        return None, events

    def on_press(key):
        if key == keyboard.Key.space:
            print("Space key pressed. Recording pre-policy waypoint...")
            events["record_move_path_waypoint"] = True
        elif key == keyboard.Key.esc:
            print("Escape key pressed. Stopping pre-policy path recording...")
            events["stop_recording"] = True

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener, events


def _extract_move_path_waypoint(obs: RobotObservation) -> list[float]:
    missing_keys = [key for key in MOVE_PATH_JOINT_KEYS if key not in obs]
    if missing_keys:
        raise KeyError(f"Cannot record pre-policy waypoint; missing robot observation keys: {missing_keys}")
    return [round(float(obs[key]), 4) for key in MOVE_PATH_JOINT_KEYS]


def _load_move_path_before_policy() -> list[list[float]]:
    tree = ast.parse(ROLLOUT_CONFIG_PATH.read_text())
    for node in tree.body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == MOVE_PATH_CONSTANT
        ):
            value = ast.literal_eval(node.value)
            return [list(point) for point in value]
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == MOVE_PATH_CONSTANT:
                    value = ast.literal_eval(node.value)
                    return [list(point) for point in value]
    return []


def _write_move_path_before_policy(path: list[list[float]]) -> None:
    source = ROLLOUT_CONFIG_PATH.read_text()
    tree = ast.parse(source)
    for node in tree.body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == MOVE_PATH_CONSTANT
        ):
            old = ast.get_source_segment(source, node)
            if old is None:
                raise RuntimeError(f"Could not update {MOVE_PATH_CONSTANT} in {ROLLOUT_CONFIG_PATH}")
            new = f"{MOVE_PATH_CONSTANT}: list[list[float]] = {pformat(path, width=120)}"
            ROLLOUT_CONFIG_PATH.write_text(source.replace(old, new, 1))
            return
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == MOVE_PATH_CONSTANT:
                    old = ast.get_source_segment(source, node)
                    if old is None:
                        raise RuntimeError(f"Could not update {MOVE_PATH_CONSTANT} in {ROLLOUT_CONFIG_PATH}")
                    new = f"{MOVE_PATH_CONSTANT}: list[list[float]] = {pformat(path, width=120)}"
                    ROLLOUT_CONFIG_PATH.write_text(source.replace(old, new, 1))
                    return
    raise RuntimeError(f"Could not find {MOVE_PATH_CONSTANT} in {ROLLOUT_CONFIG_PATH}")


@parser.wrap()
def teleoperate(cfg: TeleoperateConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="teleoperation", ip=cfg.display_ip, port=cfg.display_port)
    display_compressed_images = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )

    teleop = make_teleoperator_from_config(cfg.teleop)
    robot = make_robot_from_config(cfg.robot)
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    try:
        teleop.connect()
        robot.connect()

        teleop_loop(
            teleop=teleop,
            robot=robot,
            fps=cfg.fps,
            display_data=cfg.display_data,
            duration=cfg.teleop_time_s,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            display_compressed_images=display_compressed_images,
            move_path_before_policy_record=cfg.move_path_before_policy_record,
        )
    except KeyboardInterrupt:
        pass
    finally:
        if cfg.display_data:
            shutdown_rerun()
        if teleop.is_connected:
            teleop.disconnect()
        if robot.is_connected:
            robot.disconnect()


def main():
    register_third_party_plugins()
    teleoperate()


if __name__ == "__main__":
    main()
