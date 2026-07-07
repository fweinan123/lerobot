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
Records a dataset via teleoperation.  This is a pure data-collection
tool — no policy inference.  For deploying trained policies, use
``lerobot-rollout`` instead.

Requires: pip install 'lerobot[core_scripts]'  (includes dataset + hardware + viz extras)

Example:

```shell
lerobot-record \\
    --robot.type=so100_follower \\
    --robot.port=/dev/tty.usbmodem58760431541 \\
    --robot.cameras="{laptop: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \\
    --robot.id=black \\
    --teleop.type=so100_leader \\
    --teleop.port=/dev/tty.usbmodem58760431551 \\
    --teleop.id=blue \\
    --dataset.repo_id=<my_username>/<my_dataset_name> \\
    --dataset.num_episodes=2 \\
    --dataset.single_task="Grab the cube" \\
    --dataset.streaming_encoding=true \\
    --dataset.encoder_threads=2 \\
    --display_data=true
```

Example recording with bimanual so100:
```shell
lerobot-record \\
  --robot.type=bi_so_follower \\
  --robot.left_arm_config.port=/dev/tty.usbmodem5A460822851 \\
  --robot.right_arm_config.port=/dev/tty.usbmodem5A460814411 \\
  --robot.id=bimanual_follower \\
  --robot.left_arm_config.cameras='{
    wrist: {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30},
    top: {"type": "opencv", "index_or_path": 3, "width": 640, "height": 480, "fps": 30},
  }' --robot.right_arm_config.cameras='{
    wrist: {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30},
    front: {"type": "opencv", "index_or_path": 4, "width": 640, "height": 480, "fps": 30},
  }' \\
  --teleop.type=bi_so_leader \\
  --teleop.left_arm_config.port=/dev/tty.usbmodem5A460852721 \\
  --teleop.right_arm_config.port=/dev/tty.usbmodem5A460819811 \\
  --teleop.id=bimanual_leader \\
  --display_data=true \\
  --dataset.repo_id=${HF_USER}/bimanual-so-handover-cube \\
  --dataset.num_episodes=25 \\
  --dataset.single_task="Grab and handover the red cube to the other arm" \\
  --dataset.streaming_encoding=true \\
  --dataset.encoder_threads=2
```

Example recording with custom video encoding parameters:
```shell
lerobot-record \\
    --robot.type=so100_follower \\
    --robot.port=/dev/tty.usbmodem58760431541 \\
    --robot.cameras="{laptop: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \\
    --robot.id=black \\
    --teleop.type=so100_leader \\
    --teleop.port=/dev/tty.usbmodem58760431551 \\
    --teleop.id=blue \\
    --dataset.repo_id=<my_username>/<my_dataset_name> \\
    --dataset.num_episodes=2 \\
    --dataset.single_task="Grab the cube" \\
    --dataset.streaming_encoding=true \\
    --dataset.encoder_threads=2 \\
    --dataset.camera_encoder.vcodec=h264 \\
    --dataset.camera_encoder.preset=fast \\
    --dataset.camera_encoder.extra_options={"tune": "film", "profile:v": "high", "bf": 2} \\
    --display_data=true
```
"""

import logging
import threading
import time
from dataclasses import asdict, dataclass
from pprint import pformat

from lerobot.cameras import CameraConfig  # noqa: F401
from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.orbbec import OrbbecCameraConfig  # noqa: F401
from lerobot.cameras.reachy2_camera import Reachy2CameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq import ZMQCameraConfig  # noqa: F401
from lerobot.common.control_utils import (
    init_keyboard_listener,
    is_headless,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.configs import parser
from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.datasets import (
    LeRobotDataset,
    VideoEncodingManager,
    aggregate_pipeline_dataset_features,
    create_initial_features,
    safe_stop_image_writer,
)
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
    homunculus,
    koch_leader,
    make_teleoperator_from_config,
    omx_leader,
    openarm_leader,
    openarm_mini,
    reachy2_teleoperator,
    so_leader,
    unitree_g1,
)
from lerobot.teleoperators.keyboard import KeyboardTeleop
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import (
    init_logging,
    log_say,
)
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


@dataclass
class RecordConfig:
    robot: RobotConfig
    dataset: DatasetRecordConfig
    # Teleoperator to control the robot (required)
    teleop: TeleoperatorConfig | None = None
    # Display all cameras on screen
    display_data: bool = False
    # Display data on a remote Rerun server
    display_ip: str | None = None
    # Port of the remote Rerun server
    display_port: int | None = None
    # Whether to  display compressed images in Rerun
    display_compressed_images: bool = False
    # Use vocal synthesis to read events.
    play_sounds: bool = True
    # Resume recording on an existing dataset.
    resume: bool = False

    def __post_init__(self):
        if self.teleop is None:
            raise ValueError(
                "A teleoperator is required for recording. "
                "Use --teleop.type=... to specify one. "
                "For policy-based deployment, use lerobot-rollout instead."
            )


""" --------------- record_loop() data flow --------------------------
       [ Robot ]
           V
     [ robot.get_observation() ] ---> raw_obs
           V
     [ robot_observation_processor ] ---> processed_obs
           V
     [ Teleoperator ]
     |
     |  [teleop.get_action] -> raw_action
     |          |
     |          V
     | [teleop_action_processor]
     |          |
     '---> processed_teleop_action
                               V
                  [ robot_action_processor ] --> robot_action_to_send
                               V
                    [ robot.send_action() ] -- (Robot Executes)
                               V
                    ( Save to Dataset )
                               V
                  ( Rerun Log / Loop Wait )
"""


def make_hold_action(robot: Robot, obs: RobotObservation) -> RobotAction:
    """Hold the robot at its current action-space positions."""
    hold_action = {}
    for key in robot.action_features:
        if key in obs:
            hold_action[key] = obs[key]
        elif key.endswith(".pos") and key.removesuffix(".pos") in obs:
            hold_action[key] = obs[key.removesuffix(".pos")]

    missing_keys = sorted(set(robot.action_features) - set(hold_action))
    if missing_keys:
        raise KeyError(f"Cannot build hold action; observation is missing action keys: {missing_keys}")

    return hold_action


def clear_hold_targets(events: dict) -> None:
    events.pop("robot_hold_action", None)


def get_robot_motor_observation(robot: Robot) -> RobotObservation:
    """Read robot joint positions without reading cameras when the robot exposes a motor bus."""
    if hasattr(robot, "bus") and hasattr(robot.bus, "sync_read_all_states"):
        states = robot.bus.sync_read_all_states()
        return {
            f"{motor}.pos": state.get("position", 0.0)
            for motor, state in states.items()
            if state.get("position") is not None
        }

    return robot.get_observation()


def send_robot_hold_action(
    robot: Robot,
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    events: dict | None = None,
) -> None:
    if events is not None and "robot_hold_action" in events:
        hold_action = events["robot_hold_action"]
        obs = hold_action
    else:
        obs = get_robot_motor_observation(robot)
        hold_action = make_hold_action(robot, obs)
        if events is not None:
            events["robot_hold_action"] = hold_action
    robot_hold_action = robot_action_processor((hold_action, obs))
    robot.send_action(robot_hold_action)


def hold_teleoperator_if_supported(teleop: Teleoperator | list[Teleoperator] | None) -> None:
    """Hold teleoperator hardware in place when it exposes a hold_position hook."""
    teleops = teleop if isinstance(teleop, list) else [teleop]
    for item in teleops:
        if item is not None and hasattr(item, "hold_position"):
            item.hold_position()


def release_teleoperator_if_supported(teleop: Teleoperator | list[Teleoperator] | None) -> None:
    """Release teleoperator hardware for manual control when it exposes a release hook."""
    teleops = teleop if isinstance(teleop, list) else [teleop]
    for item in teleops:
        if item is not None and hasattr(item, "release_for_manual_control"):
            item.release_for_manual_control()


def safe_cleanup(name: str, cleanup_fn) -> None:
    """Run cleanup without preventing later resources from being released."""
    try:
        cleanup_fn()
    except Exception:
        logging.exception("Error while cleaning up %s.", name)


class HoldPositionGuard:
    """Continuously hold robot and teleoperator positions while blocking work runs."""

    def __init__(
        self,
        robot: Robot,
        teleop: Teleoperator | list[Teleoperator] | None,
        robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
        fps: int,
        events: dict | None = None,
    ):
        self.robot = robot
        self.teleop = teleop
        self.robot_action_processor = robot_action_processor
        self.period_s = 1 / fps
        self.events = events if events is not None else {}
        self.stop_event = threading.Event()
        self.robot_thread: threading.Thread | None = None
        self.teleop_thread: threading.Thread | None = None

    def __enter__(self):
        self.robot_thread = threading.Thread(target=self._run_robot_hold, name="record_robot_hold", daemon=True)
        self.teleop_thread = threading.Thread(
            target=self._run_teleop_hold, name="record_teleop_hold", daemon=True
        )
        self.robot_thread.start()
        self.teleop_thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop_event.set()
        if self.robot_thread is not None:
            self.robot_thread.join(timeout=2.0)
        if self.teleop_thread is not None:
            self.teleop_thread.join(timeout=2.0)

    def _run_robot_hold(self) -> None:
        while not self.stop_event.is_set():
            start_t = time.perf_counter()
            try:
                send_robot_hold_action(self.robot, self.robot_action_processor, self.events)
            except Exception:
                logging.exception("Failed to refresh robot hold position while saving episode.")
                self.stop_event.set()
                break

            self.stop_event.wait(max(self.period_s - (time.perf_counter() - start_t), 0.0))

    def _run_teleop_hold(self) -> None:
        while not self.stop_event.is_set():
            start_t = time.perf_counter()
            try:
                hold_teleoperator_if_supported(self.teleop)
            except Exception:
                logging.exception("Failed to refresh teleoperator hold position while saving episode.")
                self.stop_event.set()
                break

            self.stop_event.wait(max(self.period_s - (time.perf_counter() - start_t), 0.0))


def idle_control_loop(
    robot: Robot,
    events: dict,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],
    robot_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],
    teleop: Teleoperator | list[Teleoperator] | None,
    control_time_s: float | None = None,
    exit_on_start_recording: bool = False,
) -> None:
    """Control or hold the robot without reading cameras or writing dataset frames."""
    teleop_arm = teleop_keyboard = None
    if isinstance(teleop, list):
        teleop_keyboard = next((t for t in teleop if isinstance(t, KeyboardTeleop)), None)
        teleop_arm = next(
            (
                t
                for t in teleop
                if isinstance(
                    t,
                    (
                        so_leader.SO100Leader
                        | so_leader.SO101Leader
                        | koch_leader.KochLeader
                        | omx_leader.OmxLeader
                    ),
                )
            ),
            None,
        )

    control_interval = 1 / fps
    timestamp = 0.0
    start_t = time.perf_counter()
    while control_time_s is None or timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        if exit_on_start_recording and events["start_recording_episode"]:
            break
        if events["stop_recording"]:
            break
        if events["exit_early"]:
            events["exit_early"] = False
            hold_teleoperator_if_supported(teleop)
            send_robot_hold_action(robot, robot_action_processor, events)
            break

        if events.pop("teleop_release_requested", False):
            clear_hold_targets(events)
            release_teleoperator_if_supported(teleop)

        obs = get_robot_motor_observation(robot)
        if not events.get("teleop_enabled", True):
            if "robot_hold_action" not in events:
                events["robot_hold_action"] = make_hold_action(robot, obs)
            action_values = events["robot_hold_action"]
            robot_action_to_send = robot_action_processor((action_values, obs))
        elif isinstance(teleop, Teleoperator):
            act = teleop.get_action()
            act_processed_teleop = teleop_action_processor((act, obs))
            robot_action_to_send = robot_action_processor((act_processed_teleop, obs))
        elif isinstance(teleop, list) and teleop_arm is not None and teleop_keyboard is not None:
            arm_action = teleop_arm.get_action()
            arm_action = {f"arm_{k}": v for k, v in arm_action.items()}
            keyboard_action = teleop_keyboard.get_action()
            base_action = robot._from_keyboard_to_base_action(keyboard_action)
            act = {**arm_action, **base_action} if len(base_action) > 0 else arm_action
            act_processed_teleop = teleop_action_processor((act, obs))
            robot_action_to_send = robot_action_processor((act_processed_teleop, obs))
        else:
            precise_sleep(control_interval)
            timestamp = time.perf_counter() - start_t
            continue

        robot.send_action(robot_action_to_send)
        if not events.get("teleop_enabled", True):
            hold_teleoperator_if_supported(teleop)
        precise_sleep(max(control_interval - (time.perf_counter() - start_loop_t), 0.0))
        timestamp = time.perf_counter() - start_t


@safe_stop_image_writer
def record_loop(
    robot: Robot,
    events: dict,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],  # runs after teleop
    robot_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],  # runs before robot
    robot_observation_processor: RobotProcessorPipeline[
        RobotObservation, RobotObservation
    ],  # runs after robot
    dataset: LeRobotDataset | None = None,
    teleop: Teleoperator | list[Teleoperator] | None = None,
    control_time_s: float | None = None,
    single_task: str | None = None,
    display_data: bool = False,
    display_compressed_images: bool = False,
    exit_on_start_recording: bool = False,
):
    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")

    teleop_arm = teleop_keyboard = None
    if isinstance(teleop, list):
        teleop_keyboard = next((t for t in teleop if isinstance(t, KeyboardTeleop)), None)
        teleop_arm = next(
            (
                t
                for t in teleop
                if isinstance(
                    t,
                    (
                        so_leader.SO100Leader
                        | so_leader.SO101Leader
                        | koch_leader.KochLeader
                        | omx_leader.OmxLeader
                    ),
                )
            ),
            None,
        )

        if not (teleop_arm and teleop_keyboard and len(teleop) == 2 and robot.name == "lekiwi_client"):
            raise ValueError(
                "For multi-teleop, the list must contain exactly one KeyboardTeleop and one arm teleoperator. Currently only supported for LeKiwi robot."
            )

    control_interval = 1 / fps

    no_action_count = 0
    timestamp = 0
    start_episode_t = time.perf_counter()
    while control_time_s is None or timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        if exit_on_start_recording and events["start_recording_episode"]:
            break

        if events["exit_early"]:
            events["exit_early"] = False
            hold_teleoperator_if_supported(teleop)
            send_robot_hold_action(robot, robot_action_processor, events)
            break

        if events.pop("teleop_release_requested", False):
            clear_hold_targets(events)
            release_teleoperator_if_supported(teleop)

        # Get robot observation
        obs = robot.get_observation()

        # Applies a pipeline to the raw robot observation, default is IdentityProcessor
        obs_processed = robot_observation_processor(obs)

        if dataset is not None:
            observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)

        if not events.get("teleop_enabled", True):
            if "robot_hold_action" not in events:
                events["robot_hold_action"] = make_hold_action(robot, obs_processed)
            action_values = events["robot_hold_action"]
            robot_action_to_send = robot_action_processor((action_values, obs))
        # Get action from teleop
        elif isinstance(teleop, Teleoperator):
            act = teleop.get_action()
            if robot.name == "unitree_g1":
                teleop.send_feedback(obs)

            # Applies a pipeline to the raw teleop action, default is IdentityProcessor
            act_processed_teleop = teleop_action_processor((act, obs))
            action_values = act_processed_teleop
            robot_action_to_send = robot_action_processor((act_processed_teleop, obs))

        elif isinstance(teleop, list):
            arm_action = teleop_arm.get_action()
            arm_action = {f"arm_{k}": v for k, v in arm_action.items()}
            keyboard_action = teleop_keyboard.get_action()
            base_action = robot._from_keyboard_to_base_action(keyboard_action)
            act = {**arm_action, **base_action} if len(base_action) > 0 else arm_action
            act_processed_teleop = teleop_action_processor((act, obs))
            action_values = act_processed_teleop
            robot_action_to_send = robot_action_processor((act_processed_teleop, obs))
        else:
            no_action_count += 1
            if no_action_count == 1 or no_action_count % 10 == 0:
                logging.warning(
                    "No teleoperator provided, skipping action generation. "
                    "This is likely to happen when resetting the environment without a teleop device. "
                    "The robot won't be at its rest position at the start of the next episode."
                )
            continue

        # Send action to robot
        # Action can eventually be clipped using `max_relative_target`,
        # so action actually sent is saved in the dataset. action = postprocessor.process(action)
        # TODO(steven, pepijn, adil): we should use a pipeline step to clip the action, so the sent action is the action that we input to the robot.
        _sent_action = robot.send_action(robot_action_to_send)
        if not events.get("teleop_enabled", True):
            hold_teleoperator_if_supported(teleop)

        # Write to dataset
        if dataset is not None:
            action_frame = build_dataset_frame(dataset.features, action_values, prefix=ACTION)
            frame = {**observation_frame, **action_frame, "task": single_task}
            dataset.add_frame(frame)

        if display_data:
            log_rerun_data(
                observation=obs_processed, action=action_values, compress_images=display_compressed_images
            )

        dt_s = time.perf_counter() - start_loop_t

        sleep_time_s: float = control_interval - dt_s
        if sleep_time_s < 0:
            logging.warning(
                f"Record loop is running slower ({1 / dt_s:.1f} Hz) than the target FPS ({fps} Hz). Dataset frames might be dropped and robot control might be unstable. Common causes are: 1) Camera FPS not keeping up 2) Policy inference taking too long 3) CPU starvation"
            )

        precise_sleep(max(sleep_time_s, 0.0))

        timestamp = time.perf_counter() - start_episode_t


def wait_for_episode_start(
    robot: Robot,
    events: dict,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],
    robot_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],
    teleop: Teleoperator | list[Teleoperator] | None,
    play_sounds: bool,
) -> None:
    """Teleoperate without recording until the user presses space."""
    events["start_recording_episode"] = False
    log_say("Press space to start recording episode", play_sounds)
    while not events["start_recording_episode"] and not events["stop_recording"]:
        idle_control_loop(
            robot=robot,
            events=events,
            fps=fps,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            teleop=teleop,
            control_time_s=None,
            exit_on_start_recording=True,
        )
        if events["rerecord_episode"] and not events["start_recording_episode"]:
            events["rerecord_episode"] = False
    events["start_recording_episode"] = False


@parser.wrap()
def record(
    cfg: RecordConfig,
    teleop_action_processor: RobotProcessorPipeline | None = None,
    robot_action_processor: RobotProcessorPipeline | None = None,
    robot_observation_processor: RobotProcessorPipeline | None = None,
) -> LeRobotDataset:
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="recording", ip=cfg.display_ip, port=cfg.display_port)
    display_compressed_images = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )

    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop) if cfg.teleop is not None else None

    # Fall back to identity pipelines when the caller doesn't supply processors.
    if (
        teleop_action_processor is None
        or robot_action_processor is None
        or robot_observation_processor is None
    ):
        _t, _r, _o = make_default_processors()
        teleop_action_processor = teleop_action_processor or _t
        robot_action_processor = robot_action_processor or _r
        robot_observation_processor = robot_observation_processor or _o

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(
                action=robot.action_features
            ),  # TODO(steven, pepijn): in future this should be come from teleop or policy
            use_videos=cfg.dataset.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=cfg.dataset.video,
        ),
    )

    dataset = None
    listener = None

    try:
        if cfg.resume:
            num_cameras = len(robot.cameras) if hasattr(robot, "cameras") else 0
            dataset = LeRobotDataset.resume(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                camera_encoder=cfg.dataset.camera_encoder,
                encoder_threads=cfg.dataset.encoder_threads,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                image_writer_processes=cfg.dataset.num_image_writer_processes if num_cameras > 0 else 0,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * num_cameras
                if num_cameras > 0
                else 0,
            )
            sanity_check_dataset_robot_compatibility(dataset, robot, cfg.dataset.fps, dataset_features)
        else:
            # Reject eval_ prefix — for policy evaluation use lerobot-rollout
            repo_name = cfg.dataset.repo_id.split("/", 1)[-1]
            if repo_name.startswith("eval_"):
                raise ValueError(
                    "Dataset names starting with 'eval_' are reserved for policy evaluation. "
                    "lerobot-record is for data collection only. Use lerobot-rollout for policy deployment."
                )
            cfg.dataset.stamp_repo_id()
            dataset = LeRobotDataset.create(
                cfg.dataset.repo_id,
                cfg.dataset.fps,
                root=cfg.dataset.root,
                robot_type=robot.name,
                features=dataset_features,
                use_videos=cfg.dataset.video,
                image_writer_processes=cfg.dataset.num_image_writer_processes,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                camera_encoder=cfg.dataset.camera_encoder,
                encoder_threads=cfg.dataset.encoder_threads,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
            )

        robot.connect()
        if teleop is not None:
            teleop.connect()

        listener, events = init_keyboard_listener()

        if not cfg.dataset.streaming_encoding:
            logging.info(
                "Streaming encoding is disabled. If you have capable hardware, consider enabling it for way faster episode saving. --dataset.streaming_encoding=true --dataset.encoder_threads=2 # --dataset.camera_encoder.vcodec=auto. More info in the documentation: https://huggingface.co/docs/lerobot/streaming_video_encoding"
            )

        with VideoEncodingManager(dataset):
            recorded_episodes = 0
            while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
                wait_for_episode_start(
                    robot=robot,
                    events=events,
                    fps=cfg.dataset.fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    teleop=teleop,
                    play_sounds=cfg.play_sounds,
                )
                if events["stop_recording"]:
                    break

                log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)
                record_loop(
                    robot=robot,
                    events=events,
                    fps=cfg.dataset.fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=teleop,
                    dataset=dataset,
                    control_time_s=cfg.dataset.episode_time_s,
                    single_task=cfg.dataset.single_task,
                    display_data=cfg.display_data,
                    display_compressed_images=display_compressed_images,
                )

                # Execute a few seconds without recording to give time to manually reset the environment
                # Skip reset for the last episode to be recorded
                if not events["stop_recording"] and (
                    (recorded_episodes < cfg.dataset.num_episodes - 1) or events["rerecord_episode"]
                ):
                    log_say("Reset the environment", cfg.play_sounds)

                    idle_control_loop(
                        robot=robot,
                        events=events,
                        fps=cfg.dataset.fps,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        teleop=teleop,
                        control_time_s=cfg.dataset.reset_time_s,
                    )

                if events["rerecord_episode"]:
                    log_say("Re-record episode", cfg.play_sounds)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    with HoldPositionGuard(robot, teleop, robot_action_processor, cfg.dataset.fps, events):
                        dataset.clear_episode_buffer()
                    continue

                if dataset.has_pending_frames():
                    with HoldPositionGuard(robot, teleop, robot_action_processor, cfg.dataset.fps, events):
                        dataset.save_episode()
                    recorded_episodes += 1
                else:
                    empty_episode_msg = (
                        f"No frames were recorded for episode {dataset.num_episodes}; "
                        "skipping save_episode() and keeping the episode count unchanged. "
                        "This usually means recording was stopped immediately after pressing space, "
                        "or Stop recording was requested during the reset phase."
                    )
                    logging.warning(empty_episode_msg)
                    log_say(empty_episode_msg, cfg.play_sounds)
                    with HoldPositionGuard(robot, teleop, robot_action_processor, cfg.dataset.fps, events):
                        dataset.clear_episode_buffer()
    finally:
        safe_cleanup("stop recording sound", lambda: log_say("Stop recording", cfg.play_sounds, blocking=True))

        if dataset:
            safe_cleanup("dataset", dataset.finalize)

        if robot.is_connected:
            safe_cleanup("robot", robot.disconnect)
        if teleop and teleop.is_connected:
            safe_cleanup("teleoperator", teleop.disconnect)

        if not is_headless() and listener:
            safe_cleanup("keyboard listener", listener.stop)

        if cfg.dataset.push_to_hub:
            if dataset and dataset.num_episodes > 0:
                safe_cleanup(
                    "dataset push_to_hub",
                    lambda: dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private),
                )
            else:
                logging.warning("No episodes saved — skipping push to hub")

        safe_cleanup("exit sound", lambda: log_say("Exiting", cfg.play_sounds))
    return dataset


def main():
    register_third_party_plugins()
    record()


if __name__ == "__main__":
    main()
