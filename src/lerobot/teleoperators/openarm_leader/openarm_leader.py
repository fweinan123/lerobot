#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import logging
import math
from pathlib import Path
import time
from typing import Any

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.damiao import DamiaoMotorsBus
from lerobot.types import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from .config_openarm_leader import OpenArmLeaderConfig

logger = logging.getLogger(__name__)


class OpenArmLeader(Teleoperator):
    """
    OpenArm Leader/Teleoperator Arm with Damiao motors.

    This teleoperator uses CAN bus communication to read positions from
    Damiao motors that are manually moved (torque disabled).
    """

    config_class = OpenArmLeaderConfig
    name = "openarm_leader"

    def __init__(self, config: OpenArmLeaderConfig):
        super().__init__(config)
        self.config = config

        # Arm motors
        motors: dict[str, Motor] = {}
        for motor_name, (send_id, recv_id, motor_type_str) in config.motor_config.items():
            motor = Motor(
                send_id, motor_type_str, MotorNormMode.DEGREES
            )  # Always use degrees for Damiao motors
            motor.recv_id = recv_id
            motor.motor_type_str = motor_type_str
            motors[motor_name] = motor

        self.bus = DamiaoMotorsBus(
            port=self.config.port,
            motors=motors,
            calibration=self.calibration,
            can_interface=self.config.can_interface,
            use_can_fd=self.config.use_can_fd,
            bitrate=self.config.can_bitrate,
            data_bitrate=self.config.can_data_bitrate if self.config.use_can_fd else None,
        )
        self._pin = None
        self._pin_model = None
        self._pin_data = None
        self._pin_q = None
        self._pin_joint_q_indices: dict[str, int] = {}
        self._pin_joint_v_indices: dict[str, int] = {}
        self._hold_torque_enabled = False
        self._manual_gravity_compensation_paused = False
        self._hold_positions: dict[str, float] | None = None

    @property
    def action_features(self) -> dict[str, type]:
        """Features produced by this teleoperator."""
        features: dict[str, type] = {}
        for motor in self.bus.motors:
            features[f"{motor}.pos"] = float
            if self.config.use_velocity_and_torque:
                features[f"{motor}.vel"] = float
                features[f"{motor}.torque"] = float
        return features

    @property
    def feedback_features(self) -> dict[str, type]:
        """Feedback features (not implemented for OpenArms)."""
        return {}

    @property
    def is_connected(self) -> bool:
        """Check if teleoperator is connected."""
        return self.bus.is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        """
        Connect to the teleoperator.

        For manual control, we disable torque after connecting so the
        arm can be moved by hand.
        """

        # Connect to CAN bus
        logger.info(f"Connecting arm on {self.config.port}...")
        self.bus.connect()

        # Run calibration if needed
        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        self.configure()
        if not self.config.manual_control and self.config.gravity_compensation:
            self._load_pinocchio_model()

        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        """Check if teleoperator is calibrated."""
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        """
        Run calibration procedure for OpenArms leader.

        The calibration procedure:
        1. Disable torque (if not already disabled)
        2. Ask user to position arm in zero position (hanging with gripper closed)
        3. Set this as zero position
        4. Record range of motion for each joint
        5. Save calibration
        """
        if self.calibration:
            # Calibration file exists, ask user whether to use it or run new calibration
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
                self.bus.write_calibration(self.calibration)
                return

        logger.info(f"\nRunning calibration for {self}")
        self.bus.disable_torque()

        # Step 1: Set zero position
        input(
            "\nCalibration: Set Zero Position)\n"
            "Position the arm in the following configuration:\n"
            "  - Arm hanging straight down\n"
            "  - Gripper closed\n"
            "Press ENTER when ready..."
        )

        # Set current position as zero for all motors
        self.bus.set_zero_position()
        logger.info("Arm zero position set.")

        logger.info("Setting range: -90° to +90° by default for all joints")
        # TODO(Steven, Pepijn): Check if MotorCalibration is actually needed here given that we only use Degrees
        for motor_name, motor in self.bus.motors.items():
            self.calibration[motor_name] = MotorCalibration(
                id=motor.id,
                drive_mode=0,
                homing_offset=0,
                range_min=-90,
                range_max=90,
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print(f"Calibration saved to {self.calibration_fpath}")

    def configure(self) -> None:
        """
        Configure motors for manual teleoperation.

        For manual control, we disable torque so the arm can be moved by hand.
        """

        return self.bus.disable_torque() if self.config.manual_control else self.bus.configure_motors()

    def hold_position(self) -> RobotAction:
        """Hold the leader arm at its current joint positions."""
        if self._hold_positions is None:
            states = self.bus.sync_read_all_states()
            self._hold_positions = {
                motor: state.get("position")
                for motor, state in states.items()
                if state.get("position") is not None
            }
        positions = self._hold_positions
        if not positions:
            raise RuntimeError("Cannot hold OpenArm leader; no motor positions were read.")

        if not self._hold_torque_enabled:
            self.bus.enable_torque()
            self._hold_torque_enabled = True
        self._manual_gravity_compensation_paused = False

        self._send_hold_position(positions)
        return {f"{motor}.pos": position for motor, position in positions.items()}

    def _send_hold_position(self, positions: dict[str, float]) -> None:
        motor_index = {
            "joint_1": 0,
            "joint_2": 1,
            "joint_3": 2,
            "joint_4": 3,
            "joint_5": 4,
            "joint_6": 5,
            "joint_7": 6,
            "gripper": 7,
        }
        commands = {}
        for motor_name, position_degrees in positions.items():
            idx = motor_index.get(motor_name, 0)
            kp = (
                self.config.position_kp[idx]
                if isinstance(self.config.position_kp, list)
                else self.config.position_kp
            )
            kd = (
                self.config.position_kd[idx]
                if isinstance(self.config.position_kd, list)
                else self.config.position_kd
            )
            commands[motor_name] = (kp, kd, position_degrees, 0.0, 0.0)

        self.bus._mit_control_batch(commands)

    def release_for_manual_control(self) -> None:
        """Release the leader arm so it can be moved by hand."""
        if self.config.manual_control:
            self.bus.disable_torque()
            self._manual_gravity_compensation_paused = True
        elif self.config.gravity_compensation:
            self._manual_gravity_compensation_paused = False
        else:
            self.bus.disable_torque()
            self._manual_gravity_compensation_paused = True
        self._hold_torque_enabled = False
        self._hold_positions = None

    def _normalized_side(self) -> str:
        side = self.config.side or "right"
        side = side.removesuffix("_arm")
        if side not in {"left", "right"}:
            raise ValueError("OpenArm leader side must be 'left', 'right', 'left_arm', or 'right_arm'.")
        return side

    def _load_pinocchio_model(self) -> None:
        if self._pin_model is not None:
            return

        if self.config.gravity_compensation_urdf_path is None:
            raise ValueError("gravity_compensation_urdf_path must be set when gravity compensation is enabled.")

        urdf_path = Path(self.config.gravity_compensation_urdf_path)
        if not urdf_path.is_file():
            raise FileNotFoundError(f"OpenArm leader gravity compensation URDF not found: {urdf_path}")

        try:
            import pinocchio as pin
        except ImportError as e:
            raise ImportError(
                "OpenArm leader gravity compensation requires Pinocchio. Install it before running "
                "with --teleop.manual_control=false."
            ) from e

        if not hasattr(pin, "RobotWrapper") and not hasattr(pin, "buildModelFromUrdf"):
            raise ImportError(
                "The imported 'pinocchio' package is not the robotics Pinocchio library. "
                "Install the correct package with: conda install -c conda-forge pinocchio"
            )

        self._pin = pin
        if hasattr(pin, "buildModelFromUrdf"):
            self._pin_model = pin.buildModelFromUrdf(str(urdf_path))
        else:
            self._pin_model = pin.RobotWrapper.BuildFromURDF(str(urdf_path)).model
        self._pin_data = self._pin_model.createData()
        self._pin_q = pin.neutral(self._pin_model)

        side = self._normalized_side()
        for idx in range(1, 8):
            motor_name = f"joint_{idx}"
            urdf_joint_name = f"openarm_{side}_joint{idx}"
            joint_id = self._pin_model.getJointId(urdf_joint_name)
            if joint_id == 0:
                raise ValueError(f"Joint '{urdf_joint_name}' was not found in {urdf_path}.")
            self._pin_joint_q_indices[motor_name] = self._pin_model.idx_qs[joint_id]
            self._pin_joint_v_indices[motor_name] = self._pin_model.idx_vs[joint_id]

        logger.info("Loaded OpenArm leader Pinocchio model from %s for %s arm.", urdf_path, side)

    def _apply_gravity_compensation(self, states: dict[str, Any]) -> None:
        if (
            self.config.manual_control
            or not self.config.gravity_compensation
            or self._manual_gravity_compensation_paused
        ):
            return

        self._load_pinocchio_model()
        if self._pin is None or self._pin_model is None or self._pin_data is None or self._pin_q is None:
            raise RuntimeError("Pinocchio model was not initialized.")

        for motor_name, q_index in self._pin_joint_q_indices.items():
            position_deg = states.get(motor_name, {}).get("position")
            if position_deg is None:
                logger.debug("Skipping gravity compensation; missing position for %s.", motor_name)
                return
            self._pin_q[q_index] = math.radians(float(position_deg))

        gravity = self._pin.computeGeneralizedGravity(self._pin_model, self._pin_data, self._pin_q)

        commands = {}
        for motor_name in self.bus.motors:
            if motor_name == "gripper":
                commands[motor_name] = (0.0, 0.0, 0.0, 0.0, 0.0)
                continue

            v_index = self._pin_joint_v_indices[motor_name]
            commands[motor_name] = (0.0, 0.0, 0.0, 0.0, float(gravity[v_index]))

        self.bus._mit_control_batch(commands)

    def setup_motors(self) -> None:
        raise NotImplementedError(
            "Motor ID configuration is typically done via manufacturer tools for CAN motors."
        )

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        """
        Get current action from the leader arm.

        This is the main method for teleoperators - it reads the current state
        of the leader arm and returns it as an action that can be sent to a follower.

        Reads all motor states (pos/vel/torque) in one CAN refresh cycle.
        """
        start = time.perf_counter()
        self.release_for_manual_control()

        action_dict: dict[str, Any] = {}

        # Use sync_read_all_states to get pos/vel/torque in one go
        states = self.bus.sync_read_all_states()
        self._apply_gravity_compensation(states)
        for motor in self.bus.motors:
            state = states.get(motor, {})
            action_dict[f"{motor}.pos"] = state.get("position")
            if self.config.use_velocity_and_torque:
                action_dict[f"{motor}.vel"] = state.get("velocity")
                action_dict[f"{motor}.torque"] = state.get("torque")

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        return action_dict

    def send_feedback(self, feedback: dict[str, float]) -> None:
        raise NotImplementedError("Feedback is not yet implemented for OpenArm leader.")

    @check_if_not_connected
    def disconnect(self) -> None:
        """Disconnect from teleoperator."""

        # Disconnect CAN bus.
        self.bus.disconnect(disable_torque=self.config.disable_torque_on_disconnect)
        logger.info(f"{self} disconnected.")
