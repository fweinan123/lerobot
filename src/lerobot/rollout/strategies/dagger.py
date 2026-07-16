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

"""DAgger rollout strategy: Human-in-the-Loop data collection.

Implements the RaC paradigm (Recovery and Correction) for interactive
imitation learning.  Alternates between autonomous policy execution and
human intervention via teleoperator.

Input is controlled via either a keyboard or foot pedal, selected by
the ``input_device`` config field.  Each device exposes three actions:

    1. **pause_resume** — Toggle policy execution (AUTONOMOUS <-> PAUSED).
    2. **correction**   — Toggle correction recording (PAUSED <-> CORRECTING).
    3. **upload**        — Push dataset to hub on demand (corrections-only mode).
    4. **sync_leader**   — Move the leader to the follower's current joint pose.
    ESC (keyboard only) — Stop session.

Recording modes:
    ``record_autonomous=True``:  Sentry-like continuous recording with
        time-based episode rotation.  Both autonomous and correction
        frames are recorded; corrections tagged ``intervention=True``.
    ``record_autonomous=False``: Only correction windows are recorded.
        Each correction (start to stop) becomes one episode.

Teleoperator handover:
    On AUTONOMOUS → PAUSED, actuated teleops (those with non-empty
    ``feedback_features``, e.g. SO-101, OpenArmMini) are smoothly driven to
    the follower's last position via ``send_feedback`` so the operator takes
    over without a jerk.  Non-actuated teleops cannot be driven,
    so on PAUSED → CORRECTING the follower is instead slid to the teleop's
    current pose before the correction begins.  On CORRECTING → PAUSED, the
    leader is held at its current pose until PAUSED → AUTONOMOUS powers it off
    while policy control resumes.
"""

from __future__ import annotations

import contextlib
import enum
import logging
import os
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event, Lock
from typing import Any

import numpy as np

from lerobot.common.control_utils import is_headless
from lerobot.datasets import VideoEncodingManager
from lerobot.datasets.utils import DEFAULT_VIDEO_FILE_SIZE_IN_MB
from lerobot.teleoperators import Teleoperator
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.import_utils import _pynput_available
from lerobot.utils.pedal import start_pedal_listener
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say

from ..configs import DAggerKeyboardConfig, DAggerPedalConfig, DAggerStrategyConfig
from ..context import RolloutContext
from ..robot_wrapper import ThreadSafeRobot
from .core import RolloutStrategy, estimate_max_episode_seconds, safe_push_to_hub, send_next_action

PYNPUT_AVAILABLE = _pynput_available
keyboard = None
if PYNPUT_AVAILABLE:
    try:
        if ("DISPLAY" not in os.environ) and ("linux" in sys.platform):
            logging.info("No DISPLAY set. Skipping pynput import.")
            PYNPUT_AVAILABLE = False
        else:
            from pynput import keyboard
    except Exception as e:
        PYNPUT_AVAILABLE = False
        logging.info(f"Could not import pynput: {e}")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DAgger state machine
# ---------------------------------------------------------------------------


class DAggerPhase(enum.Enum):
    """Observable phases of a DAgger episode."""

    AUTONOMOUS = "autonomous"  # Policy driving
    PAUSED = "paused"  # Engine paused, teleop aligned, awaiting input
    CORRECTING = "correcting"  # Human driving via teleop, recording interventions


# Valid (current_phase, event) -> next_phase
_DAGGER_TRANSITIONS: dict[tuple[DAggerPhase, str], DAggerPhase] = {
    (DAggerPhase.AUTONOMOUS, "pause_resume"): DAggerPhase.PAUSED,
    (DAggerPhase.PAUSED, "pause_resume"): DAggerPhase.AUTONOMOUS,
    (DAggerPhase.PAUSED, "correction"): DAggerPhase.CORRECTING,
    (DAggerPhase.CORRECTING, "correction"): DAggerPhase.PAUSED,
}


class DAggerEvents:
    """Thread-safe container for DAgger input device events.

    The keyboard/pedal threads write transition requests; the main loop
    consumes them.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._phase = DAggerPhase.AUTONOMOUS
        self._pending_transition: str | None = None

        # Session-level flags
        self.stop_recording = Event()
        self.upload_requested = Event()
        self.sync_leader_requested = Event()
        self.correction_record_requested = Event()
        self.correction_recording_active = Event()
        self.leader_hold_active = Event()
        self._leader_hold_target: dict[str, float] | None = None

    # -- Thread-safe phase access ------------------------------------------

    @property
    def phase(self) -> DAggerPhase:
        """Current phase of the DAgger state machine."""
        with self._lock:
            return self._phase

    @phase.setter
    def phase(self, value: DAggerPhase) -> None:
        with self._lock:
            self._phase = value

    def request_transition(self, event: str) -> None:
        """Request a phase transition (called from keyboard/pedal threads).

        Only enqueues the request if it corresponds to a valid transition
        from the current phase, preventing impossible state changes.
        """
        with self._lock:
            if (self._phase, event) in _DAGGER_TRANSITIONS:
                self._pending_transition = event

    def consume_transition(self) -> tuple[DAggerPhase, DAggerPhase] | None:
        """Consume a pending transition (called from main loop)."""
        with self._lock:
            if self._pending_transition is None:
                return None
            key = (self._phase, self._pending_transition)
            self._pending_transition = None
            new_phase = _DAGGER_TRANSITIONS.get(key)
            if new_phase is None:
                return None
            old_phase = self._phase
            self._phase = new_phase
            return old_phase, new_phase

    def reset(self) -> None:
        """Reset all transient state for a fresh session."""
        with self._lock:
            self._phase = DAggerPhase.AUTONOMOUS
            self._pending_transition = None
            self._leader_hold_target = None
        self.upload_requested.clear()
        self.sync_leader_requested.clear()
        self.correction_record_requested.clear()
        self.correction_recording_active.clear()
        self.leader_hold_active.clear()

    def set_leader_hold_target(self, target: dict[str, float]) -> None:
        """Store the leader pose that should be actively maintained while paused."""
        with self._lock:
            self._leader_hold_target = dict(target)
        self.leader_hold_active.set()

    def clear_leader_hold(self) -> None:
        """Stop actively holding the leader pose."""
        with self._lock:
            self._leader_hold_target = None
        self.leader_hold_active.clear()

    def get_leader_hold_target(self) -> dict[str, float] | None:
        """Return the currently requested leader hold target, if any."""
        with self._lock:
            return None if self._leader_hold_target is None else dict(self._leader_hold_target)


# ---------------------------------------------------------------------------
# Teleoperator helpers
# ---------------------------------------------------------------------------


def _teleop_supports_feedback(teleop: Teleoperator) -> bool:
    """Return True when the teleop can receive position feedback (is actuated).
    TODO(Maxime): See if it is possible to unify this interface across teleops instead of duck-typing.
    """
    return (
        bool(teleop.feedback_features)
        and hasattr(teleop, "disable_torque")
        and hasattr(teleop, "enable_torque")
    )


def _release_teleop_for_manual_control(teleop: Teleoperator | None) -> None:
    """Release a teleoperator after automated alignment/holding."""
    if teleop is None:
        return
    if hasattr(teleop, "release_for_manual_control"):
        teleop.release_for_manual_control()
    elif _teleop_supports_feedback(teleop):
        teleop.disable_torque()


def _power_off_teleop_for_policy_control(teleop: Teleoperator | None) -> None:
    """Make the leader passive when autonomous policy control resumes."""
    if teleop is None:
        return
    if hasattr(teleop, "disable_torque"):
        teleop.disable_torque()
    elif hasattr(teleop, "bus") and hasattr(teleop.bus, "disable_torque"):
        teleop.bus.disable_torque()
    elif hasattr(teleop, "release_for_manual_control"):
        teleop.release_for_manual_control()
    elif _teleop_supports_feedback(teleop):
        teleop.disable_torque()

    for attr, value in (
        ("_hold_torque_enabled", False),
        ("_hold_positions", None),
        ("_manual_gravity_compensation_paused", True),
    ):
        if hasattr(teleop, attr):
            with contextlib.suppress(Exception):
                setattr(teleop, attr, value)


def _hold_teleop_at_current_position(teleop: Teleoperator | None) -> None:
    """Hold a teleoperator at its current pose until control is handed back."""
    if teleop is None:
        return
    if hasattr(teleop, "hold_position"):
        teleop.hold_position()
        return
    if _teleop_supports_feedback(teleop):
        current_position = teleop.get_action()
        teleop.enable_torque()
        teleop.send_feedback(current_position)


def _get_teleop_hold_target(teleop: Teleoperator | None) -> dict[str, float] | None:
    """Read the currently held leader target without releasing the leader."""
    if teleop is None:
        return None
    hold_positions = getattr(teleop, "_hold_positions", None)
    if hold_positions:
        return {
            key if key.endswith(".pos") else f"{key}.pos": float(value)
            for key, value in hold_positions.items()
            if value is not None
        }
    if _teleop_supports_feedback(teleop):
        return teleop.get_action()
    return None


def _hold_teleop_at_target(teleop: Teleoperator | None, target: dict[str, float] | None) -> None:
    """Continuously command a teleoperator to hold a specific target pose."""
    if teleop is None:
        return
    if target is None:
        _hold_teleop_at_current_position(teleop)
        return

    # OpenArm leader exposes position holding through its CAN MIT helper; keep
    # sending the target so the arm does not fall passive between sync and tab.
    if hasattr(teleop, "_send_hold_position") and hasattr(teleop, "bus"):
        positions = {
            key.removesuffix(".pos"): float(value)
            for key, value in target.items()
            if key.endswith(".pos")
        }
        positions.update({key: float(value) for key, value in target.items() if not key.endswith(".pos")})
        if not positions:
            raise RuntimeError("Cannot hold leader: target does not contain position keys.")
        teleop.bus.enable_torque()
        if hasattr(teleop, "_hold_torque_enabled"):
            teleop._hold_torque_enabled = True
        if hasattr(teleop, "_manual_gravity_compensation_paused"):
            teleop._manual_gravity_compensation_paused = True
        if hasattr(teleop, "_hold_positions"):
            teleop._hold_positions = positions.copy()
        teleop._send_hold_position(positions)
        return

    if _teleop_supports_feedback(teleop):
        teleop.enable_torque()
        teleop.send_feedback(target)
        return

    if hasattr(teleop, "hold_position"):
        teleop.hold_position()


def _teleop_smooth_move_to(
    teleop: Teleoperator, target_pos: dict, duration_s: float = 2.0, fps: int = 30
) -> None:
    """Smoothly move an actuated teleop to ``target_pos`` via linear interpolation.

    Requires the teleoperator to support feedback
    (i.e. have non-empty ``feedback_features`` and implement ``disable_torque`` / ``enable_torque``).

    TODO(Maxime): This blocks up to ``duration_s`` seconds, during this time
    the follower robot doesn't receive new actions, this could be an issue on LeKiwi.
    """
    teleop.enable_torque()
    current = teleop.get_action()
    steps = max(int(duration_s * fps), 1)

    for step in range(steps + 1):
        t = step / steps
        interp = {
            k: current[k] * (1 - t) + target_pos[k] * t if k in target_pos else current[k] for k in current
        }
        teleop.send_feedback(interp)
        time.sleep(1 / fps)


def _follower_smooth_move_to(
    robot: ThreadSafeRobot, current: dict, target: dict, duration_s: float = 1.0, fps: int = 30
) -> None:
    """Smoothly move the follower robot from ``current`` to ``target`` action.

    Used when the teleop is non-actuated: instead of driving the leader arm
    to the follower, we bring the follower to the teleop's current pose.
    Both ``current`` and ``target`` must be in robot-action key space.
    """
    steps = max(int(duration_s * fps), 1)

    for step in range(steps + 1):
        t = step / steps
        interp = {k: current[k] * (1 - t) + target[k] * t if k in target else current[k] for k in current}
        robot.send_action(interp)
        time.sleep(1 / fps)


def _sync_leader_to_follower(ctx: RolloutContext, duration_s: float = 1.0) -> dict[str, float] | None:
    """Move an actuated leader to the follower's current joint_1..joint_7 pose."""
    teleop = ctx.hardware.teleop
    if teleop is None:
        logger.warning("Cannot sync leader: no teleoperator configured")
        return None

    robot = ctx.hardware.robot_wrapper
    obs = robot.get_observation()
    target = {f"joint_{idx}.pos": obs[f"joint_{idx}.pos"] for idx in range(1, 8) if f"joint_{idx}.pos" in obs}
    if len(target) != 7:
        logger.warning("Cannot sync leader: follower observation is missing joint keys (%s)", sorted(target))
        return None

    if hasattr(teleop, "sync_to_action"):
        logger.info("Syncing leader to follower joint pose via teleop.sync_to_action")
        synced_target = teleop.sync_to_action(target, duration_s=duration_s, fps=max(int(ctx.runtime.cfg.fps), 1))
        return synced_target if isinstance(synced_target, dict) else target

    if _teleop_supports_feedback(teleop):
        logger.info("Syncing leader to follower joint pose via teleop feedback")
        _teleop_smooth_move_to(teleop, target, duration_s=duration_s, fps=max(int(ctx.runtime.cfg.fps), 1))
        return target

    logger.warning("Cannot sync leader: teleoperator %s does not support leader positioning", teleop.name)
    return None


def _handle_sync_leader_request(ctx: RolloutContext, events: DAggerEvents, play_sounds: bool) -> None:
    if not events.sync_leader_requested.is_set():
        return

    events.sync_leader_requested.clear()
    if events.phase != DAggerPhase.PAUSED:
        logger.warning("Leader sync requested outside PAUSED phase; press pause before syncing leader")
        log_say("Pause before syncing leader", play_sounds)
        return

    log_say("Syncing leader to follower", play_sounds, blocking=True)
    try:
        if hold_target := _sync_leader_to_follower(ctx):
            events.set_leader_hold_target(hold_target)
    except Exception as e:
        logger.warning("Failed to sync leader to follower: %s", e)
        events.clear_leader_hold()
        log_say("Failed to sync leader", play_sounds)


def _handle_correction_record_request(events: DAggerEvents) -> bool:
    """Start writing correction frames after the operator presses r."""
    if not events.correction_record_requested.is_set():
        return False

    events.correction_record_requested.clear()
    if events.phase != DAggerPhase.CORRECTING:
        logger.warning("Correction recording requested outside CORRECTING phase; press tab before r")
        return False

    if not events.correction_recording_active.is_set():
        logger.info("Correction frame recording started")
        events.correction_recording_active.set()
        return True
    return False


# ---------------------------------------------------------------------------
# Input device handlers
# ---------------------------------------------------------------------------


def _init_dagger_keyboard(events: DAggerEvents, cfg: DAggerKeyboardConfig):
    """Initialise keyboard listener with DAgger 3-key controls.

    Returns the pynput Listener (or ``None`` in headless mode or when
    pynput is unavailable).
    """
    if not PYNPUT_AVAILABLE or is_headless():
        logger.warning("Headless environment or pynput unavailable — keyboard controls disabled")
        return None

    # Map config key names to pynput Key objects for special keys
    special_keys = {
        "space": keyboard.Key.space,
        "tab": keyboard.Key.tab,
        "enter": keyboard.Key.enter,
    }

    def _resolve_key(key) -> str | None:
        """Resolve a pynput key event to a config-comparable string."""
        if key == keyboard.Key.esc:
            return "esc"
        for name, pynput_key in special_keys.items():
            if key == pynput_key:
                return name
        if hasattr(key, "char") and key.char:
            return key.char
        return None

    # Build mapping: resolved key string -> DAgger event name
    key_to_event = {
        cfg.pause_resume: "pause_resume",
        cfg.correction: "correction",
    }

    def on_press(key):
        try:
            resolved = _resolve_key(key)
            if resolved is None:
                return
            if resolved == "esc":
                logger.info("Stop recording...")
                events.stop_recording.set()
                return
            if resolved in key_to_event:
                events.request_transition(key_to_event[resolved])
            if resolved == "r":
                events.correction_record_requested.set()
            if resolved == cfg.upload:
                events.upload_requested.set()
            if resolved == cfg.sync_leader:
                events.sync_leader_requested.set()
        except Exception as e:
            logger.debug("Key error: %s", e)

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    logger.info(
        "DAgger keyboard listener started (pause_resume='%s', correction='%s', record='r', upload='%s', sync_leader='%s', ESC=stop)",
        cfg.pause_resume,
        cfg.correction,
        cfg.upload,
        cfg.sync_leader,
    )
    return listener


def _init_dagger_pedal(events: DAggerEvents, cfg: DAggerPedalConfig):
    """Initialise foot pedal listener with DAgger 3-pedal controls.

    Returns the pedal listener thread (or ``None`` if evdev is unavailable).
    """
    code_to_event = {
        cfg.pause_resume: "pause_resume",
        cfg.correction: "correction",
    }

    def on_press(code: str) -> None:
        if code in code_to_event:
            events.request_transition(code_to_event[code])
        if code == cfg.upload:
            events.upload_requested.set()

    logger.info("Initializing DAgger foot pedal listener (device=%s)", cfg.device_path)
    return start_pedal_listener(on_press, device_path=cfg.device_path)


# ---------------------------------------------------------------------------
# DAgger Strategy
# ---------------------------------------------------------------------------


class DAggerStrategy(RolloutStrategy):
    """Human-in-the-Loop data collection with intervention tagging.

    State machine::

        AUTONOMOUS --(key1)--> PAUSED --(key2)--> CORRECTING --(key2)--> PAUSED
                               --(key1)--> AUTONOMOUS

    Recording modes:
        ``record_autonomous=True``: Sentry-like continuous recording with
            time-based episode rotation.  Intervention frames tagged True.
        ``record_autonomous=False``: Only correction windows recorded.
            Each correction = one episode.  Upload on demand via key3.
    """

    config: DAggerStrategyConfig

    def __init__(self, config: DAggerStrategyConfig):
        super().__init__(config)
        self._listener = None
        self._pedal_thread = None
        self._events = DAggerEvents()
        self._push_executor: ThreadPoolExecutor | None = None
        self._pending_push: Future | None = None
        self._needs_push = Event()
        self._episode_lock = Lock()

    def setup(self, ctx: RolloutContext) -> None:
        """Initialise the inference engine and input device listener."""
        self._init_engine(ctx)
        self._push_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dagger-push")
        target_mb = self.config.target_video_file_size_mb or DEFAULT_VIDEO_FILE_SIZE_IN_MB
        self._episode_duration_s = estimate_max_episode_seconds(
            ctx.data.dataset_features, ctx.runtime.cfg.fps, target_size_mb=target_mb
        )

        if self.config.input_device == "keyboard":
            self._listener = _init_dagger_keyboard(self._events, self.config.keyboard)
        else:
            self._pedal_thread = _init_dagger_pedal(self._events, self.config.pedal)

        record_mode = "all frames (sentry-like)" if self.config.record_autonomous else "corrections only"
        logger.info(
            "DAgger strategy ready (input=%s, episodes=%d, record=%s, episode_duration=%.0fs)",
            self.config.input_device,
            self.config.num_episodes,
            record_mode,
            self._episode_duration_s,
        )

    def run(self, ctx: RolloutContext) -> None:
        """Run DAgger episodes with human-in-the-loop intervention."""
        if self.config.record_autonomous:
            self._run_continuous(ctx)
        else:
            self._run_corrections_only(ctx)

    def teardown(self, ctx: RolloutContext) -> None:
        """Stop listeners, finalise the dataset, and disconnect hardware."""
        play_sounds = ctx.runtime.cfg.play_sounds
        logger.info("Stopping DAgger recording")
        log_say("Stopping DAgger recording", play_sounds)

        if self._listener is not None and not is_headless():
            logger.info("Stopping keyboard listener")
            self._listener.stop()

        # Flush any queued/running push cleanly
        if self._push_executor is not None:
            logger.info("Shutting down push executor (waiting for pending pushes)...")
            self._push_executor.shutdown(wait=True)
            self._push_executor = None

        if ctx.data.dataset is not None:
            logger.info("Finalizing dataset...")
            ctx.data.dataset.finalize()
            if self._needs_push.is_set() and ctx.runtime.cfg.dataset and ctx.runtime.cfg.dataset.push_to_hub:
                logger.info("Pushing final dataset to hub...")
                if safe_push_to_hub(
                    ctx.data.dataset,
                    tags=ctx.runtime.cfg.dataset.tags,
                    private=ctx.runtime.cfg.dataset.private,
                ):
                    logger.info("Dataset uploaded to hub")
                    log_say("Dataset uploaded to hub", play_sounds)

        self._teardown_hardware(
            ctx.hardware,
            return_to_initial_position=ctx.runtime.cfg.return_to_initial_position,
        )
        logger.info("DAgger strategy teardown complete")

    # ------------------------------------------------------------------
    # Continuous recording mode (record_autonomous=True)
    # ------------------------------------------------------------------

    def _run_continuous(self, ctx: RolloutContext) -> None:
        """Sentry-like continuous recording with intervention tagging.

        Episodes are auto-rotated every ``episode_time_s`` seconds and
        uploaded in the background every ``upload_every_n_episodes`` episodes.
        Both autonomous and correction frames are recorded; corrections are
        tagged with ``intervention=True``.
        """
        engine = self._engine
        cfg = ctx.runtime.cfg
        robot = ctx.hardware.robot_wrapper
        teleop = ctx.hardware.teleop
        dataset = ctx.data.dataset
        events = self._events
        interpolator = self._interpolator
        features = ctx.data.dataset_features

        control_interval = interpolator.get_control_interval(cfg.fps)
        record_stride = max(1, cfg.interpolation_multiplier)
        task_str = cfg.dataset.single_task if cfg.dataset else cfg.task
        play_sounds = cfg.play_sounds

        engine.reset()
        interpolator.reset()
        events.reset()
        engine.resume()

        last_action: dict[str, Any] | None = None
        record_tick = 0
        start_time = time.perf_counter()
        episode_start = time.perf_counter()
        episodes_since_push = 0
        episode_duration_s = self._episode_duration_s
        logger.info("DAgger continuous recording started (episode_duration=%.0fs)", episode_duration_s)

        with VideoEncodingManager(dataset):
            try:
                while not events.stop_recording.is_set() and not ctx.runtime.shutdown_event.is_set():
                    loop_start = time.perf_counter()

                    if cfg.duration > 0 and (time.perf_counter() - start_time) >= cfg.duration:
                        logger.info("Duration limit reached (%.0fs)", cfg.duration)
                        break

                    # Process transitions
                    transition = events.consume_transition()
                    if transition is not None:
                        old_phase, new_phase = transition
                        self._apply_transition(
                            old_phase,
                            new_phase,
                            engine,
                            interpolator,
                            ctx,
                            last_action,
                        )
                        if new_phase == DAggerPhase.AUTONOMOUS:
                            last_action = None
                            events.clear_leader_hold()
                        if old_phase == DAggerPhase.PAUSED and new_phase == DAggerPhase.CORRECTING:
                            events.clear_leader_hold()
                            events.correction_record_requested.clear()
                            events.correction_recording_active.clear()
                            record_tick = 0
                        if old_phase == DAggerPhase.CORRECTING and new_phase == DAggerPhase.PAUSED:
                            if hold_target := _get_teleop_hold_target(teleop):
                                events.set_leader_hold_target(hold_target)
                            else:
                                events.leader_hold_active.set()
                            events.correction_record_requested.clear()
                            events.correction_recording_active.clear()

                    _handle_sync_leader_request(ctx, events, play_sounds)
                    if _handle_correction_record_request(events):
                        record_tick = 0

                    phase = events.phase

                    # --- CORRECTING: human teleop control ---
                    # TODO(Steven): teleop runs at the same FPS as the policy. To
                    # decouple the two, sample teleop at its native rate and
                    # interpolate to the control loop's tick rate.
                    if phase == DAggerPhase.CORRECTING:
                        obs = robot.get_observation()
                        obs_processed = ctx.processors.robot_observation_processor(obs)
                        teleop_action = teleop.get_action()
                        processed_teleop = ctx.processors.teleop_action_processor((teleop_action, obs))
                        robot_action_to_send = ctx.processors.robot_action_processor((processed_teleop, obs))
                        robot.send_action(robot_action_to_send)
                        last_action = robot_action_to_send
                        self._log_telemetry(obs_processed, processed_teleop, ctx.runtime)
                        if events.correction_recording_active.is_set() and record_tick % record_stride == 0:
                            obs_frame = build_dataset_frame(features, obs_processed, prefix=OBS_STR)
                            action_frame = build_dataset_frame(features, processed_teleop, prefix=ACTION)
                            frame = {
                                **obs_frame,
                                **action_frame,
                                "task": task_str,
                                "intervention": np.array([True], dtype=bool),
                            }
                            dataset.add_frame(frame)
                        if events.correction_recording_active.is_set():
                            record_tick += 1

                    # --- PAUSED: hold position ---
                    elif phase == DAggerPhase.PAUSED:
                        if events.leader_hold_active.is_set():
                            try:
                                _hold_teleop_at_target(teleop, events.get_leader_hold_target())
                            except Exception as e:
                                logger.warning("Failed to maintain leader hold: %s", e)
                                events.clear_leader_hold()
                        if last_action:
                            robot.send_action(last_action)

                    # --- AUTONOMOUS: policy control ---
                    else:
                        obs = robot.get_observation()
                        obs_processed = self._process_observation_and_notify(ctx.processors, obs)

                        if self._handle_warmup(cfg.use_torch_compile, loop_start, control_interval):
                            continue

                        action_dict = send_next_action(obs_processed, obs, ctx, interpolator)
                        if action_dict is not None:
                            self._log_telemetry(obs_processed, action_dict, ctx.runtime)
                            last_action = ctx.processors.robot_action_processor((action_dict, obs))
                            if record_tick % record_stride == 0:
                                obs_frame = build_dataset_frame(features, obs_processed, prefix=OBS_STR)
                                action_frame = build_dataset_frame(features, action_dict, prefix=ACTION)
                                frame = {
                                    **obs_frame,
                                    **action_frame,
                                    "task": task_str,
                                    "intervention": np.array([False], dtype=bool),
                                }
                                dataset.add_frame(frame)
                            record_tick += 1

                    # Episode rotation derived from the video file-size target.
                    # Saving is deferred while a correction is ongoing so the
                    # episode boundary lands on a clean autonomous frame.
                    elapsed = time.perf_counter() - episode_start
                    if elapsed >= episode_duration_s and phase != DAggerPhase.CORRECTING:
                        with self._episode_lock:
                            dataset.save_episode()
                        episodes_since_push += 1
                        self._needs_push.set()
                        logger.info(
                            "Episode saved (total: %d, elapsed: %.1fs)",
                            dataset.num_episodes,
                            elapsed,
                        )
                        log_say(f"Episode {dataset.num_episodes} saved", play_sounds)

                        if episodes_since_push >= self.config.upload_every_n_episodes:
                            self._background_push(dataset, cfg)
                            episodes_since_push = 0

                        episode_start = time.perf_counter()

                    dt = time.perf_counter() - loop_start
                    if (sleep_t := control_interval - dt) > 0:
                        precise_sleep(sleep_t)
                    else:
                        logger.warning(
                            f"Record loop is running slower ({1 / dt:.1f} Hz) than the target FPS ({cfg.fps} Hz). Dataset frames might be dropped and robot control might be unstable. Common causes are: 1) Camera FPS not keeping up 2) Policy inference taking too long 3) CPU starvation"
                        )

            finally:
                logger.info("DAgger continuous control loop ended — pausing engine")
                engine.pause()
                with contextlib.suppress(Exception):
                    with self._episode_lock:
                        dataset.save_episode()
                    self._needs_push.set()
                    logger.info("Final in-progress episode saved")

    # ------------------------------------------------------------------
    # Corrections-only mode (record_autonomous=False)
    # ------------------------------------------------------------------

    def _run_corrections_only(self, ctx: RolloutContext) -> None:
        """Record only human correction windows.  Each correction = one episode.

        The policy runs autonomously without recording.  When the user
        pauses and starts a correction, frames are recorded with
        ``intervention=True``.  Stopping the correction saves the episode.
        The dataset can be uploaded on demand via the upload key/pedal.
        """
        engine = self._engine
        cfg = ctx.runtime.cfg
        robot = ctx.hardware.robot_wrapper
        teleop = ctx.hardware.teleop
        dataset = ctx.data.dataset
        events = self._events
        interpolator = self._interpolator
        features = ctx.data.dataset_features

        control_interval = interpolator.get_control_interval(cfg.fps)
        record_stride = max(1, cfg.interpolation_multiplier)
        task_str = cfg.dataset.single_task if cfg.dataset else cfg.task
        play_sounds = cfg.play_sounds

        engine.reset()
        interpolator.reset()
        events.reset()
        engine.resume()

        last_action: dict[str, Any] | None = None
        start_time = time.perf_counter()
        record_tick = 0
        recorded = 0
        logger.info(
            "DAgger corrections-only recording started (target: %d episodes)", self.config.num_episodes
        )

        with VideoEncodingManager(dataset):
            try:
                while (
                    recorded < self.config.num_episodes
                    and not events.stop_recording.is_set()
                    and not ctx.runtime.shutdown_event.is_set()
                ):
                    loop_start = time.perf_counter()

                    if cfg.duration > 0 and (time.perf_counter() - start_time) >= cfg.duration:
                        logger.info("Duration limit reached (%.0fs)", cfg.duration)
                        break

                    # Process transitions
                    transition = events.consume_transition()
                    if transition is not None:
                        old_phase, new_phase = transition
                        correction_was_recording = events.correction_recording_active.is_set()
                        self._apply_transition(
                            old_phase,
                            new_phase,
                            engine,
                            interpolator,
                            ctx,
                            last_action,
                        )
                        if new_phase == DAggerPhase.AUTONOMOUS:
                            last_action = None
                            events.clear_leader_hold()
                        if old_phase == DAggerPhase.PAUSED and new_phase == DAggerPhase.CORRECTING:
                            events.clear_leader_hold()
                            events.correction_record_requested.clear()
                            events.correction_recording_active.clear()
                            record_tick = 0
                        if old_phase == DAggerPhase.CORRECTING and new_phase == DAggerPhase.PAUSED:
                            if hold_target := _get_teleop_hold_target(teleop):
                                events.set_leader_hold_target(hold_target)
                            else:
                                events.leader_hold_active.set()
                            events.correction_record_requested.clear()
                            events.correction_recording_active.clear()

                        # Correction ended -> save episode (blocking if not streaming)
                        if (
                            old_phase == DAggerPhase.CORRECTING
                            and new_phase == DAggerPhase.PAUSED
                            and correction_was_recording
                        ):
                            with self._episode_lock:
                                dataset.save_episode()
                            recorded += 1
                            self._needs_push.set()
                            logger.info(
                                "Correction %d/%d saved",
                                recorded,
                                self.config.num_episodes,
                            )
                            log_say(f"Correction {recorded} saved", play_sounds)

                    # On-demand upload
                    if events.upload_requested.is_set():
                        events.upload_requested.clear()
                        logger.info("Upload requested by user")
                        self._background_push(dataset, cfg)

                    _handle_sync_leader_request(ctx, events, play_sounds)
                    if _handle_correction_record_request(events):
                        record_tick = 0

                    phase = events.phase

                    # --- CORRECTING: human teleop control + recording ---
                    # TODO(Steven): teleop runs at the same FPS as the policy. To
                    # decouple the two, sample teleop at its native rate and
                    # interpolate to the control loop's tick rate.
                    if phase == DAggerPhase.CORRECTING:
                        obs = robot.get_observation()
                        obs_processed = ctx.processors.robot_observation_processor(obs)
                        teleop_action = teleop.get_action()
                        processed_teleop = ctx.processors.teleop_action_processor((teleop_action, obs))
                        robot_action_to_send = ctx.processors.robot_action_processor((processed_teleop, obs))
                        robot.send_action(robot_action_to_send)
                        last_action = robot_action_to_send
                        self._log_telemetry(obs_processed, processed_teleop, ctx.runtime)

                        if events.correction_recording_active.is_set() and record_tick % record_stride == 0:
                            obs_frame = build_dataset_frame(features, obs_processed, prefix=OBS_STR)
                            action_frame = build_dataset_frame(features, processed_teleop, prefix=ACTION)
                            dataset.add_frame(
                                {
                                    **obs_frame,
                                    **action_frame,
                                    "task": task_str,
                                    "intervention": np.array([True], dtype=bool),
                                }
                            )
                        if events.correction_recording_active.is_set():
                            record_tick += 1

                    # --- PAUSED: hold position ---
                    elif phase == DAggerPhase.PAUSED:
                        if events.leader_hold_active.is_set():
                            try:
                                _hold_teleop_at_target(teleop, events.get_leader_hold_target())
                            except Exception as e:
                                logger.warning("Failed to maintain leader hold: %s", e)
                                events.clear_leader_hold()
                        if last_action:
                            robot.send_action(last_action)

                    # --- AUTONOMOUS: policy control (no recording) ---
                    else:
                        obs = robot.get_observation()
                        obs_processed = self._process_observation_and_notify(ctx.processors, obs)

                        if self._handle_warmup(cfg.use_torch_compile, loop_start, control_interval):
                            continue

                        action_dict = send_next_action(obs_processed, obs, ctx, interpolator)
                        if action_dict is not None:
                            self._log_telemetry(obs_processed, action_dict, ctx.runtime)
                            last_action = ctx.processors.robot_action_processor((action_dict, obs))

                    dt = time.perf_counter() - loop_start
                    if (sleep_t := control_interval - dt) > 0:
                        precise_sleep(sleep_t)
                    else:
                        logger.warning(
                            f"Record loop is running slower ({1 / dt:.1f} Hz) than the target FPS ({cfg.fps} Hz). Dataset frames might be dropped and robot control might be unstable. Common causes are: 1) Camera FPS not keeping up 2) Policy inference taking too long 3) CPU starvation"
                        )

            finally:
                logger.info("DAgger corrections-only loop ended — pausing engine")
                engine.pause()
                with contextlib.suppress(Exception):
                    with self._episode_lock:
                        dataset.save_episode()
                    self._needs_push.set()
                    logger.info("Final in-progress episode saved")

    # ------------------------------------------------------------------
    # State-machine transition side-effects
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_transition(
        old_phase: DAggerPhase,
        new_phase: DAggerPhase,
        engine,
        interpolator,
        ctx: RolloutContext,
        prev_action: dict | None,
    ) -> None:
        """Execute side-effects for a validated phase transition, including smooth handovers.

        AUTONOMOUS -> PAUSED (actuated teleop):
            Pause the engine, then drive the leader arm to the follower's last
            commanded position so the operator takes over without a jerk.

        PAUSED -> CORRECTING (non-actuated teleop):
            Slide the follower to the teleop's current pose so the robot meets
            the operator's hand rather than jumping to it on the first frame.

        CORRECTING -> PAUSED:
            Hold the leader at the correction endpoint.  It is released only
            when the operator resumes autonomous policy control.

        PAUSED -> AUTONOMOUS:
            Reset and resume the inference engine, then power off the leader
            until the next pause/sync cycle.
        """
        teleop = ctx.hardware.teleop
        robot = ctx.hardware.robot_wrapper

        logger.info("Phase transition: %s -> %s", old_phase.value, new_phase.value)
        if old_phase == DAggerPhase.AUTONOMOUS and new_phase == DAggerPhase.PAUSED:
            logger.info("Pausing engine - robot holds position")
            engine.pause()

            if _teleop_supports_feedback(teleop) and prev_action is not None:
                # TODO(Maxime): prev_action is in robot action key space (output of robot_action_processor).
                # send_feedback expects teleop feedback key space. For homogeneous setups (e.g. SO-101
                # leader + SO-101 follower) the keys are identical so this works. If the processor pipeline
                # does non-trivial key renaming (e.g. a rename_map on action keys), the interpolation in
                # _teleop_smooth_move_to silently no-ops and the arm doesn't move.
                logger.info("Smooth handover: moving leader arm to follower position")
                _teleop_smooth_move_to(teleop, prev_action)

        elif old_phase == DAggerPhase.PAUSED and new_phase == DAggerPhase.CORRECTING:
            logger.info("Entering correction mode - human teleop control")
            if not _teleop_supports_feedback(teleop) and prev_action is not None:
                logger.info("Smooth handover: sliding follower to teleop position")
                obs = robot.get_observation()
                teleop_action = teleop.get_action()
                processed = ctx.processors.teleop_action_processor((teleop_action, obs))
                target = ctx.processors.robot_action_processor((processed, obs))
                _follower_smooth_move_to(robot, prev_action, target)

            # unlock the teleop for human control
            _release_teleop_for_manual_control(teleop)

        elif old_phase == DAggerPhase.CORRECTING and new_phase == DAggerPhase.PAUSED:
            logger.info("Correction ended - holding leader until policy resumes")
            try:
                _hold_teleop_at_current_position(teleop)
            except Exception as e:
                logger.warning("Failed to hold leader after correction: %s", e)
                _release_teleop_for_manual_control(teleop)

        elif new_phase == DAggerPhase.AUTONOMOUS:
            logger.info("Resuming autonomous mode - resetting engine and interpolator")
            interpolator.reset()
            engine.reset()
            engine.resume()

            # Policy is back in control; make the leader passive/down-powered.
            _power_off_teleop_for_policy_control(teleop)

    # ------------------------------------------------------------------
    # Background push (shared by both modes)
    # ------------------------------------------------------------------

    def _background_push(self, dataset, cfg) -> None:
        """Queue a Hub push on the single-worker executor.

        The executor's max_workers=1 guarantees at most one push runs at
        a time; submitted tasks are queued rather than dropped.  Pushes
        are blocked while the operator is mid-correction to avoid
        uploading a partially-recorded episode.
        """
        if self._push_executor is None:
            return

        if self._events.phase == DAggerPhase.CORRECTING:
            logger.info("Skipping push — correction in progress")
            return

        if self._pending_push is not None and not self._pending_push.done():
            logger.info("Previous push still in progress; queueing next")

        def _push():
            try:
                with self._episode_lock:
                    if safe_push_to_hub(
                        dataset,
                        tags=cfg.dataset.tags if cfg.dataset else None,
                        private=cfg.dataset.private if cfg.dataset else False,
                    ):
                        self._needs_push.clear()
                        logger.info("Background push to hub complete")
            except Exception as e:
                logger.error("Background push failed: %s", e)

        self._pending_push = self._push_executor.submit(_push)
        logger.info("Background push task submitted")
