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
Provides the OrbbecCamera class for capturing frames from Orbbec SDK cameras.
"""

import logging
import time
from threading import Event, Lock, Thread
from typing import TYPE_CHECKING, Any

import cv2  # type: ignore  # TODO: add type stubs for OpenCV
import numpy as np  # type: ignore  # TODO: add type stubs for numpy
from numpy.typing import NDArray  # type: ignore  # TODO: add type stubs for numpy.typing

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError
from lerobot.utils.import_utils import _pyorbbecsdk_available

if TYPE_CHECKING or _pyorbbecsdk_available:
    import pyorbbecsdk as ob
else:
    ob = None

from ..camera import Camera
from ..configs import ColorMode
from ..utils import get_cv2_rotation
from .configuration_orbbec import OrbbecCameraConfig

logger = logging.getLogger(__name__)


def _require_orbbec_sdk() -> None:
    if not _pyorbbecsdk_available:
        raise ImportError(
            "'pyorbbecsdk2' is required for Orbbec cameras. Install it in this environment with: "
            "uv pip install pyorbbecsdk2  # or: python -m pip install pyorbbecsdk2"
        )


_SHARED_CAPTURE_LOCK = Lock()
_SHARED_CAPTURES: dict[tuple[Any, ...], "_SharedOrbbecCapture"] = {}


def _stream_config_mode(color_sensor: str) -> str:
    return "dual_color" if color_sensor in {"left", "right"} else "color"


class _SharedOrbbecCapture:
    """One Orbbec SDK pipeline shared by all LeRobot camera views on the same device."""

    def __init__(
        self,
        key: tuple[Any, ...],
        device: Any,
        *,
        width: int | None,
        height: int | None,
        fps: int | None,
        mode: str,
        use_depth: bool,
    ) -> None:
        self.key = key
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.mode = mode
        self.use_depth = use_depth

        self.pipeline: ob.Pipeline | None = None
        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self.latest_frame_set: Any | None = None
        self.latest_timestamp: float | None = None
        self.new_frame_event: Event = Event()
        self.ref_count = 0

    @property
    def is_running(self) -> bool:
        return self.pipeline is not None and self.thread is not None and self.thread.is_alive()

    def acquire(self) -> None:
        self.ref_count += 1
        if self.pipeline is None:
            self._start()

    def release(self) -> None:
        self.ref_count -= 1
        if self.ref_count <= 0:
            self._stop()

    def _start(self) -> None:
        self.pipeline = ob.Pipeline(self.device)
        sdk_config = ob.Config()

        if self.mode == "dual_color":
            self._enable_video_stream(
                sdk_config,
                ob.OBSensorType.LEFT_COLOR_SENSOR,
                ob.OBStreamType.LEFT_COLOR_STREAM,
            )
            self._enable_video_stream(
                sdk_config,
                ob.OBSensorType.RIGHT_COLOR_SENSOR,
                ob.OBStreamType.RIGHT_COLOR_STREAM,
            )
        else:
            self._enable_video_stream(
                sdk_config,
                ob.OBSensorType.COLOR_SENSOR,
                ob.OBStreamType.COLOR_STREAM,
            )

        if self.use_depth:
            self._enable_video_stream(
                sdk_config,
                ob.OBSensorType.DEPTH_SENSOR,
                ob.OBStreamType.DEPTH_STREAM,
            )

        self._configure_frame_aggregation(sdk_config)
        self.pipeline.start(sdk_config)
        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, args=(), name=f"OrbbecSharedCapture{self.key}")
        self.thread.daemon = True
        self.thread.start()
        time.sleep(0.1)

    def _configure_frame_aggregation(self, sdk_config: Any) -> None:
        if self.mode != "dual_color" and not self.use_depth:
            return

        aggregate_mode = getattr(ob.OBFrameAggregateOutputMode, "FULL_FRAME_REQUIRE", None)
        if aggregate_mode is None:
            return

        try:
            sdk_config.set_frame_aggregate_output_mode(aggregate_mode)
        except Exception as e:
            logger.debug("Failed to enable Orbbec full-frame aggregation for %s: %s", self.key, e)

    def _enable_video_stream(self, sdk_config: Any, sensor_type: Any, stream_type: Any) -> None:
        if self.pipeline is None:
            raise RuntimeError("Orbbec pipeline must be initialized before enabling streams.")

        if self.mode == "dual_color":
            try:
                sdk_config.enable_stream(sensor_type)
                return
            except Exception as e:
                logger.debug("Failed to enable Orbbec stream by sensor type %s: %s", sensor_type, e)

        if self.width and self.height and self.fps:
            try:
                profile_list = self.pipeline.get_stream_profile_list(sensor_type)
                profile = profile_list.get_video_stream_profile(
                    self.width,
                    self.height,
                    ob.OBFormat.UNKNOWN_FORMAT,
                    self.fps,
                )
                sdk_config.enable_stream(profile)
                return
            except Exception:
                sdk_config.enable_video_stream(
                    sensor_type,
                    self.width,
                    self.height,
                    self.fps,
                    ob.OBFormat.UNKNOWN_FORMAT,
                )
                return

        profile_list = self.pipeline.get_stream_profile_list(sensor_type)
        sdk_config.enable_stream(profile_list.get_default_video_stream_profile())

    def _read_loop(self) -> None:
        if self.stop_event is None or self.pipeline is None:
            raise RuntimeError("Orbbec shared capture was not initialized correctly.")

        failure_count = 0
        while not self.stop_event.is_set():
            try:
                frame_set = self.pipeline.wait_for_frames(10000)
                if frame_set is None:
                    raise RuntimeError("Orbbec shared capture received no frame set.")
                if not self._has_required_frames(frame_set):
                    continue

                with self.frame_lock:
                    self.latest_frame_set = frame_set
                    self.latest_timestamp = time.perf_counter()
                self.new_frame_event.set()
                failure_count = 0

            except Exception as e:
                if self.stop_event.is_set():
                    break
                if failure_count <= 10:
                    failure_count += 1
                    logger.warning("Error reading Orbbec shared frame set for %s: %s", self.key, e)
                else:
                    raise RuntimeError("Orbbec shared capture exceeded maximum read failures.") from e

    def _has_required_frames(self, frame_set: Any) -> bool:
        if self.mode == "dual_color":
            if frame_set.get_left_color_frame() is None or frame_set.get_right_color_frame() is None:
                return False
        elif frame_set.get_color_frame() is None:
            return False

        return not (self.use_depth and frame_set.get_depth_frame() is None)

    def wait_latest(self, timeout_ms: float) -> tuple[Any, float]:
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError("Orbbec shared capture read thread is not running.")

        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(
                f"Timed out waiting for frame from Orbbec shared capture after {timeout_ms} ms. "
                f"Read thread alive: {self.thread.is_alive()}."
            )

        with self.frame_lock:
            frame_set = self.latest_frame_set
            timestamp = self.latest_timestamp

        if frame_set is None or timestamp is None:
            raise RuntimeError("Internal error: Orbbec shared capture has no frame set.")

        return frame_set, timestamp

    def read_latest(self, max_age_ms: int) -> tuple[Any, float]:
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError("Orbbec shared capture read thread is not running.")

        with self.frame_lock:
            frame_set = self.latest_frame_set
            timestamp = self.latest_timestamp

        if frame_set is None or timestamp is None:
            raise RuntimeError("Orbbec shared capture has not captured any frames yet.")

        age_ms = (time.perf_counter() - timestamp) * 1e3
        if age_ms > max_age_ms:
            raise TimeoutError(
                f"Orbbec shared capture latest frame is too old: {age_ms:.1f} ms "
                f"(max allowed: {max_age_ms} ms)."
            )

        return frame_set, timestamp

    def _stop(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        self.thread = None
        self.stop_event = None

        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None

        with self.frame_lock:
            self.latest_frame_set = None
            self.latest_timestamp = None
            self.new_frame_event.clear()


class OrbbecCamera(Camera):
    """
    Manages Orbbec SDK cameras using the pyorbbecsdk2 Python bindings.

    The public behavior mirrors the other LeRobot camera backends: `read()` and
    `async_read()` return a color image as a numpy array in HWC layout.
    """

    def __init__(self, config: OrbbecCameraConfig):
        _require_orbbec_sdk()
        super().__init__(config)

        self.config = config
        self.serial_number_or_name = config.serial_number_or_name
        self.preset = config.preset
        self.fps = config.fps
        self.color_mode = config.color_mode
        self.color_sensor = config.color_sensor
        self.use_depth = config.use_depth
        self.warmup_s = config.warmup_s

        self.device: ob.Device | None = None
        self.pipeline: ob.Pipeline | None = None
        self.shared_capture: _SharedOrbbecCapture | None = None

        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self.latest_color_frame: NDArray[Any] | None = None
        self.latest_depth_frame: NDArray[Any] | None = None
        self.latest_timestamp: float | None = None
        self.new_frame_event: Event = Event()

        self.rotation: int | None = get_cv2_rotation(config.rotation)

        if self.height and self.width:
            self.capture_width, self.capture_height = self.width, self.height
            if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                self.capture_width, self.capture_height = self.height, self.width

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.serial_number_or_name or 'default'}, {self.color_sensor})"

    @property
    def is_connected(self) -> bool:
        return self.shared_capture is not None and self.shared_capture.is_running

    @staticmethod
    def find_cameras(preset: str | None = None) -> list[dict[str, Any]]:
        """
        Detects available Orbbec cameras connected to the system.
        """
        _require_orbbec_sdk()

        found_cameras_info = []
        context = ob.Context()
        device_list = context.query_devices()

        color_sensors = (
            ("color", "Color", ob.OBSensorType.COLOR_SENSOR),
            ("left", "Left Color", ob.OBSensorType.LEFT_COLOR_SENSOR),
            ("right", "Right Color", ob.OBSensorType.RIGHT_COLOR_SENSOR),
        )

        for index in range(device_list.get_count()):
            device = device_list.get_device_by_index(index)
            info = device.get_device_info()
            if preset is not None:
                OrbbecCamera._load_device_preset(device, preset)
            pipeline = ob.Pipeline(device)

            for sensor_key, sensor_label, sensor_type in color_sensors:
                try:
                    profile_list = pipeline.get_stream_profile_list(sensor_type)
                except Exception as e:
                    logger.debug(
                        "Failed to read Orbbec %s profile list for %s: %s",
                        sensor_label,
                        info.get_name(),
                        e,
                    )
                    continue

                camera_info: dict[str, Any] = {
                    "name": f"{info.get_name()} {sensor_label}",
                    "type": "Orbbec",
                    "id": info.get_serial_number(),
                    "uid": info.get_uid(),
                    "color_sensor": sensor_key,
                    "firmware_version": info.get_firmware_version(),
                    "hardware_version": info.get_hardware_version(),
                    "connection_type": info.get_connection_type(),
                    "pid": f"0x{info.get_pid():04X}",
                    "vid": f"0x{info.get_vid():04X}",
                }

                profile = profile_list.get_default_video_stream_profile()
                camera_info["default_stream_profile"] = {
                    "stream_type": sensor_label,
                    "format": profile.get_format().name,
                    "width": profile.get_width(),
                    "height": profile.get_height(),
                    "fps": profile.get_fps(),
                }

                found_cameras_info.append(camera_info)

        return found_cameras_info

    @staticmethod
    def _load_device_preset(device: Any, preset: str) -> None:
        try:
            current = device.get_current_preset_name()
            if current != preset:
                device.load_preset(preset)
                logger.info("Loaded Orbbec preset '%s' (previously '%s').", preset, current)
        except Exception as e:
            raise RuntimeError(f"Failed to load Orbbec preset '{preset}'.") from e

    def _color_sensor_type(self) -> Any:
        if self.color_sensor in {"color", "main"}:
            return ob.OBSensorType.COLOR_SENSOR
        if self.color_sensor == "left":
            return ob.OBSensorType.LEFT_COLOR_SENSOR
        if self.color_sensor == "right":
            return ob.OBSensorType.RIGHT_COLOR_SENSOR
        raise ValueError(f"Unsupported Orbbec color_sensor: {self.color_sensor}")

    def _color_stream_type(self) -> Any:
        if self.color_sensor in {"color", "main"}:
            return ob.OBStreamType.COLOR_STREAM
        if self.color_sensor == "left":
            return ob.OBStreamType.LEFT_COLOR_STREAM
        if self.color_sensor == "right":
            return ob.OBStreamType.RIGHT_COLOR_STREAM
        raise ValueError(f"Unsupported Orbbec color_sensor: {self.color_sensor}")

    def _find_device(self) -> ob.Device:
        context = ob.Context()
        device_list = context.query_devices()
        count = device_list.get_count()

        if count == 0:
            raise ConnectionError("No Orbbec camera found. Run `lerobot-find-cameras orbbec`.")

        if self.serial_number_or_name is None:
            if count > 1:
                identifiers = [device_list.get_device_serial_number_by_index(i) for i in range(count)]
                raise ValueError(
                    "Multiple Orbbec cameras found. Please set serial_number_or_name. "
                    f"Available serial numbers: {identifiers}"
                )
            return device_list.get_device_by_index(0)

        matches = []
        for index in range(count):
            name = device_list.get_device_name_by_index(index)
            serial = device_list.get_device_serial_number_by_index(index)
            uid = device_list.get_device_uid_by_index(index)
            user_name = device_list.get_device_user_name_by_index(index)

            if self.serial_number_or_name in {serial, uid, name, user_name}:
                matches.append(index)

        if not matches:
            available = [
                {
                    "name": device_list.get_device_name_by_index(i),
                    "serial_number": device_list.get_device_serial_number_by_index(i),
                    "uid": device_list.get_device_uid_by_index(i),
                    "user_name": device_list.get_device_user_name_by_index(i),
                }
                for i in range(count)
            ]
            raise ValueError(
                f"No Orbbec camera found with serial/name/uid '{self.serial_number_or_name}'. "
                f"Available cameras: {available}"
            )

        if len(matches) > 1:
            raise ValueError(
                f"Multiple Orbbec cameras matched '{self.serial_number_or_name}'. Use a unique serial number."
            )

        return device_list.get_device_by_index(matches[0])

    @check_if_already_connected
    def connect(self, warmup: bool = True) -> None:
        self.device = self._find_device()
        if self.preset is not None:
            self._load_device_preset(self.device, self.preset)

        self._configure_capture_settings()
        self.shared_capture = self._get_shared_capture()
        try:
            self.shared_capture.acquire()
        except Exception as e:
            self.shared_capture = None
            self.device = None
            raise ConnectionError(
                f"Failed to open {self}. Run `lerobot-find-cameras orbbec` to find available cameras."
            ) from e

        if warmup and self.warmup_s > 0:
            start_time = time.time()
            while time.time() - start_time < self.warmup_s:
                self.async_read(timeout_ms=self.warmup_s * 1000)
                time.sleep(0.1)
            with self.frame_lock:
                if self.latest_color_frame is None or self.use_depth and self.latest_depth_frame is None:
                    raise ConnectionError(f"{self} failed to capture frames during warmup.")

        logger.info(f"{self} connected.")

    def _configure_pipeline(self, sdk_config: Any) -> None:
        if self.pipeline is None:
            raise RuntimeError(f"{self}: pipeline must be initialized before use.")

        if self.width and self.height and self.fps:
            try:
                profile_list = self.pipeline.get_stream_profile_list(self._color_sensor_type())
                color_profile = profile_list.get_video_stream_profile(
                    self.capture_width,
                    self.capture_height,
                    ob.OBFormat.UNKNOWN_FORMAT,
                    self.fps,
                )
                sdk_config.enable_stream(color_profile)
            except Exception:
                sdk_config.enable_video_stream(
                    self._color_stream_type(),
                    self.capture_width,
                    self.capture_height,
                    self.fps,
                    ob.OBFormat.UNKNOWN_FORMAT,
                )
        else:
            profile_list = self.pipeline.get_stream_profile_list(self._color_sensor_type())
            sdk_config.enable_stream(profile_list.get_default_video_stream_profile())

        if self.use_depth:
            depth_profile_list = self.pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)
            if self.width and self.height and self.fps:
                try:
                    depth_profile = depth_profile_list.get_video_stream_profile(
                        self.capture_width,
                        self.capture_height,
                        ob.OBFormat.UNKNOWN_FORMAT,
                        self.fps,
                    )
                except Exception:
                    depth_profile = depth_profile_list.get_default_video_stream_profile()
            else:
                depth_profile = depth_profile_list.get_default_video_stream_profile()
            sdk_config.enable_stream(depth_profile)

    def _shared_capture_key(self) -> tuple[Any, ...]:
        if self.device is None:
            raise DeviceNotConnectedError(f"{self} device is not initialized")

        info = self.device.get_device_info()
        return (
            info.get_serial_number(),
            self.preset,
            _stream_config_mode(self.color_sensor),
            self.capture_width if self.width is not None else None,
            self.capture_height if self.height is not None else None,
            self.fps,
            self.use_depth,
        )

    def _get_shared_capture(self) -> _SharedOrbbecCapture:
        key = self._shared_capture_key()

        with _SHARED_CAPTURE_LOCK:
            capture = _SHARED_CAPTURES.get(key)
            if capture is None:
                if self.device is None:
                    raise DeviceNotConnectedError(f"{self} device is not initialized")
                capture = _SharedOrbbecCapture(
                    key,
                    self.device,
                    width=self.capture_width if self.width is not None else None,
                    height=self.capture_height if self.height is not None else None,
                    fps=self.fps,
                    mode=_stream_config_mode(self.color_sensor),
                    use_depth=self.use_depth,
                )
                _SHARED_CAPTURES[key] = capture

        return capture

    def _configure_capture_settings(self) -> None:
        if self.device is None:
            raise DeviceNotConnectedError(f"{self} device is not initialized")

        if self.width is not None and self.height is not None and self.fps is not None:
            return

        pipeline = ob.Pipeline(self.device)
        profile_list = pipeline.get_stream_profile_list(self._color_sensor_type())
        profile = profile_list.get_default_video_stream_profile()

        if self.fps is None:
            self.fps = profile.get_fps()

        if self.width is None or self.height is None:
            actual_width = int(profile.get_width())
            actual_height = int(profile.get_height())
            if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                self.width, self.height = actual_height, actual_width
                self.capture_width, self.capture_height = actual_width, actual_height
            else:
                self.width, self.height = actual_width, actual_height
                self.capture_width, self.capture_height = actual_width, actual_height

    def _read_from_hardware(self) -> Any:
        if self.shared_capture is None:
            raise DeviceNotConnectedError(f"{self} shared capture is not initialized")

        frame_set, _ = self.shared_capture.wait_latest(10000)
        return frame_set

    @check_if_not_connected
    def read(self, color_mode: ColorMode | None = None) -> NDArray[Any]:
        start_time = time.perf_counter()

        if color_mode is not None:
            logger.warning(
                f"{self} read() color_mode parameter is deprecated and will be removed in future versions."
            )

        frame = self.async_read(timeout_ms=10000)

        read_duration_ms = (time.perf_counter() - start_time) * 1e3
        logger.debug(f"{self} read took: {read_duration_ms:.1f}ms")

        return frame

    @check_if_not_connected
    def read_depth(self, timeout_ms: int = 200) -> NDArray[Any]:
        if timeout_ms:
            logger.warning(
                f"{self} read_depth() timeout_ms parameter is deprecated and will be removed in future versions."
            )

        if not self.use_depth:
            raise RuntimeError(
                f"Failed to capture depth frame '.read_depth()'. Depth stream is not enabled for {self}."
            )

        _ = self.async_read(timeout_ms=10000)

        with self.frame_lock:
            depth_map = self.latest_depth_frame

        if depth_map is None:
            raise RuntimeError("No depth frame available. Ensure camera is streaming.")

        return depth_map

    def _color_frame_to_rgb(self, color_frame: Any) -> NDArray[Any]:
        width = color_frame.get_width()
        height = color_frame.get_height()
        color_format = color_frame.get_format()
        data = np.asanyarray(color_frame.get_data())

        if color_format == ob.OBFormat.RGB:
            return np.resize(data, (height, width, 3))
        if color_format == ob.OBFormat.BGR:
            bgr = np.resize(data, (height, width, 3))
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if color_format == ob.OBFormat.MJPG:
            bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"{self} failed to decode MJPG color frame.")
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if color_format == ob.OBFormat.YUYV:
            yuyv = np.resize(data, (height, width, 2))
            return cv2.cvtColor(yuyv, cv2.COLOR_YUV2RGB_YUYV)
        if color_format == ob.OBFormat.UYVY:
            uyvy = np.resize(data, (height, width, 2))
            return cv2.cvtColor(uyvy, cv2.COLOR_YUV2RGB_UYVY)
        if color_format == ob.OBFormat.NV12:
            nv12 = np.resize(data, (height * 3 // 2, width))
            return cv2.cvtColor(nv12, cv2.COLOR_YUV2RGB_NV12)
        if color_format == ob.OBFormat.NV21:
            nv21 = np.resize(data, (height * 3 // 2, width))
            return cv2.cvtColor(nv21, cv2.COLOR_YUV2RGB_NV21)
        if color_format == ob.OBFormat.I420:
            i420 = np.resize(data, (height * 3 // 2, width))
            return cv2.cvtColor(i420, cv2.COLOR_YUV2RGB_I420)

        raise RuntimeError(f"{self} unsupported Orbbec color format: {color_format}.")

    def _postprocess_color_image(self, image: NDArray[Any]) -> NDArray[Any]:
        if self.color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"Invalid color mode '{self.color_mode}'. Expected {ColorMode.RGB} or {ColorMode.BGR}."
            )

        h, w, c = image.shape
        if h != self.capture_height or w != self.capture_width:
            raise RuntimeError(
                f"{self} frame width={w} or height={h} do not match configured width={self.capture_width} or height={self.capture_height}."
            )
        if c != 3:
            raise RuntimeError(f"{self} frame channels={c} do not match expected 3 channels (RGB/BGR).")

        processed_image = image
        if self.color_mode == ColorMode.BGR:
            processed_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]:
            processed_image = cv2.rotate(processed_image, self.rotation)

        return processed_image

    def _postprocess_depth_image(self, depth_frame: Any) -> NDArray[Any]:
        width = depth_frame.get_width()
        height = depth_frame.get_height()
        depth_scale = depth_frame.get_depth_scale()
        depth_map = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((height, width))
        depth_map = (depth_map.astype(np.float32) * depth_scale).astype(np.uint16)

        if height != self.capture_height or width != self.capture_width:
            logger.debug(
                "%s depth width=%s height=%s differs from color width=%s height=%s.",
                self,
                width,
                height,
                self.capture_width,
                self.capture_height,
            )

        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]:
            depth_map = cv2.rotate(depth_map, self.rotation)

        return depth_map

    def _read_loop(self) -> None:
        if self.stop_event is None:
            raise RuntimeError(f"{self}: stop_event is not initialized before starting read loop.")

        failure_count = 0
        while not self.stop_event.is_set():
            try:
                frame_set = self._read_from_hardware()
                color_frame = self._get_color_frame(frame_set)
                if color_frame is None:
                    raise RuntimeError(f"{self} frame set does not contain a color frame.")

                rgb_image = self._color_frame_to_rgb(color_frame)
                processed_color_frame = self._postprocess_color_image(rgb_image)

                if self.use_depth:
                    depth_frame_raw = frame_set.get_depth_frame()
                    if depth_frame_raw is None:
                        raise RuntimeError(f"{self} frame set does not contain a depth frame.")
                    processed_depth_frame = self._postprocess_depth_image(depth_frame_raw)

                capture_time = time.perf_counter()

                with self.frame_lock:
                    self.latest_color_frame = processed_color_frame
                    if self.use_depth:
                        self.latest_depth_frame = processed_depth_frame
                    self.latest_timestamp = capture_time
                self.new_frame_event.set()
                failure_count = 0

            except DeviceNotConnectedError:
                break
            except Exception as e:
                if failure_count <= 10:
                    failure_count += 1
                    logger.warning(f"Error reading frame in background thread for {self}: {e}")
                else:
                    raise RuntimeError(f"{self} exceeded maximum consecutive read failures.") from e

    def _get_color_frame(self, frame_set: Any) -> Any:
        if self.color_sensor in {"color", "main"}:
            return frame_set.get_color_frame()
        if self.color_sensor == "left":
            return frame_set.get_left_color_frame()
        if self.color_sensor == "right":
            return frame_set.get_right_color_frame()
        raise ValueError(f"Unsupported Orbbec color_sensor: {self.color_sensor}")

    def _start_read_thread(self) -> None:
        self._stop_read_thread()

        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, args=(), name=f"{self}_read_loop")
        self.thread.daemon = True
        self.thread.start()
        time.sleep(0.1)

    def _stop_read_thread(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        self.thread = None
        self.stop_event = None

        with self.frame_lock:
            self.latest_color_frame = None
            self.latest_depth_frame = None
            self.latest_timestamp = None
            self.new_frame_event.clear()

    @check_if_not_connected
    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        if self.shared_capture is None:
            raise DeviceNotConnectedError(f"{self} shared capture is not initialized")

        frame_set, timestamp = self.shared_capture.wait_latest(timeout_ms)
        frame = self._process_frame_set(frame_set)
        with self.frame_lock:
            self.latest_timestamp = timestamp
        return frame

    @check_if_not_connected
    def read_latest(self, max_age_ms: int = 500) -> NDArray[Any]:
        if self.shared_capture is None:
            raise DeviceNotConnectedError(f"{self} shared capture is not initialized")

        frame_set, timestamp = self.shared_capture.read_latest(max_age_ms)
        frame = self._process_frame_set(frame_set)
        with self.frame_lock:
            self.latest_timestamp = timestamp
        return frame

    def _process_frame_set(self, frame_set: Any) -> NDArray[Any]:
        color_frame = self._get_color_frame(frame_set)
        if color_frame is None:
            raise RuntimeError(f"{self} frame set does not contain a color frame.")

        rgb_image = self._color_frame_to_rgb(color_frame)
        processed_color_frame = self._postprocess_color_image(rgb_image)

        with self.frame_lock:
            self.latest_color_frame = processed_color_frame
            if self.use_depth:
                depth_frame_raw = frame_set.get_depth_frame()
                if depth_frame_raw is None:
                    raise RuntimeError(f"{self} frame set does not contain a depth frame.")
                self.latest_depth_frame = self._postprocess_depth_image(depth_frame_raw)

        return processed_color_frame

    def disconnect(self) -> None:
        if not self.is_connected and self.shared_capture is None:
            raise DeviceNotConnectedError(
                f"Attempted to disconnect {self}, but it appears already disconnected."
            )

        if self.shared_capture is not None:
            key = self.shared_capture.key
            with _SHARED_CAPTURE_LOCK:
                self.shared_capture.release()
                if self.shared_capture.ref_count <= 0:
                    _SHARED_CAPTURES.pop(key, None)
            self.shared_capture = None

        self.pipeline = None
        self.device = None

        with self.frame_lock:
            self.latest_color_frame = None
            self.latest_depth_frame = None
            self.latest_timestamp = None
            self.new_frame_event.clear()

        logger.info(f"{self} disconnected.")
