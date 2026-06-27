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

from dataclasses import dataclass

from ..configs import CameraConfig, ColorMode, Cv2Rotation

__all__ = ["OrbbecCameraConfig", "ColorMode", "Cv2Rotation"]


@CameraConfig.register_subclass("orbbec")
@dataclass
class OrbbecCameraConfig(CameraConfig):
    """Configuration class for Orbbec SDK cameras.

    Example:
    ```python
    OrbbecCameraConfig(serial_number_or_name="0123456789", fps=30, width=1280, height=800)
    ```

    Attributes:
        serial_number_or_name: Orbbec camera serial number, uid, or unique device name.
            If unset and only one Orbbec camera is connected, the first device is used.
        preset: Optional Orbbec device preset to load before querying streams.
            On Gemini 305, use "Dual Color Streams" to expose left/right RGB sensors.
        fps: Requested color stream frames per second.
        width: Requested color stream width.
        height: Requested color stream height.
        color_mode: Output color mode. Defaults to RGB.
        color_sensor: Which Orbbec color sensor to use. Use "color" for the
            main color sensor, or "left"/"right" on devices exposing stereo
            color sensors.
        use_depth: Capture a depth frame alongside color frames.
        rotation: Image rotation setting.
        warmup_s: Time reading frames before returning from connect.
    """

    serial_number_or_name: str | None = None
    preset: str | None = None
    color_mode: ColorMode = ColorMode.RGB
    color_sensor: str = "color"
    use_depth: bool = False
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    warmup_s: int = 1

    def __post_init__(self) -> None:
        self.color_mode = ColorMode(self.color_mode)
        self.rotation = Cv2Rotation(self.rotation)
        self.color_sensor = self.color_sensor.lower()

        if self.color_sensor not in {"color", "main", "left", "right"}:
            raise ValueError(
                f"`color_sensor` must be one of 'color', 'main', 'left', or 'right', got '{self.color_sensor}'."
            )

        if (self.fps is None) ^ (self.width is None) or (self.fps is None) ^ (self.height is None):
            raise ValueError(
                "For Orbbec cameras, fps, width, and height must either all be set, or all be unset."
            )
