#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import pytest

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.scripts.lerobot_train import _validate_output_dir_distributed


def _make_train_config(output_dir):
    return TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="user/repo"),
        policy=DiffusionConfig(device="cpu", push_to_hub=False),
        output_dir=output_dir,
    )


def test_validate_rejects_existing_output_dir(tmp_path):
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    cfg = _make_train_config(output_dir)

    with pytest.raises(FileExistsError):
        cfg.validate()


def test_validate_can_defer_output_dir_check_for_distributed_workers(tmp_path):
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    cfg = _make_train_config(output_dir)

    cfg._validate(check_output_dir=False)

    with pytest.raises(FileExistsError):
        cfg.validate_output_dir()


class _FakeAccelerator:
    def __init__(self, *, is_main_process, num_processes):
        self.is_main_process = is_main_process
        self.num_processes = num_processes


def test_distributed_output_dir_check_broadcasts_main_process_error(tmp_path, monkeypatch):
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    cfg = _make_train_config(output_dir)
    accelerator = _FakeAccelerator(is_main_process=True, num_processes=2)
    broadcasted = []

    monkeypatch.setattr("torch.distributed.is_available", lambda: True)
    monkeypatch.setattr("torch.distributed.is_initialized", lambda: True)

    def fake_broadcast_object_list(result, src):
        broadcasted.append((result[0].copy(), src))

    monkeypatch.setattr("torch.distributed.broadcast_object_list", fake_broadcast_object_list)

    with pytest.raises(FileExistsError):
        _validate_output_dir_distributed(cfg, accelerator)

    assert broadcasted
    assert broadcasted[0][0]["error_type"] == "FileExistsError"
    assert broadcasted[0][1] == 0
