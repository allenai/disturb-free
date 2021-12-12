"""Utility classes and functions for sensory inputs used by the models."""
from typing import Any, Union, Optional

import gym
import numpy as np
from allenact.base_abstractions.sensor import Sensor
from allenact.base_abstractions.task import Task
from allenact.utils.misc_utils import prepare_locals_for_super

from allenact_plugins.manipulathor_plugin.manipulathor_environment import (
    ManipulaTHOREnvironment,
)
from projects.manipulathor_disturb_free.manipulathor_plugin.manipulathor_task import DF


class DisturbanceSensor(Sensor):
    def __init__(self, uuid: str = "disturbance_binary", **kwargs: Any):
        observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.bool)
        super().__init__(**prepare_locals_for_super(locals()))

        raw = (
            DF.groupby("scene").sum() / 200
        )  # averge vibration per step on all the objects
        raw = raw.clip(lower=0.001)
        self.vibration_distances_scene = raw.to_dict()["dist"]

    def get_observation(
        self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        scene_id = env.scene_name.split("_")[0]
        thres = self.vibration_distances_scene[scene_id]

        disturbance_distance = task.current_penalized_distance

        result = disturbance_distance >= thres  # bool

        return result
