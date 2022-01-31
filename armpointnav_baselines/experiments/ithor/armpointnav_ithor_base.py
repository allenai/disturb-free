from abc import ABC
import torch
from allenact_plugins.manipulathor_plugin.armpointnav_constants import (
    TRAIN_OBJECTS,
    TEST_OBJECTS,
)
from projects.manipulathor_disturb_free.armpointnav_baselines.experiments.armpointnav_thor_base import (
    ArmPointNavThorBaseConfig,
)


class ArmPointNaviThorBaseConfig(ArmPointNavThorBaseConfig, ABC):
    """The base config for all iTHOR ObjectNav experiments."""

    THOR_COMMIT_ID = "a84dd29471ec2201f583de00257d84fac1a03de2"

    NUM_PROCESSES = 19
    TRAIN_GPU_IDS = list(range(torch.cuda.device_count()))
    SAMPLER_GPU_IDS = TRAIN_GPU_IDS
    VALID_GPU_IDS = [torch.cuda.device_count() - 1]
    TEST_GPU_IDS = [torch.cuda.device_count() - 1]

    # add all the arguments here
    TOTAL_NUMBER_SCENES = 30

    TRAIN_SCENES = [
        "FloorPlan{}_physics".format(str(i))
        for i in range(1, TOTAL_NUMBER_SCENES + 1)
        if (i % 3 == 1 or i % 3 == 0) and i != 28
    ]  # last scenes are really bad, then it is 19 training scenes actually
    TEST_SCENES = [
        "FloorPlan{}_physics".format(str(i))
        for i in range(1, TOTAL_NUMBER_SCENES + 1)
        if i % 3 == 2 and i % 6 == 2
    ]  # 5 scenes
    VALID_SCENES = [
        "FloorPlan{}_physics".format(str(i))
        for i in range(1, TOTAL_NUMBER_SCENES + 1)
        if i % 3 == 2 and i % 6 == 5
    ]  # 5 scenes

    ALL_SCENES = TRAIN_SCENES + TEST_SCENES + VALID_SCENES

    assert (
        len(ALL_SCENES) == TOTAL_NUMBER_SCENES - 1
        and len(set(ALL_SCENES)) == TOTAL_NUMBER_SCENES - 1
    )

    OBJECT_TYPES = tuple(sorted(TRAIN_OBJECTS))

    UNSEEN_OBJECT_TYPES = tuple(sorted(TEST_OBJECTS))
