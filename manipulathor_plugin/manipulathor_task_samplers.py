"""Task Samplers for the task of ArmPointNav"""

from allenact_plugins.manipulathor_plugin.manipulathor_task_samplers import (
    ArmPointNavTaskSampler as RawArmPointNavTaskSampler,
)

from projects.manipulathor_disturb_free.manipulathor_plugin.manipulathor_task import (
    ArmPointNavTask,
    RotateArmPointNavTask,
    CamRotateArmPointNavTask,
)


class ArmPointNavTaskSampler(RawArmPointNavTaskSampler):
    _TASK_TYPE = ArmPointNavTask


class RotateArmPointNavTaskSampler(ArmPointNavTaskSampler):
    _TASK_TYPE = RotateArmPointNavTask


class CamRotateArmPointNavTaskSampler(ArmPointNavTaskSampler):
    _TASK_TYPE = CamRotateArmPointNavTask
