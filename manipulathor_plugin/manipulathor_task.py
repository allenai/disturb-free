from allenact_plugins.manipulathor_plugin.manipulathor_tasks import (
    ArmPointNavTask as RawArmPointNavTask,
    RotateArmPointNavTask as RawRotateArmPointNavTask,
    CamRotateArmPointNavTask as RawCamRotateArmPointNavTask,
)

import pandas as pd

DF = pd.read_csv(
    "projects/manipulathor_disturb_free/manipulathor_plugin/vibrations.csv"
)

# use dict is much faster to query than dataframe
VIBRATION_DISTANCES = {}
for i in range(DF.shape[0]):
    VIBRATION_DISTANCES[DF.at[i, "scene"] + "-" + DF.at[i, "object"]] = DF.at[i, "dist"]


class ArmPointNavTask(RawArmPointNavTask):
    _vibration_dist_dict = VIBRATION_DISTANCES


class RotateArmPointNavTask(RawRotateArmPointNavTask):
    _vibration_dist_dict = VIBRATION_DISTANCES


class CamRotateArmPointNavTask(RawCamRotateArmPointNavTask):
    _vibration_dist_dict = VIBRATION_DISTANCES
