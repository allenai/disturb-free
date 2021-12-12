from allenact_plugins.manipulathor_plugin.manipulathor_constants import ENV_ARGS
from allenact_plugins.manipulathor_plugin.manipulathor_sensors import (
    DepthSensorThor,
    RelativeAgentArmToObjectSensor,
    RelativeObjectToGoalSensor,
    PickedUpObjSensor,
)
from projects.manipulathor_disturb_free.manipulathor_plugin.disturb_sensor import (
    DisturbanceSensor,
)

from projects.manipulathor_disturb_free.manipulathor_plugin.manipulathor_task_samplers import (
    ArmPointNavTaskSampler,
    CamRotateArmPointNavTaskSampler,
)
from projects.manipulathor_disturb_free.armpointnav_baselines.experiments.armpointnav_mixin_ddppo import (
    ArmPointNavMixInPPOConfig,
)
from projects.manipulathor_disturb_free.armpointnav_baselines.experiments.armpointnav_mixin_actorcritic import (
    ArmPointNavAdvancedACConfig,
)
from projects.manipulathor_disturb_free.armpointnav_baselines.experiments.ithor.armpointnav_ithor_base import (
    ArmPointNaviThorBaseConfig,
)
from projects.manipulathor_disturb_free.armpointnav_baselines.models.disturb_pred_loss import (
    DisturbPredictionLoss,
)


class ArmPointNavDepth(
    ArmPointNaviThorBaseConfig, ArmPointNavMixInPPOConfig, ArmPointNavAdvancedACConfig,
):
    """An Object Navigation experiment configuration in iThor with Depth
    input."""

    ACTION_SPACE = (
        # "original"
        "cam_rotate"
    )
    if ACTION_SPACE == "original":
        TASK_SAMPLER = ArmPointNavTaskSampler
    else:
        TASK_SAMPLER = CamRotateArmPointNavTaskSampler

    DISTURB_PEN = (
        # -25.0
        # -20.0
        # -15.0
        # -10.0
        # -5.0
        # -1.0
        0.0
    )
    DISTURB_VIS = False
    DISTURB_FOCAL_GAMMA = 2.0

    BACKBONE = (
        # "simple_cnn"
        "gnresnet18"
    )
    load_pretrained_weights = (
        # True
        False
    )
    weights_path = ()

    coord_system = (
        # "xyz_unsigned"
        "polar_radian"
    )

    goal_space_mode = "man_sel"

    SENSORS = [
        DepthSensorThor(
            height=ArmPointNaviThorBaseConfig.SCREEN_SIZE,
            width=ArmPointNaviThorBaseConfig.SCREEN_SIZE,
            use_normalization=True,
            uuid="depth_lowres",
        ),
        RelativeAgentArmToObjectSensor(coord_system=coord_system,),
        RelativeObjectToGoalSensor(coord_system=coord_system,),
        PickedUpObjSensor(),
        DisturbanceSensor(),
    ]

    MAX_STEPS = 200

    THOR_COMMIT_ID = "2c61316364ab784027b47def9c7b96d203074725"

    def __init__(self):
        super().__init__()

        assert (
            self.CAMERA_WIDTH == 224
            and self.CAMERA_HEIGHT == 224
            and self.VISIBILITY_DISTANCE == 1
            and self.STEP_SIZE == 0.25
        )

        self.ENV_ARGS = {**ENV_ARGS, "renderDepthImage": True}

        if self.THOR_COMMIT_ID is not None:
            self.ENV_ARGS["commit_id"] = self.THOR_COMMIT_ID

    @classmethod
    def tag(cls):
        # some basic assumptions
        assert cls.normalize_advantage == False
        assert cls.add_prev_actions == True
        assert cls.BACKBONE == "gnresnet18"
        # assert cls.load_pretrained_weights == False
        assert cls.coord_system == "polar_radian"
        # assert cls.ACTION_SPACE == "cam_rotate"
        assert cls.INFERENCE_COEF == 0.0

        aux_tag = cls.BACKBONE

        if cls.normalize_advantage:
            aux_tag += "-NormAdv"
        else:
            aux_tag += "-woNormAdv"

        if cls.add_prev_actions:
            aux_tag += "-wact"
        else:
            aux_tag += "-woact"

        aux_tag += "-" + cls.goal_space_mode
        aux_tag += "-" + cls.coord_system

        if cls.load_pretrained_weights:
            aux_tag += "-finetune"
        else:
            aux_tag += "-scratch"

        aux_tag += f"-disturb_pen{abs(cls.DISTURB_PEN)}"
        if cls.DISTURB_VIS:
            aux_tag += "_vis"
        else:
            aux_tag += "_all"

        if cls.AUXILIARY_UUIDS is None or (
            isinstance(cls.AUXILIARY_UUIDS, list) and len(cls.AUXILIARY_UUIDS) == 0
        ):
            aux_tag += "-no_aux"
        else:
            aux_tag += "-" + "-".join(cls.AUXILIARY_UUIDS)
            if DisturbPredictionLoss.UUID in cls.AUXILIARY_UUIDS:
                aux_tag += "-gamma" + str(cls.DISTURB_FOCAL_GAMMA)
            if len(cls.AUXILIARY_UUIDS) > 1 and cls.multiple_beliefs:
                aux_tag += "-mulbelief-" + cls.beliefs_fusion

        return aux_tag
