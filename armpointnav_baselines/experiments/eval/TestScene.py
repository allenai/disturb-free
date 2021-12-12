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

from allenact.embodiedai.aux_losses.losses import (
    InverseDynamicsLoss,
    CPCA16Loss,
)
from projects.manipulathor_disturb_free.armpointnav_baselines.models.disturb_pred_loss import (
    DisturbPredictionLoss,
)

from allenact.embodiedai.models.fusion_models import AverageFusion
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
from allenact_plugins.manipulathor_plugin.manipulathor_viz import (
    ImageVisualizer,
    TestMetricLogger,
)
from typing import Optional


class TestScene(
    ArmPointNaviThorBaseConfig, ArmPointNavMixInPPOConfig, ArmPointNavAdvancedACConfig,
):
    VISUALIZERS = [
        # lambda exp_name: ImageVisualizer(exp_name,
        #     add_top_down_view=True
        # ),
        # TestMetricLogger,
    ]

    NUM_TASK_PER_SCENE = None  # min(60, 10)

    NUMBER_OF_TEST_PROCESS = 5
    TEST_GPU_IDS = [0]  # has to be one gpu

    TEST_SCENES_DICT = {
        "ValidScene": ArmPointNaviThorBaseConfig.VALID_SCENES,
        "TestScene": ArmPointNaviThorBaseConfig.TEST_SCENES,
    }
    OBJECT_TYPES_DICT = {
        "novel": ArmPointNaviThorBaseConfig.UNSEEN_OBJECT_TYPES,
        "seen": ArmPointNaviThorBaseConfig.OBJECT_TYPES,
        "all": ArmPointNaviThorBaseConfig.OBJECT_TYPES
        + ArmPointNaviThorBaseConfig.UNSEEN_OBJECT_TYPES,
    }
    TEST_SCENES_NAME = (
        "ValidScene"
        # "TestScene"
    )
    OBJECT_TYPES_NAME = (
        "novel"
        # "seen"
        # "all"
    )
    TEST_SCENES = TEST_SCENES_DICT[TEST_SCENES_NAME]
    OBJECT_TYPES = OBJECT_TYPES_DICT[OBJECT_TYPES_NAME]

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
        -15.0
        # -10.0
        # -5.0
        # -1.0
        # 0.0
    )
    DISTURB_VIS = False
    INFERENCE_COEF = (
        # 0.1
        0.0
    )

    # selected auxiliary uuids
    AUXILIARY_UUIDS = [
        # InverseDynamicsLoss.UUID,
        # CPCA16Loss.UUID,
        DisturbPredictionLoss.UUID,
    ]
    multiple_beliefs = False  # True #
    beliefs_fusion = None

    MAX_STEPS = 200

    BACKBONE = (
        # "simple_cnn"
        "gnresnet18"
    )
    add_prev_actions = (
        True
        # False
    )

    coord_system = (
        # "xyz_unsigned" # used in CVPR 2021 paper
        "polar_radian"  # used in our method
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

    THOR_COMMIT_ID = "2c61316364ab784027b47def9c7b96d203074725"

    def __init__(self, test_ckpt: Optional[str] = None):
        super().__init__()
        self.test_ckpt = test_ckpt

        assert (
            self.CAMERA_WIDTH == 224
            and self.CAMERA_HEIGHT == 224
            and self.VISIBILITY_DISTANCE == 1
            and self.STEP_SIZE == 0.25
        )

        self.ENV_ARGS = ENV_ARGS

        depth_uuid = next(
            (s.uuid for s in self.SENSORS if isinstance(s, DepthSensorThor)), None
        )
        if depth_uuid is not None:
            self.ENV_ARGS["renderDepthImage"] = True

        if self.THOR_COMMIT_ID is not None:
            self.ENV_ARGS["commit_id"] = self.THOR_COMMIT_ID

    @classmethod
    def tag(cls):
        assert cls.NUM_TASK_PER_SCENE == None
        tag_name = cls.TEST_SCENES_NAME + "-objects_" + str(cls.OBJECT_TYPES_NAME)
        if cls.INFERENCE_COEF > 0.0:
            tag_name += "-safety" + str(cls.INFERENCE_COEF)
        return tag_name
