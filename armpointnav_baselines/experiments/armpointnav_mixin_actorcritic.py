from typing import Sequence, Union

import gym
import torch.nn as nn
from allenact.base_abstractions.preprocessor import Preprocessor
from allenact.embodiedai.sensors.vision_sensors import RGBSensor, DepthSensor
from allenact_plugins.manipulathor_plugin.manipulathor_sensors import (
    RelativeAgentArmToObjectSensor,
    RelativeObjectToGoalSensor,
    PickedUpObjSensor,
    DisturbanceSensor,
)
from allenact.utils.experiment_utils import Builder
from projects.manipulathor_disturb_free.armpointnav_baselines.experiments.armpointnav_base import (
    ArmPointNavBaseConfig,
)
from projects.manipulathor_disturb_free.armpointnav_baselines.models.arm_pointnav_models import (
    ArmPointNavBaselineActorCritic,
)

class ArmPointNavAdvancedACConfig(ArmPointNavBaseConfig):
    @classmethod
    def preprocessors(cls) -> Sequence[Union[Preprocessor, Builder[Preprocessor]]]:
        preprocessors = []
        return preprocessors

    BACKBONE = "resnet18"
    load_pretrained_weights = False
    weights_path = None
    INFERENCE_COEF = 0.0
    
    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        rgb_uuid = next(
            (s.uuid for s in cls.SENSORS if isinstance(s, RGBSensor)), None
        )
        depth_uuid = next(
            (s.uuid for s in cls.SENSORS if isinstance(s, DepthSensor)), None
        )
        arm2obj_uuid = next(
            (s.uuid for s in cls.SENSORS if isinstance(s, RelativeAgentArmToObjectSensor)),
            None,
        )
        obj2goal_uuid = next(
            (s.uuid for s in cls.SENSORS if isinstance(s, RelativeObjectToGoalSensor)),
            None,
        )
        pickedup_uuid = next(
            (s.uuid for s in cls.SENSORS if isinstance(s, PickedUpObjSensor)),
            None,
        )
        disturbance_uuid = next(
            (s.uuid for s in cls.SENSORS if isinstance(s, DisturbanceSensor)),
            None,
        )

        return ArmPointNavBaselineActorCritic(
            # Env and Task
            action_space=gym.spaces.Discrete(
                len(cls.TASK_SAMPLER._TASK_TYPE.class_action_names())
            ),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            rgb_uuid=rgb_uuid,
            depth_uuid=depth_uuid,
            arm2obj_uuid=arm2obj_uuid,
            obj2goal_uuid=obj2goal_uuid,
            pickedup_uuid=pickedup_uuid,
            disturbance_uuid=disturbance_uuid,
            # RNN
            hidden_size=512 if cls.multiple_beliefs == False or len(cls.AUXILIARY_UUIDS) <= 1 else 256,
            num_rnn_layers=1,
            rnn_type="GRU",
            add_prev_actions=cls.add_prev_actions,
            action_embed_size=16,
            # CNN            
            backbone=cls.BACKBONE,
            resnet_baseplanes=32,
            load_pretrained_weights=cls.load_pretrained_weights,
            weights_path=cls.weights_path,
            # goal sensor
            goal_embedding_size=32, # change it smaller
            goal_space_mode=cls.goal_space_mode,
            # Aux
            auxiliary_uuids=cls.AUXILIARY_UUIDS,
            multiple_beliefs=cls.multiple_beliefs,
            beliefs_fusion=cls.beliefs_fusion,
            inference_coef=cls.INFERENCE_COEF,
        )
