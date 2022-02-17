"""Baseline models for use in the Arm Point Navigation task.

Arm Point Navigation is currently available as a Task in ManipulaTHOR.
"""
from typing import Tuple, Dict, Optional, cast, List
from collections import OrderedDict
from allenact.utils.system import get_logger

import gym
import torch
import torch.nn as nn
from gym.spaces.dict import Dict as SpaceDict
from allenact.algorithms.onpolicy_sync.policy import (
    ObservationType,
    DistributionType,
)

from allenact.base_abstractions.distributions import CategoricalDistr
from allenact.base_abstractions.misc import ActorCriticOutput, Memory
from allenact.embodiedai.models.basic_models import SimpleCNN
import allenact.embodiedai.models.resnet as resnet
from allenact.embodiedai.models.visual_nav_models import (
    VisualNavActorCritic,
    FusionType,
)
from allenact.embodiedai.models.aux_models import AuxiliaryModel
from projects.manipulathor_disturb_free.armpointnav_baselines.models.disturb_pred_loss import (
    DisturbPredictionLoss,
)
from projects.manipulathor_disturb_free.armpointnav_baselines.models.aux_model import (
    AuxiliaryModel as DisturbAuxiliaryModel,
)

from projects.manipulathor_disturb_free.armpointnav_baselines.models.manipulathor_net_utils import (
    input_embedding_net,
)

from projects.manipulathor_disturb_free.armpointnav_baselines.models.disturb_pred_loss import (
    DisturbPredictionLoss,
)


class ArmPointNavBaselineActorCritic(VisualNavActorCritic):
    """Baseline recurrent actor critic model for armpointnav task.

    # Attributes
    action_space : The space of actions available to the agent. Currently only discrete
        actions are allowed (so this space will always be of type `gym.spaces.Discrete`).
    observation_space : The observation space expected by the agent. This observation space
        should include (optionally) 'rgb' images and 'depth' images.
    hidden_size : The hidden size of the GRU RNN.
    object_type_embedding_dim: The dimensionality of the embedding corresponding to the goal
        object type.
    """

    def __init__(
        self,
        # Env and Task
        action_space: gym.spaces.Discrete,
        observation_space: SpaceDict,
        arm2obj_uuid: str,
        obj2goal_uuid: str,
        pickedup_uuid: str,
        disturbance_uuid: str,
        # RNN
        hidden_size=512,
        num_rnn_layers=1,
        rnn_type="GRU",
        add_prev_actions=False,
        action_embed_size=16,
        # Aux loss
        multiple_beliefs=False,
        beliefs_fusion: Optional[FusionType] = None,
        auxiliary_uuids: Optional[List[str]] = None,
        # safety inference with the disturbance prediction task
        inference_coef: float = 0.0,
        # below are custom params
        rgb_uuid: Optional[str] = None,
        depth_uuid: Optional[str] = None,
        goal_embedding_size=32,
        goal_space_mode=None,
        trainable_masked_hidden_state: bool = False,
        # perception backbone params,
        backbone="gnresnet18",
        resnet_baseplanes=32,
    ):
        """Initializer.

        See class documentation for parameter definitions.
        """
        super().__init__(
            action_space=action_space,
            observation_space=observation_space,
            hidden_size=hidden_size,
            multiple_beliefs=multiple_beliefs,
            beliefs_fusion=beliefs_fusion,
            auxiliary_uuids=auxiliary_uuids,
        )

        self.goal_embedding_size = goal_embedding_size
        self.goal_space_mode = goal_space_mode
        self.backbone = backbone
        self.rgb_uuid = rgb_uuid
        self.depth_uuid = depth_uuid
        self.arm2obj_uuid = arm2obj_uuid
        self.obj2goal_uuid = obj2goal_uuid
        self.pickedup_uuid = pickedup_uuid
        self.disturbance_uuid = disturbance_uuid
        assert inference_coef >= 0.0
        self.inference_coef = inference_coef

        if backbone == "simple_cnn":
            self.visual_encoder = SimpleCNN(
                observation_space=observation_space,
                output_size=hidden_size,
                rgb_uuid=rgb_uuid,
                depth_uuid=depth_uuid,
            )
        else:  # resnet family
            self.visual_encoder = resnet.GroupNormResNetEncoder(
                observation_space=observation_space,
                output_size=hidden_size,
                rgb_uuid=rgb_uuid,
                depth_uuid=depth_uuid,
                baseplanes=resnet_baseplanes,
                ngroups=resnet_baseplanes // 2,
                make_backbone=getattr(resnet, backbone),
            )

        self.create_state_encoders(
            obs_embed_size=self.goal_visual_encoder_output_dims,
            num_rnn_layers=num_rnn_layers,
            rnn_type=rnn_type,
            add_prev_actions=add_prev_actions,
            prev_action_embed_size=action_embed_size,
            trainable_masked_hidden_state=trainable_masked_hidden_state,
        )

        self.create_actorcritic_head()

        self.create_aux_models(
            obs_embed_size=self.goal_visual_encoder_output_dims,
            action_embed_size=action_embed_size,
        )

        self.create_goal_sensor_model()

        self.train()
        get_logger().debug(self)

    def create_goal_sensor_model(self):
        assert self.goal_space_mode in ["man_sel", "pickup_obs", "coords_only"]
        goal_sensor_dim = self.observation_space[self.arm2obj_uuid].shape[0]
        assert goal_sensor_dim == self.observation_space[self.obj2goal_uuid].shape[0]

        if (
            self.goal_space_mode == "man_sel"
        ):  # manual select the coord by boolean selector
            goal_embedding_sizes = torch.Tensor(
                [goal_sensor_dim, 100, self.goal_embedding_size]
            )
        elif (
            self.goal_space_mode == "pickup_obs"
        ):  # observe the boolean selector to learn selection
            goal_embedding_sizes = torch.Tensor(
                [goal_sensor_dim * 2 + 1, 100, self.goal_embedding_size]
            )
        else:  # only observe two coords
            goal_embedding_sizes = torch.Tensor(
                [goal_sensor_dim * 2, 100, self.goal_embedding_size]
            )

        self.goal_embedder = input_embedding_net(
            goal_embedding_sizes.long().tolist(), dropout=0
        )

    def create_aux_models(self, obs_embed_size: int, action_embed_size: int):
        if self.auxiliary_uuids is None:
            return
        aux_models = OrderedDict()
        for aux_uuid in self.auxiliary_uuids:
            if aux_uuid == DisturbPredictionLoss.UUID:
                model_class = DisturbAuxiliaryModel
            else:
                model_class = AuxiliaryModel
            aux_models[aux_uuid] = model_class(
                aux_uuid=aux_uuid,
                action_dim=self.action_space.n,
                obs_embed_dim=obs_embed_size,
                belief_dim=self._hidden_size,
                action_embed_size=action_embed_size,
            )

        self.aux_models = nn.ModuleDict(aux_models)

    @property
    def is_blind(self) -> bool:
        """True if the model is blind (e.g. neither 'depth' or 'rgb' is an
        input observation type)."""
        return self.visual_encoder.is_blind

    @property
    def goal_visual_encoder_output_dims(self):
        dims = self.goal_embedding_size
        if self.is_blind:
            return dims
        if self.backbone == "simple_cnn":
            input_visual_feature_num = int(self.rgb_uuid is not None) + int(
                self.depth_uuid is not None
            )
        else:  # resnet
            input_visual_feature_num = 1
        return dims + self.recurrent_hidden_state_size * input_visual_feature_num

    def forward_encoder(self, observations: ObservationType) -> torch.FloatTensor:

        arm2obj_dist_raw = observations[self.arm2obj_uuid]
        obj2goal_dist_raw = observations[self.obj2goal_uuid]
        pickup_bool_raw = observations[self.pickedup_uuid]

        if self.goal_space_mode == "man_sel":
            arm2obj_dist_embed = self.goal_embedder(arm2obj_dist_raw)
            obj2goal_dist_embed = self.goal_embedder(obj2goal_dist_raw)
            # use partial obj state space
            after_pickup = pickup_bool_raw == 1
            distances = arm2obj_dist_embed
            distances[after_pickup] = obj2goal_dist_embed[after_pickup]

        elif self.goal_space_mode == "pickup_obs":
            inputs_raw = torch.cat(
                [
                    pickup_bool_raw.unsqueeze(-1),  # (T, N, 1)
                    arm2obj_dist_raw,
                    obj2goal_dist_raw,
                ],
                dim=-1,
            )
            distances = self.goal_embedder(inputs_raw)

        else:  # coords_only
            inputs_raw = torch.cat([arm2obj_dist_raw, obj2goal_dist_raw], dim=-1)
            distances = self.goal_embedder(inputs_raw)

        obs_embeds = [distances]

        if not self.is_blind:
            perception_embed = self.visual_encoder(observations)
            obs_embeds = [perception_embed] + obs_embeds

        obs_embeds = torch.cat(obs_embeds, dim=-1)
        return obs_embeds

    def forward(  # type:ignore
        self,
        observations: ObservationType,
        memory: Memory,
        prev_actions: torch.Tensor,
        masks: torch.FloatTensor,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:

        actor_critic_output, memory = super().forward(
            observations, memory, prev_actions, masks,
        )

        if (
            self.auxiliary_uuids is not None
            and DisturbPredictionLoss.UUID in self.auxiliary_uuids
            and self.inference_coef > 0.0
        ):
            actor_critic_output.distributions = DisturbPredictionLoss.inference(
                actor_critic_output, self.inference_coef,
            )

        return actor_critic_output, memory
