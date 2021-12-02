from typing import Tuple, Dict, Optional, Union, List, cast

import torch
import torch.nn as nn
from gym.spaces.dict import Dict as SpaceDict

from projects.manipulathor_disturb_free.armpointnav_baselines.models.disturb_pred_loss import (
    DisturbPredictionLoss,
)


class AuxiliaryModel(nn.Module):
    def __init__(
        self,
        aux_uuid: str,
        action_dim: int,
        obs_embed_dim: int,
        belief_dim: int,
        disturb_hidden_dim: int = 128,
        **kwargs
    ):
        super().__init__()
        self.aux_uuid = aux_uuid
        self.action_dim = action_dim
        self.obs_embed_dim = obs_embed_dim
        self.belief_dim = belief_dim

        assert self.aux_uuid == DisturbPredictionLoss.UUID
        self.classifier = nn.ModuleList(
            [
                nn.Linear(self.belief_dim, disturb_hidden_dim),
                nn.ReLU(),
                nn.Linear(disturb_hidden_dim, self.action_dim),
            ]
        )
        # follow focal loss trick: initialize the bias a large value, so sigmoid is 0.01
        # correct on the majority samples (which are background)
        torch.nn.init.constant_(self.classifier[-1].bias, -4.5)

    def forward(self, features: torch.FloatTensor):
        x = features
        for m in self.classifier:
            x = m(x)
        return x
