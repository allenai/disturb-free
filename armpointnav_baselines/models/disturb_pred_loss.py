"""Defining the auxiliary loss for actor critic type models."""

from typing import Dict, cast, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

from allenact.embodiedai.aux_losses.losses import AuxiliaryLoss
from allenact.algorithms.onpolicy_sync.losses.abstract_loss import ObservationType
from allenact.base_abstractions.distributions import CategoricalDistr
from allenact.base_abstractions.misc import ActorCriticOutput

from projects.manipulathor_disturb_free.manipulathor_plugin.disturb_sensor import (
    DisturbanceSensor,
)


class DisturbPredictionLoss(AuxiliaryLoss):
    UUID = "Disturb_Pred"

    def __init__(self, gamma=2.0, *args, **kwargs):
        super().__init__(auxiliary_uuid=self.UUID, *args, **kwargs)
        self.gamma = gamma

    def get_aux_loss(
        self,
        aux_model: nn.Module,
        observations: ObservationType,
        obs_embeds: torch.FloatTensor,
        actions: torch.FloatTensor,
        beliefs: torch.FloatTensor,
        masks: torch.FloatTensor,
        *args,
        **kwargs
    ):
        # num_steps, num_sampler = actions.shape  # T, B
        # NOTE: alignment:
        # bt = RNN(ot, a(t-1))
        # d(t+1) <- M(bt, at)
        actions = cast(torch.LongTensor, actions)
        actions = actions.unsqueeze(-1)  # (T, B, 1) for gather

        ## get disturbance prediction logits
        raw_logits = aux_model(beliefs)  # (T, B, dim) -> (T, B, A)
        logits = torch.gather(input=raw_logits, dim=-1, index=actions)  # (T, B, 1)
        logits = logits.squeeze(-1)[
            :-1
        ]  # (T, B, 1) -> (T-1, B) final action does not have label

        raw_disturb = observations[DisturbanceSensor().uuid].float()  # (T, B)
        next_disturb = raw_disturb[1:]  # (T-1, B) next-step disturbance signal

        # raw BCE loss -> focal loss
        # https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/5
        raw_loss = F.binary_cross_entropy_with_logits(
            logits, next_disturb, reduction="none"
        )  # (T-1, B), -log(pt)
        probs = torch.exp(-raw_loss)
        raw_focal_loss = (1.0 - probs) ** self.gamma * raw_loss

        # NOTE: mask = 0.0 <-> the start of one episode (m1 = 0)
        # a1, a2, ..., aN-1, aN, a1, a2, ...
        # d2, d3, ..., dN,   d1, d2, d3, ...
        # m2, m3, ..., mN,   m1, m2, m3, ...
        masks = masks.squeeze(-1)  # (T, B)
        loss_masks = masks[1:]  # (T-1, B)
        num_valid_losses = torch.count_nonzero(loss_masks)
        avg_loss = (raw_focal_loss * loss_masks).sum() / torch.clamp(
            num_valid_losses, min=1.0
        )

        # report accuracy metrics
        with torch.no_grad():
            loss_masks = loss_masks.bool().flatten()  # (T-1 * B)
            disturb_preds = (torch.sigmoid(logits) > 0.5).int().flatten()  # (T-1 * B)
            disturb_preds = disturb_preds[loss_masks].cpu().numpy()
            disturb_targets = next_disturb.int().flatten()[loss_masks].cpu().numpy()

            matrix = confusion_matrix(
                y_true=disturb_targets,
                y_pred=disturb_preds,
                labels=(0, 1),  # in case of NaN
            )
            # real neg: TN | FP
            # real pos: FN | TP
            no_disturb_recall = matrix[0, 0] / max(matrix[0, 0] + matrix[0, 1], 1.0)
            has_disturb_recall = matrix[1, 1] / max(matrix[1, 0] + matrix[1, 1], 1.0)
            has_disturb_precision = matrix[1, 1] / max(matrix[0, 1] + matrix[1, 1], 1.0)
            overall_acc = (matrix[0, 0] + matrix[1, 1]) / matrix.sum()
            disturb_gt_ratio = (matrix[1, 0] + matrix[1, 1]) / matrix.sum()
            disturb_pred_ratio = (matrix[0, 1] + matrix[1, 1]) / matrix.sum()

        # from fpdb import ForkedPdb; ForkedPdb().set_trace()

        return (
            avg_loss,
            {
                "focal_loss": cast(torch.Tensor, avg_loss).item(),
                "no_disturb_recall": cast(torch.Tensor, no_disturb_recall).item(),
                "has_disturb_recall": cast(torch.Tensor, has_disturb_recall).item(),
                "has_disturb_precision": cast(
                    torch.Tensor, has_disturb_precision
                ).item(),
                "overall_acc": cast(torch.Tensor, overall_acc).item(),
                "disturb_gt_ratio": cast(torch.Tensor, disturb_gt_ratio).item(),
                "disturb_pred_ratio": cast(torch.Tensor, disturb_pred_ratio).item(),
            },
        )

    @classmethod
    def inference(
        cls, actor_critic_output: ActorCriticOutput[CategoricalDistr], inference_coef,
    ):
        # one-step inference
        beliefs = actor_critic_output.extras[cls.UUID]["beliefs"]  # (1, B, -1)
        aux_model = actor_critic_output.extras[cls.UUID][
            "aux_model"
        ]  # given the trained model
        raw_action_logits = actor_critic_output.distributions.logits  # (1, B, A)

        # NOTE: we don't need masks, because belief has reset if mask = 0.0
        # the larger the logit, the higher prob to being predicted disturb
        logits = aux_model(beliefs)  # (1, B, A)

        assert inference_coef > 0.0
        new_logits = raw_action_logits - inference_coef * logits
        # ignore the negative prediction logits
        # new_logits = raw_action_logits - inference_coef * torch.clamp(logits, min=0.0)

        return CategoricalDistr(logits=new_logits)
