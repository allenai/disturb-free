from typing import Dict, Tuple
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from allenact.algorithms.onpolicy_sync.losses.abstract_loss import (
    AbstractActorCriticLoss,
)
from allenact.algorithms.onpolicy_sync.losses import PPO
from allenact.algorithms.onpolicy_sync.losses.ppo import PPOConfig
from allenact.embodiedai.aux_losses.losses import (
    InverseDynamicsLoss,
    CPCA16Loss,
)
from projects.manipulathor_disturb_free.armpointnav_baselines.models.disturb_pred_loss import (
    DisturbPredictionLoss,
)

# noinspection PyUnresolvedReferences
from allenact.embodiedai.models.fusion_models import AverageFusion
from allenact.utils.experiment_utils import (
    Builder,
    PipelineStage,
    TrainingPipeline,
    LinearDecay,
)

from projects.manipulathor_disturb_free.armpointnav_baselines.experiments.armpointnav_base import (
    ArmPointNavBaseConfig,
)


class ArmPointNavMixInPPOConfig(ArmPointNavBaseConfig):

    NORMALIZE_ADVANTAGE = (
        # True
        False
    )
    ADD_PREV_ACTIONS = (
        True
        # False
    )

    # selected auxiliary uuids
    ## if comment all the keys, then it's vanilla DD-PPO
    AUXILIARY_UUIDS = [
        # InverseDynamicsLoss.UUID,
        # CPCA16Loss.UUID,
        DisturbPredictionLoss.UUID,
    ]
    MULTIPLE_BELIEFS = False
    BELIEF_FUSION = None

    def training_pipeline(self, **kwargs):
        ppo_steps = int(30000000)  # 30M
        lr = 3e-4
        num_mini_batch = 1
        update_repeats = 4
        num_steps = self.MAX_STEPS
        save_interval = 1000000  # 1M
        log_interval = 10000
        gamma = 0.99
        use_gae = True
        gae_lambda = 0.95
        max_grad_norm = 0.5
        PPOConfig["normalize_advantage"] = self.NORMALIZE_ADVANTAGE

        # Total losses
        named_losses = {"ppo_loss": (PPO(**PPOConfig), 1.0)}
        named_losses = self._update_with_auxiliary_losses(named_losses)

        return TrainingPipeline(
            save_interval=save_interval,
            metric_accumulate_interval=log_interval,
            optimizer_builder=Builder(optim.Adam, dict(lr=lr)),
            num_mini_batch=num_mini_batch,
            update_repeats=update_repeats,
            max_grad_norm=max_grad_norm,
            num_steps=num_steps,
            named_losses={key: val[0] for key, val in named_losses.items()},
            gamma=gamma,
            use_gae=use_gae,
            gae_lambda=gae_lambda,
            advance_scene_rollout_period=self.ADVANCE_SCENE_ROLLOUT_PERIOD,
            pipeline_stages=[
                PipelineStage(
                    loss_names=list(named_losses.keys()),
                    max_stage_steps=ppo_steps,
                    loss_weights=[val[1] for val in named_losses.values()],
                )
            ],
            lr_scheduler_builder=Builder(  # lambda lr: lambda * base_lr
                LambdaLR,
                {"lr_lambda": LinearDecay(steps=ppo_steps, startp=1.0, endp=1.0 / 3)},
            ),
        )

    @classmethod
    def _update_with_auxiliary_losses(cls, named_losses):
        # auxliary losses
        aux_loss_total_weight = 2.0

        # Total losses
        total_aux_losses: Dict[str, Tuple[AbstractActorCriticLoss, float]] = {
            InverseDynamicsLoss.UUID: (
                InverseDynamicsLoss(subsample_rate=0.2, subsample_min_num=10,),
                0.05 * aux_loss_total_weight,
            ),
            CPCA16Loss.UUID: (
                CPCA16Loss(subsample_rate=0.2,),
                0.05 * aux_loss_total_weight,
            ),
            DisturbPredictionLoss.UUID: (
                DisturbPredictionLoss(gamma=cls.DISTURB_FOCAL_GAMMA),
                0.05 * aux_loss_total_weight,
            ),
        }
        named_losses.update(
            {uuid: total_aux_losses[uuid] for uuid in cls.AUXILIARY_UUIDS}
        )

        return named_losses
