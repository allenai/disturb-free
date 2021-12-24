# Towards Disturbance-Free Visual Mobile Manipulation
[[Project Site]](https://sites.google.com/view/disturb-free/home)  [[arXiv]](https://arxiv.org/abs/2112.12612)

## Install
Please git clone [AllenAct](https://github.com/allenai/allenact) repo with this submodule and change to a specific commit for reproducibility:
```bash
git clone --recurse-submodules https://github.com/allenai/allenact.git
git checkout f00445e4ae8724ccc001b3300af5c56a7f882614
```

We recommend using a virtual environment like conda to install the dependencies. Typically, our experiment needs a 8-GPU machine with around 60 CPU cores. Below are scripts for installation by conda:

```
conda create -n disturb
conda activate disturb
conda install python==3.6.12
pip install -r projects/disturb-free/requirements.txt
```

Then download the [APND dataset](https://github.com/allenai/manipulathor/tree/main/datasets) to the allenact folder.

## Stage I: Pre-Training
The main configuration is in `projects/manipulathor_disturb_free/armpointnav_baselines/experiments/ithor/armpointnav_depth.py`. 

First keep all the configurations fixed to run pre-training experiments. By choosing different auxiliary task or commenting all the tasks in `AUXILIARY_UUIDS` in `projects/manipulathor_disturb_free/armpointnav_baselines/experiments/armpointnav_mixin_ddppo.py`, we can run experiment with CPC|A, Inverse Dynamics, Disturbance Prediction, or without auxiliary task, i.e., 

```python
    AUXILIARY_UUIDS = [
        # InverseDynamicsLoss.UUID,
        # CPCA16Loss.UUID,
        # DisturbPredictionLoss.UUID,
    ]
```

Run pre-training by the following script:

```bash
python main.py projects/manipulathor_disturb_free/armpointnav_baselines/experiments/ithor/armpointnav_depth.py \
    -s 302 --save_dir_fmt NESTED \
    -o experiment_output/armpointnav/
```

Then we will see the experiment results in `experiment_output/armpointnav/` folder. We can use tensorboard to visualize the results on training scenes.

## Stage II: Fine-Tuning
Now we switch to the fine-tuning stage, where we use the pre-trained checkpoint saved in the experiments results folder `*/checkpoints/*.pt`. Choose one checkpoint and copy its path in `WEIGHTS_PATH` and set `LOAD_PRETRAINED_WEIGHTS = True` and `DISTURB_PEN=15.0` in `armpointnav_depth.py`.

Then run fine-tuning using the same script above.


## Stage III: Online Evaluation
Finally, the online evaluation configuration is `projects/manipulathor_disturb_free/armpointnav_baselines/experiments/eval/TestScene.py`, where we need to set the configurations same as the checkpoint we want to evaluate. Then we can set `INFERENCE_COEF=0.1` if using disturbance prediction task.

Run online evaluation by the script:

```bash
python main.py projects/manipulathor_disturb_free/armpointnav_baselines/experiments/eval/TestScene.py \
    -s 1 --save_dir_fmt NESTED --infer_output_dir --eval \
    -c <checkpoint_path>
```

## Evaluate and Visualize the Best Checkpoints
Please download the best checkpoints of our method (`Disturb_Pred`), the compared methods (`CPCA_16` and `InvDyn`) and baseline (`no_aux`) using the script:
```bash
sh projects/manipulathor_disturb_free/download_ckpts.sh armpointnav-disturb-free-2021
```

### Evaluation

Run the following script to evaluate (take our disturbance prediction task as example) after setting the same configurations in `TestScene.py`:

```bash
TEST_CKPT=pretrained_model_ckpts/armpointnav-disturb-free-2021/best_models/gnresnet18-woNormAdv-wact-man_sel-polar_radian-finetune-disturb_pen15.0_all-Disturb_Pred-gamma2.0/checkpoints/exp_resnet18-woNormAdv-wact-man_sel-polar_radian-finetune-disturb_pen15.0_all-Disturb_Pred-gamma2.0__stage_00__steps_000025054444.pt

python main.py projects/manipulathor_disturb_free/armpointnav_baselines/experiments/eval/TestScene.py \
    -s 1 --save_dir_fmt NESTED --infer_output_dir --eval -i \
    -c $TEST_CKPT
```

### Visualization

For visualization, please use the following configs in `TestScene.py`:
```python
    VISUALIZERS = [
        lambda exp_name: ImageVisualizer(exp_name, 
            add_top_down_view=True,
            add_depth_map=True,
        ),
        lambda exp_name: TestMetricLogger(exp_name),
    ]
    CAMERA_WIDTH = (
        # 224
        224 * 2 # higher resolution for better visualization, won't change the agent obs shape
    )
    CAMERA_HEIGHT = (
        # 224
        224 * 2
    )

    NUM_TASK_PER_SCENE = (
        # None
        6 # set a small number for light storage
    )
```
Then run the following script (we take our best model as example):
```bash
# Currently we only support visualization on a single checkpoint
python main.py projects/manipulathor_disturb_free/armpointnav_baselines/experiments/eval/TestScene.py \
    -s 1 --save_dir_fmt NESTED --infer_output_dir --eval -i \
    -c $TEST_CKPT \
    --config_kwargs '{"test_ckpt": "'$TEST_CKPT'"}'
```
It will generate the gifs of both egocentric and top-down view with annotation of current disturbance distance in meter on the top-left corner. It will also generate the start and goal image in RGB and depth, and all the evaluation metrics in json.

## Core Functions
* [Disturbance distance sensor](https://github.com/allenai/disturb-free/blob/main/manipulathor_plugin/disturb_sensor.py)
* [Disturbance distance penalty](https://github.com/allenai/allenact/blob/f00445e4ae8724ccc001b3300af5c56a7f882614/allenact_plugins/manipulathor_plugin/manipulathor_tasks.py#L441)
* [Disturbance distance prediction task](https://github.com/allenai/disturb-free/blob/main/armpointnav_baselines/models/disturb_pred_loss.py)

## Acknowledgement
This repository uses [AllenAct](https://github.com/allenai/allenact) as codebase and [ManipulaTHOR](https://github.com/allenai/manipulathor) as testbed. GroupNormResNet18 and auxiliary tasks are heavily based on [Auxiliary Tasks Speed Up Learning PointGoal Navigation](https://github.com/joel99/habitat-pointnav-aux). 
