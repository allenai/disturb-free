# Towards Disturbance-Free Visual Mobile Manipulation

## Install
Please git clone allenact repo with this submodule:
```bash
git clone --recurse-submodules https://github.com/allenai/allenact.git
```

We recommend using a virtual environment like conda to install the dependencies. Typically, our experiment needs a 8-GPU machine with around 60 CPU cores. Below are scripts for installation by conda:

```
conda create -n disturb
conda activate disturb
conda install python==3.6.12
pip install -r projects/disturb-free/requirements.txt
```

Then download the APND dataset from https://github.com/allenai/manipulathor/tree/main/datasets

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
python main.py projects/manipulathor_disturb_free/armpointnav_baselines/experiments/ithor/armpointnav_depth.py -s 302 --save_dir_fmt NESTED \
    -o experiment_output/armpointnav/
```

Then we will see the experiment results in `experiment_output/armpointnav/` folder. We can use tensorboard to visualize the results on training scenes.

## Stage II: Fine-Tuning
Now we switch to the fine-tuning stage, where we use the pre-trained checkpoint saved in the experiments results folder `*/checkpoints/*.pt`. Choose one checkpoint and copy its path in `weights_path` and set `load_pretrained_weights = True` and `DISTURB_PEN=15.0` in `armpointnav_depth.py`.

Then run fine-tuning using the same script above.


## Stage III: Online Evaluation
Finally, the online evaluation configuration is `projects/manipulathor_disturb_free/armpointnav_baselines/experiments/eval/TestScene.py`, where we need to set the configurations same as the checkpoint we want to evaluate. Then we can set `INFERENCE_COEF=0.1` if using disturbance prediction task.

Run online evaluation by the script:

```bash
python main.py projects/manipulathor_disturb_free/armpointnav_baselines/experiments/eval/TestScene.py -s 1 --save_dir_fmt NESTED \
    --eval \
    -c <checkpoint_path>
```

## Evaluate and Visualize the Best Checkpoints
Please download the best checkpoint from AWS [TODO]

Then locate them into `experiment_output/armpointnav/best_models/`. Run the following script to evaluate (take our disturbance prediction task as example) after setting the same configurations in `TestScene.py`:

```bash
python main.py projects/manipulathor_disturb_free/armpointnav_baselines/experiments/eval/TestScene.py -s 1 --save_dir_fmt NESTED \
    --eval -i \
    -c experiment_output/armpointnav/best_models/gnresnet18-woNormAdv-wact-man_sel-polar_radian-finetune-disturb_pen15.0_all-Disturb_Pred-gamma2.0/checkpoints
```



## Acknowledgement
This repository uses AllenAct framework https://github.com/allenai/allenact and ManipulaTHOR framework https://github.com/allenai/manipulathor. ResNet18 and auxiliary tasks are heavily based on https://github.com/joel99/habitat-pointnav-aux. 
