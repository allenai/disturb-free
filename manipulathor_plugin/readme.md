## Disturb Dist Dict
In [our experimented version](https://github.com/allenai/ai2thor/commit/a84dd29471ec2201f583de00257d84fac1a03de2) of AI2THOR, there exist some vibrations in some objects in some scenes without external force. To eliminate this simulation bias from disturbance checking, we calculated the vibration distances of all the objects of all the scenes using the following:

```bash
python projects/manipulathor_disturb_free/manipulathor_plugin/disturb_dist_dict.py
```

It will generate a csv file and then used in our task.
