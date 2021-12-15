import pandas as pd

import ai2thor.controller

ENV_ARGS = dict(
    gridSize=0.25,
    width=224,
    height=224,
    visibilityDistance=1.0,
    agentMode="arm",
    fieldOfView=100,
    agentControllerType="mid-level",
    server_class=ai2thor.fifo_server.FifoServer,
    useMassThreshold=True,
    massThreshold=10,
    autoSimulation=False,
    autoSyncTransforms=True,
    renderInstanceSegmentation=True,
)

commit_id = "a84dd29471ec2201f583de00257d84fac1a03de2"
ENV_ARGS["commit_id"] = commit_id
controller = ai2thor.controller.Controller(**ENV_ARGS)

kitchens = [f"FloorPlan{i}" for i in range(1, 31)]
living_rooms = [f"FloorPlan{200 + i}" for i in range(1, 31)]
bedrooms = [f"FloorPlan{300 + i}" for i in range(1, 31)]
bathrooms = [f"FloorPlan{400 + i}" for i in range(1, 31)]

scenes = kitchens
#  + living_rooms + bedrooms + bathrooms


def make_dict_from_object(object_list):
    result = {}
    for obj in object_list:
        result[obj["objectId"]] = dict(position=obj["position"])
    return result


def position_distance(s1, s2):
    position1 = s1["position"]
    position2 = s2["position"]
    dist = (
        (position1["x"] - position2["x"]) ** 2
        + (position1["y"] - position2["y"]) ** 2
        + (position1["z"] - position2["z"]) ** 2
    ) ** 0.5
    return dist


def object_vibration_list(d1, d2):
    vib = {"object": [], "dist": []}
    for object in d1.keys():
        vib["object"].append(object)
        vib["dist"].append(position_distance(d1[object], d2[object]))
    return vib


results = []
for scene in scenes:
    print(scene)

    controller.reset(scene)
    total = 200

    initial_objects = make_dict_from_object(controller.last_event.metadata["objects"])
    for i in range(total):
        controller.step("AdvancePhysicsStep")
    final_objects = make_dict_from_object(controller.last_event.metadata["objects"])

    vib = object_vibration_list(initial_objects, final_objects)
    df = pd.DataFrame.from_dict(vib)
    df["scene"] = scene

    results.append(df)

results = pd.concat(results)
results.to_csv(
    "projects/manipulathor_disturb_free/manipulathor_plugin/vibrations.csv", index=False
)
