import os
import json
import cv2
from detectron2.data import MetadataCatalog, DatasetCatalog

_root = os.getenv("DETECTRON2_DATASETS", "datasets")

def get_blfloors_dicts_from_json(jsonpath):
    print("JSON PATH: " + jsonpath)
    dataset = json.load(open(jsonpath))
    for x in dataset:
        x['file_name'] = os.path.join(_root, "vmdatasets", x['file_name'])
        x['sem_seg_file_name'] = os.path.join(_root, "vmdatasets", x['sem_seg_file_name'])
    return dataset

for d in ["train", "val"]:
    DatasetCatalog.register("blfloors_sem_seg_" + d, lambda d=d:get_blfloors_dicts_from_json(os.path.join(_root, "vmdatasets/broadlumefloors", "blfloors_detectrondict_" + d + ".json")))
    MetadataCatalog.get("blfloors_sem_seg_" + d).set(
        stuff_classes = ["Background", "Floor"],
        stuff_colors = [(0,0,0), (0,255,0)],
        thing_classes = [],
        thing_dataset_id_to_contiguous_id = {},
        ignore_label = 255,
        evaluator_type = "sem_seg"
    )
