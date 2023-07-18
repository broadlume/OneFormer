import os
import json
import cv2
from detectron2.data import MetadataCatalog, DatasetCatalog

_root = os.getenv("DETECTRON2_DATASETS", "datasets")

def get_bvrooms_dicts_from_json(jsonpath):
    print("JSON PATH: " + jsonpath)
    dataset = json.load(open(jsonpath))
    for x in dataset:
        x['file_name'] = os.path.join(_root, "vmdatasets", x['file_name'])
        x['sem_seg_file_name'] = os.path.join(_root, "vmdatasets", x['sem_seg_file_name'])
    return dataset

for d in ["train", "val"]:
    DatasetCatalog.register("bvrooms_sem_seg_" + d, lambda d=d:get_bvrooms_dicts_from_json(os.path.join(_root, "vmdatasets/bvrooms", "bvrooms_detectrondict_" + d + ".json")))
    MetadataCatalog.get("bvrooms_sem_seg_" + d).set(
        stuff_classes = ["Background", "Floor", "Wall"],
        stuff_colors = [(0,0,0), (0,255,0), (0,0, 255)],
        thing_classes = [],
        thing_dataset_id_to_contiguous_id = {},
        ignore_label = 255,
        evaluator_type = "sem_seg"
    )
