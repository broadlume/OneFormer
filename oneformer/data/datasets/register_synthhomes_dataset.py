import os
import json
import cv2
from detectron2.data import MetadataCatalog, DatasetCatalog

_root = os.getenv("DETECTRON2_DATASETS", "datasets")

def get_synthhomes_dicts(img_dir):
    
    dataset_dicts = []
    for root, subFolders, files in os.walk(img_dir, topdown=True):
        record = {}
        frame_data = json.load(os.path.join(root, "step1.frame_data.json"))

        if not "annotations" in frame_data:
            continue

        seg_annotations = None
        for idx, x in enumerate(frame_data["annotations"]):
            if "@type" in x and x["@type"] == "type.unity.com/unity.solo.SemanticSegmentationAnnotation":
                seg_annotations = x
                break

        if seg_annotations == None: continue

        segid = seg_annotations["id"]
        segfilename = seg_annotations["filename"]
        segdimension = seg_annotations["dimension"]

        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = os.path.join(root, "step1.camera.jpg")
        record["sem_seg_file_name"] = os.path.join(root, segfilename)
        record["image_id"] = segid
        record["height"] = segdimension[1]
        record["width"] = segdimension[0]

        dataset_dicts.append(record)
    return dataset_dicts

def get_synthhomes_dicts_from_json(jsonpath):
    print("JSON PATH: " + jsonpath)
    dataset = json.load(open(jsonpath))
    for x in dataset:
        x['file_name'] = os.path.join(_root, "vmdatasets", x['file_name'])
        x['sem_seg_file_name'] = os.path.join(_root, "vmdatasets", x['sem_seg_file_name'])
    return dataset


for d in ["train", "val"]:
    DatasetCatalog.register("synthhomes_sem_seg_" + d, lambda d=d:get_synthhomes_dicts_from_json(os.path.join(_root, "vmdatasets/vmsynthhomes", "synthhomes_detectrondict_" + d + ".json")))
    MetadataCatalog.get("synthhomes_sem_seg_" + d).set(
        stuff_classes = ["Floor", "Wall"],
        stuff_colors = [(0,255,0), (0,0,255)],
        thing_classes = [],
        thing_dataset_id_to_contiguous_id = {},
        ignore_label = 0,
        evaluator_type = "sem_seg"
    )
