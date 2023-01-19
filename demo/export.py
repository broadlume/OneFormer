
import argparse
import numpy as np
import torch
from torch import Tensor, nn
import torch.onnx
from defaults import DefaultPredictor

from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.export import dump_torchscript_IR, scripting_with_instances
from detectron2.structures import Boxes, Instances

# fmt: off
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

from oneformer import (
    add_oneformer_config,
    add_common_config,
    add_swin_config,
    add_dinat_config,
    add_convnext_config,
)

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_common_config(cfg)
    add_swin_config(cfg)
    add_dinat_config(cfg)
    add_convnext_config(cfg)
    add_oneformer_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="oneformer exporter for builtin configs")
    parser.add_argument(
        "--config-file",
        default="../configs/ade20k/swin/oneformer_swin_large_IN21k_384_bs16_160k.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--task", help="Task type")
    parser.add_argument(
        "--output",
        help="File path to save the exported model",
        default="./output/oneformer_frozen.pt"
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

if __name__ == "__main__":

    args = get_parser().parse_args()
    cfg = setup_cfg(args)

    predictor = DefaultPredictor(cfg)

    img = np.zeros([512,512,3],dtype=np.uint8)
    img = predictor.aug.get_transform(img).apply_image(img)
    img = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
    inputs = {"image": img, "height": 512, "width": 512, "task": 'semantic'}

    # Export the model
    torch.onnx.export(predictor.model,       # model being run
                  [img],                         # model input (or a tuple for multiple inputs)
                  "export.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})

    #fields = {
    #    "proposal_boxes": Boxes,
    #    "objectness_logits": Tensor,
    #    "pred_boxes": Boxes,
    #    "scores": Tensor,
    #    "pred_classes": Tensor,
    #    "pred_masks": Tensor,
    #}
    #model_scripted = scripting_with_instances(predictor.model, fields)

    #model_scripted = torch.jit.script(predictor.model) # Export to TorchScript
    
    #model_scripted.save(args.output) # Save