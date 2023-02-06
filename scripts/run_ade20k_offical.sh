#!/bin/bash

export task=panoptic

python ./demo/demo.py --config-file ./configs/ade20k/convnext/oneformer_convnext_large_bs16_160k.yaml \
  --input ../datasets/test_images/*.jpg \
  --output "./output/ade20k_official" \
  --task $task \
  --opts MODEL.IS_TRAIN False MODEL.IS_DEMO True MODEL.WEIGHTS "./checkpoints/ade20k/convnext/250_16_convnext_l_oneformer_ade20k_160k.pth"