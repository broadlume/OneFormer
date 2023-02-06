#!/bin/bash

export task=panoptic

traindir="./output/ade20k_panoptic"
lastcheckpoint=`cat $traindir/last_checkpoint`
lastname=${lastcheckpoint%????}
outputdir="${traindir}/${lastname}"
weights="${traindir}/${lastcheckpoint}"

echo "$outputdir --> $weights"

python ./demo/demo.py --config-file ./configs/ade20k/convnext/oneformer_convnext_large_bs16_160k_2gpu.yaml \
  --input ../datasets/test_images/*.jpg \
  --output "${outputdir}" \
  --task $task \
  --opts MODEL.IS_TRAIN False MODEL.IS_DEMO True MODEL.WEIGHTS "${weights}"