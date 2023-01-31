#!/bin/bash

export task=semantic

traindir="./output/ade20k_semseg"
lastcheckpoint=`cat $traindir/last_checkpoint`
lastname=${lastcheckpoint%????}
outputdir="${traindir}/${lastname}"
weights="${traindir}/${lastcheckpoint}"

echo "$outputdir --> $weights"

python ./demo/demo.py --config-file ./configs/ade20k_semseg/oneformer_R50_bs16_160k.yaml \
  --input ../datasets/test_images/*.jpg \
  --output "${outputdir}" \
  --task $task \
  --opts MODEL.IS_TRAIN False MODEL.IS_DEMO True MODEL.WEIGHTS "${weights}"