#!/bin/bash

export task=semantic

traindir="./output/synthhomes_v3_640_semseg_lr10-e4"
lastcheckpoint=`cat $traindir/last_checkpoint`
lastname=${lastcheckpoint%????}
outputdir="${traindir}/${lastname}"
weights="${traindir}/${lastcheckpoint}"

echo "$outputdir --> $weights"

python ./demo/demo.py --config-file "${traindir}/config.yaml" \
  --input ../datasets/test_images/*.jpg \
  --output "${outputdir}" \
  --task $task \
  --opts MODEL.IS_TRAIN False MODEL.IS_DEMO True MODEL.WEIGHTS "${weights}"