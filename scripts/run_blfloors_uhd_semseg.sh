#!/bin/bash

export task=semantic

traindir="./output/blfloors_v2_uhd_semseg_lr10-e4"
lastcheckpoint=`cat $traindir/last_checkpoint`
lastname=${lastcheckpoint%????}
outputdir="${traindir}/${lastname}"
weights="${traindir}/${lastcheckpoint}"

echo "$outputdir --> $weights"

python ./demo/demo.py --config-file "${traindir}/config.yaml" \
  --input ../datasets/fail_images/*.jpg \
  --output "${outputdir}_fails" \
  --task $task \
  --opts MODEL.IS_TRAIN False MODEL.IS_DEMO True MODEL.WEIGHTS "${weights}"