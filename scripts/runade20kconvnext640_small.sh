export task=panoptic

python ./demo/demo.py --config-file ./configs/ade20k/convnext/oneformer_convnext_large_bs16_160k_2gpu.yaml \
  --input ../datasets/test_images/*.jpg \
  --output ./output/ade20k/convnext640_small_0009999 \
  --task $task \
  --opts MODEL.DEVICE cpu MODEL.IS_TRAIN False MODEL.IS_DEMO True MODEL.WEIGHTS ./output/ade20k_convnext_large/model_0009999.pth