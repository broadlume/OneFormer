export task=panoptic

python ./demo/demo.py --config-file ./configs/ade20k/dinat/oneformer_dinat_large_bs16_160k_1280x1280.yaml \
  --input ../datasets/test_images/*.jpg \
  --output ./output/ade20k/dinat1280 \
  --task $task \
  --opts MODEL.IS_TRAIN False MODEL.IS_DEMO True MODEL.WEIGHTS ./checkpoints/ade20k/dinat/1280x1280_250_16_dinat_l_oneformer_ade20k_160k.pth