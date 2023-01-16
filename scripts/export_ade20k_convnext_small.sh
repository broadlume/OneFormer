export task=panoptic

CUDA_VISIBLE_DEVICES=0 python ./demo/export.py --config-file ./configs/ade20k/convnext/oneformer_convnext_small_bs16_160k.yaml \
  --output ./output/ade20k/convnext640_small/model.pt \
  --task $task \
  --opts MODEL.IS_TRAIN False MODEL.IS_DEMO True MODEL.WEIGHTS ./output/ade20k_convnext_large/model_0079999.pth