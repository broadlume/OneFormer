export task=semantic

#CUDA_VISIBLE_DEVICES=0 python ./demo/export.py --config-file ./configs/ade20k/convnext/oneformer_convnext_large_bs16_160k_2gpu.yaml \
#  --output ./output/ade20k/convnext640_small/model_0099999.pt \
#  --task $task \
#  --opts MODEL.IS_TRAIN False MODEL.IS_DEMO True MODEL.IS_EXPORT True MODEL.WEIGHTS ./output/ade20k_convnext_large/model_0099999.pth
python ./demo/export_model.py --config-file ./configs/ade20k/convnext/oneformer_convnext_large_bs16_160k_2gpu.yaml \
   --format onnx \
   --export-method tracing \
   --sample-image ../datasets/512x512.jpg \
   --run-eval \
   --output ./output \
   MODEL.IS_TRAIN False MODEL.IS_DEMO True MODEL.WEIGHTS ./output/ade20k_convnext_large/model_0099999.pth