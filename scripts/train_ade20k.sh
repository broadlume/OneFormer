export DETECTRON2_DATASETS=../datasetsX

python ./train_net.py --dist-url 'tcp://127.0.0.1:50163' \
    --num-gpus 2 \
    --config-file ./configs/ade20k/convnext/oneformer_convnext_large_bs16_160k_2gpu.yaml \
    OUTPUT_DIR ./output/ade20k_convnext_large