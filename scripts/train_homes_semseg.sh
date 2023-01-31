export DETECTRON2_DATASETS=../datasets

python ./train_net.py --dist-url 'tcp://127.0.0.1:50163' \
    --num-gpus 2 \
    --config-file ./configs/synthhomes/oneformer_convnext_large_bs16_160k.yaml \
    --resume \
    OUTPUT_DIR ./output/synthhomes_semseg