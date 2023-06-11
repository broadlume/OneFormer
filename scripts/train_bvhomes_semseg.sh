export DETECTRON2_DATASETS=/media/tom/Data/datasets

python ./train_net.py --dist-url 'tcp://127.0.0.1:50163' \
    --num-gpus 2 \
    --config-file ./configs/bvhomes/oneformer_convnext_large_bs16_160k.yaml \
    --resume \
    OUTPUT_DIR ./output/bvhomes_01