export DETECTRON2_DATASETS=/media/tom/Data/datasets

python ./train_net.py --dist-url 'tcp://127.0.0.1:50163' \
    --num-gpus 2 \
    --config-file ./configs/blfloors/oneformer_convnext_large_bs16_160k.yaml \
    --resume \
    OUTPUT_DIR ./output/blfloors_v2_semseg_lr10-e4 SOLVER.BASE_LR 1.0e-04