export DETECTRON2_DATASETS=/mnt/mlvolume/datasets

python ./train_net.py --dist-url 'tcp://127.0.0.1:50163' \
    --num-gpus 8 \
    --config-file ./configs/blfoors/oneformer_convnext_large_bs16_160k.yaml \
    --resume \
    OUTPUT_DIR /mnt/mlvolume/output/blfloors_v2_640_semseg_lr10-e4 SOLVER.BASE_LR 1.0e-04