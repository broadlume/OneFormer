wget -P ./checkpoints https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_1k_384.pth
python ./tools/convert-pretrained-model-to-d2.py ./checkpoints/convnext_large_22k_1k_384.pth ./checkpoints/convnext_large_22k_1k_384.pkl

wget -P ./checkpoints https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_1k_384_ema.pth
python ./tools/convert-pretrained-model-to-d2.py ./checkpoints/convnext_xlarge_22k_1k_384_ema.pth ./checkpoints/convnext_xlarge_22k_1k_384_ema.pkl