pip install timm

wget -P ./checkpoints https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth
python ./tools/convert-pretrained-model-to-d2.py ./checkpoints/swin_large_patch4_window12_384_22k.pth ./checkpoints/swin_large_patch4_window12_384_22k.pkl

wget -P ./checkpoints https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22kto1k.pth
python ./tools/convert-pretrained-model-to-d2.py ./checkpoints/swin_large_patch4_window12_384_22kto1k.pth ./checkpoints/swin_large_patch4_window12_384_22kto1k.pkl