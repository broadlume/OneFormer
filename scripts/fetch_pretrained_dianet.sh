wget -P ./checkpoints https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_large_in22k_in1k_384_11x11.pth
python ./tools/convert-pretrained-nat-model-to-d2.py ./checkpoints/dinat_large_in22k_in1k_384_11x11.pth ./checkpoints/dinat_large_in22k_in1k_384_11x11.pkl

wget -P ./checkpoints https://shi-labs.com/projects/dinat/checkpoints/imagenet22k/dinat_large_in22k_224.pth
python ./tools/convert-pretrained-nat-model-to-d2.py ./checkpoints/dinat_large_in22k_224.pth ./checkpoints/dinat_large_in22k_224.pkl