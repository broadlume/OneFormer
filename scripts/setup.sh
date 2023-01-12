
conda create --name oneformerX python=3.8 -y
conda activate oneformerX

# Install Pytorch
conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -c conda-forge

# Install opencv (required for running the demo)
pip3 install -U opencv-python

# Install detectron2
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

# Install other dependencies
pip3 install git+https://github.com/cocodataset/panopticapi.git
pip3 install git+https://github.com/mcordts/cityscapesScripts.git
pip3 install -r requirements.txt

cd oneformer/modeling/pixel_decoder/ops
sh make.sh
cd ../../../..

# fetch the pretained weights for training
/bin/bash ./scripts/fetch_pretrained_convnext.sh
/bin/bash ./scripts/fetch_pretrained_dianet.sh
/bin/bash ./scripts/fetch_pretrained_swin.sh
