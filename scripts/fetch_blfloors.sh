DATASET_DIR="/mnt/mlvolume/datasets/vmdatasets"
OUTPUT_PATH="${DATASET_DIR}blfloors.zip"

wget -O $OUTPUT_PATH https://www.dropbox.com/s/5utaqvbokky6kst/datasets.zip?dl=1
unzip $OUTPUT_PATH -d $DATASET_DIR
