DATASET_DIR="/mnt/mlvolume/datasets/vmdatasets"
OUTPUT_PATH="${DATASET_DIR}blfloors.zip"

wget -O $OUTPUT_PATH https://www.dropbox.com/s/lydzu7dfh8806il/blfloors.zip?dl=1
unzip $OUTPUT_PATH -d $DATASET_DIR
