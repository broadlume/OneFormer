DATASET_DIR=../datasetsX
export DETECTRON2_DATASETS=$DATASET_DIR

wget -P $DATASET_DIR http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
unzip $DATASET_DIR/ADEChallengeData2016.zip -d $DATASET_DIR

wget -P $DATASET_DIR http://sceneparsing.csail.mit.edu/data/ChallengeData2017/annotations_instance.tar
tar -xvf $DATASET_DIR/annotations_instance.tar -C $DATASET_DIR/ADEChallengeData2016

python ./datasets/prepare_ade20k_sem_seg.py
python ./datasets/prepare_ade20k_pan_seg.py
python ./datasets/prepare_ade20k_ins_seg.py