TRAIN_NAME="blfloors_v2_uhd_semseg_lr10-e4"
SRVR_OUTPUT_DIR="/mnt/mlvolume/output"
SRVR_TRAIN_DIR="${SRVR_OUTPUT_DIR}/${TRAIN_NAME}"
SRVR_LAST_CHECK_PATH="${SRVR_TRAIN_DIR}/last_checkpoint"

OUTPUT_DIR="./output/${TRAIN_NAME}"
LAST_CHECK_PATH="${OUTPUT_DIR}/last_checkpoint.ext"

sshpass -p 'tH033208053' scp "thomashogarth@216.153.61.211:${SRVR_LAST_CHECK_PATH}" "${LAST_CHECK_PATH}"

lastcheckpoint=`cat $LAST_CHECK_PATH`
SRVR_CHECK_PATH="${SRVR_TRAIN_DIR}/${lastcheckpoint}"
CHECK_PATH="${OUTPUT_DIR}/${lastcheckpoint}"

sshpass -p 'tH033208053' scp "thomashogarth@216.153.61.211:${SRVR_CHECK_PATH}" "${CHECK_PATH}"
