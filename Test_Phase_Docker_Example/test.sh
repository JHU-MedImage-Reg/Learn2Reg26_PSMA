#!/usr/bin/env bash
bash ./build.sh
#docker load --input psmareg_convexadam.tar.gz

# The container registers ONE moving/fixed PET+CT set per run (five paths: fixed CT,
# fixed PET, moving CT, moving PET, output). This script iterates the VALIDATION set
# described by a dataset JSON and calls the container once per pair.
#   fixed  = Baseline    (CT 0000, PET 0001, timepoint 00)
#   moving = Follow-up 01 (CT 0000, PET 0001, timepoint 01)
#   output = disp_<id>_00_<id>_01.nii.gz
DATA_DIR=/scratch2/jchen/DATA/PSMA_autoPET/PSMAReg_PSMA_preprocessed_27344_327_192x192x288
IMAGES_DIR=${DATA_DIR}/imagesVal
DOCKER_DIR=/scratch/jchen/python_projects/custom_packages/MIR/tutorials/PSMAReg/Docker_Example
DATASET_JSON=${DOCKER_DIR}/PSMAReg_val_dataset.json
OUTPUT_DIR=${DOCKER_DIR}/PSMAReg_convexadam_TestPhase
mkdir -p "${OUTPUT_DIR}"

# Extract one line per validation pair (basenames):
#   fixed_ct  fixed_pet  moving_ct  moving_pet  output
while read -r FCT FPT MCT MPT OUT; do
    [ -z "${FCT}" ] && continue
    echo "=== Registering ${OUT} ==="
    docker run --rm \
        --ipc=host \
        --memory 256g \
        --gpus "device=0" \
        --user $(id -u):$(id -g) \
        --network=none \
        --mount type=bind,source=${IMAGES_DIR},target=/app/input,readonly \
        --mount type=bind,source=${OUTPUT_DIR},target=/app/output \
        psmareg_convexadam \
            /app/input/${FCT} \
            /app/input/${FPT} \
            /app/input/${MCT} \
            /app/input/${MPT} \
            /app/output/${OUT} < /dev/null
done < <(python3 - "${DATASET_JSON}" <<'PY'
import json, os, sys
d = json.load(open(sys.argv[1]))
for e in d["validation_paired"]:
    sid = e["subject"].split("_")[-1]
    print(os.path.basename(e["Baseline CT"]),
          os.path.basename(e["Baseline PET"]),
          os.path.basename(e["Follow-up 01 CT"]),
          os.path.basename(e["Follow-up 01 PET"]),
          "disp_{0}_00_{0}_01.nii.gz".format(sid))
PY
)

echo "All validation displacement fields written to ${OUTPUT_DIR}"
