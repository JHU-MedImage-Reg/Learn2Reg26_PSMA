#!/usr/bin/env bash
set -euo pipefail

REFERENCE_DIR="${1:?Usage: run_baseline.sh /path/to/reference_data /path/to/output_dir /path/to/MIR/src}"
OUTPUT_DIR="${2:?Usage: run_baseline.sh /path/to/reference_data /path/to/output_dir /path/to/MIR/src}"
MIR_SRC="${3:?Usage: run_baseline.sh /path/to/reference_data /path/to/output_dir /path/to/MIR/src}"

python3.8 "$(dirname "$0")/estimate_displacements.py" \
  --reference-dir "${REFERENCE_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --mir-src "${MIR_SRC}" \
  --stage affine-convex \
  --ants-transform Affine \
  --affine-field-transform fwd \
  --downsample-factor 2 \
  --convex-grid-sp 4 \
  --convex-niter 80 \
  --convex-smooth 3 \
  --dtype float32 \
  --overwrite

python3.8 "$(dirname "$0")/zip_submission.py" \
  --prediction-dir "${OUTPUT_DIR}/predictions" \
  --zip-path "${OUTPUT_DIR}/predictions.zip"

python3.8 "$(dirname "$0")/validate_submission.py" \
  --zip-path "${OUTPUT_DIR}/predictions.zip" \
  --dataset-json "${REFERENCE_DIR}/PSMAReg_dataset.json"
