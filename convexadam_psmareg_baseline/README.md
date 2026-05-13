# PSMAReg ANTs Affine + ConvexAdam Baseline

Standalone baseline for the Learn2Reg26 PSMAReg task. It estimates moving-to-fixed displacement fields named
`disp_<subject>_00_<subject>_01.nii.gz` and packages them into a Codabench-style submission zip.

The method includes:

1. CT bed/table suppression.
2. CT window normalization, default `[-300, 300] -> [0, 1]`.
3. ANTsPy affine registration, `moving` follow-up CT to `fixed` baseline CT.
4. Optional ConvexAdam MIND/SVF residual registration on the affine-warped moving CT.
5. Composition into one forward moving-to-fixed voxel-displacement field.

## Official Submission Convention

The generated displacement fields follow the official PSMAReg evaluation format:

- File names: `disp_<subject>_00_<subject>_01.nii.gz`.
- Files live at the root of the submission zip, with no nested directory.
- Default field shape: `(3, 96, 96, 144)` for `--downsample-factor 2`.
- Channel order: displacement components first, then spatial dimensions.
- Direction: forward moving-to-fixed, so warping follow-up/moving images with the field aligns them to baseline/fixed images.
- Header spacing is set to the fixed image spacing multiplied by the downsample factor.

## Requirements

Python 3.8 is recommended.

[MIR package](https://github.com/junyuchen245/MIR) is required to run this baseline.

```bash
pip install -r requirements.txt
```

The script also requires the MIR package, either installed in the Python environment or supplied with:

```bash
--mir-src /path/to/custom_packages/MIR/src
```

## Run

```bash
python3.8 estimate_displacements.py \
  --reference-dir /path/to/reference_data \
  --output-dir ./outputs/ants_affine_convexadam \
  --mir-src /scratch/jchen/python_projects/custom_packages/MIR/src \
  --stage affine-convex \
  --ants-transform Affine \
  --affine-field-transform fwd \
  --downsample-factor 2 \
  --convex-grid-sp 4 \
  --convex-niter 80 \
  --convex-smooth 3 \
  --dtype float32 \
  --overwrite

python3.8 zip_submission.py \
  --prediction-dir ./outputs/ants_affine_convexadam/predictions \
  --zip-path ./outputs/ants_affine_convexadam/predictions.zip
```

Submit `predictions.zip`.

Validate before submission:

```bash
python3.8 validate_submission.py \
  --zip-path ./outputs/ants_affine_convexadam/predictions.zip \
  --dataset-json /path/to/reference_data/PSMAReg_dataset.json
```

## Notes

- Use `--affine-field-transform fwd`; this is the calibrated forward moving-to-fixed convention for this baseline.
- `--stage affine` runs affine-only. `--stage affine-convex` runs affine plus ConvexAdam.
- By default, no intermediate warped images or dense debug fields are saved. Keep this behavior for disk-light challenge runs.
- Add `--save-previews` if you want compact PNG diagnostics.
