# PSMAReg (Learn2Reg 2026) — Docker Submission Instructions & Example

This folder is a **ready-to-run template** for containerizing a PSMAReg registration
method with Docker. It packages a working baseline — **ANTs Affine + ConvexAdam-MIND
(SVF)** — that you can build, run, and adapt into your own submission.

> The test set (~200 PET/CT pairs) will **not** be shared with participants. Instead you
> submit a Docker container; the organizers run it internally on the hidden test set. Your
> code is used only for evaluation and is never shared.

---

## 0. What the container must do (interface contract)

Your container registers **one moving/fixed set per run**. The organizers invoke it once
per test pair with **five positional arguments** (absolute paths inside the container):

```
<your_image>  <fixed_ct>  <fixed_pet>  <moving_ct>  <moving_pet>  <output_disp>
```

| # | Argument       | Meaning                                                        |
|---|----------------|----------------------------------------------------------------|
| 1 | `fixed_ct`     | Fixed (baseline) CT, `..._0000_00.nii.gz`                      |
| 2 | `fixed_pet`    | Fixed (baseline) PSMA-PET, `..._0001_00.nii.gz`               |
| 3 | `moving_ct`    | Moving (follow-up) CT, `..._0000_01.nii.gz`                    |
| 4 | `moving_pet`   | Moving (follow-up) PSMA-PET, `..._0001_01.nii.gz`            |
| 5 | `output_disp`  | Where to write the displacement field (path is chosen by us)   |

- **Both CT and PET are provided for the fixed and the moving image.** Your method may use
  either or both — the provided baseline drives registration with CT and reads PET for a QA
  preview, but a fully multimodal method is welcome.
- Input images are mounted under `/app/input` (read-only) and the output directory under
  `/app/output`. **Write the displacement field to exactly the 5th-argument path** — do not
  invent your own filename; the organizers pick it.
- The container must **not** assume any dataset layout, filenames, or the number of pairs.
  It receives explicit paths and processes exactly that one pair.

### Output format

- A single NIfTI displacement field, **channel-first `(3, X, Y, Z)`**, `float32`.
- **Original / full input resolution** — the same spatial grid as the fixed image
  (e.g. `192 × 192 × 288`), not a downsampled grid.
- Values are **displacements in voxels** on the fixed-image grid that map the **moving**
  image onto the **fixed** image (i.e. warping `moving` by the field aligns it to `fixed`).

The images are geometry-normalized to a common voxel size and shape
(`target_spacing = [2.7344, 2.7344, 3.27] mm`, `target_shape = [192, 192, 288]`); modality
code `0000 = CT`, `0001 = PSMA PET`; timepoint `00 = baseline (fixed)`,
`01 = follow-up (moving)`.

---

## 1. Files in this template

```
Docker_Example/
├── Dockerfile                 # builds the image (base: anibali/pytorch 2.0.1 / CUDA 11.8)
├── requirements.txt           # Python dependencies
├── infer_convexadam.py        # the inference entrypoint (one PET+CT set -> displacement)
├── MIR/                        # the MIR package (bundled, pip-installed inside the image)
├── build.sh                   # docker build helper
├── export.sh                  # docker save -> .tar.gz helper
├── test.sh                    # iterate the validation set locally (see §5)
├── PSMAReg_val_dataset.json   # validation manifest test.sh loops over
└── README.md                  # this file
```

You will typically **replace `infer_convexadam.py`** (and `requirements.txt` / `MIR/`) with
your own method, keeping the same 5-argument interface and output format.

---

## 2. The baseline method (what the example does)

`infer_convexadam.py` reproduces the `ants_affine_convexadam_svf` baseline:

1. Remove the CT table/bed, apply a `[-300, 300] HU` window, and downsample the CT by a
   factor of 2.
2. **ANTs affine** registration on the low-resolution CT.
3. **ConvexAdam-MIND** deformable registration (SVF parameterization) on the affine-warped
   low-resolution CT.
4. **Compose** the deformable field *after* the affine field into a single transform.
5. **Up-sample** the composed field back to the **original resolution** and save it.

It is optimization-based (no learned weights, no GPU training). Typical runtime is
**~15 s/pair** on an NVIDIA GPU. Registration is CT-driven; the PET inputs are read and
warped by the final field only for the optional QA preview (`--save-preview`).

---

## 3. Build your Docker image

Install [Docker](https://docs.docker.com/get-docker/) first. Then, from this directory:

```bash
docker build -f Dockerfile -t psmareg_convexadam .
# or:
bash build.sh
```

The example `Dockerfile` (adapt as needed for your method):

```dockerfile
FROM anibali/pytorch:2.0.1-cuda11.8-ubuntu22.04

# Optional: set working directory
WORKDIR /app

# Set up time zone
ENV TZ=UTC
RUN sudo ln -snf /usr/share/zoneinfo/$TZ /etc/localtime

# Copy requirements first for caching
COPY ./requirements.txt .

# Install system and Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy only necessary files for runtime
COPY ./infer_convexadam.py /app/
COPY ./MIR /app/MIR
RUN sudo chmod -R a+rw /app/MIR
# Install the MIR package (editable if you plan to modify)
WORKDIR /app/MIR
RUN pip install --no-cache-dir -e .

RUN mkdir -p /app/input /app/output
RUN chmod -R 777 /app/output
WORKDIR /app

# One moving/fixed PET+CT set per run. The caller passes five paths:
#   fixed CT, fixed PET, moving CT, moving PET, output displacement.
ENTRYPOINT ["python3","-u","/app/infer_convexadam.py"]
```

Key points if you build your own:

- Use a **GPU-based PyTorch** base image (recommended for reasonable runtime).
- Use **`ENTRYPOINT`** (not `CMD`) so the five positional path arguments are forwarded to
  your script. Test it: `docker run --rm your_image --help` should reach your argument
  parser.
- If your method needs learned weights, `COPY` them into the image (they are not downloaded
  at runtime — evaluation containers run **offline**, see §4).

---

## 4. How the organizers run your container

Once per test pair, using a command of this form (paths are examples):

```bash
docker run --rm \
    --ipc=host \
    --memory 60g \
    --gpus "device=0" \
    --user $(id -u):$(id -g) \
    --network=none \
    --mount type=bind,source=[test image dir],target=/app/input,readonly \
    --mount type=bind,source=[predictions dir],target=/app/output \
    [your image name] \
        /app/input/PSMARegPSMA_XXXX_0000_00.nii.gz \
        /app/input/PSMARegPSMA_XXXX_0001_00.nii.gz \
        /app/input/PSMARegPSMA_XXXX_0000_01.nii.gz \
        /app/input/PSMARegPSMA_XXXX_0001_01.nii.gz \
        /app/output/disp_XXXX_00_XXXX_01.nii.gz
```

Notes:

- `--network=none`: containers run **offline**. Bundle everything you need at build time.
- `--gpus "device=0"`: one GPU is available.
- Your container must exit cleanly and leave the displacement field at the given output
  path.

### Runtime budget

The test set is **~200 PET/CT pairs**. Please estimate your total runtime and keep the
per-pair time reasonable so the full set finishes comfortably (confirm any hard limit on the
official challenge page). The baseline here runs in **~15 s/pair** on GPU, i.e. roughly
**~50 min** for the whole test set — leaving ample margin.

### Reference evaluation environment

> The evaluation hardware/software below is provided as a reference; confirm the exact
> specification on the official challenge page.

- CPU: Intel(R) Xeon(R) Silver 4410Y
- GPU: NVIDIA H100 (80 GB VRAM)
- RAM: 60 GB
- Driver 545.23.06, CUDA 12.3, Docker 26.1.3, Rocky Linux 8

---

## 5. Test your container on the validation set

The container itself does **one** pair per run, so validation is done by looping over the
validation pairs. `test.sh` does exactly this: it builds the image, then reads
`PSMAReg_val_dataset.json` and calls the container once per pair.

```bash
bash test.sh
```

Edit the paths at the top of `test.sh` first:

- `DATA_DIR` — the dataset root containing `imagesVal/` (mounted at `/app/input`).
- `DATASET_JSON` — `PSMAReg_val_dataset.json` (the validation manifest).
- `OUTPUT_DIR` — where predictions are written (mounted at `/app/output`).

`PSMAReg_val_dataset.json` mirrors the official dataset JSON but lists only the validation
split. Each entry defines one longitudinal pair:

```json
{
  "subject": "PSMARegPSMA_0001",
  "Baseline CT":  "./imagesVal/PSMARegPSMA_0001_0000_00.nii.gz",
  "Baseline PET": "./imagesVal/PSMARegPSMA_0001_0001_00.nii.gz",
  "Follow-up 01 CT":  "./imagesVal/PSMARegPSMA_0001_0000_01.nii.gz",
  "Follow-up 01 PET": "./imagesVal/PSMARegPSMA_0001_0001_01.nii.gz"
}
```

`test.sh` maps each entry to the container arguments as:

```
fixed  = Baseline     ->  Baseline CT ,  Baseline PET
moving = Follow-up 01 ->  Follow-up 01 CT ,  Follow-up 01 PET
output =  disp_<id>_00_<id>_01.nii.gz
```

After it finishes, `OUTPUT_DIR` contains one `disp_XXXX_00_XXXX_01.nii.gz` per validation
subject — these are your **validation predictions** for submission.

---

## 6. Export your image

```bash
docker save psmareg_convexadam | gzip -c > psmareg_convexadam.tar.gz
# or:
bash export.sh
```

---

## 7. Submission checklist

Package a single `.zip`:

```
PSMAReg_[your Grand Challenge username]_TestPhase.zip
├── [your image name].tar.gz     <-- your exported Docker container (docker save | gzip)
├── README.txt                    <-- requirements to run your model: #CPUs, RAM,
│                                     estimated time/pair, GPU VRAM, and the exact
│                                     `docker run` argument order you expect
└── validation_predictions.zip    <-- your validation displacement fields
    ├── disp_0001_00_0001_01.nii.gz
    ├── disp_0003_00_0003_01.nii.gz
    └── ...
```

Because the organizers reproduce your validation predictions locally, please make the
container **deterministic** where possible and document any nondeterminism in `README.txt`.

---

## 8. Paper submission (Springer LNCS proceedings)

This year the challenge runs **full paper proceedings**. To be eligible for evaluation and
prizes, submit a full method paper prepared with the **Springer LNCS templates** (LaTeX or
Word), available from the
[Springer LNCS author guidelines](https://www.springer.com/gp/computer-science/lncs/conference-proceedings-guidelines).
Note that LNCS itself does **not** fix a universal page count — the page limits are set by
the challenge, not by Springer. The PSMAReg requirement is **at least 6 pages**; confirm the
exact minimum/maximum on the official challenge call for papers.

Your paper should give a clear, reproducible description of the method, the computing
infrastructure used (hardware and software), and validation-leaderboard results
(mean ± std per metric).

**Peer review and originality gating.** Papers are submitted through **OpenReview** and
undergo **peer review**. Submissions are gated for originality: a paper found to have
**significant overlap with existing publications will not be accepted** into the proceedings,
and its authors will be **ineligible to compete for prizes** and to be included in any
challenge summary papers.

> Deadlines, the OpenReview submission link, and any required citations are announced on the
> official Learn2Reg 2026 / PSMAReg challenge page — follow those for the authoritative dates
> and requirements.

---

## Questions

For questions about the Docker submission, contact the organizers at **jchen245 [at]
jhmi.edu**.
