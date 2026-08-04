"""PSMAReg (Learn2Reg 2026) baseline registration: ANTs Affine + ConvexAdam-MIND (SVF).

One moving/fixed set per run. Given the CT and PSMA-PET of a fixed (baseline) and a
moving (follow-up) scan, it reads the four images and generates ONE displacement field:

    1. remove CT table/bed, window-normalize, downsample CT (factor 2),
    2. ANTs affine registration on the low-resolution CT,
    3. ConvexAdam-MIND deformable (SVF) registration on the affine-warped CT,
    4. compose the deformable field AFTER the affine field,
    5. up-sample the composed field to the ORIGINAL image resolution and save it.

Both fixed and moving provide CT + PET (the challenge input interface participants will
use). This baseline drives the registration with CT only and warps PET for the optional
QA preview, but PET is a required input so the example mirrors a real PET+CT submission.
Output: ``(3, X, Y, Z)`` float32 displacement field at the original input resolution, in
voxel units - the composed ANTs-affine + ConvexAdam field.

Hyper-parameters reproduce the validated experiment
``ants_affine_convexadam_svf / val_affine_convex_svf_df2_g4_n80_smooth3_fwd``.

The container makes NO assumption about dataset layout, filenames, or naming
conventions: the caller passes five paths - fixed CT, fixed PET, moving CT, moving PET,
output - as the Docker ENTRYPOINT arguments::

    <image>  /app/input/FIXED_CT.nii.gz  /app/input/FIXED_PET.nii.gz \
             /app/input/MOVING_CT.nii.gz /app/input/MOVING_PET.nii.gz \
             /app/output/DISP.nii.gz
"""

import argparse
import os
import time
from pathlib import Path

import ants
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from scipy.ndimage import zoom

from MIR.models.convexAdam.convex_adam_MIND_SVF import convex_adam_pt_svf
from MIR.models.registration_utils import SpatialTransformer


# --------------------------------------------------------------------------- #
# Registration hyper-parameters
# (match ants_affine_convexadam_svf / val_affine_convex_svf_df2_g4_n80_smooth3_fwd)
# --------------------------------------------------------------------------- #
DOWNSAMPLE_FACTOR = 2
CT_WINDOW = (-300.0, 300.0)
REMOVE_BED = True

ANTS_TRANSFORM = "Affine"          # Rigid | Affine | TRSAA
AFF_METRIC = "mattes"
AFF_SAMPLING = 64
AFFINE_FIELD_TRANSFORM = "fwd"     # "fwd" (forward) | "inverse"

CONVEX_MIND_R = 1
CONVEX_MIND_D = 2
CONVEX_LAMBDA = 2.0
CONVEX_GRID_SP = 4
CONVEX_DISP_HW = 4
CONVEX_NITER = 80
CONVEX_SMOOTH = 3
CONVEX_GRID_SP_ADAM = 2
SVF_STEPS = 7
IC = True
DTYPE = "float32"                  # float32 | float16 (float16 only on CUDA)


# --------------------------------------------------------------------------- #
# Pre-processing helpers (verbatim from the validated baseline)
# --------------------------------------------------------------------------- #
def robust_normalize(volume):
    volume = np.asarray(volume, dtype=np.float32)
    mask = np.isfinite(volume)
    if not mask.any():
        return np.zeros_like(volume, dtype=np.float32)
    values = volume[mask]
    lo, hi = np.percentile(values, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        out = np.zeros_like(volume, dtype=np.float32)
    else:
        out = (np.clip(volume, lo, hi) - lo) / (hi - lo)
    out[~mask] = 0.0
    return out.astype(np.float32)


def ct_window_normalize(volume, window):
    lo, hi = [float(value) for value in window]
    if hi <= lo:
        raise ValueError("Invalid CT window: {}".format(window))
    volume = np.asarray(volume, dtype=np.float32)
    out = (np.clip(volume, lo, hi) - lo) / (hi - lo)
    out[~np.isfinite(volume)] = 0.0
    return out.astype(np.float32)


def get_largest_cc(segmentation):
    labels, num_labels = ndimage.label(segmentation)
    if num_labels == 0:
        return np.zeros_like(segmentation, dtype=bool)
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    return labels == int(np.argmax(counts))


def slice_center_mask(shape):
    center_mask = np.zeros(shape, dtype=bool)
    x0 = int(round(shape[0] * 0.2))
    x1 = int(round(shape[0] * 0.8))
    y0 = int(round(shape[1] * 0.2))
    y1 = int(round(shape[1] * 0.8))
    center_mask[x0:x1, y0:y1] = True
    return center_mask


def slice_border_hits(component):
    return int(component[0, :].sum() + component[-1, :].sum() + component[:, 0].sum() + component[:, -1].sum())


def component_extent(coords):
    return coords.max(axis=0) - coords.min(axis=0) + 1


def select_body_components(mask2d, center_mask, prev_support):
    labels, num_labels = ndimage.label(mask2d)
    if num_labels == 0:
        return np.zeros_like(mask2d, dtype=bool)

    selected = np.zeros_like(mask2d, dtype=bool)
    support = None if prev_support is None else ndimage.binary_dilation(prev_support, structure=np.ones((9, 9), dtype=bool))
    fallback_component = np.zeros_like(mask2d, dtype=bool)
    fallback_score = -np.inf
    for label_idx in range(1, num_labels + 1):
        component = labels == label_idx
        coords = np.argwhere(component)
        area = int(coords.shape[0])
        if area < 64:
            continue
        extent = component_extent(coords)
        if int(extent.min()) < 6:
            continue
        center_hits = int(np.logical_and(component, center_mask).sum())
        border_hits = slice_border_hits(component)
        overlap_hits = 0 if support is None else int(np.logical_and(component, support).sum())
        score = float(area + 4 * center_hits + 8 * extent.min() + 6 * overlap_hits - 3 * border_hits)
        if support is not None and overlap_hits > 0:
            selected |= component
            continue
        if center_hits > 0 and border_hits < int(0.35 * max(area, 1)):
            selected |= component
            continue
        if score > fallback_score:
            fallback_score = score
            fallback_component = component

    if selected.sum() == 0:
        selected = fallback_component
    return selected


def remove_bed(img):
    img = np.asarray(img, dtype=np.float32).copy()
    body_candidate = img >= -700
    body_candidate = ndimage.binary_opening(body_candidate, structure=np.ones((3, 3, 3), dtype=bool))

    tracked_mask = np.zeros_like(body_candidate, dtype=bool)
    center_mask = slice_center_mask(body_candidate.shape[:2])
    mid_slice = body_candidate.shape[2] // 2

    prev_support = None
    for z_idx in range(mid_slice, -1, -1):
        current = select_body_components(body_candidate[:, :, z_idx], center_mask, prev_support)
        tracked_mask[:, :, z_idx] = current
        if current.any():
            prev_support = current

    prev_support = None
    for z_idx in range(mid_slice + 1, body_candidate.shape[2]):
        current = select_body_components(body_candidate[:, :, z_idx], center_mask, prev_support)
        tracked_mask[:, :, z_idx] = current
        if current.any():
            prev_support = current

    if tracked_mask.sum() == 0:
        tracked_mask = get_largest_cc(body_candidate)
    else:
        tracked_mask = ndimage.binary_fill_holes(tracked_mask)
        tracked_mask = get_largest_cc(tracked_mask)

    tracked_mask = ndimage.binary_closing(tracked_mask, structure=np.ones((5, 5, 3), dtype=bool))
    tracked_mask = ndimage.binary_fill_holes(tracked_mask)
    tracked_mask = ndimage.binary_dilation(tracked_mask, structure=np.ones((3, 3, 3), dtype=bool))
    img[tracked_mask == 0] = np.percentile(img, 0.5)
    return img


def downsample_volume(volume, factor, order=1):
    if factor == 1:
        return np.asarray(volume, dtype=np.float32)
    scale = tuple(1.0 / factor for _ in range(3))
    return zoom(volume, zoom=scale, order=order).astype(np.float32)


def preprocess_ct(volume):
    volume = np.asarray(volume, dtype=np.float32)
    if REMOVE_BED:
        volume = remove_bed(volume)
    return volume


def make_lowres_ants_image(volume, spacing, factor, ct_window):
    lowres = downsample_volume(ct_window_normalize(volume, ct_window), factor=factor, order=1)
    image = ants.from_numpy(lowres)
    image.set_spacing(tuple(float(s) * factor for s in spacing))
    image.set_origin((0.0, 0.0, 0.0))
    image.set_direction(np.diag([-1.0, -1.0, 1.0]))
    return image


# --------------------------------------------------------------------------- #
# ANTs affine -> full-resolution voxel displacement (verbatim from the baseline)
# --------------------------------------------------------------------------- #
def ants_affine_to_fullres_voxel_disp(transform, reference_image, fullres_spacing):
    params = np.asarray(transform.parameters, dtype=np.float32)
    fixed_params = np.asarray(transform.fixed_parameters, dtype=np.float32)
    if params.size != 12 or fixed_params.size < 3:
        raise ValueError(
            "Expected a 3D affine transform with 12 parameters and center fixed parameters; "
            "got params={} fixed={}".format(params.shape, fixed_params.shape)
        )
    matrix = params[:9].reshape(3, 3)
    translation = params[9:12]
    center = fixed_params[:3]
    shape = tuple(int(v) for v in reference_image.shape)
    spacing_lr = np.asarray(reference_image.spacing, dtype=np.float32)
    direction = np.asarray(reference_image.direction, dtype=np.float32)
    origin = np.asarray(reference_image.origin, dtype=np.float32)
    grids = np.meshgrid(
        np.arange(shape[0], dtype=np.float32),
        np.arange(shape[1], dtype=np.float32),
        np.arange(shape[2], dtype=np.float32),
        indexing="ij",
    )
    index = np.stack(grids, axis=-1).reshape(-1, 3)
    physical = origin + (index * spacing_lr).dot(direction.T)
    moved = (physical - center).dot(matrix.T) + center + translation
    disp = (moved - physical).reshape(shape + (3,)).astype(np.float32)
    return ants_physical_delta_to_fullres_voxel_disp(disp, direction, fullres_spacing)


def ants_physical_delta_to_fullres_voxel_disp(disp, direction, fullres_spacing):
    spacing = np.asarray(fullres_spacing, dtype=np.float32)
    flat = disp.reshape(-1, 3)
    # ANTs vector components are physical-space. Convert physical deltas to index
    # deltas in the scorer's numpy axis order, preserving the LPS direction flips.
    index_delta = flat.dot(direction) / spacing
    return index_delta.reshape(disp.shape).astype(np.float32)


def channel_last_to_channel_first(field):
    if field.shape[-1] == 3:
        return np.moveaxis(field, -1, 0)
    if field.shape[0] == 3:
        return field
    raise ValueError("Expected displacement field with 3 channels.")


def compose_deform_after_affine(deform_fullres_vox, affine_fullres_vox):
    """Total field = deform + (affine sampled at deform locations) -> deform AFTER affine."""
    deform = torch.from_numpy(channel_last_to_channel_first(deform_fullres_vox)).unsqueeze(0).float()
    affine = torch.from_numpy(channel_last_to_channel_first(affine_fullres_vox)).unsqueeze(0).float()
    transformer = SpatialTransformer(tuple(deform.shape[2:]), mode="bilinear")
    with torch.no_grad():
        sampled_affine = transformer(affine, deform, padding_mode="border")
        total = deform + sampled_affine
    return total.squeeze(0).numpy().astype(np.float32)


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #
def upsample_field_to(field_ch_first, target_shape, factor):
    """Resample a low-resolution (3, x, y, z) voxel field to the original grid.

    The grid is trilinearly interpolated to ``target_shape`` and the vector
    magnitudes are scaled by ``factor`` to convert low-resolution voxel units to
    original-resolution voxel units.
    """
    field = torch.from_numpy(np.ascontiguousarray(field_ch_first)).unsqueeze(0).float()
    with torch.no_grad():
        up = F.interpolate(
            field, size=tuple(int(s) for s in target_shape), mode="trilinear", align_corners=False
        )
    return (up * float(factor)).squeeze(0).numpy().astype(np.float32)


def save_disp(path, field_ch_first, affine, spacing):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(field_ch_first.astype(np.float32), affine)
    image.header["pixdim"][1:4] = [float(s) for s in spacing]
    nib.save(image, str(path))


def warp_volume(volume, field_ch_first, mode="bilinear"):
    field = torch.from_numpy(np.ascontiguousarray(field_ch_first)).unsqueeze(0).float()
    src = torch.from_numpy(volume.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    transformer = SpatialTransformer(tuple(field.shape[2:]), mode=mode)
    with torch.no_grad():
        warped = transformer(src, field, padding_mode="border").squeeze().numpy()
    return warped.astype(np.float32)


def mid_slices(volume):
    x, y, z = [int(dim) // 2 for dim in volume.shape[:3]]
    return [
        ("sag", np.rot90(volume[x, :, :])),
        ("cor", np.rot90(volume[:, y, :])),
        ("axi", np.rot90(volume[:, :, z])),
    ]


def save_preview_png(path, rows, title):
    fig, axes = plt.subplots(len(rows), 3, figsize=(9, 2.4 * len(rows)), squeeze=False)
    fig.suptitle(title, fontsize=10)
    for r, (row_name, volume) in enumerate(rows):
        for c, (view_name, image) in enumerate(mid_slices(volume)):
            axes[r, c].imshow(image, cmap="gray", origin="lower", vmin=0.0, vmax=1.0)
            axes[r, c].set_title("{} {}".format(row_name, view_name), fontsize=8)
            axes[r, c].axis("off")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(path), dpi=160, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Single-pair registration
# --------------------------------------------------------------------------- #
def register(fixed_ct, moving_ct, output, fixed_pet=None, moving_pet=None,
             device=None, save_preview=False, preview_path=None):
    """Register moving -> fixed with CT-driven ANTs affine + ConvexAdam-MIND (SVF).

    Args:
        fixed_ct / moving_ct: paths to the fixed and moving CT NIfTI files (drive the
            registration).
        output: path for the composed, original-resolution displacement field.
        fixed_pet / moving_pet: optional PET paths - accepted for the challenge input
            interface; used only for the optional QA preview, not for the registration.
        device: torch device (defaults to CUDA when available).
        save_preview: if True, also warp the moving CT + PET and dump a QA PNG.
        preview_path: where to write the QA PNG (defaults next to ``output``).

    Returns:
        dict with basic diagnostics about the produced field.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    # 1. Load CT (drives registration). PET is loaded on demand for the preview only.
    fixed_ct_nii = nib.load(str(fixed_ct))
    moving_ct_nii = nib.load(str(moving_ct))
    fixed_ct_arr = fixed_ct_nii.get_fdata(dtype=np.float32)
    moving_ct_arr = moving_ct_nii.get_fdata(dtype=np.float32)
    fullres_shape = tuple(int(s) for s in fixed_ct_arr.shape[:3])
    fullres_spacing = tuple(float(s) for s in fixed_ct_nii.header.get_zooms()[:3])

    # 2. ANTs affine on the low-resolution CT.
    fixed_pre = preprocess_ct(fixed_ct_arr)
    moving_pre = preprocess_ct(moving_ct_arr)
    fixed_ants = make_lowres_ants_image(fixed_pre, fullres_spacing, DOWNSAMPLE_FACTOR, CT_WINDOW)
    moving_ants = make_lowres_ants_image(moving_pre, fullres_spacing, DOWNSAMPLE_FACTOR, CT_WINDOW)

    tx = ants.registration(
        fixed=fixed_ants,
        moving=moving_ants,
        type_of_transform=ANTS_TRANSFORM,
        aff_metric=AFF_METRIC,
        aff_sampling=AFF_SAMPLING,
        verbose=False,
    )
    warped_moving_ants = tx["warpedmovout"]

    affine_transform = ants.read_transform(tx["fwdtransforms"][0])
    if AFFINE_FIELD_TRANSFORM == "inverse":
        affine_transform = ants.invert_ants_transform(affine_transform)
    affine_disp_fullres_vox = ants_affine_to_fullres_voxel_disp(
        affine_transform, fixed_ants, fullres_spacing=fullres_spacing
    )

    # 3. ConvexAdam-MIND (SVF) deformable on the affine-warped CT.
    fixed_low = fixed_ants.numpy().astype(np.float32)
    warped_moving_low = warped_moving_ants.numpy().astype(np.float32)
    dtype = torch.float16 if DTYPE == "float16" and device.type == "cuda" else torch.float32

    deform_fwd, _deform_rev = convex_adam_pt_svf(
        img_fixed=torch.from_numpy(fixed_low),
        img_moving=torch.from_numpy(warped_moving_low),
        mind_r=CONVEX_MIND_R,
        mind_d=CONVEX_MIND_D,
        lambda_weight=CONVEX_LAMBDA,
        grid_sp=CONVEX_GRID_SP,
        disp_hw=CONVEX_DISP_HW,
        selected_niter=CONVEX_NITER,
        selected_smooth=CONVEX_SMOOTH,
        grid_sp_adam=CONVEX_GRID_SP_ADAM,
        ic=IC,
        svf_steps=SVF_STEPS,
        dtype=dtype,
        device=device,
        verbose=False,
        save_disp=True,
    )
    deform_fullres_vox = np.asarray(deform_fwd, dtype=np.float32) * float(DOWNSAMPLE_FACTOR)

    # 4. Compose deformable AFTER affine (single low-resolution field).
    total_field_low = compose_deform_after_affine(deform_fullres_vox, affine_disp_fullres_vox)

    # 5. Up-sample the composed field to the ORIGINAL image resolution and save.
    disp_fullres = upsample_field_to(total_field_low, fullres_shape, DOWNSAMPLE_FACTOR)
    save_disp(output, disp_fullres, affine=fixed_ct_nii.affine, spacing=fullres_spacing)

    if save_preview:
        rows = [
            ("fixed CT", ct_window_normalize(fixed_ct_arr, CT_WINDOW)),
            ("moving CT", ct_window_normalize(moving_ct_arr, CT_WINDOW)),
            ("warped CT", ct_window_normalize(warp_volume(moving_ct_arr, disp_fullres), CT_WINDOW)),
        ]
        if fixed_pet is not None and moving_pet is not None:
            fixed_pet_arr = nib.load(str(fixed_pet)).get_fdata(dtype=np.float32)
            moving_pet_arr = nib.load(str(moving_pet)).get_fdata(dtype=np.float32)
            rows += [
                ("fixed PET", robust_normalize(fixed_pet_arr)),
                ("moving PET", robust_normalize(moving_pet_arr)),
                ("warped PET", robust_normalize(warp_volume(moving_pet_arr, disp_fullres))),
            ]
        if preview_path is None:
            preview_path = Path(output).with_suffix("").with_suffix(".png")
        save_preview_png(preview_path, rows, title=Path(output).name)

    return {
        "output": str(output),
        "field_shape": list(disp_fullres.shape),
        "field_abs_mean": float(np.mean(np.abs(disp_fullres))),
        "field_abs_max": float(np.max(np.abs(disp_fullres))),
        "seconds": round(time.time() - t0, 2),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="ANTs Affine + ConvexAdam-MIND (SVF) registration of one moving/fixed PET+CT set.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="example:\n  infer_convexadam.py fixed_ct.nii.gz fixed_pet.nii.gz moving_ct.nii.gz moving_pet.nii.gz disp.nii.gz",
    )
    parser.add_argument("fixed_ct", help="Fixed CT NIfTI (path). Drives the registration.")
    parser.add_argument("fixed_pet", help="Fixed PSMA-PET NIfTI (path). Read for the PET+CT interface / QA preview.")
    parser.add_argument("moving_ct", help="Moving CT NIfTI (path). Drives the registration.")
    parser.add_argument("moving_pet", help="Moving PSMA-PET NIfTI (path). Read for the PET+CT interface / QA preview.")
    parser.add_argument("output", help="Output displacement field (.nii.gz), original resolution.")
    parser.add_argument("--save-preview", action="store_true", help="Also write a QA PNG (warps moving CT + PET).")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device), flush=True)

    for label, path in (("fixed_ct", args.fixed_ct), ("fixed_pet", args.fixed_pet),
                        ("moving_ct", args.moving_ct), ("moving_pet", args.moving_pet)):
        if not os.path.exists(path):
            raise FileNotFoundError("{} not found: {}".format(label, path))

    info = register(
        fixed_ct=args.fixed_ct,
        moving_ct=args.moving_ct,
        output=args.output,
        fixed_pet=args.fixed_pet,
        moving_pet=args.moving_pet,
        device=device,
        save_preview=args.save_preview,
    )
    print("Saved {}  (|u| mean={:.3f} max={:.3f}, {:.1f}s)".format(
        info["output"], info["field_abs_mean"], info["field_abs_max"], info["seconds"]), flush=True)


if __name__ == "__main__":
    main()
