from typing import Optional, Tuple, List, Dict
import json
import os
import numpy as np
from html import escape
from pathlib import Path
import zipfile
import glob
import torch
import torch.nn.functional as F
import nibabel as nib
import digital_diffeomorphism as dd
from surface_distance import metrics as surface_metrics

WORST_SCORE_FALLBACKS = {
    'DSC': 0.0,
    'HD95': 1_000_000.0,
    'MTV Percent Error': 1_000_000.0,
    'TLG Percent Error': 1_000_000.0,
    'Percent NDV': 1_000_000.0,
}

DEFAULT_REFERENCE_DIR = Path("/scratch/jchen/python_projects/Learn2Reg26/results_submission_bundle/psmareg_val_phase/reference_data")
DEFAULT_PREDICTION_DIR = Path("/scratch/jchen/python_projects/Learn2Reg26/results_submission_bundle/psmareg_val_phase/input_data")
DEFAULT_OUTPUT_DIR = Path("/scratch/jchen/python_projects/Learn2Reg26/results_submission_bundle/psmareg_val_phase/output")

def first_existing_path(candidates, expect_file: Optional[str] = None) -> Optional[Path]:
    for candidate in candidates:
        if candidate is None:
            continue
        path = Path(candidate)
        if expect_file is None:
            if path.exists():
                return path
        else:
            if (path / expect_file).exists():
                return path
    return None


INPUT_DIR = Path(os.environ.get("CODABENCH_INPUT_DIR", "/app/input"))

REFERENCE_DIR = first_existing_path(
    [
        os.environ.get("CODABENCH_REF_DIR"),
        INPUT_DIR / "ref",
        "/app/input/ref",
        "/app/input_data/ref",
        DEFAULT_REFERENCE_DIR,
    ],
    expect_file="PSMAReg_dataset.json",
)
if REFERENCE_DIR is None:
    raise FileNotFoundError(
        "Could not locate reference data directory containing PSMAReg_dataset.json. "
        "Checked CODABENCH_REF_DIR, standard Codabench mount points, and local fallback paths."
    )

PREDICTION_DIR = first_existing_path(
    [
        os.environ.get("CODABENCH_RES_DIR"),
        INPUT_DIR / "res",
        "/app/input/res",
        "/app/input_data/res",
        DEFAULT_PREDICTION_DIR,
    ]
)
if PREDICTION_DIR is None:
    raise FileNotFoundError(
        "Could not locate prediction directory. Checked CODABENCH_RES_DIR, standard Codabench mount points, and local fallback paths."
    )

OUTPUT_DIR = first_existing_path(
    [
        os.environ.get("CODABENCH_OUTPUT_DIR"),
        "/app/output",
        DEFAULT_OUTPUT_DIR,
    ]
)
if OUTPUT_DIR is None:
    OUTPUT_DIR = DEFAULT_OUTPUT_DIR

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class SpatialTransformer(torch.nn.Module):
    """
    N-D Spatial Transformer
    Obtained from https://github.com/voxelmorph/voxelmorph
    """

    def __init__(self, size, mode='bilinear'):
        """Initialize a spatial transformer.

        Args:
            size: Spatial size tuple (H, W[, D]).
            mode: Interpolation mode ('bilinear' or 'nearest').
        """
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(*vectors, indexing='ij')
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        """Warp a source image with a displacement field.

        Args:
            src: Source tensor (B, C, ...).
            flow: Displacement field (B, ndim, ...).

        Returns:
            Warped source tensor.
        """
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=False, mode=self.mode)

pred_zip_matches = glob.glob(str(PREDICTION_DIR / "*.zip"))
if pred_zip_matches:
    pred_zip = pred_zip_matches[0]
    with zipfile.ZipFile(pred_zip, 'r') as zip_ref:
        zip_ref.extractall(PREDICTION_DIR)

dataset_json_path = REFERENCE_DIR / "PSMAReg_dataset.json"

with dataset_json_path.open("r", encoding="utf-8") as handle:
	dataset = json.load(handle)

validation_entries = dataset.get("validation_paired", [])
if not validation_entries:
    raise RuntimeError(f"No validation_paired entries found in {dataset_json_path}")


def compute_average_ct_label_dice(
    fixed_labels: np.ndarray,
    warped_labels: np.ndarray,
    label_ids=range(1, 118),
) -> float:
    dice_scores = []
    for label_id in label_ids:
        dice_score = surface_metrics.compute_dice_coefficient(
            fixed_labels == label_id,
            warped_labels == label_id,
        )
        if not np.isnan(dice_score):
            dice_scores.append(float(dice_score))
    if not dice_scores:
        return float("nan")
    return float(np.mean(dice_scores))


def compute_average_ct_label_hd95(
    fixed_labels: np.ndarray,
    moving_labels: np.ndarray,
    warped_labels: np.ndarray,
    spacing_mm: Tuple[float, float, float],
    label_ids=range(1, 118),
) -> float:
    hd95_scores = []
    for label_id in label_ids:
        fixed_mask = fixed_labels == label_id
        moving_mask = moving_labels == label_id
        if not fixed_mask.any() or not moving_mask.any():
            continue

        warped_mask = warped_labels == label_id
        surface_distances = surface_metrics.compute_surface_distances(
            fixed_mask,
            warped_mask,
            spacing_mm,
        )
        hd95_score = surface_metrics.compute_robust_hausdorff(surface_distances, 95.0)
        if np.isfinite(hd95_score):
            hd95_scores.append(float(hd95_score))

    if not hd95_scores:
        return float("nan")
    return float(np.mean(hd95_scores))


def sanitize_for_json(value):
    if isinstance(value, dict):
        return {key: sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, (np.floating, float)):
        value = float(value)
        if not np.isfinite(value):
            return None
        return value
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def make_json_safe_score(metric_name: str, value: float) -> float:
    value = float(value)
    if np.isfinite(value):
        return value
    return WORST_SCORE_FALLBACKS[metric_name]


def summarize_metric(values: List[float]) -> Dict[str, float | int]:
    values_array = np.asarray(values, dtype=float)
    finite_values = values_array[np.isfinite(values_array)]
    summary = {
        "count": int(values_array.size),
        "finite_count": int(finite_values.size),
        "non_finite_count": int(values_array.size - finite_values.size),
    }
    if finite_values.size == 0:
        summary.update(
            {
                "mean": float("nan"),
                "std": float("nan"),
                "median": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
            }
        )
        return summary

    summary.update(
        {
            "mean": float(np.mean(finite_values)),
            "std": float(np.std(finite_values)),
            "median": float(np.median(finite_values)),
            "min": float(np.min(finite_values)),
            "max": float(np.max(finite_values)),
        }
    )
    return summary


def format_metric_value(value: float | int) -> str:
    value = float(value)
    if np.isnan(value):
        return "nan"
    if np.isposinf(value):
        return "inf"
    if np.isneginf(value):
        return "-inf"
    return f"{value:.4f}"


def build_detailed_results_html(
    metric_summaries: Dict[str, Dict[str, float | int]],
    per_subject_rows: List[Dict[str, str | float]],
) -> str:
    summary_rows = []
    for metric_name, summary in metric_summaries.items():
        summary_rows.append(
            "".join(
                [
                    "<tr>",
                    f"<td>{escape(metric_name)}</td>",
                    f"<td>{format_metric_value(summary['mean'])}</td>",
                    f"<td>{format_metric_value(summary['std'])}</td>",
                    f"<td>{format_metric_value(summary['median'])}</td>",
                    f"<td>{format_metric_value(summary['min'])}</td>",
                    f"<td>{format_metric_value(summary['max'])}</td>",
                    f"<td>{summary['count']}</td>",
                    f"<td>{summary['non_finite_count']}</td>",
                    "</tr>",
                ]
            )
        )

    per_subject_html_rows = []
    for row in per_subject_rows:
        per_subject_html_rows.append(
            "".join(
                [
                    "<tr>",
                    f"<td>{escape(str(row['subject_id']))}</td>",
                    f"<td>{format_metric_value(row['DSC'])}</td>",
                    f"<td>{format_metric_value(row['HD95'])}</td>",
                    f"<td>{format_metric_value(row['MTV Percent Error'])}</td>",
                    f"<td>{format_metric_value(row['TLG Percent Error'])}</td>",
                    f"<td>{format_metric_value(row['Percent NDV'])}</td>",
                    "</tr>",
                ]
            )
        )

    return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <title>PSMAReg Detailed Results</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2937; }}
    h1, h2 {{ margin-bottom: 0.4rem; }}
    p {{ margin-top: 0; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px; }}
    th, td {{ border: 1px solid #d1d5db; padding: 8px 10px; text-align: left; }}
    th {{ background: #f3f4f6; }}
    tbody tr:nth-child(even) {{ background: #f9fafb; }}
    .note {{ color: #4b5563; font-size: 0.95rem; }}
  </style>
</head>
<body>
  <h1>PSMAReg Detailed Results</h1>
  <p class=\"note\">Codabench reads this file because the competition bundle enables detailed results. The leaderboard still uses <code>scores.json</code>; this page exposes summary statistics and per-subject values.</p>

  <h2>Summary Statistics</h2>
  <table>
    <thead>
      <tr>
        <th>Metric</th>
        <th>Mean</th>
        <th>Std</th>
        <th>Median</th>
        <th>Min</th>
        <th>Max</th>
        <th>N</th>
        <th>Non-finite</th>
      </tr>
    </thead>
    <tbody>
      {''.join(summary_rows)}
    </tbody>
  </table>

  <h2>Per-Subject Metrics</h2>
  <table>
    <thead>
      <tr>
        <th>Subject</th>
        <th>DSC</th>
        <th>HD95</th>
        <th>MTV Percent Error</th>
        <th>TLG Percent Error</th>
        <th>Percent NDV</th>
      </tr>
    </thead>
    <tbody>
      {''.join(per_subject_html_rows)}
    </tbody>
  </table>
</body>
</html>
"""


def get_compute_device() -> torch.device:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    return torch.device("cpu")


def to_device_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(array).float().to(device=device, non_blocking=device.type == "cuda")

if __name__ == "__main__":
    print('Reading prediction')
    print(f"Resolved reference dir: {REFERENCE_DIR}")
    print(f"Resolved prediction dir: {PREDICTION_DIR}")
    print(f"Resolved output dir: {OUTPUT_DIR}")
    device = get_compute_device()
    if device.type == "cuda":
        print(f"Using CUDA for warping and tensor metrics: {torch.cuda.get_device_name(device)}")
    else:
        print("CUDA not available, using CPU for scoring")

    spatial_trans = SpatialTransformer(size=(192, 192, 288), mode='nearest').to(device)
    spatial_trans.eval()
    ct_dice_scores = []
    ct_hd95_scores = []
    percent_ndv_scores = []
    mtv_percent_err_scores = []
    tlg_percent_err_scores = []
    per_subject_results = []
    for entry in validation_entries:
        subject_id = entry["subject"].split("_")[-1]
        print(f"Processing subject {subject_id}")
        
        # Step 1: Load predicted displacement field and relevant images/labels
        pred_disp_path = PREDICTION_DIR / f"disp_{subject_id}_00_{subject_id}_01.nii.gz"
        if not pred_disp_path.exists():
            raise FileNotFoundError(f"Missing predicted displacement field for {subject_id}: {pred_disp_path}")
        pred_disp_torch = to_device_tensor(nib.load(str(pred_disp_path)).get_fdata(), device)
        if pred_disp_torch.shape[0] != 3:
            pred_disp_torch = pred_disp_torch.permute(3, 0, 1, 2)
        else:
            pred_disp_torch = torch.clone(pred_disp_torch)
        with torch.inference_mode():
            pred_disp_torch = F.interpolate(
                pred_disp_torch.unsqueeze(0),
                scale_factor=2,
                mode='trilinear',
                align_corners=False,
            )
        #print(f"Predicted displacement field shape: {pred_disp_torch.shape}")
        
        ct_lbls_fx_nii = nib.load(str(REFERENCE_DIR / entry["CT Label"]))
        ct_lbls_fx = ct_lbls_fx_nii.get_fdata()
        ct_lbls_mv = nib.load(str(REFERENCE_DIR / entry["Follow-up 01 CT Label"])).get_fdata()
        #print(f"Fixed labels shape: {ct_lbls_fx.shape}, Moving labels shape: {ct_lbls_mv.shape}")
        
        pet_lbls_mv = nib.load(str(REFERENCE_DIR / entry["Follow-up 01 PET Label"])).get_fdata()
        #print(f"Fixed labels shape: {pet_lbls_fx.shape}, Moving labels shape: {pet_lbls_mv.shape}")
        
        ct_img_fx = nib.load(str(REFERENCE_DIR / entry["CT"])).get_fdata()
        pet_img_mv = nib.load(str(REFERENCE_DIR / entry["Follow-up 01 PET"])).get_fdata()
        
        # Step 2: Warp moving labels with predicted displacement field
        ct_lbls_mv_torch = to_device_tensor(ct_lbls_mv, device)
        pet_lbls_mv_torch = to_device_tensor(pet_lbls_mv, device)
        pet_img_mv_torch = to_device_tensor(pet_img_mv, device)

        with torch.inference_mode():
            warped_ct_lbls_mv_torch = spatial_trans(
                ct_lbls_mv_torch.unsqueeze(0).unsqueeze(0),
                pred_disp_torch,
            ).squeeze(0).squeeze(0)
            warped_pet_lbls_mv_torch = spatial_trans(
                pet_lbls_mv_torch.unsqueeze(0).unsqueeze(0),
                pred_disp_torch,
            ).squeeze(0).squeeze(0)

            total_metabolic_tumor_volume_before = torch.count_nonzero(pet_lbls_mv_torch > 0).item()
            total_metabolic_tumor_volume_after = torch.count_nonzero(warped_pet_lbls_mv_torch > 0).item()

            total_lesion_glycolysis_before = torch.sum((pet_lbls_mv_torch > 0).float() * pet_img_mv_torch).item()
            total_lesion_glycolysis_after = torch.sum((warped_pet_lbls_mv_torch > 0).float() * pet_img_mv_torch).item()

        warped_ct_lbls_mv = torch.round(warped_ct_lbls_mv_torch).cpu().numpy()
        
        # Step 3: Compute evaluation metrics
        average_ct_dice = compute_average_ct_label_dice(ct_lbls_fx, warped_ct_lbls_mv)
        ct_dice_scores.append(average_ct_dice)

        average_ct_hd95 = compute_average_ct_label_hd95(
            ct_lbls_fx,
            ct_lbls_mv,
            warped_ct_lbls_mv,
            tuple(float(spacing) for spacing in ct_lbls_fx_nii.header.get_zooms()[:3]),
        )
        ct_hd95_scores.append(average_ct_hd95)

        mtv_percent_err = 100 * abs(total_metabolic_tumor_volume_after - total_metabolic_tumor_volume_before) / max(total_metabolic_tumor_volume_before, 1e-6)
        mtv_percent_err_scores.append(mtv_percent_err)

        tlg_percent_err = 100 * abs(total_lesion_glycolysis_after - total_lesion_glycolysis_before) / max(total_lesion_glycolysis_before, 1e-6)
        tlg_percent_err_scores.append(tlg_percent_err)
        
        
        pred_disp_np = pred_disp_torch.squeeze(0).cpu().numpy()
        mask = ct_img_fx[1:-1, 1:-1, 1:-1]
        mask = mask > 0
        
        trans_ = pred_disp_np + dd.get_identity_grid(pred_disp_np)
        jac_dets = dd.calc_jac_dets(trans_)
        non_diff_voxels, non_diff_tetrahedra, non_diff_volume, non_diff_volume_map = dd.calc_measurements(jac_dets, mask)
        total_voxels = np.sum(mask)
        percent_ndv = non_diff_volume / total_voxels * 100
        percent_ndv_scores.append(percent_ndv)

        per_subject_results.append(
            {
                "subject_id": subject_id,
                "DSC": average_ct_dice,
                "HD95": average_ct_hd95,
                "MTV Percent Error": mtv_percent_err,
                "TLG Percent Error": tlg_percent_err,
                "Percent NDV": percent_ndv,
            }
        )
        
        print(f"Subject {subject_id} - Average CT Label Dice: {average_ct_dice:.4f}, HD95: {average_ct_hd95:.4f}, MTV Percent Error: {mtv_percent_err:.4f}%, TLG Percent Error: {tlg_percent_err:.4f}%, Percent NDV: {percent_ndv:.4f}%")
    
    print('Scores:')
    metric_summaries = {
        'DSC': summarize_metric(ct_dice_scores),
        'HD95': summarize_metric(ct_hd95_scores),
        'MTV Percent Error': summarize_metric(mtv_percent_err_scores),
        'TLG Percent Error': summarize_metric(tlg_percent_err_scores),
        'Percent NDV': summarize_metric(percent_ndv_scores),
    }
    scores = {
        metric_name: make_json_safe_score(metric_name, float(summary['mean']))
        for metric_name, summary in metric_summaries.items()
    }
    score_summary = {
        'DSC': f"{scores['DSC']:.4f}+-{np.std(ct_dice_scores):.4f}",
        'HD95': f"{scores['HD95']:.4f}+-{np.std(ct_hd95_scores):.4f}",
        'MTV Percent Error': f"{scores['MTV Percent Error']:.4f}+-{np.std(mtv_percent_err_scores):.4f}",
        'TLG Percent Error': f"{scores['TLG Percent Error']:.4f}+-{np.std(tlg_percent_err_scores):.4f}",
        'Percent NDV': f"{scores['Percent NDV']:.4f}+-{np.std(percent_ndv_scores):.4f}",
    }
    print(score_summary)

    with (OUTPUT_DIR / 'scores.json').open('w', encoding='utf-8') as score_file:
        json.dump(sanitize_for_json(scores), score_file, allow_nan=False)

    with (OUTPUT_DIR / 'detailed_results.json').open('w', encoding='utf-8') as detailed_json_file:
        json.dump(
            sanitize_for_json({
                'summary': metric_summaries,
                'per_subject': per_subject_results,
            }),
            detailed_json_file,
            indent=2,
            allow_nan=False,
        )

    with (OUTPUT_DIR / 'detailed_results.html').open('w', encoding='utf-8') as detailed_html_file:
        detailed_html_file.write(build_detailed_results_html(metric_summaries, per_subject_results))

        
        
        
        
        
        
        
        