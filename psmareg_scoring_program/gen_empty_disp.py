from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np


DEFAULT_DATASET_JSON = Path(
	"/scratch/jchen/python_projects/Learn2Reg26/results_submission_bundle/dev_phase/reference_data/PSMAReg_dataset.json"
)
DEFAULT_LABEL_ROOT = Path(
	"/scratch/jchen/python_projects/Learn2Reg26/results_submission_bundle/dev_phase/reference_data/labelsVal"
)
DEFAULT_OUTPUT_DIR = Path(
	"/scratch/jchen/python_projects/Learn2Reg26/results_submission_bundle/dev_phase/input_data"
)
DEFAULT_ZIP_PATH = Path(
	"/scratch/jchen/python_projects/Learn2Reg26/results_submission_bundle/dev_phase/input_data/input_data.zip"
)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Generate MIR-style empty displacement fields for PSMAReg validation subjects "
			"using baseline CT labels as shape references."
		)
	)
	parser.add_argument("--dataset-json", type=Path, default=DEFAULT_DATASET_JSON)
	parser.add_argument("--label-root", type=Path, default=DEFAULT_LABEL_ROOT)
	parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
	parser.add_argument("--zip-path", type=Path, default=DEFAULT_ZIP_PATH)
	parser.add_argument(
		"--downsample-factor",
		type=int,
		default=2,
		help="Integer spatial downsampling factor for the empty displacement grid.",
	)
	parser.add_argument(
		"--overwrite",
		action="store_true",
		help="Overwrite existing displacement fields and zip output.",
	)
	return parser.parse_args()


def downsample_shape(shape: tuple[int, int, int], factor: int) -> tuple[int, int, int]:
	if factor <= 0:
		raise ValueError("downsample factor must be positive")
	return tuple(max(1, dim // factor) for dim in shape)


def resolve_relative_path(root: Path, relative_path: str) -> Path:
	if relative_path.startswith("./"):
		relative_path = relative_path[2:]
	return (root / relative_path).resolve()


def save_mir_disp(disp: np.ndarray, output_path: Path, pix_dim: tuple[float, float, float]) -> None:
	image = nib.Nifti1Image(disp, np.eye(4))
	image.header["pixdim"][1:4] = pix_dim
	nib.save(image, str(output_path))


def find_followup_ct_keys(entry: dict[str, object]) -> list[str]:
	return sorted(
		key
		for key in entry
		if key.startswith("Follow-up ") and key.endswith(" CT")
	)


def main() -> None:
	args = parse_args()
	dataset_json_path = args.dataset_json.expanduser().resolve()
	label_root = args.label_root.expanduser().resolve()
	output_dir = args.output_dir.expanduser().resolve()
	zip_path = args.zip_path.expanduser().resolve()

	with dataset_json_path.open("r", encoding="utf-8") as handle:
		dataset = json.load(handle)

	validation_entries = dataset.get("validation_paired", [])
	if not validation_entries:
		raise RuntimeError(f"No validation_paired entries found in {dataset_json_path}")

	output_dir.mkdir(parents=True, exist_ok=True)
	if args.overwrite:
		for existing_file in output_dir.glob("disp_*.nii.gz"):
			existing_file.unlink()
	generated_files: list[Path] = []

	for entry in validation_entries:
		subject = entry["subject"]
		subject_id = subject.split("_")[-1]
		followup_ct_keys = find_followup_ct_keys(entry)
		if len(followup_ct_keys) != 1:
			raise RuntimeError(
				f"Expected exactly one follow-up CT for {subject}, found {len(followup_ct_keys)}"
			)
		moving_timepoint = followup_ct_keys[0].split()[1]
		ct_label_path = resolve_relative_path(label_root.parent, entry["CT Label"])
		if not ct_label_path.exists():
			raise FileNotFoundError(f"Missing CT label for {subject}: {ct_label_path}")

		label_nib = nib.load(str(ct_label_path))
		label_shape = tuple(int(value) for value in label_nib.shape[:3])
		field_shape = downsample_shape(label_shape, args.downsample_factor)
		field = np.zeros((3,) + field_shape, dtype=np.float32)
		spacing = tuple(float(value) * args.downsample_factor for value in label_nib.header.get_zooms()[:3])

		output_path = output_dir / (
			f"disp_{subject_id}_00_{subject_id}_{moving_timepoint}.nii.gz"
		)
		if output_path.exists() and not args.overwrite:
			raise FileExistsError(f"Output already exists: {output_path}")
		save_mir_disp(field, output_path, spacing)
		generated_files.append(output_path)

	if zip_path.exists():
		if not args.overwrite:
			raise FileExistsError(f"Zip file already exists: {zip_path}")
		zip_path.unlink()

	with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
		for output_path in sorted(generated_files):
			archive.write(output_path, arcname=output_path.name)

	print(f"Generated {len(generated_files)} empty displacement fields in {output_dir}")
	print(f"Created zip archive: {zip_path}")


if __name__ == "__main__":
	main()
