#!/usr/bin/env python3.8
"""Validate PSMAReg displacement files before submission."""

import argparse
import json
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

import nibabel as nib


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", type=Path, default=None)
    parser.add_argument("--zip-path", type=Path, default=None)
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--expected-shape", type=int, nargs=4, default=[3, 96, 96, 144])
    return parser.parse_args()


def expected_names(dataset_json):
    with dataset_json.open("r", encoding="utf-8") as handle:
        dataset = json.load(handle)
    names = []
    for entry in dataset.get("validation_paired", []):
        subject_id = entry["subject"].split("_")[-1]
        names.append("disp_{0}_00_{0}_01.nii.gz".format(subject_id))
    if not names:
        raise RuntimeError("No validation_paired entries found in {}".format(dataset_json))
    return sorted(names)


def validate_prediction_dir(prediction_dir, names, expected_shape):
    found = sorted(path.name for path in prediction_dir.glob("disp_*.nii.gz"))
    missing = sorted(set(names) - set(found))
    extra = sorted(set(found) - set(names))
    if missing:
        raise RuntimeError("Missing displacement files: {}".format(", ".join(missing)))
    if extra:
        raise RuntimeError("Unexpected displacement files: {}".format(", ".join(extra)))

    for name in names:
        path = prediction_dir / name
        image = nib.load(str(path))
        if tuple(image.shape) != tuple(expected_shape):
            raise RuntimeError("{} has shape {}; expected {}".format(name, image.shape, tuple(expected_shape)))
    print("OK: {} displacement files match expected names and shape {}.".format(len(names), tuple(expected_shape)))


def validate_zip(zip_path, names, expected_shape):
    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        with zipfile.ZipFile(str(zip_path), "r") as archive:
            members = archive.namelist()
            nested = [name for name in members if "/" in name.strip("/")]
            if nested:
                raise RuntimeError("Zip contains nested paths; files must be at root: {}".format(nested[:5]))
            archive.extractall(str(tmpdir))
        validate_prediction_dir(tmpdir, names, expected_shape)
    print("OK: {} is submission-ready.".format(zip_path))


def main():
    args = parse_args()
    if (args.prediction_dir is None) == (args.zip_path is None):
        raise ValueError("Provide exactly one of --prediction-dir or --zip-path.")
    names = expected_names(args.dataset_json)
    expected_shape = tuple(int(value) for value in args.expected_shape)
    if args.prediction_dir is not None:
        validate_prediction_dir(args.prediction_dir, names, expected_shape)
    else:
        validate_zip(args.zip_path, names, expected_shape)


if __name__ == "__main__":
    main()
