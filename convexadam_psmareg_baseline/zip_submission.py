#!/usr/bin/env python3.8
"""Zip PSMAReg displacement predictions for submission."""

import argparse
import zipfile
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--zip-path", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    fields = sorted(args.prediction_dir.glob("disp_*.nii.gz"))
    if not fields:
        raise FileNotFoundError("No disp_*.nii.gz files found in {}".format(args.prediction_dir))
    args.zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(args.zip_path), "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for field in fields:
            archive.write(str(field), arcname=field.name)
    print("Wrote {} files to {}".format(len(fields), args.zip_path))


if __name__ == "__main__":
    main()

