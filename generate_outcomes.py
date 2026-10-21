#!/usr/bin/env python3
"""
generate_outcomes.py - Fetal Distress Prediction Script
=======================================================

Loads a trained FetalDistressModel from the artifacts/ directory and produces
predictions on one or more CTG recordings.

Input
-----
- A path to a directory containing WFDB-format CTG recordings (.dat + .hea files),
  OR a path to a single .hea file (the .dat must sit beside it).
- Each recording has two channels sampled at 4 Hz: FHR (fetal heart rate) and UC
  (uterine contractions).

Output
------
- A CSV file (predictions.csv) with columns:
    record_id   : the recording identifier (e.g. "1001")
    distress_prob : model's estimated probability of fetal distress (0.0 – 1.0)
    prediction   : binary label — 1 = distressed, 0 = normal (threshold 0.5)

- Also printed to stdout.

Usage
-----
    # Predict on a single recording:
    python generate_outcomes.py --input data/raw/1001.hea

    # Predict on all recordings in a directory:
    python generate_outcomes.py --input data/raw/

    # Specify a custom artifacts directory:
    python generate_outcomes.py --input data/raw/ --artifacts artifacts/

    # Change output path:
    python generate_outcomes.py --input data/raw/ --output results.csv
"""

import argparse
import os
import sys
import glob
import json

import numpy as np
import pandas as pd
import torch
import wfdb

from model import FetalDistressModel


# ---------- Signal preprocessing (must match training pipeline) ----------

def load_ctg_signal(record_path: str) -> tuple:
    """
    Load FHR and UC signals from a WFDB record.

    Parameters
    ----------
    record_path : str
        Path to the record WITHOUT the file extension (e.g. "data/raw/1001").

    Returns
    -------
    fhr : np.ndarray   (shape: n_samples,)
    uc  : np.ndarray   (shape: n_samples,)
    fs  : int           sampling frequency (4 Hz)
    """
    record = wfdb.rdrecord(record_path)
    fhr = record.p_signal[:, 0]  # channel 0: FHR
    uc = record.p_signal[:, 1]   # channel 1: UC
    fs = record.fs
    return fhr, uc, fs


def preprocess_signal(fhr: np.ndarray, uc: np.ndarray, target_length: int = 4800,
                      scaler_path: str = None) -> np.ndarray:
    """
    Clean and normalise a CTG signal pair for model input.

    Steps:
    1. Replace missing/invalid FHR values (0 or out of physiological range) with NaN,
       then interpolate.
    2. Clip FHR to [50, 210] bpm and UC to [0, 100].
    3. Z-normalise using saved training statistics (mean, std) if available.
    4. Truncate or pad to target_length (take the last `target_length` samples —
       the period closest to delivery is the most clinically relevant).

    Parameters
    ----------
    fhr, uc : np.ndarray
        Raw signal arrays.
    target_length : int
        Fixed sequence length (default 4800 = 20 minutes at 4 Hz).
    scaler_path : str or None
        Path to scaler_stats.json saved during training.

    Returns
    -------
    np.ndarray, shape (2, target_length)
        Preprocessed 2-channel signal ready for the model.
    """
    # --- Step 1: handle missing FHR values ---
    fhr = fhr.copy().astype(np.float64)
    uc = uc.copy().astype(np.float64)

    # FHR = 0 means the sensor lost the signal
    fhr[fhr == 0] = np.nan
    fhr[(fhr < 50) | (fhr > 210)] = np.nan

    # Interpolate NaNs
    if np.any(np.isnan(fhr)):
        nans = np.isnan(fhr)
        if not np.all(nans):
            x = np.arange(len(fhr))
            fhr[nans] = np.interp(x[nans], x[~nans], fhr[~nans])
        else:
            fhr[:] = 140.0  # fallback: use baseline FHR

    # --- Step 2: clip to physiological range ---
    fhr = np.clip(fhr, 50, 210)
    uc = np.clip(uc, 0, 100)

    # --- Step 3: normalise ---
    if scaler_path and os.path.exists(scaler_path):
        with open(scaler_path, "r") as f:
            stats = json.load(f)
        fhr = (fhr - stats["fhr_mean"]) / (stats["fhr_std"] + 1e-8)
        uc = (uc - stats["uc_mean"]) / (stats["uc_std"] + 1e-8)
    else:
        # Fallback: per-signal normalisation
        fhr = (fhr - np.mean(fhr)) / (np.std(fhr) + 1e-8)
        uc = (uc - np.mean(uc)) / (np.std(uc) + 1e-8)

    # --- Step 4: take last `target_length` samples (closest to delivery) ---
    if len(fhr) >= target_length:
        fhr = fhr[-target_length:]
        uc = uc[-target_length:]
    else:
        # Pad with zeros (left-pad so delivery end is at the right)
        pad_len = target_length - len(fhr)
        fhr = np.pad(fhr, (pad_len, 0), mode="constant", constant_values=0)
        uc = np.pad(uc, (pad_len, 0), mode="constant", constant_values=0)

    signal = np.stack([fhr, uc], axis=0)  # shape (2, target_length)
    return signal


def discover_records(input_path: str) -> list:
    """
    Find all WFDB record base-paths under the given path.

    Parameters
    ----------
    input_path : str
        Either a path to a single .hea file, or a directory.

    Returns
    -------
    list[str]
        List of record base paths (without extension).
    """
    if os.path.isfile(input_path):
        if input_path.endswith(".hea"):
            return [input_path.replace(".hea", "")]
        else:
            raise ValueError(f"Expected a .hea file, got: {input_path}")

    if os.path.isdir(input_path):
        hea_files = sorted(glob.glob(os.path.join(input_path, "*.hea")))
        if not hea_files:
            raise FileNotFoundError(f"No .hea files found in {input_path}")
        return [f.replace(".hea", "") for f in hea_files]

    raise FileNotFoundError(f"Path does not exist: {input_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate fetal distress predictions from CTG recordings."
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to a single .hea file or a directory of WFDB records.",
    )
    parser.add_argument(
        "--artifacts", "-a",
        default="artifacts",
        help="Directory containing model weights and config (default: artifacts/).",
    )
    parser.add_argument(
        "--output", "-o",
        default="predictions.csv",
        help="Output CSV path (default: predictions.csv).",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.5,
        help="Classification threshold (default: 0.5).",
    )
    parser.add_argument(
        "--seq-length", "-s",
        type=int,
        default=4800,
        help="Sequence length in samples (default: 4800 = 20 min @ 4 Hz).",
    )
    args = parser.parse_args()

    # --- Load model ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading model from {args.artifacts}/ ...")

    model = FetalDistressModel.load(args.artifacts, device=device)

    # Scaler stats path
    scaler_path = os.path.join(args.artifacts, "scaler_stats.json")

    # --- Discover records ---
    records = discover_records(args.input)
    print(f"Found {len(records)} recording(s).\n")

    # --- Run predictions ---
    results = []
    for rec_path in records:
        rec_id = os.path.basename(rec_path)
        try:
            fhr, uc, fs = load_ctg_signal(rec_path)
            signal = preprocess_signal(fhr, uc, target_length=args.seq_length,
                                       scaler_path=scaler_path)
            tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).to(device)
            prob = model.predict_proba(tensor).item()
            pred = int(prob >= args.threshold)
            results.append({"record_id": rec_id, "distress_prob": round(prob, 4),
                            "prediction": pred})
        except Exception as e:
            print(f"  [WARN] Skipping {rec_id}: {e}")
            results.append({"record_id": rec_id, "distress_prob": None,
                            "prediction": None})

    # --- Output ---
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(df.to_string(index=False))
    print(f"\nPredictions saved to {args.output}")


if __name__ == "__main__":
    main()
