"""Deterministic synthetic inputs for software tests, never participant records.

The column names follow the supported schema. Values and class effects are
arbitrary generated numbers, unsuitable for scientific performance claims.
"""

import numpy as np
import pandas as pd


REGIONS = (
    "Cerebellum", "III", "III_A", "IV", "V", "VI", "CrusI", "CrusII",
    "VIIB", "VIIIA", "VIIIB", "IX", "X",
)


def _column(region, measurement):
    if region == "III_A":
        return f"III{measurement}_A"
    return f"{region}{measurement}"


def make_synthetic_frame(n=200, seed=42):
    """Return a balanced fixture with all raw and normalized schema columns."""
    if n < 2:
        raise ValueError("Synthetic data requires at least two rows")
    rng = np.random.RandomState(seed)
    dx = np.r_[np.zeros(n // 2, dtype=int), np.ones(n - n // 2, dtype=int)]
    age = rng.uniform(7, 15, n)
    gender = rng.binomial(1, 0.5, n)
    icv = rng.uniform(1000, 1800, n)
    columns = {
        "Sub": [f"SYNTHETIC_{index:04d}" for index in range(n)],
        "Gender": gender,
        "Age": age,
        "DX": dx,
        "ADHDIndex": rng.uniform(0, 100, n),
        "Inattentive": rng.uniform(0, 100, n),
        "HyperImpulsive": rng.uniform(0, 100, n),
        "MedStatus": rng.binomial(1, 0.5, n),
        "ICVcm3": icv,
    }
    morphometry = (
        "totalcm3", "rightcm3", "leftcm3", "total", "right", "left",
        "asymmetry", "corticalthicknessasymmetry", "greymattercm3",
        "rightgreymattercm3", "leftgreymattercm3", "greymatter",
        "rightgreymatter", "leftgreymatter", "greymatterasymmetry",
    )
    for region in REGIONS:
        for measurement in morphometry:
            columns[_column(region, measurement)] = rng.uniform(0.5, 10, n)
        for side in ("mean", "right", "left"):
            raw = (
                2.0 + 0.03 * age - 0.05 * gender
                + 0.1 * dx + rng.normal(0, 0.2, n)
            )
            columns[_column(region, f"{side}corticalthickness")] = raw
            norm_measurement = f"{side}corticalthicknessnorm"
            if region == "Cerebellum" and side == "mean":
                norm_measurement = "corticalthicknessnorm"
            columns[_column(region, norm_measurement)] = (
                raw * np.power(1000.0 / icv, 1.0 / 3.0)
            )
    return pd.DataFrame(columns)
