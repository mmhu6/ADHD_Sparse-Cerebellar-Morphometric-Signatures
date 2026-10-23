"""Control-defined normative deviation ADHD screening experiment.

Every normative reference is fitted using controls from the current training
partition only.  The module remains separate from all previous experiments.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from cereb_adhd_screen import (
    FoldResidualizer,
    LEGACY_FEATURES,
    LOCKED_XGB_PARAMS,
    _bootstrap_delta,
    _fit_legacy_svm,
    _fit_locked_xgb,
    _make_xgb,
    _metrics,
)


MODEL_SELECTION_SEED = 842
FUSION_SELECTION_SEED = 942
VALIDATION_SEED = 42
DIRECT_COVARIATES = ("Age", "Gender", "ICVcm3")
FORBIDDEN_MODEL_INPUTS = frozenset(
    {"DX", "ADHDIndex", "Inattentive", "HyperImpulsive", "MedStatus", "Sub"}
)
NORM_FEATURES = tuple(LEGACY_FEATURES)
NORM_VIEWS = ("z_only", "raw_plus_z")
MODEL_FAMILIES = ("elastic_net", "rbf_svm", "xgboost")


def _numeric(series: pd.Series, fill: float | None = None) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(float)
    if fill is not None:
        values = np.where(np.isfinite(values), values, fill)
    return values


class ControlNormativeTransformer:
    """Map brain measures to control-defined conditional z deviations."""

    def __init__(
        self,
        features: Sequence[str] = NORM_FEATURES,
        z_clip: float = 5.0,
    ):
        self.features = tuple(features)
        self.z_clip = float(z_clip)

    def _filled_covariate(self, X: pd.DataFrame, name: str) -> np.ndarray:
        return _numeric(X[name], self.covariate_medians_[name])

    def _design(self, X: pd.DataFrame) -> np.ndarray:
        age = self._filled_covariate(X, "Age")
        gender = self._filled_covariate(X, "Gender")
        icv = self._filled_covariate(X, "ICVcm3")
        age_z = (age - self.age_mean_) / self.age_sd_
        icv_z = (icv - self.icv_mean_) / self.icv_sd_
        return np.column_stack(
            [np.ones(len(X)), age_z, age_z**2, gender, icv_z]
        )

    def fit(self, X: pd.DataFrame, y: Sequence[int]):
        X = pd.DataFrame(X)
        y_array = np.asarray(y, dtype=int)
        if len(X) != len(y_array):
            raise ValueError("X and y length mismatch")
        missing = (set(self.features) | set(DIRECT_COVARIATES)).difference(X.columns)
        if missing:
            raise ValueError(f"Missing normative columns: {sorted(missing)}")
        control_mask = y_array == 0
        self.n_controls_ = int(control_mask.sum())
        if self.n_controls_ < 20:
            raise ValueError("At least 20 training controls are required")
        controls = X.loc[control_mask]

        self.covariate_medians_ = {}
        for name in DIRECT_COVARIATES:
            values = _numeric(controls[name])
            finite = values[np.isfinite(values)]
            if not len(finite):
                raise ValueError(f"Control covariate {name} has no finite values")
            self.covariate_medians_[name] = float(np.median(finite))
        control_age = self._filled_covariate(controls, "Age")
        control_icv = self._filled_covariate(controls, "ICVcm3")
        self.age_mean_ = float(np.mean(control_age))
        self.age_sd_ = float(np.std(control_age)) or 1.0
        self.icv_mean_ = float(np.mean(control_icv))
        self.icv_sd_ = float(np.std(control_icv)) or 1.0

        self.feature_medians_ = {}
        brain_parts = []
        for name in self.features:
            values = _numeric(controls[name])
            finite = values[np.isfinite(values)]
            if not len(finite):
                raise ValueError(f"Control feature {name} has no finite values")
            median = float(np.median(finite))
            self.feature_medians_[name] = median
            brain_parts.append(np.where(np.isfinite(values), values, median))
        control_brain = np.column_stack(brain_parts)
        design = self._design(controls)
        self.beta_ = np.linalg.lstsq(design, control_brain, rcond=None)[0]
        residuals = control_brain - design @ self.beta_
        residual_sd = np.std(residuals, axis=0, ddof=1)
        self.residual_sd_ = np.where(
            np.isfinite(residual_sd) & (residual_sd > 1e-8), residual_sd, 1.0
        )
        return self

    @property
    def z_feature_names(self) -> tuple[str, ...]:
        return tuple(f"z__{name}" for name in self.features)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X)
        brain = np.column_stack(
            [
                _numeric(X[name], self.feature_medians_[name])
                for name in self.features
            ]
        )
        expected = self._design(X) @ self.beta_
        z = np.clip(
            (brain - expected) / self.residual_sd_, -self.z_clip, self.z_clip
        )
        output = pd.DataFrame(z, columns=self.z_feature_names, index=X.index)
        output["norm_mean_abs_z"] = np.mean(np.abs(z), axis=1)
        output["norm_extreme_count"] = np.sum(np.abs(z) > 2.0, axis=1)
        return output


@dataclass
class NormativeMatrixTransformer:
    view: str
    normative: ControlNormativeTransformer
    residualizer: FoldResidualizer | None
    covariate_medians: dict[str, float]

    @classmethod
    def fit(cls, X: pd.DataFrame, y: pd.Series, view: str):
        if view not in NORM_VIEWS:
            raise ValueError(f"Unknown normative view: {view}")
        normative = ControlNormativeTransformer().fit(X, y)
        residualizer = (
            FoldResidualizer(NORM_FEATURES, include_icv=True).fit(X)
            if view == "raw_plus_z"
            else None
        )
        medians = {}
        for name in DIRECT_COVARIATES:
            values = _numeric(X[name])
            finite = values[np.isfinite(values)]
            if not len(finite):
                raise ValueError(f"Covariate {name} has no finite values")
            medians[name] = float(np.median(finite))
        return cls(view, normative, residualizer, medians)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        z_frame = self.normative.transform(X)
        parts = []
        if self.residualizer is not None:
            parts.append(self.residualizer.transform(X).to_numpy(float))
        parts.append(z_frame.to_numpy(float))
        parts.append(
            np.column_stack(
                [_numeric(X[name], self.covariate_medians[name]) for name in DIRECT_COVARIATES]
            )
        )
        return np.column_stack(parts)


@dataclass
class FittedNormModel:
    view: str
    family: str
    transformer: NormativeMatrixTransformer
    scaler: StandardScaler | None
    model: Any
    input_columns: tuple[str, ...]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        matrix = self.transformer.transform(X)
        if self.scaler is not None:
            matrix = self.scaler.transform(matrix)
        probability = self.model.predict_proba(matrix)[:, 1]
        return np.clip(np.asarray(probability, dtype=float), 0.0, 1.0)


def _fit_norm_model(
    X: pd.DataFrame,
    y: pd.Series,
    view: str,
    family: str,
    params: dict[str, Any],
    seed: int,
) -> FittedNormModel:
    if family not in MODEL_FAMILIES:
        raise ValueError(f"Unknown model family: {family}")
    transformer = NormativeMatrixTransformer.fit(X, y, view)
    matrix = transformer.transform(X)
    scaler: StandardScaler | None = None
    if family in {"elastic_net", "rbf_svm"}:
        scaler = StandardScaler().fit(matrix)
        matrix = scaler.transform(matrix)
    if family == "elastic_net":
        model = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            class_weight="balanced",
            max_iter=5000,
            tol=1e-4,
            random_state=seed,
            **params,
        )
    elif family == "rbf_svm":
        model = SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=seed,
            **params,
        )
    else:
        model = _make_xgb(dict(params), seed)
    model.fit(matrix, y)
    inputs = tuple(NORM_FEATURES) + DIRECT_COVARIATES
    if set(inputs).intersection(FORBIDDEN_MODEL_INPUTS):
        raise ValueError("Forbidden normative model input")
    return FittedNormModel(view, family, transformer, scaler, model, inputs)


@dataclass(frozen=True)
class ExperimentMode:
    name: str
    development_splits: int
    outer_splits: int
    outer_repeats: int
    bootstrap_samples: int
    full_candidate_library: bool


SMOKE_MODE = ExperimentMode("smoke", 3, 3, 1, 300, False)
FULL_MODE = ExperimentMode("full", 5, 5, 3, 5000, True)


@dataclass(frozen=True)
class NormModelConfig:
    family: str
    view: str
    parameter_name: str
    params: dict[str, Any]

    @property
    def name(self) -> str:
        return f"{self.family}__{self.view}__{self.parameter_name}"


XGB_NORM_LIBRARY: tuple[tuple[str, dict[str, Any]], ...] = (
    ("locked_reference", dict(LOCKED_XGB_PARAMS)),
    (
        "shallow_high_regularization",
        {
            "n_estimators": 260,
            "max_depth": 1,
            "learning_rate": 0.04,
            "min_child_weight": 8,
            "gamma": 0.05,
            "subsample": 0.85,
            "colsample_bytree": 0.75,
            "reg_alpha": 0.75,
            "reg_lambda": 25.0,
            "scale_pos_weight": 1.0,
        },
    ),
)


def _prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    required = set(NORM_FEATURES) | set(DIRECT_COVARIATES) | {"DX"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    clean = data.copy()
    clean["DX"] = pd.to_numeric(clean["DX"], errors="raise").astype(int)
    clean = clean.dropna(subset=["DX", *DIRECT_COVARIATES]).reset_index(drop=True)
    if set(clean["DX"].unique()) != {0, 1}:
        raise ValueError("DX must contain binary 0/1 values")
    return clean


def _fixed_splits(y: pd.Series, n_splits: int, seed: int):
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(y)), y))


def _candidate_configs(mode: ExperimentMode) -> list[NormModelConfig]:
    elastic_cs = (
        (0.03, 0.1, 0.3, 1.0) if mode.full_candidate_library else (0.1, 0.3)
    )
    svm_cs = (0.3, 1.0, 3.0) if mode.full_candidate_library else (0.3, 1.0)
    xgb_library = (
        XGB_NORM_LIBRARY if mode.full_candidate_library else XGB_NORM_LIBRARY[:1]
    )
    configs: list[NormModelConfig] = []
    for view in NORM_VIEWS:
        for C in elastic_cs:
            configs.append(
                NormModelConfig(
                    "elastic_net",
                    view,
                    f"C{C:g}_l1r0.5",
                    {"C": C, "l1_ratio": 0.5},
                )
            )
        for C in svm_cs:
            configs.append(
                NormModelConfig(
                    "rbf_svm",
                    view,
                    f"C{C:g}_gamma_scale",
                    {"C": C, "gamma": "scale"},
                )
            )
        for parameter_name, params in xgb_library:
            configs.append(
                NormModelConfig("xgboost", view, parameter_name, dict(params))
            )
    return configs


def _norm_oof(
    data: pd.DataFrame,
    y: pd.Series,
    config: NormModelConfig,
    splits,
    seed: int,
) -> tuple[np.ndarray, list[float]]:
    probability = np.full(len(y), np.nan, dtype=float)
    fold_auc: list[float] = []
    for fold_id, (train_idx, valid_idx) in enumerate(splits, start=1):
        model = _fit_norm_model(
            data.iloc[train_idx],
            y.iloc[train_idx],
            config.view,
            config.family,
            config.params,
            seed + fold_id,
        )
        fold_probability = model.predict_proba(data.iloc[valid_idx])
        probability[valid_idx] = fold_probability
        fold_auc.append(float(roc_auc_score(y.iloc[valid_idx], fold_probability)))
    if not np.isfinite(probability).all():
        raise RuntimeError(f"Non-finite OOF probability for {config.name}")
    return probability, fold_auc


def _score_oof(y: pd.Series, probability: np.ndarray, fold_auc: Sequence[float]):
    return {
        "pooled_auc": float(roc_auc_score(y, probability)),
        "mean_fold_auc": float(np.mean(fold_auc)),
        "sd_fold_auc": (
            float(np.std(fold_auc, ddof=1)) if len(fold_auc) > 1 else 0.0
        ),
    }


def select_family_configs(
    data: pd.DataFrame,
    y: pd.Series,
    mode: ExperimentMode,
) -> tuple[dict[str, NormModelConfig], pd.DataFrame]:
    splits = _fixed_splits(y, mode.development_splits, MODEL_SELECTION_SEED)
    configs = _candidate_configs(mode)
    rows: list[dict[str, Any]] = []
    for trial_id, config in enumerate(configs, start=1):
        probability, fold_auc = _norm_oof(
            data,
            y,
            config,
            splits,
            MODEL_SELECTION_SEED + trial_id * 100,
        )
        rows.append(
            {
                "config": config.name,
                "family": config.family,
                "view": config.view,
                "parameter_name": config.parameter_name,
                "params": json.dumps(config.params, sort_keys=True),
                **_score_oof(y, probability, fold_auc),
            }
        )
        print(
            f"Norm model trial {trial_id:02d}/{len(configs)}: "
            f"{config.name} AUC={rows[-1]['pooled_auc']:.4f}",
            flush=True,
        )
    trials = pd.DataFrame(rows)
    config_by_name = {config.name: config for config in configs}
    selected = {}
    for family in MODEL_FAMILIES:
        subset = trials[trials.family == family]
        best = subset.loc[subset.pooled_auc.idxmax()]
        selected[family] = config_by_name[str(best["config"])]
    return selected, trials.sort_values(
        ["family", "pooled_auc"], ascending=[True, False]
    ).reset_index(drop=True)

def simplex_weight_grid(
    step: float = 0.25, dimensions: int = 4
) -> list[np.ndarray]:
    units = int(round(1.0 / step))
    if dimensions < 2 or not np.isclose(units * step, 1.0):
        raise ValueError("Invalid simplex grid")
    return [
        np.asarray(values, dtype=float) / units
        for values in itertools.product(range(units + 1), repeat=dimensions)
        if sum(values) == units
    ]


def _existing_screen_probability(
    train: pd.DataFrame,
    y_train: pd.Series,
    valid: pd.DataFrame,
    seed: int,
) -> np.ndarray:
    svm = _fit_legacy_svm(train, y_train, seed + 875, direct_icv=True)
    xgb = _fit_locked_xgb(train, y_train, seed + 900)
    return np.clip(
        0.4 * svm.predict_proba(valid) + 0.6 * xgb.predict_proba(valid),
        0.0,
        1.0,
    )


FUSION_BRANCHES = (*MODEL_FAMILIES, "existing_screen")


def select_fusion(
    data: pd.DataFrame,
    y: pd.Series,
    selected: dict[str, NormModelConfig],
    mode: ExperimentMode,
) -> tuple[np.ndarray, pd.DataFrame]:
    splits = _fixed_splits(y, mode.development_splits, FUSION_SELECTION_SEED)
    matrix = np.full((len(y), len(FUSION_BRANCHES)), np.nan, dtype=float)
    for fold_id, (train_idx, valid_idx) in enumerate(splits, start=1):
        train, valid = data.iloc[train_idx], data.iloc[valid_idx]
        y_train = y.iloc[train_idx]
        for family_id, family in enumerate(MODEL_FAMILIES):
            config = selected[family]
            model = _fit_norm_model(
                train,
                y_train,
                config.view,
                config.family,
                config.params,
                FUSION_SELECTION_SEED + fold_id * 1000 + family_id,
            )
            matrix[valid_idx, family_id] = model.predict_proba(valid)
        matrix[valid_idx, -1] = _existing_screen_probability(
            train,
            y_train,
            valid,
            FUSION_SELECTION_SEED + fold_id * 1000,
        )
        print(
            f"Norm fusion development fold {fold_id}/{len(splits)} complete",
            flush=True,
        )
    if not np.isfinite(matrix).all():
        raise RuntimeError("Non-finite fusion development matrix")
    valid_indices = [valid_idx for _, valid_idx in splits]
    rows = []
    for weights in simplex_weight_grid(0.25, len(FUSION_BRANCHES)):
        score = matrix @ weights
        fold_auc = [
            float(roc_auc_score(y.iloc[index], score[index]))
            for index in valid_indices
        ]
        rows.append(
            {
                **{
                    f"weight_{branch}": float(weight)
                    for branch, weight in zip(FUSION_BRANCHES, weights)
                },
                **_score_oof(y, score, fold_auc),
            }
        )
    trials = pd.DataFrame(rows)
    best_index = int(trials.pooled_auc.to_numpy().argmax())
    best = trials.iloc[best_index]
    weights = np.asarray(
        [best[f"weight_{branch}"] for branch in FUSION_BRANCHES], dtype=float
    )
    return weights, trials.sort_values("pooled_auc", ascending=False).reset_index(
        drop=True
    )

NORM_MODEL_NAMES = {
    "elastic_net": "NormElasticNet",
    "rbf_svm": "NormRbfSVM",
    "xgboost": "NormXGBoost",
}


def frozen_repeated_validation(
    data: pd.DataFrame,
    y: pd.Series,
    selected: dict[str, NormModelConfig],
    fusion_weights: np.ndarray,
    mode: ExperimentMode,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    splitter = RepeatedStratifiedKFold(
        n_splits=mode.outer_splits,
        n_repeats=mode.outer_repeats,
        random_state=VALIDATION_SEED,
    )
    splits = list(splitter.split(data, y))
    names = tuple(NORM_MODEL_NAMES.values()) + (
        "CerebNormFusion",
        "ExistingCerebADHDScreen",
    )
    score_sum = {name: np.zeros(len(y), dtype=float) for name in names}
    score_count = np.zeros(len(y), dtype=int)
    fold_rows: list[dict[str, Any]] = []

    for fold_id, (train_idx, valid_idx) in enumerate(splits, start=1):
        train, valid = data.iloc[train_idx], data.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        branch_probability: dict[str, np.ndarray] = {}
        for family_id, family in enumerate(MODEL_FAMILIES):
            config = selected[family]
            model = _fit_norm_model(
                train,
                y_train,
                config.view,
                config.family,
                config.params,
                VALIDATION_SEED + fold_id * 1000 + family_id,
            )
            branch_probability[family] = model.predict_proba(valid)

        existing_svm = _fit_legacy_svm(
            train,
            y_train,
            VALIDATION_SEED + fold_id * 875,
            direct_icv=True,
        )
        existing_xgb = _fit_locked_xgb(
            train, y_train, VALIDATION_SEED + fold_id * 900
        )
        existing_probability = np.clip(
            0.4 * existing_svm.predict_proba(valid)
            + 0.6 * existing_xgb.predict_proba(valid),
            0.0,
            1.0,
        )
        fusion_matrix = np.column_stack(
            [branch_probability[family] for family in MODEL_FAMILIES]
            + [existing_probability]
        )
        fusion_probability = np.clip(fusion_matrix @ fusion_weights, 0.0, 1.0)
        scores = {
            **{
                NORM_MODEL_NAMES[family]: branch_probability[family]
                for family in MODEL_FAMILIES
            },
            "CerebNormFusion": fusion_probability,
            "ExistingCerebADHDScreen": existing_probability,
        }
        for name, probability in scores.items():
            score_sum[name][valid_idx] += probability
            fold_rows.append(
                {
                    "outer_fold": fold_id,
                    "model": name,
                    **_metrics(y_valid, probability),
                }
            )
        score_count[valid_idx] += 1
        print(
            f"Validation fold {fold_id:02d}/{len(splits)}: "
            f"norm={roc_auc_score(y_valid, fusion_probability):.3f}; "
            f"baseline={roc_auc_score(y_valid, existing_probability):.3f}",
            flush=True,
        )

    if not np.all(score_count == mode.outer_repeats):
        raise RuntimeError("Each participant must be predicted once per repeat")
    aggregated = {name: score_sum[name] / score_count for name in names}
    fold_metrics = pd.DataFrame(fold_rows)
    fold_summary = (
        fold_metrics.groupby("model")["AUC"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "mean_fold_AUC", "std": "sd_fold_AUC"})
    )
    comparison = pd.DataFrame(
        [{"model": name, **_metrics(y, aggregated[name])} for name in names]
    ).merge(fold_summary, on="model", how="left")
    comparison = comparison.sort_values("AUC", ascending=False).reset_index(drop=True)
    predictions = pd.DataFrame(
        {
            "row_id": np.arange(len(y)),
            "Sub": data["Sub"] if "Sub" in data else "",
            "DX": y,
            "Age": pd.to_numeric(data["Age"], errors="coerce"),
            "Gender": pd.to_numeric(data["Gender"], errors="coerce"),
            "prediction_count": score_count,
            **aggregated,
        }
    )
    bootstrap_rows = []
    for comparator in ("ExistingCerebADHDScreen", *NORM_MODEL_NAMES.values()):
        bootstrap_rows.append(
            {
                "comparison": f"CerebNormFusion - {comparator}",
                **_bootstrap_delta(
                    y.to_numpy(),
                    aggregated["CerebNormFusion"],
                    aggregated[comparator],
                    mode.bootstrap_samples,
                    VALIDATION_SEED + len(bootstrap_rows),
                ),
            }
        )
    return fold_metrics, comparison, predictions, pd.DataFrame(bootstrap_rows)


def run_experiment(
    data: pd.DataFrame,
    mode: ExperimentMode,
    output_dir: Path,
) -> dict[str, Any]:
    started = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    data = _prepare_data(data)
    y = data["DX"].astype(int).reset_index(drop=True)

    selected, model_trials = select_family_configs(data, y, mode)
    fusion_weights, fusion_trials = select_fusion(data, y, selected, mode)
    print(
        "Frozen CerebNorm: "
        + "; ".join(
            f"{family}={selected[family].name}" for family in MODEL_FAMILIES
        )
        + "; weights="
        + str(
            {
                branch: float(weight)
                for branch, weight in zip(FUSION_BRANCHES, fusion_weights)
            }
        ),
        flush=True,
    )
    fold_metrics, comparison, predictions, bootstrap = frozen_repeated_validation(
        data, y, selected, fusion_weights, mode
    )

    model_trials.to_csv(output_dir / "development_model_trials.csv", index=False)
    fusion_trials.to_csv(output_dir / "development_fusion_trials.csv", index=False)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    predictions.to_csv(output_dir / "oof_predictions.csv", index=False)
    bootstrap.to_csv(output_dir / "bootstrap_comparisons.csv", index=False)

    config = {
        "mode": mode.__dict__,
        "n": len(data),
        "adhd": int((y == 1).sum()),
        "control": int((y == 0).sum()),
        "normative_reference": "training_partition_controls_only",
        "normative_features": list(NORM_FEATURES),
        "normative_covariates": ["Age", "Age_squared", "Gender", "ICVcm3"],
        "z_clip": 5.0,
        "derived_summary_features": ["norm_mean_abs_z", "norm_extreme_count"],
        "model_selection_seed": MODEL_SELECTION_SEED,
        "fusion_selection_seed": FUSION_SELECTION_SEED,
        "validation_seed": VALIDATION_SEED,
        "selected_models": {
            family: {
                "name": selected[family].name,
                "view": selected[family].view,
                "parameter_name": selected[family].parameter_name,
                "params": selected[family].params,
            }
            for family in MODEL_FAMILIES
        },
        "fusion_weights": {
            branch: float(weight)
            for branch, weight in zip(FUSION_BRANCHES, fusion_weights)
        },
        "forbidden_model_inputs": sorted(FORBIDDEN_MODEL_INPUTS),
        "elapsed_seconds": time.time() - started,
        "baseline_auc_reference": 0.722773,
        "stretch_target_auc": 0.75,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    norm_auc = float(
        comparison.loc[comparison.model == "CerebNormFusion", "AUC"].iloc[0]
    )
    baseline_auc = float(
        comparison.loc[
            comparison.model == "ExistingCerebADHDScreen", "AUC"
        ].iloc[0]
    )
    main_delta = bootstrap.iloc[0]
    promote = (
        norm_auc - baseline_auc >= 0.01 and float(main_delta["ci_lower"]) > 0
    )
    lines = [
        "CerebNorm-ADHDScreen exploratory internal-validation summary",
        "",
        "Selected models: "
        + "; ".join(config.name for config in selected.values()),
        "Fusion weights: "
        + ", ".join(
            f"{branch}={weight:.2f}"
            for branch, weight in zip(FUSION_BRANCHES, fusion_weights)
        ),
        "",
        comparison.to_string(index=False),
        "",
        bootstrap.to_string(index=False),
        "",
        f"Predefined promotion rule (delta >= 0.01 and CI lower > 0): "
        f"{'MET' if promote else 'NOT MET'}",
        f"Stretch target AUC >= 0.750: {'MET' if norm_auc >= 0.75 else 'NOT MET'}",
        f"All {len(data)} participants received predictions; no rejection option was used.",
        "This is exploratory internal validation, not independent validation.",
    ]
    (output_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    return {
        "model_trials": model_trials,
        "fusion_trials": fusion_trials,
        "comparison": comparison,
        "predictions": predictions,
        "bootstrap": bootstrap,
        "config": config,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("cerebellum.csv"))
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode = SMOKE_MODE if args.mode == "smoke" else FULL_MODE
    output = args.output or Path(f"cerebnorm_adhd_screen_outputs_{mode.name}")
    data = pd.read_csv(args.data)
    print(
        f"CerebNorm-ADHDScreen | mode={mode.name} | N={len(data)} | "
        f"output={output}",
        flush=True,
    )
    result = run_experiment(data, mode, output)
    print(result["comparison"].to_string(index=False), flush=True)
    print(f"Finished. Outputs: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
