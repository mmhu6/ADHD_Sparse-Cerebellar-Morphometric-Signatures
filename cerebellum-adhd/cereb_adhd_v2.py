"""Cereb-ADHDScreen v2: anatomy-aware bounded optimization experiment.

The pipeline uses only ``cerebellum.csv``.  Diagnosis, symptom scores,
medication status, and subject identifiers are never model inputs.  Global
stability-selection development intentionally follows the paper's full-cohort
exploratory convention and must not be described as independent validation.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
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
    feature_groups,
)


STABILITY_SEED = 642
MODEL_SELECTION_SEED = 442
FUSION_SELECTION_SEED = 542
VALIDATION_SEED = 42

DIRECT_COVARIATES = ("Age", "Gender", "ICVcm3")
FORBIDDEN_MODEL_INPUTS = frozenset(
    {"DX", "ADHDIndex", "Inattentive", "HyperImpulsive", "MedStatus", "Sub"}
)
MODEL_FAMILIES = ("elastic_net", "rbf_svm", "xgboost")

REGIONS = (
    "III",
    "III_A",
    "IV",
    "V",
    "VI",
    "CrusI",
    "CrusII",
    "VIIB",
    "VIIIA",
    "VIIIB",
    "IX",
    "X",
)
THICKNESS_SIDES = ("mean", "right", "left")
ENGINEERED_FEATURE_COUNT = len(REGIONS) * (3 + len(THICKNESS_SIDES))


def _region_column(region: str, measurement: str) -> str:
    """Map a logical region/measurement to the source CSV column."""

    if region == "III_A":
        return f"III{measurement}_A"
    return f"{region}{measurement}"


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> np.ndarray:
    numerator = pd.to_numeric(numerator, errors="coerce").to_numpy(float)
    denominator = pd.to_numeric(denominator, errors="coerce").to_numpy(float)
    ratio = np.divide(
        numerator,
        denominator,
        out=np.full(len(numerator), np.nan, dtype=float),
        where=np.isfinite(denominator) & (np.abs(denominator) > 1e-12),
    )
    return ratio


class AnatomicalFeatureEngineer:
    """Create row-wise anatomical ratios that are absent from the source."""

    @staticmethod
    def engineered_feature_names() -> tuple[str, ...]:
        names: list[str] = []
        for region in REGIONS:
            names.extend(
                [
                    f"relative_volume__{region}",
                    f"relative_greymatter__{region}",
                    f"local_greymatter_fraction__{region}",
                ]
            )
            names.extend(
                f"relative_thickness_{side}__{region}"
                for side in THICKNESS_SIDES
            )
        return tuple(names)

    @staticmethod
    def required_source_columns() -> tuple[str, ...]:
        columns = {
            "Cerebellumtotalcm3",
            "Cerebellumgreymattercm3",
            *(
                f"Cerebellum{side}corticalthickness"
                for side in THICKNESS_SIDES
            ),
        }
        for region in REGIONS:
            columns.update(
                {
                    _region_column(region, "totalcm3"),
                    _region_column(region, "greymattercm3"),
                    *(
                        _region_column(region, f"{side}corticalthickness")
                        for side in THICKNESS_SIDES
                    ),
                }
            )
        return tuple(sorted(columns))

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X)
        missing = set(self.required_source_columns()).difference(X.columns)
        if missing:
            raise ValueError(f"Missing anatomy source columns: {sorted(missing)}")
        output = X.copy()
        for region in REGIONS:
            total = _region_column(region, "totalcm3")
            grey = _region_column(region, "greymattercm3")
            output[f"relative_volume__{region}"] = _safe_ratio(
                X[total], X["Cerebellumtotalcm3"]
            )
            output[f"relative_greymatter__{region}"] = _safe_ratio(
                X[grey], X["Cerebellumgreymattercm3"]
            )
            output[f"local_greymatter_fraction__{region}"] = _safe_ratio(
                X[grey], X[total]
            )
            for side in THICKNESS_SIDES:
                output[f"relative_thickness_{side}__{region}"] = _safe_ratio(
                    X[_region_column(region, f"{side}corticalthickness")],
                    X[f"Cerebellum{side}corticalthickness"],
                )
        return output


def curated_raw_feature_names(columns: Iterable[str]) -> tuple[str, ...]:
    """Return all 234 paper-compatible raw brain candidates.

    The 39 separately encoded norm-thickness columns remain excluded, while
    known useful ICV-normalized volume columns such as ``Xleft`` are retained.
    """

    return tuple(feature_groups(columns)["brain_raw"])


def candidate_feature_names(columns: Iterable[str]) -> tuple[str, ...]:
    return curated_raw_feature_names(columns) + AnatomicalFeatureEngineer.engineered_feature_names()


def model_source_columns(columns: Iterable[str]) -> tuple[str, ...]:
    columns = tuple(columns)
    available = set(columns)
    sources = (
        set(curated_raw_feature_names(columns))
        | set(AnatomicalFeatureEngineer.required_source_columns())
        | set(DIRECT_COVARIATES)
    )
    missing = sources.difference(available)
    if missing:
        raise ValueError(f"Missing model source columns: {sorted(missing)}")
    if sources.intersection(FORBIDDEN_MODEL_INPUTS):
        raise ValueError("Forbidden model input detected")
    return tuple(name for name in columns if name in sources)


def _numeric_with_fill(series: pd.Series, fill: float) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(float)
    return np.where(np.isfinite(values), values, fill)


@dataclass
class V2MatrixTransformer:
    features: tuple[str, ...]
    engineer: AnatomicalFeatureEngineer
    residualizer: FoldResidualizer
    covariate_medians: dict[str, float]

    @classmethod
    def fit(cls, X: pd.DataFrame, features: Sequence[str]):
        features = tuple(features)
        valid = set(candidate_feature_names(X.columns))
        invalid = set(features).difference(valid)
        if invalid:
            raise ValueError(f"Invalid v2 feature names: {sorted(invalid)}")
        engineer = AnatomicalFeatureEngineer()
        transformed = engineer.transform(X)
        residualizer = FoldResidualizer(features, include_icv=True).fit(transformed)
        medians: dict[str, float] = {}
        for name in DIRECT_COVARIATES:
            values = pd.to_numeric(X[name], errors="coerce").to_numpy(float)
            finite = values[np.isfinite(values)]
            if not len(finite):
                raise ValueError(f"Covariate {name} has no finite values")
            medians[name] = float(np.median(finite))
        return cls(features, engineer, residualizer, medians)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        transformed = self.engineer.transform(X)
        brain = self.residualizer.transform(transformed).to_numpy(float)
        direct = np.column_stack(
            [
                _numeric_with_fill(X[name], self.covariate_medians[name])
                for name in DIRECT_COVARIATES
            ]
        )
        return np.column_stack([brain, direct])


@dataclass
class FittedV2Model:
    family: str
    transformer: V2MatrixTransformer
    scaler: StandardScaler | None
    model: Any
    input_columns: tuple[str, ...]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        matrix = self.transformer.transform(X)
        if self.scaler is not None:
            matrix = self.scaler.transform(matrix)
        probability = self.model.predict_proba(matrix)[:, 1]
        return np.clip(np.asarray(probability, dtype=float), 0.0, 1.0)


def _fit_v2_model(
    X: pd.DataFrame,
    y: pd.Series,
    features: Sequence[str],
    family: str,
    params: dict[str, Any],
    seed: int,
) -> FittedV2Model:
    if family not in MODEL_FAMILIES:
        raise ValueError(f"Unknown model family: {family}")
    transformer = V2MatrixTransformer.fit(X, features)
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
    return FittedV2Model(
        family,
        transformer,
        scaler,
        model,
        model_source_columns(X.columns),
    )


def _ecdf_transform(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference, dtype=float)
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(values)):
        raise ValueError("ECDF inputs must be finite")
    if not len(reference):
        raise ValueError("ECDF reference cannot be empty")
    ordered = np.sort(reference)
    return np.searchsorted(ordered, values, side="right") / float(len(ordered))


def simplex_weight_grid(step: float = 0.25, dimensions: int = 3) -> list[np.ndarray]:
    if dimensions < 2 or not 0 < step <= 1:
        raise ValueError("Invalid simplex grid specification")
    units = int(round(1.0 / step))
    if not np.isclose(units * step, 1.0):
        raise ValueError("step must divide 1 exactly")
    weights = []
    for values in itertools.product(range(units + 1), repeat=dimensions):
        if sum(values) == units:
            weights.append(np.asarray(values, dtype=float) / units)
    return weights


@dataclass(frozen=True)
class ExperimentMode:
    name: str
    stability_splits: int
    stability_repeats: int
    stability_top_k: int
    panel_sizes: tuple[int, ...]
    development_splits: int
    fusion_inner_splits: int
    outer_splits: int
    outer_repeats: int
    bootstrap_samples: int
    full_candidate_library: bool


SMOKE_MODE = ExperimentMode(
    "smoke", 3, 2, 12, (8, 12), 3, 2, 3, 1, 300, False
)
FULL_MODE = ExperimentMode(
    "full", 5, 10, 20, (8, 12, 20), 5, 3, 5, 3, 5000, True
)


@dataclass(frozen=True)
class ModelConfig:
    family: str
    panel_size: int
    parameter_name: str
    params: dict[str, Any]

    @property
    def name(self) -> str:
        return f"{self.family}__top{self.panel_size}__{self.parameter_name}"


XGB_V2_LIBRARY: tuple[tuple[str, dict[str, Any]], ...] = (
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
    (
        "depth2_sparse",
        {
            "n_estimators": 340,
            "max_depth": 2,
            "learning_rate": 0.02,
            "min_child_weight": 10,
            "gamma": 0.15,
            "subsample": 0.80,
            "colsample_bytree": 0.65,
            "reg_alpha": 1.0,
            "reg_lambda": 35.0,
            "scale_pos_weight": 1.0,
        },
    ),
)


def _prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    needed = set(AnatomicalFeatureEngineer.required_source_columns()) | set(
        DIRECT_COVARIATES
    ) | {"DX"}
    missing = needed.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    clean = data.copy()
    clean["DX"] = pd.to_numeric(clean["DX"], errors="raise").astype(int)
    clean = clean.dropna(subset=["DX", *DIRECT_COVARIATES]).reset_index(drop=True)
    if set(clean["DX"].unique()) != {0, 1}:
        raise ValueError("DX must be binary 0/1")
    return clean


def _feature_source_type(name: str) -> str:
    if name.startswith("relative_") or name.startswith("local_"):
        return "engineered_anatomical"
    if "corticalthickness" in name.lower():
        return "raw_thickness"
    if "greymatter" in name.lower():
        return "raw_greymatter"
    return "raw_volume_or_asymmetry"


def _rank_and_correlation_select(
    matrix: np.ndarray,
    y: pd.Series,
    feature_names: Sequence[str],
    top_k: int,
    corr_threshold: float = 0.98,
) -> tuple[list[int], np.ndarray]:
    """Univariate rank then greedily remove near-duplicate ranked features."""

    matrix = np.asarray(matrix, dtype=float)
    scores = np.asarray(f_classif(matrix, np.asarray(y, dtype=int))[0], dtype=float)
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    order = np.argsort(-scores, kind="mergesort")
    ranks = np.empty(len(feature_names), dtype=int)
    ranks[order] = np.arange(1, len(feature_names) + 1)

    means = np.mean(matrix, axis=0)
    sd = np.std(matrix, axis=0)
    usable = np.isfinite(sd) & (sd > 1e-12)
    standardized = np.zeros_like(matrix, dtype=float)
    standardized[:, usable] = (matrix[:, usable] - means[usable]) / sd[usable]

    selected: list[int] = []
    for index in order:
        if len(selected) >= top_k:
            break
        if not usable[index]:
            continue
        if selected:
            correlations = np.abs(
                standardized[:, selected].T @ standardized[:, index] / len(matrix)
            )
            if np.any(correlations >= corr_threshold):
                continue
        selected.append(int(index))
    if len(selected) < top_k:
        raise RuntimeError(
            f"Only {len(selected)} non-redundant features available for top-{top_k}"
        )
    return selected, ranks


def repeated_stability_selection(
    data: pd.DataFrame,
    y: pd.Series,
    mode: ExperimentMode,
) -> pd.DataFrame:
    """Aggregate selection frequency over one predefined repeated CV object."""

    engineer = AnatomicalFeatureEngineer()
    transformed = engineer.transform(data)
    features = candidate_feature_names(data.columns)
    count = np.zeros(len(features), dtype=int)
    rank_sum = np.zeros(len(features), dtype=float)
    splitter = RepeatedStratifiedKFold(
        n_splits=mode.stability_splits,
        n_repeats=mode.stability_repeats,
        random_state=STABILITY_SEED,
    )
    n_folds = 0
    for fold_id, (train_idx, _) in enumerate(splitter.split(data, y), start=1):
        train = transformed.iloc[train_idx]
        residualizer = FoldResidualizer(features, include_icv=True).fit(train)
        matrix = residualizer.transform(train).to_numpy(float)
        selected, ranks = _rank_and_correlation_select(
            matrix,
            y.iloc[train_idx],
            features,
            top_k=mode.stability_top_k,
        )
        count[selected] += 1
        rank_sum += ranks
        n_folds += 1
        if fold_id % mode.stability_splits == 0:
            print(
                f"Stability repeat {fold_id // mode.stability_splits}/"
                f"{mode.stability_repeats} complete",
                flush=True,
            )
    table = pd.DataFrame(
        {
            "feature": features,
            "source_type": [_feature_source_type(name) for name in features],
            "selection_count": count,
            "selection_frequency": count / float(n_folds),
            "mean_univariate_rank": rank_sum / float(n_folds),
        }
    )
    return table.sort_values(
        ["selection_frequency", "mean_univariate_rank", "feature"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def build_feature_panels(
    stability: pd.DataFrame, panel_sizes: Sequence[int]
) -> dict[int, tuple[str, ...]]:
    ordered = stability["feature"].astype(str).tolist()
    anchors = tuple(LEGACY_FEATURES)
    missing_anchors = set(anchors).difference(ordered)
    if missing_anchors:
        raise ValueError(f"Missing paper anchors: {sorted(missing_anchors)}")
    extended_order = list(anchors) + [name for name in ordered if name not in anchors]
    panels = {}
    for size in panel_sizes:
        if size < len(anchors):
            raise ValueError("Panel cannot be smaller than the paper anchor set")
        if size > len(extended_order):
            raise ValueError(f"Panel size {size} exceeds candidate count")
        panels[int(size)] = tuple(extended_order[: int(size)])
    return panels

def _candidate_configs(mode: ExperimentMode) -> list[ModelConfig]:
    configs: list[ModelConfig] = []
    elastic_cs = (
        (0.03, 0.1, 0.3, 1.0) if mode.full_candidate_library else (0.1, 0.3)
    )
    svm_cs = (0.3, 1.0, 3.0) if mode.full_candidate_library else (0.3, 1.0)
    xgb_library = (
        XGB_V2_LIBRARY if mode.full_candidate_library else XGB_V2_LIBRARY[:1]
    )
    for panel_size in mode.panel_sizes:
        for C in elastic_cs:
            configs.append(
                ModelConfig(
                    "elastic_net",
                    panel_size,
                    f"C{C:g}_l1r0.5",
                    {"C": C, "l1_ratio": 0.5},
                )
            )
        for C in svm_cs:
            configs.append(
                ModelConfig(
                    "rbf_svm",
                    panel_size,
                    f"C{C:g}_gamma_scale",
                    {"C": C, "gamma": "scale"},
                )
            )
        for parameter_name, params in xgb_library:
            configs.append(
                ModelConfig("xgboost", panel_size, parameter_name, dict(params))
            )
    return configs


def _fixed_splits(y: pd.Series, n_splits: int, seed: int):
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(y)), y))


def _model_oof(
    data: pd.DataFrame,
    y: pd.Series,
    config: ModelConfig,
    features: Sequence[str],
    splits,
    seed: int,
) -> tuple[np.ndarray, list[float]]:
    probability = np.full(len(y), np.nan, dtype=float)
    fold_auc: list[float] = []
    for fold_id, (train_idx, valid_idx) in enumerate(splits, start=1):
        model = _fit_v2_model(
            data.iloc[train_idx],
            y.iloc[train_idx],
            features,
            config.family,
            config.params,
            seed + fold_id,
        )
        fold_probability = model.predict_proba(data.iloc[valid_idx])
        probability[valid_idx] = fold_probability
        fold_auc.append(float(roc_auc_score(y.iloc[valid_idx], fold_probability)))
    if not np.isfinite(probability).all():
        raise RuntimeError(f"Non-finite OOF probabilities for {config.name}")
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
    panels: dict[int, tuple[str, ...]],
    mode: ExperimentMode,
) -> tuple[dict[str, ModelConfig], pd.DataFrame]:
    splits = _fixed_splits(y, mode.development_splits, MODEL_SELECTION_SEED)
    rows: list[dict[str, Any]] = []
    configs = _candidate_configs(mode)
    for trial_id, config in enumerate(configs, start=1):
        probability, fold_auc = _model_oof(
            data,
            y,
            config,
            panels[config.panel_size],
            splits,
            MODEL_SELECTION_SEED + trial_id * 100,
        )
        rows.append(
            {
                "config": config.name,
                "family": config.family,
                "panel_size": config.panel_size,
                "parameter_name": config.parameter_name,
                "params": json.dumps(config.params, sort_keys=True),
                **_score_oof(y, probability, fold_auc),
            }
        )
        print(
            f"Model trial {trial_id:02d}/{len(configs)}: {config.name} "
            f"AUC={rows[-1]['pooled_auc']:.4f}",
            flush=True,
        )
    trials = pd.DataFrame(rows)
    config_by_name = {config.name: config for config in configs}
    selected: dict[str, ModelConfig] = {}
    for family in MODEL_FAMILIES:
        subset = trials[trials.family == family]
        best_row = subset.loc[subset.pooled_auc.idxmax()]
        selected[family] = config_by_name[str(best_row["config"])]
    return selected, trials.sort_values(
        ["family", "pooled_auc"], ascending=[True, False]
    ).reset_index(drop=True)

def _fit_fold_base_scores(
    train: pd.DataFrame,
    y_train: pd.Series,
    valid: pd.DataFrame,
    selected_configs: dict[str, ModelConfig],
    panels: dict[int, tuple[str, ...]],
    inner_splits: int,
    seed: int,
    need_rank: bool,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    probability: dict[str, np.ndarray] = {}
    rank_score: dict[str, np.ndarray] = {}
    for family_id, family in enumerate(MODEL_FAMILIES, start=1):
        config = selected_configs[family]
        features = panels[config.panel_size]
        reference = None
        if need_rank:
            inner = _fixed_splits(
                y_train.reset_index(drop=True), inner_splits, seed + family_id * 100
            )
            reference, _ = _model_oof(
                train.reset_index(drop=True),
                y_train.reset_index(drop=True),
                config,
                features,
                inner,
                seed + family_id * 1000,
            )
        model = _fit_v2_model(
            train,
            y_train,
            features,
            family,
            config.params,
            seed + family_id,
        )
        values = model.predict_proba(valid)
        probability[family] = values
        if need_rank:
            assert reference is not None
            rank_score[family] = _ecdf_transform(reference, values)
    return probability, rank_score


def select_fusion(
    data: pd.DataFrame,
    y: pd.Series,
    selected_configs: dict[str, ModelConfig],
    panels: dict[int, tuple[str, ...]],
    mode: ExperimentMode,
) -> tuple[str, np.ndarray, pd.DataFrame]:
    splits = _fixed_splits(y, mode.development_splits, FUSION_SELECTION_SEED)
    probability = np.full((len(y), len(MODEL_FAMILIES)), np.nan, dtype=float)
    rank_score = np.full_like(probability, np.nan)
    for fold_id, (train_idx, valid_idx) in enumerate(splits, start=1):
        fold_probability, fold_rank = _fit_fold_base_scores(
            data.iloc[train_idx],
            y.iloc[train_idx],
            data.iloc[valid_idx],
            selected_configs,
            panels,
            mode.fusion_inner_splits,
            FUSION_SELECTION_SEED + fold_id * 10000,
            need_rank=True,
        )
        for family_id, family in enumerate(MODEL_FAMILIES):
            probability[valid_idx, family_id] = fold_probability[family]
            rank_score[valid_idx, family_id] = fold_rank[family]
        print(
            f"Fusion development fold {fold_id}/{len(splits)} complete",
            flush=True,
        )
    if not np.isfinite(probability).all() or not np.isfinite(rank_score).all():
        raise RuntimeError("Fusion development contains non-finite scores")

    rows: list[dict[str, Any]] = []
    valid_indices = [valid_idx for _, valid_idx in splits]
    for representation, matrix in (
        ("probability", probability),
        ("rank_ecdf", rank_score),
    ):
        for weights in simplex_weight_grid(
            step=0.25, dimensions=len(MODEL_FAMILIES)
        ):
            score = matrix @ weights
            fold_auc = [
                float(roc_auc_score(y.iloc[index], score[index]))
                for index in valid_indices
            ]
            rows.append(
                {
                    "representation": representation,
                    **{
                        f"weight_{family}": float(weight)
                        for family, weight in zip(MODEL_FAMILIES, weights)
                    },
                    **_score_oof(y, score, fold_auc),
                }
            )
    trials = pd.DataFrame(rows)
    best_index = int(trials["pooled_auc"].to_numpy().argmax())
    best = trials.iloc[best_index]
    weights = np.asarray(
        [best[f"weight_{family}"] for family in MODEL_FAMILIES], dtype=float
    )
    return (
        str(best["representation"]),
        weights,
        trials.sort_values("pooled_auc", ascending=False).reset_index(drop=True),
    )

V2_MODEL_NAMES = {
    "elastic_net": "V2ElasticNet",
    "rbf_svm": "V2RbfSVM",
    "xgboost": "V2XGBoost",
}


def _selective_analysis(
    y: pd.Series,
    model_scores: dict[str, np.ndarray],
    rejection_fractions: Sequence[float] = (0.0, 0.1, 0.2, 0.3),
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    y_array = np.asarray(y, dtype=int)
    for model_name, score in model_scores.items():
        score = np.asarray(score, dtype=float)
        percentile = pd.Series(score).rank(method="average", pct=True).to_numpy()
        confidence = np.abs(percentile - 0.5)
        order = np.argsort(-confidence, kind="mergesort")
        for rejection in rejection_fractions:
            keep_n = int(round((1.0 - rejection) * len(y_array)))
            keep = np.sort(order[:keep_n])
            rows.append(
                {
                    "model": model_name,
                    "rejection_fraction": float(rejection),
                    "coverage": keep_n / float(len(y_array)),
                    "n_retained": keep_n,
                    "adhd_retained": int((y_array[keep] == 1).sum()),
                    "control_retained": int((y_array[keep] == 0).sum()),
                    **_metrics(y_array[keep], score[keep]),
                }
            )
    return pd.DataFrame(rows)


def frozen_repeated_validation(
    data: pd.DataFrame,
    y: pd.Series,
    selected_configs: dict[str, ModelConfig],
    panels: dict[int, tuple[str, ...]],
    fusion_representation: str,
    fusion_weights: np.ndarray,
    mode: ExperimentMode,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    splitter = RepeatedStratifiedKFold(
        n_splits=mode.outer_splits,
        n_repeats=mode.outer_repeats,
        random_state=VALIDATION_SEED,
    )
    splits = list(splitter.split(data, y))
    names = tuple(V2_MODEL_NAMES.values()) + (
        "V2Fusion",
        "ExistingCerebADHDScreen",
    )
    score_sum = {name: np.zeros(len(y), dtype=float) for name in names}
    score_count = np.zeros(len(y), dtype=int)
    fold_rows: list[dict[str, Any]] = []
    need_rank = fusion_representation == "rank_ecdf"

    for fold_id, (train_idx, valid_idx) in enumerate(splits, start=1):
        train, valid = data.iloc[train_idx], data.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        base_probability, base_rank = _fit_fold_base_scores(
            train,
            y_train,
            valid,
            selected_configs,
            panels,
            mode.fusion_inner_splits,
            VALIDATION_SEED + fold_id * 20000,
            need_rank=need_rank,
        )
        fusion_source = base_rank if need_rank else base_probability
        fusion_matrix = np.column_stack(
            [fusion_source[family] for family in MODEL_FAMILIES]
        )
        fusion_score = np.clip(fusion_matrix @ fusion_weights, 0.0, 1.0)

        existing_svm = _fit_legacy_svm(
            train,
            y_train,
            VALIDATION_SEED + fold_id * 875,
            direct_icv=True,
        )
        existing_xgb = _fit_locked_xgb(
            train, y_train, VALIDATION_SEED + fold_id * 900
        )
        existing_score = np.clip(
            0.4 * existing_svm.predict_proba(valid)
            + 0.6 * existing_xgb.predict_proba(valid),
            0.0,
            1.0,
        )
        scores = {
            **{
                V2_MODEL_NAMES[family]: base_probability[family]
                for family in MODEL_FAMILIES
            },
            "V2Fusion": fusion_score,
            "ExistingCerebADHDScreen": existing_score,
        }
        for name, score in scores.items():
            score_sum[name][valid_idx] += score
            fold_rows.append(
                {"outer_fold": fold_id, "model": name, **_metrics(y_valid, score)}
            )
        score_count[valid_idx] += 1
        print(
            f"Validation fold {fold_id:02d}/{len(splits)}: "
            f"v2={roc_auc_score(y_valid, fusion_score):.3f}; "
            f"baseline={roc_auc_score(y_valid, existing_score):.3f}",
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
    for comparator in ("ExistingCerebADHDScreen", *V2_MODEL_NAMES.values()):
        bootstrap_rows.append(
            {
                "comparison": f"V2Fusion - {comparator}",
                **_bootstrap_delta(
                    y.to_numpy(),
                    aggregated["V2Fusion"],
                    aggregated[comparator],
                    mode.bootstrap_samples,
                    VALIDATION_SEED + len(bootstrap_rows),
                ),
            }
        )
    bootstrap = pd.DataFrame(bootstrap_rows)
    selective = _selective_analysis(
        y,
        {
            "V2Fusion": aggregated["V2Fusion"],
            "ExistingCerebADHDScreen": aggregated["ExistingCerebADHDScreen"],
        },
    )
    return fold_metrics, comparison, predictions, bootstrap, selective


def run_experiment(
    data: pd.DataFrame,
    mode: ExperimentMode,
    output_dir: Path,
) -> dict[str, Any]:
    started = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    data = _prepare_data(data)
    y = data["DX"].astype(int).reset_index(drop=True)

    stability = repeated_stability_selection(data, y, mode)
    panels = build_feature_panels(stability, mode.panel_sizes)
    selected_configs, model_trials = select_family_configs(
        data, y, panels, mode
    )
    fusion_representation, fusion_weights, fusion_trials = select_fusion(
        data, y, selected_configs, panels, mode
    )
    print(
        "Frozen v2: "
        + "; ".join(
            f"{family}={selected_configs[family].name}"
            for family in MODEL_FAMILIES
        )
        + f"; fusion={fusion_representation}; weights={fusion_weights.tolist()}",
        flush=True,
    )

    fold_metrics, comparison, predictions, bootstrap, selective = (
        frozen_repeated_validation(
            data,
            y,
            selected_configs,
            panels,
            fusion_representation,
            fusion_weights,
            mode,
        )
    )

    stability.to_csv(output_dir / "stability_selection.csv", index=False)
    pd.DataFrame(
        [
            {"panel_size": size, "features": ";".join(features)}
            for size, features in panels.items()
        ]
    ).to_csv(output_dir / "feature_panels.csv", index=False)
    model_trials.to_csv(output_dir / "development_model_trials.csv", index=False)
    fusion_trials.to_csv(output_dir / "development_fusion_trials.csv", index=False)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    predictions.to_csv(output_dir / "oof_predictions.csv", index=False)
    bootstrap.to_csv(output_dir / "bootstrap_comparisons.csv", index=False)
    selective.to_csv(output_dir / "grey_zone_analysis.csv", index=False)

    config = {
        "mode": mode.__dict__,
        "n": len(data),
        "adhd": int((y == 1).sum()),
        "control": int((y == 0).sum()),
        "development_scope": "global_full_sample_exploratory_internal_development",
        "stability_seed": STABILITY_SEED,
        "model_selection_seed": MODEL_SELECTION_SEED,
        "fusion_selection_seed": FUSION_SELECTION_SEED,
        "validation_seed": VALIDATION_SEED,
        "raw_candidate_count": len(curated_raw_feature_names(data.columns)),
        "engineered_candidate_count": ENGINEERED_FEATURE_COUNT,
        "total_candidate_count": len(candidate_feature_names(data.columns)),
        "panels": {str(size): list(features) for size, features in panels.items()},
        "selected_models": {
            family: {
                "name": selected_configs[family].name,
                "panel_size": selected_configs[family].panel_size,
                "parameter_name": selected_configs[family].parameter_name,
                "params": selected_configs[family].params,
            }
            for family in MODEL_FAMILIES
        },
        "fusion_representation": fusion_representation,
        "fusion_weights": {
            family: float(weight)
            for family, weight in zip(MODEL_FAMILIES, fusion_weights)
        },
        "forbidden_model_inputs": sorted(FORBIDDEN_MODEL_INPUTS),
        "elapsed_seconds": time.time() - started,
        "baseline_auc_reference": 0.722773,
        "stretch_target_auc": 0.75,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    v2_auc = float(comparison.loc[comparison.model == "V2Fusion", "AUC"].iloc[0])
    baseline_auc = float(
        comparison.loc[
            comparison.model == "ExistingCerebADHDScreen", "AUC"
        ].iloc[0]
    )
    main_delta = bootstrap.iloc[0]
    promote = (
        v2_auc - baseline_auc >= 0.01 and float(main_delta["ci_lower"]) > 0
    )
    lines = [
        "Cereb-ADHDScreen v2 exploratory internal-validation summary",
        "",
        f"Fusion representation: {fusion_representation}",
        "Fusion weights: "
        + ", ".join(
            f"{family}={weight:.2f}"
            for family, weight in zip(MODEL_FAMILIES, fusion_weights)
        ),
        "Selected models: "
        + "; ".join(config.name for config in selected_configs.values()),
        "",
        comparison.to_string(index=False),
        "",
        bootstrap.to_string(index=False),
        "",
        f"Predefined promotion rule (delta >= 0.01 and CI lower > 0): "
        f"{'MET' if promote else 'NOT MET'}",
        f"Stretch target AUC >= 0.750: {'MET' if v2_auc >= 0.75 else 'NOT MET'}",
        f"Feature panels were developed using all {len(data)} participants.",
        "This is exploratory internal validation, not independent validation.",
    ]
    (output_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    return {
        "stability": stability,
        "panels": panels,
        "model_trials": model_trials,
        "fusion_trials": fusion_trials,
        "comparison": comparison,
        "predictions": predictions,
        "bootstrap": bootstrap,
        "selective": selective,
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
    output = args.output or Path(f"cereb_adhd_v2_outputs_{mode.name}")
    data = pd.read_csv(args.data)
    print(
        f"Cereb-ADHDScreen v2 | mode={mode.name} | N={len(data)} | "
        f"output={output}",
        flush=True,
    )
    result = run_experiment(data, mode, output)
    print(result["comparison"].to_string(index=False), flush=True)
    print(f"Finished. Outputs: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
