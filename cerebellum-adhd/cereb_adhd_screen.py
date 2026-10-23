"""Cereb-ADHDScreen: leakage-aware multi-view ADHD screening research model.

The module is intentionally separate from ``ADHD_c.py`` so the published
analysis and its outputs remain unchanged.  All data-dependent preprocessing
is fitted within the supplied training partition.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


warnings.filterwarnings("ignore", category=UserWarning)


SEED = 42
NON_PREDICTIVE_COLUMNS = {
    "Sub",
    "Gender",
    "Age",
    "DX",
    "ADHDIndex",
    "Inattentive",
    "HyperImpulsive",
    "MedStatus",
    "ICVcm3",
}


def raw_for_norm(norm_name: str) -> str:
    """Return the raw cortical-thickness column paired with a norm column."""

    if "norm" not in norm_name.lower():
        raise ValueError(f"Not a norm feature: {norm_name}")
    raw = norm_name.replace("norm", "")
    if raw == "Cerebellumcorticalthickness":
        raw = "Cerebellummeancorticalthickness"
    return raw


def feature_groups(columns: Iterable[str]) -> dict[str, list[str]]:
    """Partition the source schema into model views without overlap."""

    columns = list(columns)
    norm = [c for c in columns if "corticalthicknessnorm" in c.lower()]
    brain_raw = [
        c
        for c in columns
        if c not in NON_PREDICTIVE_COLUMNS and "norm" not in c.lower()
    ]
    thickness_raw = [
        c
        for c in brain_raw
        if "corticalthickness" in c.lower() and "asymmetry" not in c.lower()
    ]
    other = [c for c in brain_raw if c not in set(thickness_raw)]
    return {
        "thickness_raw": thickness_raw,
        "thickness_norm": norm,
        "other_morphometry": other,
        "brain_raw": brain_raw,
    }


def _numeric(series: pd.Series, fill: float | None = None) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(float)
    if fill is not None:
        values = np.where(np.isfinite(values), values, fill)
    return values


class FoldResidualizer(BaseEstimator, TransformerMixin):
    """Fold-fitted median imputation and nuisance regression for brain views."""

    def __init__(self, features: Sequence[str], include_icv: bool = True):
        self.features = tuple(features)
        self.include_icv = include_icv

    def _filled_covariate(self, X: pd.DataFrame, name: str) -> np.ndarray:
        return _numeric(X[name], self.covariate_medians_[name])

    def _design(self, X: pd.DataFrame) -> np.ndarray:
        age = self._filled_covariate(X, "Age")
        gender = self._filled_covariate(X, "Gender")
        age_z = (age - self.age_mean_) / self.age_sd_
        parts = [np.ones(len(X)), age_z, age_z**2, gender]
        if self.include_icv:
            icv = self._filled_covariate(X, "ICVcm3")
            icv_z = (icv - self.icv_mean_) / self.icv_sd_
            parts.append(icv_z)
        return np.column_stack(parts)

    def fit(self, X: pd.DataFrame, y=None):
        X = pd.DataFrame(X)
        if not self.features:
            raise ValueError("Residualizer requires at least one brain feature")
        needed = {"Age", "Gender"} | ({"ICVcm3"} if self.include_icv else set())
        missing = needed.difference(X.columns)
        if missing:
            raise ValueError(f"Missing nuisance columns: {sorted(missing)}")

        self.covariate_medians_ = {}
        for name in needed:
            values = _numeric(X[name])
            finite = values[np.isfinite(values)]
            if not len(finite):
                raise ValueError(f"Nuisance column {name} has no numeric values")
            self.covariate_medians_[name] = float(np.median(finite))

        age = self._filled_covariate(X, "Age")
        self.age_mean_ = float(np.mean(age))
        self.age_sd_ = float(np.std(age)) or 1.0
        if self.include_icv:
            icv = self._filled_covariate(X, "ICVcm3")
            self.icv_mean_ = float(np.mean(icv))
            self.icv_sd_ = float(np.std(icv)) or 1.0

        self.feature_medians_ = {}
        brain = []
        for name in self.features:
            if name not in X:
                raise ValueError(f"Missing brain feature: {name}")
            values = _numeric(X[name])
            finite = values[np.isfinite(values)]
            median = float(np.median(finite)) if len(finite) else 0.0
            self.feature_medians_[name] = median
            brain.append(np.where(np.isfinite(values), values, median))
        matrix = np.column_stack(brain)
        self.beta_ = np.linalg.lstsq(self._design(X), matrix, rcond=None)[0]
        self.feature_names_in_ = np.asarray(self.features, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X)
        brain = []
        for name in self.features:
            values = _numeric(X[name], self.feature_medians_[name])
            brain.append(values)
        residuals = np.column_stack(brain) - self._design(X) @ self.beta_
        return pd.DataFrame(residuals, columns=self.features, index=X.index)


class BrainViewTransformer(BaseEstimator, TransformerMixin):
    """Residualize, rank, correlation-prune, select, and scale one brain view."""

    def __init__(
        self,
        features: Sequence[str],
        include_icv: bool = True,
        max_features: int | None = None,
        corr_threshold: float = 0.95,
    ):
        self.features = tuple(features)
        self.include_icv = include_icv
        self.max_features = max_features
        self.corr_threshold = corr_threshold

    def fit(self, X: pd.DataFrame, y):
        self.residualizer_ = FoldResidualizer(
            self.features, include_icv=self.include_icv
        ).fit(X)
        residuals = self.residualizer_.transform(X)
        variance = residuals.var(axis=0)
        viable = variance[variance > 1e-10].index.tolist()
        if not viable:
            raise ValueError("No non-constant features remain in brain view")

        scores, _ = f_classif(residuals[viable], np.asarray(y))
        scores = np.nan_to_num(scores, nan=-np.inf, neginf=-np.inf)
        ranked = [viable[i] for i in np.argsort(scores)[::-1]]
        corr = residuals[ranked].corr().abs()
        keep: list[str] = []
        limit = len(ranked) if self.max_features is None else min(self.max_features, len(ranked))
        for name in ranked:
            if len(keep) >= limit:
                break
            if not keep or np.all(corr.loc[name, keep].fillna(0).to_numpy() < self.corr_threshold):
                keep.append(name)
        if not keep:
            raise ValueError("Correlation pruning removed every feature")

        self.selected_features_ = keep
        self.scaler_ = StandardScaler().fit(residuals[keep])
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        residuals = self.residualizer_.transform(X)
        return self.scaler_.transform(residuals[self.selected_features_])


@dataclass(frozen=True)
class RunMode:
    name: str
    outer_splits: int
    outer_repeats: int
    inner_splits: int
    xgb_trials: int
    bootstrap_samples: int


FAST_MODE = RunMode("fast", 3, 1, 2, 4, 300)
# The final locked screen does not use the two exploratory view branches;
# four fixed candidates keep their sensitivity-analysis comparison bounded.
FULL_MODE = RunMode("full", 5, 3, 3, 4, 5000)


@dataclass(frozen=True)
class BranchConfig:
    view: str
    max_features: int | None
    corr_threshold: float
    params: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "view": self.view,
            "max_features": self.max_features,
            "corr_threshold": self.corr_threshold,
            "params": self.params,
        }


@dataclass
class FittedBranch:
    transformer: BrainViewTransformer
    model: XGBClassifier

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict_proba(self.transformer.transform(X)))[:, 1]


def _view_features(groups: dict[str, list[str]], view: str) -> tuple[list[str], bool]:
    if view == "thickness_raw":
        return groups["thickness_raw"], True
    if view == "thickness_norm":
        return groups["thickness_norm"], False
    if view == "other_morphometry":
        return groups["other_morphometry"], True
    if view == "brain_raw":
        return groups["brain_raw"], True
    raise ValueError(f"Unknown brain view: {view}")


def _make_xgb(params: dict[str, Any], seed: int) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        n_jobs=1,
        random_state=seed,
        verbosity=0,
        **params,
    )


def _branch_candidates(
    views: Sequence[str],
    n_features_by_view: dict[str, int],
    n_trials: int,
    class_ratio: float,
    seed: int,
) -> list[BranchConfig]:
    """Generate deterministic, regularized XGBoost candidates."""

    rng = np.random.RandomState(seed)
    configs: list[BranchConfig] = []
    anchors = [
        {
            "n_estimators": 180, "max_depth": 2, "learning_rate": 0.03,
            "min_child_weight": 5, "gamma": 0.0, "subsample": 0.80,
            "colsample_bytree": 0.80, "reg_alpha": 0.10,
            "reg_lambda": 10.0, "scale_pos_weight": class_ratio,
        },
        {
            "n_estimators": 260, "max_depth": 1, "learning_rate": 0.05,
            "min_child_weight": 8, "gamma": 0.05, "subsample": 0.85,
            "colsample_bytree": 0.70, "reg_alpha": 0.50,
            "reg_lambda": 20.0, "scale_pos_weight": 1.0,
        },
    ]
    for index in range(n_trials):
        view = views[index % len(views)]
        size = n_features_by_view[view]
        feature_choices = (
            sorted(set([8, 12, 20, 30, size]))
            if size <= 50 else [12, 20, 40, 60, min(100, size)]
        )
        feature_choices = [value for value in feature_choices if value <= size]
        max_features = feature_choices[index % len(feature_choices)]
        if index < len(anchors):
            params = dict(anchors[index])
        else:
            params = {
                "n_estimators": int(rng.choice([120, 180, 240, 320, 450, 600])),
                "max_depth": int(rng.choice([1, 2, 3, 4])),
                "learning_rate": float(10 ** rng.uniform(-2.25, -0.90)),
                "min_child_weight": int(rng.choice([1, 2, 3, 5, 8, 12, 20])),
                "gamma": float(rng.choice([0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 1.0])),
                "subsample": float(rng.choice([0.55, 0.70, 0.85, 1.0])),
                "colsample_bytree": float(rng.choice([0.40, 0.55, 0.70, 0.85, 1.0])),
                "reg_alpha": float(rng.choice([0.0, 0.001, 0.01, 0.10, 1.0, 5.0, 10.0])),
                "reg_lambda": float(rng.choice([0.10, 1.0, 3.0, 10.0, 30.0, 100.0])),
                "scale_pos_weight": float(rng.choice([1.0, class_ratio])),
            }
        configs.append(BranchConfig(
            view=view,
            max_features=max_features,
            corr_threshold=float(rng.choice([0.85, 0.90, 0.95, 0.99])),
            params=params,
        ))
    return configs


def _fit_branch(
    X: pd.DataFrame,
    y: Sequence[int],
    groups: dict[str, list[str]],
    config: BranchConfig,
    seed: int,
) -> FittedBranch:
    features, include_icv = _view_features(groups, config.view)
    transformer = BrainViewTransformer(
        features,
        include_icv=include_icv,
        max_features=config.max_features,
        corr_threshold=config.corr_threshold,
    ).fit(X, y)
    matrix = transformer.transform(X)
    model = _make_xgb(config.params, seed).fit(matrix, np.asarray(y, dtype=int))
    return FittedBranch(transformer, model)


def _inner_splits(y: Sequence[int], n_splits: int, seed: int):
    y = np.asarray(y, dtype=int)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(y)), y))


def _tune_branch(
    X: pd.DataFrame,
    y: pd.Series,
    groups: dict[str, list[str]],
    views: Sequence[str],
    mode: RunMode,
    splits,
    seed: int,
) -> tuple[BranchConfig, pd.DataFrame]:
    ratio = float((y == 0).sum() / max((y == 1).sum(), 1))
    sizes = {view: len(_view_features(groups, view)[0]) for view in views}
    candidates = _branch_candidates(views, sizes, mode.xgb_trials, ratio, seed)
    rows = []
    best: BranchConfig | None = None
    best_auc = -np.inf
    for candidate_id, config in enumerate(candidates, start=1):
        scores = []
        for split_id, (train_idx, valid_idx) in enumerate(splits, start=1):
            fitted = _fit_branch(
                X.iloc[train_idx], y.iloc[train_idx], groups, config,
                seed + candidate_id * 101 + split_id,
            )
            probability = fitted.predict_proba(X.iloc[valid_idx])
            scores.append(roc_auc_score(y.iloc[valid_idx], probability))
        mean_auc = float(np.mean(scores))
        rows.append({
            "candidate": candidate_id,
            "view": config.view,
            "max_features": config.max_features,
            "corr_threshold": config.corr_threshold,
            "mean_inner_auc": mean_auc,
            "sd_inner_auc": float(np.std(scores, ddof=0)),
            "params": json.dumps(config.params, sort_keys=True),
        })
        if mean_auc > best_auc:
            best_auc, best = mean_auc, config
    assert best is not None
    return best, pd.DataFrame(rows).sort_values("mean_inner_auc", ascending=False)


def _branch_oof(
    X: pd.DataFrame,
    y: pd.Series,
    groups: dict[str, list[str]],
    config: BranchConfig,
    splits,
    seed: int,
) -> np.ndarray:
    probability = np.full(len(y), np.nan, dtype=float)
    for split_id, (train_idx, valid_idx) in enumerate(splits, start=1):
        fitted = _fit_branch(
            X.iloc[train_idx], y.iloc[train_idx], groups, config, seed + split_id
        )
        probability[valid_idx] = fitted.predict_proba(X.iloc[valid_idx])
    if not np.all(np.isfinite(probability)):
        raise RuntimeError("Branch OOF predictions are incomplete")
    return probability


def _demo_pipeline(C: float, seed: int) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(
            C=C, penalty="l2", class_weight="balanced", max_iter=3000,
            random_state=seed,
        )),
    ])


def _demo_matrix(X: pd.DataFrame) -> np.ndarray:
    return np.column_stack([
        pd.to_numeric(X["Age"], errors="coerce").to_numpy(float),
        pd.to_numeric(X["Gender"], errors="coerce").to_numpy(float),
    ])


def _tune_demo(X: pd.DataFrame, y: pd.Series, splits, seed: int) -> float:
    matrix = _demo_matrix(X)
    best_C, best_auc = 1.0, -np.inf
    for C in [0.01, 0.03, 0.10, 0.30, 1.0, 3.0, 10.0]:
        scores = []
        for split_id, (train_idx, valid_idx) in enumerate(splits, start=1):
            model = _demo_pipeline(C, seed + split_id).fit(
                matrix[train_idx], y.iloc[train_idx]
            )
            probability = model.predict_proba(matrix[valid_idx])[:, 1]
            scores.append(roc_auc_score(y.iloc[valid_idx], probability))
        auc = float(np.mean(scores))
        if auc > best_auc:
            best_C, best_auc = C, auc
    return best_C


def _demo_oof(
    X: pd.DataFrame, y: pd.Series, splits, C: float, seed: int
) -> np.ndarray:
    matrix = _demo_matrix(X)
    probability = np.full(len(y), np.nan, dtype=float)
    for split_id, (train_idx, valid_idx) in enumerate(splits, start=1):
        model = _demo_pipeline(C, seed + split_id).fit(
            matrix[train_idx], y.iloc[train_idx]
        )
        probability[valid_idx] = model.predict_proba(matrix[valid_idx])[:, 1]
    return probability


def _fit_meta(features: np.ndarray, y: pd.Series, seed: int) -> LogisticRegression:
    best_model = None
    best_auc = -np.inf
    n_splits = min(3, int(np.bincount(np.asarray(y, dtype=int)).min()))
    splits = _inner_splits(y, n_splits, seed)
    for C in [0.01, 0.03, 0.10, 0.30, 1.0, 3.0, 10.0]:
        scores = []
        for split_id, (train_idx, valid_idx) in enumerate(splits, start=1):
            model = LogisticRegression(
                C=C, penalty="l2", class_weight="balanced", max_iter=3000,
                random_state=seed + split_id,
            ).fit(features[train_idx], y.iloc[train_idx])
            scores.append(roc_auc_score(
                y.iloc[valid_idx], model.predict_proba(features[valid_idx])[:, 1]
            ))
        auc = float(np.mean(scores))
        if auc > best_auc:
            best_auc = auc
            best_model = LogisticRegression(
                C=C, penalty="l2", class_weight="balanced", max_iter=3000,
                random_state=seed,
            )
    assert best_model is not None
    return best_model.fit(features, y)


@dataclass
class AdaptiveMeta:
    weights: np.ndarray
    feature_names: tuple[str, ...]
    inner_auc: float

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        probability = np.asarray(features, dtype=float) @ self.weights
        return np.clip(probability, 0.0, 1.0)

    @property
    def feature_indices(self) -> tuple[int, ...]:
        return tuple(np.flatnonzero(self.weights > 0).tolist())


FUSION_NAMES = (
    "optimized_svm", "combined_xgb", "demographics",
    "thickness_xgb", "other_xgb",
)


def _fit_adaptive_meta(
    features: np.ndarray, y: pd.Series, seed: int
) -> AdaptiveMeta:
    """Select a stable non-negative blend from outer-training OOF scores.

    Candidate weights are fixed in advance.  A multi-branch blend must improve
    the primary optimized-SVM branch by at least 0.005 inner OOF AUC; otherwise
    the screening score falls back to that branch unchanged.
    """

    del seed  # Candidate construction is deterministic.
    features = np.asarray(features, dtype=float)
    names = FUSION_NAMES
    if features.shape[1] != len(names):
        raise ValueError(f"Expected {len(names)} fusion branches, got {features.shape[1]}")

    candidates: list[np.ndarray] = []
    for index in range(len(names)):
        weight = np.zeros(len(names), dtype=float)
        weight[index] = 1.0
        candidates.append(weight)
    for secondary in range(1, len(names)):
        for primary_weight in (0.50, 0.60, 0.70, 0.80, 0.90):
            weight = np.zeros(len(names), dtype=float)
            weight[0] = primary_weight
            weight[secondary] = 1.0 - primary_weight
            candidates.append(weight)
    for secondary in range(1, len(names)):
        for tertiary in range(secondary + 1, len(names)):
            for primary_weight, secondary_weight in ((0.60, 0.30), (0.70, 0.20), (0.80, 0.10)):
                weight = np.zeros(len(names), dtype=float)
                weight[0] = primary_weight
                weight[secondary] = secondary_weight
                weight[tertiary] = 1.0 - primary_weight - secondary_weight
                candidates.append(weight)

    y_array = np.asarray(y, dtype=int)
    base_weight = candidates[0]
    base_auc = float(roc_auc_score(y_array, features @ base_weight))
    best_weight = base_weight
    best_auc = base_auc
    for weight in candidates[1:]:
        auc = float(roc_auc_score(y_array, features @ weight))
        if auc > best_auc + 1e-12:
            best_auc, best_weight = auc, weight
    if best_auc < base_auc + 0.005:
        best_auc, best_weight = base_auc, base_weight
    selected_names = tuple(
        name for name, weight in zip(names, best_weight) if weight > 0
    )
    return AdaptiveMeta(best_weight.copy(), selected_names, best_auc)


LEGACY_FEATURES = [
    "VIleftcorticalthickness",
    "VIIIBrightcorticalthickness",
    "Xleft",
    "VIIIBtotalcm3",
    "VIIIBleftcorticalthickness",
]

LOCKED_XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 2,
    "learning_rate": 0.02,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 5,
    "gamma": 0.1,
    "reg_alpha": 0.5,
    "reg_lambda": 20.0,
    "scale_pos_weight": 1.0,
}
SCREEN_SVM_WEIGHT = 0.40


@dataclass
class FittedLegacySVM:
    residualizer: FoldResidualizer
    scaler: StandardScaler
    model: SVC
    direct_icv: bool = False

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        brain = self.residualizer.transform(X).to_numpy()
        parts = [brain, _demo_matrix(X)]
        if self.direct_icv:
            parts.append(pd.to_numeric(X["ICVcm3"], errors="coerce").to_numpy(float))
        matrix = np.column_stack(parts)
        return self.model.predict_proba(self.scaler.transform(matrix))[:, 1]


def _fit_legacy_svm(
    X: pd.DataFrame, y: pd.Series, seed: int, direct_icv: bool = False
) -> FittedLegacySVM:
    residualizer = FoldResidualizer(LEGACY_FEATURES, include_icv=True).fit(X)
    brain = residualizer.transform(X).to_numpy()
    parts = [brain, _demo_matrix(X)]
    if direct_icv:
        parts.append(pd.to_numeric(X["ICVcm3"], errors="coerce").to_numpy(float))
    matrix = np.column_stack(parts)
    scaler = StandardScaler().fit(matrix)
    model = SVC(
        C=1, gamma="scale", kernel="rbf", class_weight="balanced",
        probability=True, random_state=seed,
    ).fit(scaler.transform(matrix), y)
    return FittedLegacySVM(residualizer, scaler, model, direct_icv)


def _legacy_svm_oof(
    X: pd.DataFrame, y: pd.Series, splits, seed: int, direct_icv: bool = False
) -> np.ndarray:
    probability = np.full(len(y), np.nan, dtype=float)
    for split_id, (train_idx, valid_idx) in enumerate(splits, start=1):
        model = _fit_legacy_svm(
            X.iloc[train_idx], y.iloc[train_idx], seed + split_id,
            direct_icv=direct_icv,
        )
        probability[valid_idx] = model.predict_proba(X.iloc[valid_idx])
    return probability


def _legacy_svm_probability(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    seed: int,
    direct_icv: bool = False,
) -> np.ndarray:
    return _fit_legacy_svm(
        X_train, y_train, seed, direct_icv=direct_icv
    ).predict_proba(X_test)


@dataclass
class FittedLockedXGB:
    residualizer: FoldResidualizer
    model: XGBClassifier

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        brain = self.residualizer.transform(X).to_numpy()
        matrix = np.column_stack([
            brain,
            _demo_matrix(X),
            pd.to_numeric(X["ICVcm3"], errors="coerce").to_numpy(float),
        ])
        return self.model.predict_proba(matrix)[:, 1]


def _fit_locked_xgb(
    X: pd.DataFrame, y: pd.Series, seed: int
) -> FittedLockedXGB:
    """Fit the paper-locked five-feature regularized XGBoost branch."""

    residualizer = FoldResidualizer(LEGACY_FEATURES, include_icv=True).fit(X)
    brain = residualizer.transform(X).to_numpy()
    matrix = np.column_stack([
        brain,
        _demo_matrix(X),
        pd.to_numeric(X["ICVcm3"], errors="coerce").to_numpy(float),
    ])
    model = _make_xgb(LOCKED_XGB_PARAMS, seed).fit(matrix, y)
    return FittedLockedXGB(residualizer, model)


def _screen_probability(
    svm_probability: np.ndarray, xgb_probability: np.ndarray
) -> np.ndarray:
    probability = (
        SCREEN_SVM_WEIGHT * np.asarray(svm_probability, dtype=float)
        + (1.0 - SCREEN_SVM_WEIGHT) * np.asarray(xgb_probability, dtype=float)
    )
    return np.clip(probability, 0.0, 1.0)


def _conventional_xgb_probability(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    groups: dict[str, list[str]],
    seed: int,
) -> np.ndarray:
    transformer = BrainViewTransformer(
        groups["brain_raw"], include_icv=True, max_features=8,
        corr_threshold=0.85,
    ).fit(X_train, y_train)
    train = np.column_stack([transformer.transform(X_train), _demo_matrix(X_train)])
    test = np.column_stack([transformer.transform(X_test), _demo_matrix(X_test)])
    params = {
        "n_estimators": 100, "max_depth": 2, "learning_rate": 0.03,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "min_child_weight": 1, "gamma": 0.0, "reg_alpha": 0.0,
        "reg_lambda": 1.0, "scale_pos_weight": 1.0,
    }
    model = _make_xgb(params, seed).fit(train, y_train)
    return model.predict_proba(test)[:, 1]


def _conventional_xgb_oof(
    X: pd.DataFrame,
    y: pd.Series,
    groups: dict[str, list[str]],
    splits,
    seed: int,
) -> np.ndarray:
    probability = np.full(len(y), np.nan, dtype=float)
    for split_id, (train_idx, valid_idx) in enumerate(splits, start=1):
        probability[valid_idx] = _conventional_xgb_probability(
            X.iloc[train_idx], y.iloc[train_idx], X.iloc[valid_idx], groups,
            seed + split_id,
        )
    return probability


def _metrics(y: Sequence[int], probability: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    probability = np.asarray(probability, dtype=float)
    prediction = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "AUC": float(roc_auc_score(y, probability)),
        "AP": float(average_precision_score(y, probability)),
        "BACC": float(balanced_accuracy_score(y, prediction)),
        "Sensitivity": float(recall_score(y, prediction, zero_division=0)),
        "Specificity": float(tn / max(tn + fp, 1)),
        "F1": float(f1_score(y, prediction, zero_division=0)),
    }


def _bootstrap_delta(
    y: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    observed = roc_auc_score(y, candidate) - roc_auc_score(y, baseline)
    deltas = []
    while len(deltas) < n_boot:
        index = rng.randint(0, len(y), len(y))
        if len(np.unique(y[index])) < 2:
            continue
        deltas.append(
            roc_auc_score(y[index], candidate[index])
            - roc_auc_score(y[index], baseline[index])
        )
    values = np.asarray(deltas)
    lower, upper = np.percentile(values, [2.5, 97.5])
    p = min(1.0, 2 * min(
        (np.sum(values <= 0) + 1) / (len(values) + 1),
        (np.sum(values >= 0) + 1) / (len(values) + 1),
    ))
    return {
        "delta_auc": float(observed), "ci_lower": float(lower),
        "ci_upper": float(upper), "p": float(p),
    }


MODEL_NAMES = [
    "Demographics",
    "LegacyCombinedSVM",
    "OptimizedCombinedSVM",
    "OptimizedLockedXGB",
    "ConventionalCombinedXGB",
    "ThicknessXGB",
    "OtherMorphometryXGB",
    "OptimizedBrainFusion",
    "CerebADHDScreen",
]


def run_validation(
    data: pd.DataFrame,
    mode: RunMode,
    output_dir: Path,
    seed: int = SEED,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = data.copy()
    data["DX"] = pd.to_numeric(data["DX"], errors="raise").astype(int)
    data["Gender"] = pd.to_numeric(data["Gender"], errors="raise").astype(int)
    data = data.dropna(subset=["DX", "Age", "Gender", "ICVcm3"]).reset_index(drop=True)
    y = data["DX"].copy()
    groups = feature_groups(data.columns)
    outer = RepeatedStratifiedKFold(
        n_splits=mode.outer_splits,
        n_repeats=mode.outer_repeats,
        random_state=seed,
    )
    score_sum = {name: np.zeros(len(y), dtype=float) for name in MODEL_NAMES}
    score_count = np.zeros(len(y), dtype=int)
    fold_rows: list[dict[str, Any]] = []
    tuning_rows: list[dict[str, Any]] = []
    started = time.time()

    for fold_id, (train_idx, test_idx) in enumerate(outer.split(data, y), start=1):
        X_train, X_test = data.iloc[train_idx], data.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        inner = _inner_splits(y_train, mode.inner_splits, seed + fold_id * 1000)

        demo_C = _tune_demo(X_train, y_train, inner, seed + fold_id)
        thickness_config, thickness_trials = _tune_branch(
            X_train, y_train, groups,
            ["thickness_raw", "thickness_norm"], mode, inner,
            seed + fold_id * 100,
        )
        other_config, other_trials = _tune_branch(
            X_train, y_train, groups, ["other_morphometry"], mode, inner,
            seed + fold_id * 200,
        )
        for label, trials in [("thickness", thickness_trials), ("other", other_trials)]:
            trial_table = trials.copy()
            trial_table.insert(0, "outer_fold", fold_id)
            trial_table.insert(1, "branch", label)
            tuning_rows.extend(trial_table.to_dict("records"))

        thickness_oof = _branch_oof(
            X_train, y_train, groups, thickness_config, inner,
            seed + fold_id * 300,
        )
        other_oof = _branch_oof(
            X_train, y_train, groups, other_config, inner,
            seed + fold_id * 400,
        )
        local_y = y_train.reset_index(drop=True)
        brain_meta = _fit_meta(
            np.column_stack([thickness_oof, other_oof]),
            local_y, seed + fold_id * 500,
        )

        demo_model = _demo_pipeline(demo_C, seed + fold_id).fit(
            _demo_matrix(X_train), y_train
        )
        thickness_model = _fit_branch(
            X_train, y_train, groups, thickness_config, seed + fold_id * 700
        )
        other_model = _fit_branch(
            X_train, y_train, groups, other_config, seed + fold_id * 800
        )
        legacy_model = _fit_legacy_svm(X_train, y_train, seed + fold_id * 850)
        optimized_svm_model = _fit_legacy_svm(
            X_train, y_train, seed + fold_id * 875, direct_icv=True
        )
        locked_xgb_model = _fit_locked_xgb(
            X_train, y_train, seed + fold_id * 900
        )
        demo_test = demo_model.predict_proba(_demo_matrix(X_test))[:, 1]
        legacy_test = legacy_model.predict_proba(X_test)
        optimized_svm_test = optimized_svm_model.predict_proba(X_test)
        locked_xgb_test = locked_xgb_model.predict_proba(X_test)
        combined_xgb_test = _conventional_xgb_probability(
            X_train, y_train, X_test, groups, seed + fold_id
        )
        thickness_test = thickness_model.predict_proba(X_test)
        other_test = other_model.predict_proba(X_test)
        brain_test = brain_meta.predict_proba(
            np.column_stack([thickness_test, other_test])
        )[:, 1]
        screen_test = _screen_probability(optimized_svm_test, locked_xgb_test)

        probabilities = {
            "Demographics": demo_test,
            "LegacyCombinedSVM": legacy_test,
            "OptimizedCombinedSVM": optimized_svm_test,
            "OptimizedLockedXGB": locked_xgb_test,
            "ConventionalCombinedXGB": combined_xgb_test,
            "ThicknessXGB": thickness_test,
            "OtherMorphometryXGB": other_test,
            "OptimizedBrainFusion": brain_test,
            "CerebADHDScreen": screen_test,
        }
        for name, probability in probabilities.items():
            score_sum[name][test_idx] += probability
            row = {"outer_fold": fold_id, "model": name, **_metrics(y_test, probability)}
            if name == "CerebADHDScreen":
                meta_weights = {
                    "optimized_svm": SCREEN_SVM_WEIGHT,
                    "combined_xgb": 0.0,
                    "demographics": 0.0,
                    "thickness_xgb": 0.0,
                    "other_xgb": 0.0,
                }
                row.update({
                    "thickness_view": thickness_config.view,
                    "thickness_n": len(thickness_model.transformer.selected_features_),
                    "other_n": len(other_model.transformer.selected_features_),
                    "meta_features": "optimized_svm;optimized_locked_xgb",
                    "meta_inner_auc": np.nan,
                    "meta_optimized_svm_weight": meta_weights["optimized_svm"],
                    "meta_optimized_locked_xgb_weight": 1.0 - SCREEN_SVM_WEIGHT,
                    "meta_combined_xgb_weight": meta_weights["combined_xgb"],
                    "meta_demo_weight": meta_weights["demographics"],
                    "meta_thickness_weight": meta_weights["thickness_xgb"],
                    "meta_other_weight": meta_weights["other_xgb"],
                })
            fold_rows.append(row)
        score_count[test_idx] += 1
        print(
            f"Fold {fold_id:02d}: screen AUC={roc_auc_score(y_test, screen_test):.3f}; "
            f"thickness={thickness_config.view}/"
            f"{len(thickness_model.transformer.selected_features_)}; "
            f"other={len(other_model.transformer.selected_features_)}",
            flush=True,
        )

    aggregated = {name: score_sum[name] / score_count for name in MODEL_NAMES}
    comparison = pd.DataFrame([
        {"model": name, **_metrics(y, aggregated[name])} for name in MODEL_NAMES
    ]).sort_values("AUC", ascending=False)
    bootstrap = pd.DataFrame([
        {
            "comparison": f"CerebADHDScreen - {baseline}",
            **_bootstrap_delta(
                y.to_numpy(), aggregated["CerebADHDScreen"], aggregated[baseline],
                mode.bootstrap_samples, seed,
            ),
        }
        for baseline in [
            "ConventionalCombinedXGB", "LegacyCombinedSVM",
            "OptimizedCombinedSVM", "OptimizedLockedXGB", "Demographics",
        ]
    ])
    predictions = pd.DataFrame({
        "row_id": np.arange(len(y)), "DX": y,
        "Age": pd.to_numeric(data["Age"], errors="coerce"),
        "Gender": data["Gender"],
        **{name: aggregated[name] for name in MODEL_NAMES},
    })
    pd.DataFrame(fold_rows).to_csv(output_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(tuning_rows).to_csv(output_dir / "inner_tuning_trials.csv", index=False)
    predictions.to_csv(output_dir / "oof_predictions.csv", index=False)
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    bootstrap.to_csv(output_dir / "bootstrap_comparisons.csv", index=False)
    config = {
        "mode": mode.__dict__, "seed": seed, "n": len(data),
        "adhd": int((y == 1).sum()), "control": int((y == 0).sum()),
        "feature_group_counts": {key: len(value) for key, value in groups.items()},
        "optimized_svm_direct_icv": True,
        "feature_selection_scope": "global_full_sample_paper_locked",
        "locked_features": LEGACY_FEATURES,
        "locked_xgb_params": LOCKED_XGB_PARAMS,
        "screen_weights": {
            "optimized_svm": SCREEN_SVM_WEIGHT,
            "optimized_locked_xgb": 1.0 - SCREEN_SVM_WEIGHT,
        },
        "elapsed_seconds": time.time() - started,
        "stretch_target_auc": 0.750,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    screen_auc = float(
        comparison.loc[comparison.model == "CerebADHDScreen", "AUC"].iloc[0]
    )
    lines = [
        "Cereb-ADHDScreen internal-validation summary", "",
        comparison.to_string(index=False), "", bootstrap.to_string(index=False), "",
        f"Stretch target AUC >= 0.750: {'MET' if screen_auc >= 0.750 else 'NOT MET'}",
        "This is internal validation, not external clinical validation.",
    ]
    (output_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    return {
        "comparison": comparison, "bootstrap": bootstrap,
        "predictions": predictions, "config": config,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("cerebellum.csv"))
    parser.add_argument("--mode", choices=["fast", "full"], default="fast")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode = FAST_MODE if args.mode == "fast" else FULL_MODE
    output = args.output or Path(f"cereb_adhd_screen_outputs_{mode.name}")
    data = pd.read_csv(args.data)
    print(
        f"Cereb-ADHDScreen | mode={mode.name} | N={len(data)} | output={output}",
        flush=True,
    )
    result = run_validation(data, mode, output, seed=args.seed)
    print(result["comparison"].to_string(index=False), flush=True)
    print(f"Finished. Outputs: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
