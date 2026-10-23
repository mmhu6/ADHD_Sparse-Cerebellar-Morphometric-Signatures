import unittest

import numpy as np
import pandas as pd

from tests.synthetic_data import make_synthetic_frame

from cereb_adhd_screen import LEGACY_FEATURES
from cerebnorm_adhd_screen import (
    FORBIDDEN_MODEL_INPUTS,
    ControlNormativeTransformer,
    _fit_norm_model,
)


class ControlNormativeTransformerTests(unittest.TestCase):
    @classmethod
    def make_frame(cls, seed=31):
        rng = np.random.RandomState(seed)
        n = 90
        age = rng.uniform(7, 15, n)
        gender = rng.binomial(1, 0.6, n)
        icv = rng.normal(1450, 110, n)
        dx = np.r_[np.zeros(60, dtype=int), np.ones(30, dtype=int)]
        frame = pd.DataFrame(
            {
                "Age": age,
                "Gender": gender,
                "ICVcm3": icv,
                "DX": dx,
                "ADHDIndex": rng.normal(50, 10, n),
                "Sub": [f"N{i}" for i in range(n)],
            }
        )
        for feature_id, feature in enumerate(LEGACY_FEATURES):
            frame[feature] = (
                2.0
                + feature_id
                + 0.08 * age
                - 0.15 * gender
                + 0.0003 * icv
                + rng.normal(0, 0.12 + feature_id * 0.01, n)
            )
        return frame

    def test_only_training_controls_define_normative_reference(self):
        frame = self.make_frame()
        y = frame["DX"].astype(int)
        first = ControlNormativeTransformer(LEGACY_FEATURES).fit(frame, y)
        changed = frame.copy()
        changed.loc[y == 1, list(LEGACY_FEATURES)] += 1000
        second = ControlNormativeTransformer(LEGACY_FEATURES).fit(changed, y)
        np.testing.assert_allclose(first.beta_, second.beta_)
        np.testing.assert_allclose(first.residual_sd_, second.residual_sd_)
        self.assertEqual(first.n_controls_, 60)

    def test_exact_z_summaries_and_transform_ignores_labels_and_symptoms(self):
        frame = self.make_frame()
        train, test = frame.iloc[:75], frame.iloc[75:].copy()
        transformer = ControlNormativeTransformer(LEGACY_FEATURES).fit(
            train, train["DX"].astype(int)
        )
        observed = transformer.transform(test)
        brain = np.column_stack(
            [
                pd.to_numeric(test[name], errors="coerce").to_numpy(float)
                for name in LEGACY_FEATURES
            ]
        )
        expected = transformer._design(test) @ transformer.beta_
        manual = np.clip(
            (brain - expected) / transformer.residual_sd_,
            -transformer.z_clip,
            transformer.z_clip,
        )
        np.testing.assert_allclose(
            observed[[f"z__{name}" for name in LEGACY_FEATURES]].to_numpy(),
            manual,
        )
        np.testing.assert_allclose(observed["norm_mean_abs_z"], np.mean(np.abs(manual), axis=1))
        np.testing.assert_array_equal(observed["norm_extreme_count"], np.sum(np.abs(manual) > 2, axis=1))

        changed = test.copy()
        changed["DX"] = 1 - changed["DX"]
        changed["ADHDIndex"] = 9999
        changed["Sub"] = "changed"
        pd.testing.assert_frame_equal(observed, transformer.transform(changed))


class NormativeModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = make_synthetic_frame()
        cls.data = pd.concat(
            [data[data.DX == 0].head(70), data[data.DX == 1].head(70)]
        ).sample(frac=1, random_state=37).reset_index(drop=True)

    def test_three_model_families_emit_finite_probabilities_without_forbidden_inputs(self):
        train, test = self.data.iloc[:100], self.data.iloc[100:]
        y = train["DX"].astype(int)
        candidates = (
            ("elastic_net", {"C": 0.1, "l1_ratio": 0.5}),
            ("rbf_svm", {"C": 1.0, "gamma": "scale"}),
            (
                "xgboost",
                {
                    "n_estimators": 20,
                    "max_depth": 1,
                    "learning_rate": 0.05,
                    "min_child_weight": 5,
                    "gamma": 0.0,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "reg_alpha": 0.5,
                    "reg_lambda": 10.0,
                    "scale_pos_weight": 1.0,
                },
            ),
        )
        for family, params in candidates:
            model = _fit_norm_model(
                train, y, "raw_plus_z", family, params, seed=13
            )
            probability = model.predict_proba(test)
            self.assertEqual(probability.shape, (len(test),))
            self.assertTrue(np.isfinite(probability).all())
            self.assertTrue(np.all((probability >= 0) & (probability <= 1)))
            self.assertTrue(set(model.input_columns).isdisjoint(FORBIDDEN_MODEL_INPUTS))


if __name__ == "__main__":
    unittest.main()
