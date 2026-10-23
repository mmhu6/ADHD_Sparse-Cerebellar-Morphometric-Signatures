import unittest

import numpy as np
import pandas as pd

from tests.synthetic_data import make_synthetic_frame

from cereb_adhd_optimization import (
    CANDIDATE_FEATURES,
    DIRECT_COVARIATES,
    FORBIDDEN_MODEL_INPUTS,
    INITIAL_EIGHT_FEATURES,
    PAPER_LOCKED_FEATURES,
    STABILITY_FEATURES,
    _fit_candidate_svm,
    _fit_candidate_xgb,
    fixed_probability_fusion,
    model_input_columns,
)
from cereb_adhd_screen import feature_groups


class CandidateSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = make_synthetic_frame()

    def test_fixed_pool_is_valid_raw_brain_data(self):
        raw_brain = set(feature_groups(self.data.columns)["brain_raw"])
        self.assertEqual(len(CANDIDATE_FEATURES), 12)
        self.assertEqual(len(set(CANDIDATE_FEATURES)), 12)
        self.assertTrue(set(CANDIDATE_FEATURES).issubset(raw_brain))
        self.assertTrue(set(PAPER_LOCKED_FEATURES).issubset(CANDIDATE_FEATURES))
        self.assertEqual(
            set(CANDIDATE_FEATURES),
            set(INITIAL_EIGHT_FEATURES) | set(STABILITY_FEATURES),
        )

    def test_model_schema_excludes_labels_symptoms_medication_and_id(self):
        inputs = set(model_input_columns(CANDIDATE_FEATURES))
        self.assertTrue(set(DIRECT_COVARIATES).issubset(inputs))
        self.assertTrue(set(CANDIDATE_FEATURES).issubset(inputs))
        self.assertTrue(inputs.isdisjoint(FORBIDDEN_MODEL_INPUTS))


class CandidateModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = make_synthetic_frame()
        cls.data = pd.concat(
            [data[data.DX == 0].head(70), data[data.DX == 1].head(70)]
        ).sample(frac=1, random_state=19).reset_index(drop=True)

    def test_xgb_and_svm_emit_finite_probabilities(self):
        train, test = self.data.iloc[:100], self.data.iloc[100:]
        y = train["DX"].astype(int)
        xgb = _fit_candidate_xgb(
            train,
            y,
            PAPER_LOCKED_FEATURES,
            params={
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
            seed=7,
        )
        svm = _fit_candidate_svm(train, y, seed=7)
        for probability in (xgb.predict_proba(test), svm.predict_proba(test)):
            self.assertEqual(probability.shape, (len(test),))
            self.assertTrue(np.all(np.isfinite(probability)))
            self.assertTrue(np.all((probability >= 0) & (probability <= 1)))
        self.assertTrue(set(xgb.input_columns).isdisjoint(FORBIDDEN_MODEL_INPUTS))

    def test_fixed_fusion_is_exact_bounded_and_validates_weight(self):
        svm = np.array([0.1, 0.9, 1.2])
        xgb = np.array([0.8, 0.2, -0.2])
        observed = fixed_probability_fusion(svm, xgb, svm_weight=0.3)
        expected = np.clip(0.3 * svm + 0.7 * xgb, 0.0, 1.0)
        np.testing.assert_allclose(observed, expected)
        self.assertTrue(np.all((observed >= 0) & (observed <= 1)))
        with self.assertRaises(ValueError):
            fixed_probability_fusion(svm, xgb, svm_weight=1.1)


if __name__ == "__main__":
    unittest.main()
