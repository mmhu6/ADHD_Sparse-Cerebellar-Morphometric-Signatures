import argparse
import json
import os
import warnings
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from scipy.stats import chi2_contingency, spearmanr, ttest_ind
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')
DATA_PATH = 'cerebellum.csv'
OUT = 'adhd_cerebellum_paper_outputs_v5'
SEED = 42
PUBLICATION_MODE = True
N_JOBS = -1
OUTER_FOLDS = 5
MODEL_SELECT_REPEATS = 3 if PUBLICATION_MODE else 1
INNER_FOLDS = 3
PRESELECT_K = 80
K_GRID = [6, 8, 10, 12]
CORR_THRESHOLD = 0.85
ABLATION_TOLERANCE = 0.002
MIN_FINAL_FEATURES = 4
ABLATION_FOLDS = 5
FINAL_REPEATS = 3 if PUBLICATION_MODE else 1
N_BOOT = 5000 if PUBLICATION_MODE else 1500
N_PERM = 1000 if PUBLICATION_MODE else 100
PERM_N_JOBS = -1
RUN_SHAP = PUBLICATION_MODE
SHAP_N = 250 if PUBLICATION_MODE else 100
RUN_IMBALANCE = True
RUN_OUTLIER = True
RUN_FEATURE_COUNT = True
RUN_FAMILY_ABLATION = True
META = {'Sub', 'Gender', 'Age', 'DX', 'ADHDIndex', 'Inattentive', 'HyperImpulsive', 'MedStatus', 'ICVcm3'}

def pstr(p):
    return '<0.001' if p < 0.001 else f'{p:.4f}'

def specificity(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return tn / (tn + fp)

def metric_dict(y, score, pred):
    return {'AUC': roc_auc_score(y, score), 'Accuracy': accuracy_score(y, pred), 'BACC': balanced_accuracy_score(y, pred), 'F1': f1_score(y, pred, zero_division=0), 'Sensitivity': recall_score(y, pred, zero_division=0), 'Specificity': specificity(y, pred), 'Precision': precision_score(y, pred, zero_division=0), 'AP': average_precision_score(y, score)}

def model_score(estimator, X):
    return np.asarray(estimator.predict_proba(X))[:, 1]

def bh_fdr(p_values):
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty_like(q)
    out[order] = q
    return out

def clean_params(params):
    return {k.replace('model__', '').replace('selector__', 'selector_'): v for k, v in params.items()}

def model_params(params):
    return {k.replace('model__', ''): v for k, v in params.items() if k.startswith('model__')}

def save_table(df, title, filename, note=None, max_rows=30):
    d = df.head(max_rows).copy()
    h = max(3.2, 1.7 + 0.42 * (len(d) + 1) + (0.55 if note else 0))
    w = max(10, min(22, 2 + 1.65 * len(d.columns)))
    fig, ax = plt.subplots(figsize=(w, h))
    ax.axis('off')
    ax.text(0, 1.03, title, transform=ax.transAxes, fontsize=14, fontweight='bold', va='bottom')
    tab = ax.table(cellText=d.astype(str).values, colLabels=d.columns, cellLoc='left', colLoc='left', bbox=[0, 0.13 if note else 0.02, 1, 0.85 if note else 0.94])
    tab.auto_set_font_size(False)
    tab.set_fontsize(8.8)
    tab.scale(1, 1.18)
    for (r, _), cell in tab.get_celld().items():
        cell.set_linewidth(0.4)
        if r == 0:
            cell.set_text_props(weight='bold')
    if note:
        ax.text(0, 0.02, note, transform=ax.transAxes, fontsize=8.4, va='bottom', wrap=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, filename), dpi=300, bbox_inches='tight')
    plt.close(fig)

def region(feature):
    if feature.startswith('III') and '_A' in feature:
        return 'III_A'
    for r in ['Cerebellum', 'CrusII', 'CrusI', 'VIIIA', 'VIIIB', 'VIIB', 'III', 'IV', 'VI', 'IX', 'V', 'X']:
        if feature.startswith(r):
            return r
    return 'Other'

def family(feature):
    x = feature.lower()

    if 'greymatterasymmetry' in x:
        return 'Gray-matter asymmetry'
    if 'corticalthicknessasymmetry' in x:
        return 'Thickness asymmetry'
    if 'corticalthickness' in x:
        return 'Cortical thickness'
    if 'greymatter' in x and 'cm3' in x:
        return 'Gray-matter volume (cm3)'
    if 'greymatter' in x:
        return 'Gray-matter volume (%ICV)'
    if 'asymmetry' in x:
        return 'Volume asymmetry'
    if 'cm3' in x:
        return 'Absolute volume (cm3)'
    if 'total' in x or 'right' in x or 'left' in x:
        return 'Relative volume (%ICV)'

    return 'Other'  

NON_PREDICTIVE_COLUMNS = {
    'Sub',
    'Gender',
    'Age',
    'DX',
    'ADHDIndex',
    'Inattentive',
    'HyperImpulsive',
    'MedStatus',
    'ICVcm3'
}

def candidate_features(columns):
    return [
        c for c in columns
        if c not in NON_PREDICTIVE_COLUMNS
        and 'norm' not in c.lower()
    ]

class DemoTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame({'DEMO__Age': pd.to_numeric(X['Age']), 'DEMO__Gender': pd.to_numeric(X['Gender'])}, index=X.index)

class Residualizer(BaseEstimator, TransformerMixin):
    def __init__(self, features, add_demo=False):
        self.features = features
        self.add_demo = add_demo

    def _design(self, age, gender, icv):
        age_z = (age - self.age_mean_) / self.age_sd_
        icv_z = (icv - self.icv_mean_) / self.icv_sd_
        return np.c_[np.ones(len(age_z)), age_z, age_z ** 2, gender, icv_z]

    def fit(self, X, y=None):
        age = pd.to_numeric(X['Age']).to_numpy(float)
        gender = pd.to_numeric(X['Gender']).to_numpy(float)
        icv = pd.to_numeric(X['ICVcm3']).to_numpy(float)

        self.age_mean_ = age.mean()
        self.age_sd_ = age.std()
        self.icv_mean_ = icv.mean()
        self.icv_sd_ = icv.std()

        self.medians_ = {}
        brain = []
        for c in self.features:
            v = pd.to_numeric(X[c], errors='coerce')
            med = float(v.median())
            self.medians_[c] = med
            brain.append(v.fillna(med).to_numpy(float))

        brain = np.column_stack(brain)
        self.beta_ = np.linalg.lstsq(self._design(age, gender, icv), brain, rcond=None)[0]
        return self

    def transform(self, X):
        age = pd.to_numeric(X['Age']).to_numpy(float)
        gender = pd.to_numeric(X['Gender']).to_numpy(float)
        icv = pd.to_numeric(X['ICVcm3']).to_numpy(float)

        brain = []
        for c in self.features:
            v = pd.to_numeric(X[c], errors='coerce').fillna(self.medians_[c])
            brain.append(v.to_numpy(float))

        resid = np.column_stack(brain) - self._design(age, gender, icv) @ self.beta_
        out = pd.DataFrame(resid, columns=self.features, index=X.index)

        if self.add_demo:
            out['DEMO__Age'] = age
            out['DEMO__Gender'] = gender

        return out

class SparseSelector(BaseEstimator, TransformerMixin):

    def __init__(self, k=10, preselect=50, corr=0.85):
        self.k = k
        self.preselect = preselect
        self.corr = corr

    def fit(self, X, y):
        X = pd.DataFrame(X)
        self.demo_ = [c for c in X if str(c).startswith('DEMO__')]
        brain = [c for c in X if c not in self.demo_]
        var = X[brain].var()
        brain = var[var > 1e-08].index.tolist()
        f, _ = f_classif(X[brain], y)
        f = np.nan_to_num(f, nan=-np.inf)
        ranked = [brain[i] for i in np.argsort(f)[::-1]][:min(self.preselect, len(brain))]
        corr = X[ranked].corr().abs()
        keep = []
        for c in ranked:
            if len(keep) >= self.k:
                break
            if not keep or np.all(corr.loc[c, keep].fillna(0).values < self.corr):
                keep.append(c)
        self.brain_ = keep
        self.cols_ = keep + self.demo_
        return self

    def transform(self, X):
        return pd.DataFrame(X)[self.cols_]

class IQRClipper(BaseEstimator, TransformerMixin):

    def __init__(self, multiplier=3.0):
        self.multiplier = multiplier

    def fit(self, X, y=None):
        A = np.asarray(X, float)
        q1 = np.percentile(A, 25, axis=0)
        q3 = np.percentile(A, 75, axis=0)
        iqr = q3 - q1
        self.lo_ = q1 - self.multiplier * iqr
        self.hi_ = q3 + self.multiplier * iqr
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, float), self.lo_, self.hi_)

class CatBoostCompat(ClassifierMixin, BaseEstimator):

    def __init__(self, iterations=200, depth=5, learning_rate=0.05, l2_leaf_reg=5, balanced=True, random_state=SEED):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.balanced = balanced
        self.random_state = random_state

    def fit(self, X, y):
        self.model_ = CatBoostClassifier(iterations=self.iterations, depth=self.depth, learning_rate=self.learning_rate, l2_leaf_reg=self.l2_leaf_reg, verbose=0, random_seed=self.random_state, auto_class_weights='Balanced' if self.balanced else None, allow_writing_files=False, thread_count=1)
        self.model_.fit(X, y)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.asarray(self.model_.predict(X)).astype(int).ravel()

    def predict_proba(self, X):
        return np.asarray(self.model_.predict_proba(X))

def available_models():
    names = ['SVM', 'RandomForest', 'ElasticNet', 'XGBoost', 'CatBoost']
    return names

def make_model(name, balanced=True):
    if name == 'SVM':
        return SVC(kernel='rbf', class_weight='balanced' if balanced else None, probability=True, random_state=SEED)
    if name == 'RandomForest':
        return RandomForestClassifier(n_estimators=250, class_weight='balanced' if balanced else None, random_state=SEED, n_jobs=1)
    if name == 'ElasticNet':
        return LogisticRegression(penalty='elasticnet', solver='saga', class_weight='balanced' if balanced else None, max_iter=5000, random_state=SEED)
    if name == 'XGBoost':
        return XGBClassifier(objective='binary:logistic', eval_metric='logloss', n_jobs=1, random_state=SEED, verbosity=0)
    if name == 'CatBoost':
        return CatBoostCompat(balanced=balanced, random_state=SEED)
    raise ValueError(f'Unknown model: {name}')

def search_grid(name, include_k=True):
    if name == 'SVM':
        g = {'model__C': [0.1, 1, 10], 'model__gamma': ['scale', 0.01]}
    elif name == 'RandomForest':
        g = {'model__max_depth': [3, 5, None], 'model__min_samples_leaf': [2, 5], 'model__max_features': ['sqrt']}
    elif name == 'ElasticNet':
        g = {'model__C': [0.03, 0.1, 1, 10], 'model__l1_ratio': [0.25, 0.5, 0.75]}
    elif name == 'XGBoost':
        g = {'model__n_estimators': [100, 200], 'model__max_depth': [2, 3], 'model__learning_rate': [0.03, 0.1], 'model__subsample': [0.8], 'model__colsample_bytree': [0.8]}
    elif name == 'CatBoost':
        g = {'model__iterations': [100, 200], 'model__depth': [3, 5], 'model__learning_rate': [0.03, 0.1], 'model__l2_leaf_reg': [5]}
    else:
        raise ValueError(name)
    if include_k:
        g = {'selector__k': K_GRID, **g}
    return g

def selection_pipeline(scope, name, features, balanced=True):
    if scope == 'demo':
        return Pipeline([('demo', DemoTransformer()), ('scale', StandardScaler()), ('model', make_model(name, balanced))])
    return Pipeline([('resid', Residualizer(features, add_demo=scope == 'combined')), ('selector', SparseSelector(10, PRESELECT_K, CORR_THRESHOLD)), ('scale', StandardScaler()), ('model', make_model(name, balanced))])

def locked_pipeline(scope, name, features, params, balanced=True, clip=None, sampler=None):
    if scope == 'demo':
        steps = [('demo', DemoTransformer()), ('scale', StandardScaler())]
    else:
        steps = [('resid', Residualizer(features, add_demo=scope == 'combined'))]
        if clip is not None:
            steps.append(('clip', IQRClipper(clip)))
        steps.append(('scale', StandardScaler()))
    if sampler is not None:
        steps.append(('sampler', sampler))
    m = make_model(name, balanced)
    m.set_params(**params)
    steps.append(('model', m))
    if sampler is None:
        return Pipeline(steps)
    from imblearn.pipeline import Pipeline as ImbPipeline
    return ImbPipeline(steps)

def baseline_table(D):
    control = D[D.DX == 0]
    adhd = D[D.DX == 1]
    rows = [['Sample Size (N)', len(control), len(adhd), '-']]
    a0, a1 = (control.Age, adhd.Age)
    rows.append(['Age (Mean ± SD)', f'{a0.mean():.2f} ± {a0.std():.2f}', f'{a1.mean():.2f} ± {a1.std():.2f}', pstr(ttest_ind(a0, a1, equal_var=False).pvalue)])
    pg = chi2_contingency(pd.crosstab(D.DX, D.Gender)).pvalue
    m0, f0 = ((control.Gender == 1).sum(), (control.Gender == 0).sum())
    m1, f1 = ((adhd.Gender == 1).sum(), (adhd.Gender == 0).sum())
    rows.append(['Gender (Male / Female)', f'{m0} ({100 * m0 / (m0 + f0):.1f}%) / {f0} ({100 * f0 / (m0 + f0):.1f}%)', f'{m1} ({100 * m1 / (m1 + f1):.1f}%) / {f1} ({100 * f1 / (m1 + f1):.1f}%)', pstr(pg)])
    for s in ['ADHDIndex', 'HyperImpulsive', 'Inattentive']:
        x0 = pd.to_numeric(control[s], errors='coerce').dropna()
        x1 = pd.to_numeric(adhd[s], errors='coerce').dropna()
        rows.append([f'{s} (Mean ± SD)', f'{x0.mean():.2f} ± {x0.std():.2f}', f'{x1.mean():.2f} ± {x1.std():.2f}', pstr(ttest_ind(x0, x1, equal_var=False).pvalue)])
    df = pd.DataFrame(rows, columns=['Characteristic', f'Control (N={len(control)})', f'ADHD (N={len(adhd)})', 'p-value'])
    df.to_csv(os.path.join(OUT, 'table1_baseline_characteristics.csv'), index=False)
    save_table(df, 'Table 1. Demographic and Clinical Characteristics', 'table1_baseline_characteristics.jpg', 'Clinical symptom scales are descriptive only and never enter a prediction model.')
    return df

def architecture_table(D, features):
    A = D[features].apply(pd.to_numeric, errors='coerce')
    A = A.fillna(A.median())
    Z = StandardScaler().fit_transform(A)
    C = pd.DataFrame(Z, columns=features).corr().abs()
    upper = C.where(np.triu(np.ones(C.shape), 1).astype(bool))
    pca = PCA().fit(Z)
    cum = np.cumsum(pca.explained_variance_ratio_)
    eig = pca.explained_variance_
    rows = [['Participants', len(D), 'Current cohort'], ['ADHD / Control', f'{(D.DX == 1).sum()} / {(D.DX == 0).sum()}', 'Moderate class imbalance'], ['Raw columns', len(D.columns), 'Before exclusions'], ['Candidate brain candidates', len(features), 'Minimal a priori exclusions; redundancy handled during training-fold feature selection'], ['Pairs |r| > 0.85', int((upper > 0.85).sum().sum()), 'Residual redundancy'], ['Pairs |r| > 0.95', int((upper > 0.95).sum().sum()), 'Very high redundancy'], ['PCs for 80% / 90% / 95% variance', f'{np.searchsorted(cum, 0.8) + 1} / {np.searchsorted(cum, 0.9) + 1} / {np.searchsorted(cum, 0.95) + 1}', 'Effective dimension below nominal p'], ['Participation-ratio effective dimension', f'{eig.sum() ** 2 / (eig @ eig):.1f}', 'Approximate independent variance directions']]
    df = pd.DataFrame(rows, columns=['Dataset property', 'Value', 'Interpretation'])
    df.to_csv(os.path.join(OUT, 'table2_dataset_architecture.csv'), index=False)
    save_table(df, 'Table 2. Dataset Architecture and Redundancy', 'table2_dataset_architecture.jpg')
    plt.figure(figsize=(7, 5))
    plt.plot(np.arange(1, len(cum) + 1), cum)
    for y in [0.8, 0.9, 0.95]:
        plt.axhline(y, linestyle='--', linewidth=1)
    plt.xlabel('Principal components')
    plt.ylabel('Cumulative explained variance')
    plt.title('Candidate feature-space redundancy')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'figure_pca_cumulative_variance.jpg'), dpi=300)
    plt.close()
    return df

def nested_model_selection(X, y, features, name):
    outer = RepeatedStratifiedKFold(n_splits=OUTER_FOLDS, n_repeats=MODEL_SELECT_REPEATS, random_state=SEED)
    score_sum = np.zeros(len(y))
    count = np.zeros(len(y))
    fold_rows, params, selected = ([], [], [])
    print(f'\n{name}')
    for fold, (tr, te) in enumerate(outer.split(X, y), 1):
        inner = StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=SEED + fold)
        gs = GridSearchCV(selection_pipeline('brain', name, features), search_grid(name, include_k=True), scoring='roc_auc', cv=inner, n_jobs=N_JOBS, refit=True)
        gs.fit(X.iloc[tr], y.iloc[tr])
        est = gs.best_estimator_
        s = model_score(est, X.iloc[te])
        pred = est.predict(X.iloc[te])
        m = metric_dict(y.iloc[te], s, pred)
        chosen = list(est.named_steps['selector'].brain_)
        fold_rows.append({**m, 'Fold': fold, 'InnerBestAUC': gs.best_score_, 'NFeatures': len(chosen)})
        params.append(gs.best_params_)
        selected.append(chosen)
        score_sum[te] += s
        count[te] += 1
        print(f"  Fold {fold:02d}: AUC {m['AUC']:.3f}, F1 {m['F1']:.3f}, features {len(chosen)}")
    fd = pd.DataFrame(fold_rows)
    oof = score_sum / np.maximum(count, 1)
    summary = {'AUC': fd.AUC.mean(), 'AUC_SD': fd.AUC.std(ddof=0), 'F1': fd.F1.mean(), 'Sensitivity': fd.Sensitivity.mean(), 'Specificity': fd.Specificity.mean(), 'OOF_AUC': roc_auc_score(y, oof)}
    print(f"  Mean AUC {summary['AUC']:.3f} ± {summary['AUC_SD']:.3f}; OOF AUC {summary['OOF_AUC']:.3f}")
    common = Counter((json.dumps(p, sort_keys=True) for p in params)).most_common(2)
    print('  Most common parameter sets:')
    for p, n in common:
        print(f'    {n}/{len(params)}: {json.loads(p)}')
    return {'name': name, 'fold': fd, 'score': oof, 'summary': summary, 'params': params, 'selected': selected}

def choose_best_model(X, y, features):
    results = {}
    for name in available_models():
        results[name] = nested_model_selection(X, y, features, name)
    rows = []
    for name, r in results.items():
        s = r['summary']
        rows.append([name, s['AUC'], s['AUC_SD'], s['OOF_AUC'], s['F1'], s['Sensitivity'], s['Specificity']])
    table = pd.DataFrame(rows, columns=['Classifier', 'Mean AUC', 'AUC SD', 'OOF AUC', 'F1', 'Sensitivity', 'Specificity']).sort_values('Mean AUC', ascending=False)
    table.to_csv(os.path.join(OUT, 'table3_model_selection.csv'), index=False)
    show = table.copy()
    for c in show.columns[1:]:
        show[c] = show[c].map(lambda v: f'{v:.3f}')
    save_table(show, 'Table 3. Fair Brain-Only Classifier Comparison', 'table3_model_selection.jpg', 'Development stage: same candidate features, selector and outer folds for all classifiers.')
    best = table.iloc[0]['Classifier']
    print()
    print(f'Selected model: {best}')
    print(f"Mean nested-CV AUC: {results[best]['summary']['AUC']:.4f}")
    print()
    cv_df = pd.DataFrame({k: pd.Series(v['fold'].AUC.values) for k, v in results.items()})
    cv_df.to_csv(os.path.join(OUT, 'model_selection_fold_auc.csv'), index=False)
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=cv_df)
    sns.stripplot(data=cv_df, jitter=0.15, alpha=0.55)
    plt.ylabel('Outer-fold AUC')
    plt.title('Classifier performance across model-selection folds')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'boxplot_model_selection_auc.jpg'), dpi=300)
    plt.close()
    return (best, results, table)

def tune_initial_model(X, y, features, best_name):
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    gs = GridSearchCV(selection_pipeline('brain', best_name, features), search_grid(best_name, include_k=True), scoring='roc_auc', cv=cv, n_jobs=N_JOBS, refit=True)
    gs.fit(X, y)
    initial_features = list(gs.best_estimator_.named_steps['selector'].brain_)
    print('\nDevelopment fit')
    for k, v in gs.best_params_.items():
        print(f'  {k}: {v}')
    print(f'  CV AUC: {gs.best_score_:.4f}')
    print(f'  Features ({len(initial_features)}):')
    for i, f in enumerate(initial_features, 1):
        print(f'    {i}. {f}')
    df = pd.DataFrame([[best_name, gs.best_score_, len(initial_features), json.dumps(clean_params(gs.best_params_))]], columns=['Best classifier', 'Development CV AUC', 'Initial N features', 'Best hyperparameters'])
    df.to_csv(os.path.join(OUT, 'table4_initial_best_model.csv'), index=False)
    save_table(df, 'Table 4. Selected Classifier and Initial Hyperparameters', 'table4_initial_best_model.jpg')
    return (gs.best_params_, initial_features)

def locked_cv_auc(X, y, features, name, params, folds=5, seed=SEED, balanced=True, clip=None, sampler=None):
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in cv.split(X, y):
        est = locked_pipeline('brain', name, features, params, balanced, clip, sampler)
        est.fit(X.iloc[tr], y.iloc[tr])
        aucs.append(roc_auc_score(y.iloc[te], model_score(est, X.iloc[te])))
    return (float(np.mean(aucs)), float(np.std(aucs, ddof=0)))

def backward_ablation(X, y, initial_features, name, params):
    current = list(initial_features)
    current_auc, _ = locked_cv_auc(X, y, current, name, params, folds=ABLATION_FOLDS, seed=SEED + 100)
    accepted = [[0, 'Baseline', len(current), current_auc, np.nan, 'Start']]
    all_trials = []
    step = 0
    print('\nBackward ablation')
    print(f'  Start: {len(current)} features, AUC {current_auc:.4f}')
    while len(current) > MIN_FINAL_FEATURES:
        trials = []
        for f in current:
            subset = [x for x in current if x != f]
            auc, sd = locked_cv_auc(X, y, subset, name, params, folds=ABLATION_FOLDS, seed=SEED + 100)
            trials.append((f, auc, sd))
            all_trials.append([step + 1, f, len(subset), auc, auc - current_auc, sd])
        drop, candidate_auc, _ = max(trials, key=lambda x: x[1])
        delta = candidate_auc - current_auc
        if delta >= -ABLATION_TOLERANCE:
            step += 1
            current.remove(drop)
            decision = 'Remove: improved' if delta > 0 else 'Remove: non-inferior'
            accepted.append([step, drop, len(current), candidate_auc, delta, decision])
            print(f'  Step {step}: remove {drop}, AUC {candidate_auc:.4f} ({delta:+.4f})')
            current_auc = candidate_auc
        else:
            print(f'  Stop: best deletion changes AUC by {delta:+.4f}')
            break
    accepted_df = pd.DataFrame(accepted, columns=['Step', 'Removed feature', 'N features', 'CV AUC', 'Delta AUC', 'Decision'])
    trials_df = pd.DataFrame(all_trials, columns=['Step', 'Tested removal', 'N features', 'CV AUC', 'Delta AUC', 'AUC SD'])
    accepted_df.to_csv(os.path.join(OUT, 'table5_backward_ablation.csv'), index=False)
    trials_df.to_csv(os.path.join(OUT, 'supplement_ablation_all_trials.csv'), index=False)
    save_table(accepted_df, 'Table 5. Backward Feature Ablation', 'table5_backward_ablation.jpg', 'Exploratory development result only. The highest ablation AUC is not the final reported AUC.')
    pd.DataFrame({'Final_Locked_Feature': current}).to_csv(os.path.join(OUT, 'final_optimized_features.csv'), index=False)
    return (current, accepted_df)

def tune_locked_model(X, y, features, name):
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED + 200)
    base = locked_pipeline('brain', name, features, {}, balanced=True)
    gs = GridSearchCV(base, search_grid(name, include_k=False), scoring='roc_auc', cv=cv, n_jobs=N_JOBS, refit=True)
    gs.fit(X, y)
    final_params = model_params(gs.best_params_)
    print('\nLocked model')
    print(f'  Classifier: {name}')
    print(f'  Features ({len(features)}): {features}')
    print('  Hyperparameters:')
    for k, v in final_params.items():
        print(f'  {k}: {v}')
    print(f'  Development CV AUC: {gs.best_score_:.4f}')
    df = pd.DataFrame([[name, len(features), gs.best_score_, json.dumps(final_params), '; '.join(features)]], columns=['Classifier', 'N locked features', 'Development CV AUC', 'Final hyperparameters', 'Locked features'])
    df.to_csv(os.path.join(OUT, 'table6_locked_model.csv'), index=False)
    save_table(df, 'Table 6. Locked Final Brain Model', 'table6_locked_model.jpg')
    return (final_params, gs.best_estimator_)

def locked_repeated_eval(X, y, features, scope, name, params, label):
    cv = RepeatedStratifiedKFold(n_splits=OUTER_FOLDS, n_repeats=FINAL_REPEATS, random_state=SEED + 500)
    score_sum = np.zeros(len(y))
    pred_sum = np.zeros(len(y))
    count = np.zeros(len(y))
    rows = []
    for fold, (tr, te) in enumerate(cv.split(X, y), 1):
        est = locked_pipeline(scope, name, features, params, balanced=True)
        est.fit(X.iloc[tr], y.iloc[tr])
        score = model_score(est, X.iloc[te])
        pred = est.predict(X.iloc[te])
        rows.append({**metric_dict(y.iloc[te], score, pred), 'Fold': fold})
        score_sum[te] += score
        pred_sum[te] += pred
        count[te] += 1
    fd = pd.DataFrame(rows)
    oof = score_sum / np.maximum(count, 1)
    pred = (pred_sum / np.maximum(count, 1) >= 0.5).astype(int)
    summary = {c: fd[c].mean() for c in ['AUC', 'Accuracy', 'BACC', 'F1', 'Sensitivity', 'Specificity', 'Precision', 'AP']}
    summary['AUC_SD'] = fd.AUC.std(ddof=0)
    summary['OOF_AUC'] = roc_auc_score(y, oof)
    summary['OOF_AP'] = average_precision_score(y, oof)
    print(f"{label}: AUC {summary['AUC']:.3f} ± {summary['AUC_SD']:.3f}; OOF AUC {summary['OOF_AUC']:.3f}")
    return {'label': label, 'fold': fd, 'score': oof, 'pred': pred, 'summary': summary}

def bootstrap_delta(y, a, b, n=N_BOOT):
    y = np.asarray(y)
    a = np.asarray(a)
    b = np.asarray(b)
    rng = np.random.RandomState(SEED)
    obs = roc_auc_score(y, a) - roc_auc_score(y, b)
    vals = []
    for _ in range(n):
        ix = rng.randint(0, len(y), len(y))
        vals.append(roc_auc_score(y[ix], a[ix]) - roc_auc_score(y[ix], b[ix]))
    vals = np.asarray(vals)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    p = min(1, 2 * min((np.sum(vals <= 0) + 1) / (len(vals) + 1), (np.sum(vals >= 0) + 1) / (len(vals) + 1)))
    return (obs, lo, hi, p)

def information_set_tables(X, y, features, name, params):
    R = {'Demographics': locked_repeated_eval(X, y, features, 'demo', name, params, 'Demographics only'), 'Cerebellum': locked_repeated_eval(X, y, features, 'brain', name, params, 'Cerebellum only'), 'Combined': locked_repeated_eval(X, y, features, 'combined', name, params, 'Age + Gender + Cerebellum')}
    rows = []
    for key in ['Demographics', 'Cerebellum', 'Combined']:
        s = R[key]['summary']
        rows.append([key, name, f"{s['AUC']:.3f} ± {s['AUC_SD']:.3f}", f"{s['OOF_AUC']:.3f}", f"{s['BACC']:.3f}", f"{s['Sensitivity']:.3f}", f"{s['Specificity']:.3f}", f"{s['F1']:.3f}", f"{s['OOF_AP']:.3f}"])
    table = pd.DataFrame(rows, columns=['Information set', 'Locked classifier', 'AUC mean ± SD', 'OOF AUC', 'BACC', 'Sensitivity', 'Specificity', 'F1', 'AP'])
    table.to_csv(os.path.join(OUT, 'table7_primary_information_sets.csv'), index=False)
    save_table(table, 'Table 7. Locked Information-Set Comparison', 'table7_primary_information_sets.jpg', 'Primary internal-validation table. Same locked classifier and hyperparameters across information sets.')
    comp = []
    for label, a, b in [('Cerebellum - Demographics', 'Cerebellum', 'Demographics'), ('Combined - Demographics', 'Combined', 'Demographics'), ('Combined - Cerebellum', 'Combined', 'Cerebellum')]:
        d, lo, hi, p = bootstrap_delta(y.values, R[a]['score'], R[b]['score'])
        comp.append([label, d, lo, hi, p, 'Evidence of gain' if lo > 0 else 'CI includes no gain'])
    inc = pd.DataFrame(comp, columns=['Comparison', 'Delta OOF AUC', '95% CI lower', '95% CI upper', 'Bootstrap p', 'Interpretation'])
    inc.to_csv(os.path.join(OUT, 'table8_incremental_value.csv'), index=False)
    show = inc.copy()
    for c in ['Delta OOF AUC', '95% CI lower', '95% CI upper']:
        show[c] = show[c].map(lambda v: f'{v:+.4f}')
    show['Bootstrap p'] = show['Bootstrap p'].map(pstr)
    save_table(show, 'Table 8. Incremental Predictive Value', 'table8_incremental_value.jpg', 'Central contrast: Combined - Demographics.')
    return (R, table, inc)

def stability_table(best_selection_result, candidate_features, locked_features):
    counter = Counter((f for fold in best_selection_result['selected'] for f in fold))
    n = len(best_selection_result['selected'])
    rows = []
    for f in candidate_features:
        rows.append([f, counter[f], counter[f] / n, 'Yes' if f in locked_features else 'No', region(f), family(f)])
    d = pd.DataFrame(rows, columns=['Feature', 'Selection count', 'Selection frequency', 'Final locked', 'Region', 'Family']).sort_values(['Final locked', 'Selection frequency'], ascending=[False, False])
    d.to_csv(os.path.join(OUT, 'table9_feature_selection_stability.csv'), index=False)
    show = d.head(25).copy()
    show['Selection frequency'] = show['Selection frequency'].map(lambda x: f'{x:.1%}')
    save_table(show, 'Table 9. Feature Selection Stability', 'table9_feature_selection_stability.jpg')
    top = d.sort_values('Selection frequency', ascending=False).head(20).iloc[::-1]
    plt.figure(figsize=(9, max(5, 0.36 * len(top))))
    plt.barh(top.Feature, top['Selection frequency'])
    plt.xlabel('Selection frequency across best-model outer folds')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'figure_feature_selection_stability.jpg'), dpi=300)
    plt.close()
    pivot = d.pivot_table(index='Region', columns='Family', values='Selection frequency', aggfunc='max', fill_value=0)
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='viridis')
    plt.title('Maximum selection frequency by region and feature family')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'figure_region_family_stability_heatmap.jpg'), dpi=300)
    plt.close()
    return d

def full_residualized(D, features):
    r = Residualizer(features, add_demo=False).fit(D)
    return r.transform(D)

def association_table(D, y, locked_features, stability):
    R = full_residualized(D, locked_features)
    rows = []
    pvals = []
    for f in locked_features:
        x0 = R.loc[y.values == 0, f]
        x1 = R.loc[y.values == 1, f]
        pooled = np.sqrt(((len(x0) - 1) * x0.var(ddof=1) + (len(x1) - 1) * x1.var(ddof=1)) / max(len(x0) + len(x1) - 2, 1))
        d = (x1.mean() - x0.mean()) / pooled if pooled > 0 else np.nan
        auc = roc_auc_score(y, R[f])
        if auc < 0.5:
            auc = 1 - auc
        rho, psp = spearmanr(R[f], y)
        pt = ttest_ind(x0, x1, equal_var=False).pvalue
        pvals.append(pt)
        freq = stability.loc[stability.Feature == f, 'Selection frequency'].iloc[0]
        rows.append([f, freq, d, auc, rho, psp, pt])
    df = pd.DataFrame(rows, columns=['Feature', 'Selection frequency', 'Cohen d', 'Absolute univariate AUC', 'Spearman rho', 'Spearman p', 'Group p'])
    df['Group FDR'] = bh_fdr(pvals)
    df.to_csv(os.path.join(OUT, 'table10_feature_associations.csv'), index=False)
    show = df.copy()
    for c in ['Selection frequency', 'Cohen d', 'Absolute univariate AUC', 'Spearman rho']:
        show[c] = show[c].map(lambda v: f'{v:.3f}')
    show['Group p'] = show['Group p'].map(pstr)
    show['Group FDR'] = show['Group FDR'].map(pstr)
    save_table(show, 'Table 10. Locked-Feature Associations', 'table10_feature_associations.jpg', 'Effect-size analysis is descriptive; predictive performance is reported separately.')
    return (df, R)

def prediction_figures(y, R, best_name):
    plt.figure(figsize=(7, 6))
    for key in ['Demographics', 'Cerebellum', 'Combined']:
        fpr, tpr, _ = roc_curve(y, R[key]['score'])
        auc = roc_auc_score(y, R[key]['score'])
        plt.plot(fpr, tpr, label=f'{key} (AUC={auc:.3f})')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Locked information-set ROC curves — {best_name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'roc_information_sets.jpg'), dpi=300)
    plt.close()
    plt.figure(figsize=(7, 6))
    for key in ['Demographics', 'Cerebellum', 'Combined']:
        p, r, _ = precision_recall_curve(y, R[key]['score'])
        ap = average_precision_score(y, R[key]['score'])
        plt.plot(r, p, label=f'{key} (AP={ap:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-recall curves')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'precision_recall_information_sets.jpg'), dpi=300)
    plt.close()
    cm = confusion_matrix(y, R['Cerebellum']['pred'], labels=[0, 1])
    plt.figure(figsize=(5.5, 4.8))
    sns.heatmap(cm, annot=True, fmt='d', cbar=False, xticklabels=['Control', 'ADHD'], yticklabels=['Control', 'ADHD'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Cerebellum-only confusion matrix — {best_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'confusion_matrix_cerebellum.jpg'), dpi=300)
    plt.close()

def subgroup_table(D, y, R):
    age_cut = D.Age.median()
    groups = [('Female', D.Gender.values == 0), ('Male', D.Gender.values == 1), ('Younger', D.Age.values <= age_cut), ('Older', D.Age.values > age_cut)]
    rows = []
    for gname, mask in groups:
        yy = y.values[mask]
        for key in ['Demographics', 'Cerebellum', 'Combined']:
            score = R[key]['score'][mask]
            pred = R[key]['pred'][mask]
            m = metric_dict(yy, score, pred)
            rows.append([gname, key, mask.sum(), m['AUC'], m['BACC'], m['Sensitivity'], m['Specificity'], m['F1']])
    df = pd.DataFrame(rows, columns=['Subgroup', 'Information set', 'N', 'AUC', 'BACC', 'Sensitivity', 'Specificity', 'F1'])
    df.to_csv(os.path.join(OUT, 'table11_subgroup_performance.csv'), index=False)
    show = df.copy()
    for c in ['AUC', 'BACC', 'Sensitivity', 'Specificity', 'F1']:
        show[c] = show[c].map(lambda v: f'{v:.3f}')
    save_table(show, 'Table 11. Subgroup Performance', 'table11_subgroup_performance.jpg', 'Same locked model predictions are stratified post hoc; subgroup models are not re-trained.')
    return df

def imbalance_test(X, y, features, name, params):
    if not RUN_IMBALANCE:
        return None
    from imblearn.over_sampling import RandomOverSampler, SMOTE
    settings = [('Class-weighted', True, None), ('Unweighted', False, None), ('Random oversampling 1:1', False, RandomOverSampler(random_state=SEED)), ('SMOTE 1:1', False, SMOTE(random_state=SEED, k_neighbors=5))]
    rows = []
    for label, balanced, sampler in settings:
        auc, sd = locked_cv_auc(X, y, features, name, params, balanced=balanced, sampler=sampler, seed=SEED + 700)
        rows.append([label, auc, sd])
    df = pd.DataFrame(rows, columns=['Strategy', 'CV AUC', 'AUC SD'])
    df.to_csv(os.path.join(OUT, 'table12_imbalance_robustness.csv'), index=False)
    save_table(df.round(4), 'Table 12. Class-Imbalance Robustness', 'table12_imbalance_robustness.jpg', 'Resampling occurs inside each training fold only.')
    return df

def outlier_test(X, y, features, name, params):
    if not RUN_OUTLIER:
        return None
    rows = []
    for label, clip in [('Original', None), ('±1.5 IQR', 1.5), ('±3 IQR', 3.0), ('±4 IQR', 4.0)]:
        auc, sd = locked_cv_auc(X, y, features, name, params, clip=clip, seed=SEED + 800)
        rows.append([label, auc, sd])
    df = pd.DataFrame(rows, columns=['Outlier handling', 'CV AUC', 'AUC SD'])
    df.to_csv(os.path.join(OUT, 'table13_outlier_robustness.csv'), index=False)
    save_table(df.round(4), 'Table 13. Outlier Robustness', 'table13_outlier_robustness.jpg', 'IQR limits are estimated from the training fold and applied to the test fold.')
    return df

def feature_count_sensitivity(X, y, candidates, name, params):
    if not RUN_FEATURE_COUNT:
        return None
    rows = []
    for k in [2, 4, 6, 8, 10, 12, 16]:
        cv = StratifiedKFold(5, shuffle=True, random_state=SEED + 900)
        aucs = []
        for tr, te in cv.split(X, y):
            est = selection_pipeline('brain', name, candidates)
            est.set_params(selector__k=k, **{f'model__{p}': v for p, v in params.items()})
            est.fit(X.iloc[tr], y.iloc[tr])
            aucs.append(roc_auc_score(y.iloc[te], model_score(est, X.iloc[te])))
        rows.append([k, np.mean(aucs), np.std(aucs, ddof=0)])
    df = pd.DataFrame(rows, columns=['Requested features', 'CV AUC', 'AUC SD'])
    df.to_csv(os.path.join(OUT, 'supplement_feature_count_sensitivity.csv'), index=False)
    plt.figure(figsize=(7, 5))
    plt.errorbar(df['Requested features'], df['CV AUC'], yerr=df['AUC SD'], marker='o')
    plt.xlabel('Requested non-redundant brain features')
    plt.ylabel('CV AUC')
    plt.title('Feature-count sensitivity (exploratory)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'figure_feature_count_sensitivity.jpg'), dpi=300)
    plt.close()
    return df

def family_ablation(X, y, candidates, name, params):
    if not RUN_FAMILY_ABLATION:
        return None
    fams = sorted(set((family(f) for f in candidates)))
    rows = []
    for fam in ['All'] + fams:
        F = candidates if fam == 'All' else [f for f in candidates if family(f) == fam]
        k = min(10, len(F))
        cv = StratifiedKFold(5, shuffle=True, random_state=SEED + 950)
        aucs = []
        for tr, te in cv.split(X, y):
            est = selection_pipeline('brain', name, F)
            est.set_params(selector__k=k, **{f'model__{p}': v for p, v in params.items()})
            est.fit(X.iloc[tr], y.iloc[tr])
            aucs.append(roc_auc_score(y.iloc[te], model_score(est, X.iloc[te])))
        rows.append([fam, len(F), np.mean(aucs), np.std(aucs, ddof=0)])
    df = pd.DataFrame(rows, columns=['Feature family', 'Candidate N', 'CV AUC', 'AUC SD'])
    df.to_csv(os.path.join(OUT, 'supplement_morphometric_family_sensitivity.csv'), index=False)
    save_table(df.round(4), 'Supplement. Morphometric-Family Sensitivity', 'supplement_morphometric_family_sensitivity.jpg', 'Exploratory family-level analysis; not used to select the primary final model.')
    return df

def permutation_test(X, y, features, name, params):
    splits = list(StratifiedKFold(5, shuffle=True, random_state=SEED + 1000).split(X, y))

    def auc_for_labels(y_values):
        yy = pd.Series(np.asarray(y_values), index=y.index)
        aucs = []
        for tr, te in splits:
            est = locked_pipeline('brain', name, features, params, balanced=True)
            est.fit(X.iloc[tr], yy.iloc[tr])
            aucs.append(roc_auc_score(yy.iloc[te], model_score(est, X.iloc[te])))
        return float(np.mean(aucs))
    true_auc = auc_for_labels(y.values)
    print(f'\nPermutation test: AUC {true_auc:.4f}, permutations {N_PERM}')

    def one(seed):
        rng = np.random.RandomState(seed)
        return auc_for_labels(rng.permutation(y.values))
    perm = Parallel(n_jobs=PERM_N_JOBS)((delayed(one)(SEED + 2000 + i) for i in range(N_PERM)))
    perm = np.asarray(perm)
    p = (np.sum(perm >= true_auc) + 1) / (len(perm) + 1)
    df = pd.DataFrame([[true_auc, perm.mean(), perm.std(ddof=0), p, N_PERM]], columns=['True locked CV AUC', 'Permutation mean AUC', 'Permutation SD', 'Permutation p', 'N permutations'])
    df.to_csv(os.path.join(OUT, 'table14_permutation_test.csv'), index=False)
    save_table(df.round(4), 'Table 14. Locked-Signature Permutation Test', 'table14_permutation_test.jpg', 'This test holds the selected model/features fixed; it does not repeat the full development search per permutation.')
    plt.figure(figsize=(7, 5))
    plt.hist(perm, bins=20, density=True, alpha=0.75)
    plt.axvline(true_auc, linestyle='--', linewidth=2, label=f'True AUC={true_auc:.3f}')
    plt.xlabel('Mean CV AUC under label permutation')
    plt.ylabel('Density')
    plt.title(f'Permutation distribution (p={p:.4f})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'permutation.jpg'), dpi=300)
    plt.close()
    return df

def interpretability(D, y, features, name, params, residualized):
    plt.figure(figsize=(max(7, 0.75 * len(features)), max(6, 0.65 * len(features))))
    sns.heatmap(residualized[features].corr(), annot=True, fmt='.2f', cmap='coolwarm', square=True, annot_kws={'fontsize': 7})
    plt.title('Correlation among locked residualized cerebellar features')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'heatmap_correlation.jpg'), dpi=300)
    plt.close()
    sp = []
    for f in features:
        rho, p = spearmanr(residualized[f], y)
        sp.append([f, rho, p])
    sp = pd.DataFrame(sp, columns=['Feature', 'Spearman', 'p'])
    sp['FDR'] = bh_fdr(sp.p)
    sp.to_csv(os.path.join(OUT, 'spearman_selected_features.csv'), index=False)
    sp = sp.sort_values('Spearman')
    plt.figure(figsize=(8, max(4, 0.42 * len(sp))))
    plt.barh(sp.Feature, sp.Spearman)
    plt.axvline(0, linewidth=1)
    plt.xlabel('Spearman correlation with diagnosis')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'spearman_correlation.jpg'), dpi=300)
    plt.close()
    est = locked_pipeline('brain', name, features, params, balanced=True)
    est.fit(D, y)
    pre = est[:-1]
    final_model = est.named_steps['model']
    Xt = np.asarray(pre.transform(D), float)
    imp = permutation_importance(final_model, Xt, y, scoring='roc_auc', n_repeats=20, random_state=SEED, n_jobs=N_JOBS)
    pi = pd.DataFrame({'Feature': features, 'Importance': imp.importances_mean, 'SD': imp.importances_std}).sort_values('Importance', ascending=False)
    pi.to_csv(os.path.join(OUT, 'model_permutation_importance.csv'), index=False)
    plt.figure(figsize=(8, max(4, 0.4 * len(pi))))
    q = pi.iloc[::-1]
    plt.barh(q.Feature, q.Importance, xerr=q.SD)
    plt.xlabel('Decrease in AUC after permutation')
    plt.title(f'Permutation importance — {name}')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'permutation_importance.jpg'), dpi=300)
    plt.close()
    if RUN_SHAP:
        import shap
        rng = np.random.RandomState(SEED)
        ix = rng.choice(len(Xt), min(SHAP_N, len(Xt)), replace=False)
        sample = Xt[ix]
        if name in ['RandomForest', 'XGBoost', 'CatBoost']:
            shap_model = final_model.model_ if name == 'CatBoost' else final_model
            explainer = shap.TreeExplainer(shap_model)
            sv = explainer.shap_values(sample)
            if isinstance(sv, list):
                sv = sv[-1]
        elif name == 'ElasticNet':
            explainer = shap.LinearExplainer(final_model, Xt)
            sv = explainer.shap_values(sample)
        else:
            bg = Xt[rng.choice(len(Xt), min(35, len(Xt)), replace=False)]
            explainer = shap.KernelExplainer(final_model.predict_proba, bg)
            sv = explainer.shap_values(sample, nsamples=150)
            if isinstance(sv, list):
                sv = sv[-1]
            elif np.asarray(sv).ndim == 3:
                sv = np.asarray(sv)[:, :, 1]
        shap.summary_plot(sv, sample, feature_names=features, show=False)
        plt.title(f'SHAP summary — locked {name}')
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, 'shap_summary.jpg'), dpi=300, bbox_inches='tight')
        plt.close()
    return pi

def literature_table(R):
    rows = [['Current study', 'Cerebellum-only morphometry; locked repeated CV', f"{R['Cerebellum']['summary']['OOF_AUC']:.3f}", 'Direct result', 'Current analysis'], ['Current study + demographics', 'Age + Gender + cerebellum; locked repeated CV', f"{R['Combined']['summary']['OOF_AUC']:.3f}", 'Incremental result', 'Current analysis'], ['Zhang-James et al. 2021', 'Large multi-site structural MRI; child held-out test', '0.64', 'Strong structural-MRI comparator', 'doi:10.1038/s41398-021-01201-4'], ['Zhang-James et al. 2021 + age/sex', 'sMRI + age + sex; child held-out test', '0.65', 'Strong combined comparator', 'doi:10.1038/s41398-021-01201-4'], ['Bohland et al. 2012', 'Whole-brain anatomy; separate ADHD-200 test', '~0.74', 'Broader anatomical information', 'doi:10.3389/fnsys.2012.00078'], ['Tian et al. 2024 meta-analysis', 'Heterogeneous sMRI/fMRI studies; pooled SROC', '0.81', 'Context only; not a direct benchmark', 'doi:10.1016/j.jad.2024.03.111']]
    df = pd.DataFrame(rows, columns=['Study', 'Input / validation', 'AUC', 'Comparability', 'Reference'])
    df.to_csv(os.path.join(OUT, 'table15_literature_context.csv'), index=False)
    save_table(df, 'Table 15. Literature Context', 'table15_literature_context.jpg', 'The pooled 0.81 SROC mixes modalities and validation designs; held-out structural-MRI results are more comparable.')
    return df

def study_design_figure(n, p):
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.axis('off')
    boxes = [(0.02, 0.36, 0.12, 0.28, f'Cohort\nN={n}'), (0.17, 0.36, 0.15, 0.28, f'Candidate brain space\n{p} candidates'), (0.35, 0.36, 0.15, 0.28, '5-model comparison\nSVM / RF / EN / XGB / Cat'), (0.53, 0.36, 0.14, 0.28, 'Best model\n+ backward ablation\nPIPELINE LOCK'), (0.7, 0.56, 0.12, 0.2, 'Demographics'), (0.7, 0.3, 0.12, 0.2, 'Cerebellum'), (0.7, 0.04, 0.12, 0.2, 'Combined'), (0.85, 0.36, 0.13, 0.28, 'Inference\nDelta AUC / permutation\nrobustness / stability')]
    for x, y0, w, h, text in boxes:
        ax.add_patch(plt.Rectangle((x, y0), w, h, fill=False, lw=1.3, transform=ax.transAxes))
        ax.text(x + w / 2, y0 + h / 2, text, ha='center', va='center', transform=ax.transAxes, fontsize=9.2)
    arrows = [((0.14, 0.5), (0.17, 0.5)), ((0.32, 0.5), (0.35, 0.5)), ((0.5, 0.5), (0.53, 0.5)), ((0.67, 0.5), (0.7, 0.66)), ((0.67, 0.5), (0.7, 0.4)), ((0.67, 0.5), (0.7, 0.14)), ((0.82, 0.66), (0.85, 0.56)), ((0.82, 0.4), (0.85, 0.5)), ((0.82, 0.14), (0.85, 0.44))]
    for a, b in arrows:
        ax.annotate('', xy=b, xytext=a, xycoords=ax.transAxes, arrowprops=dict(arrowstyle='->', lw=1.1))
    ax.set_title('Study design: select, simplify, lock, then validate', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'figure_study_design.jpg'), dpi=300, bbox_inches='tight')
    plt.close()

def output_map():
    rows = [['Table 1', 'Baseline characteristics', 'Cohort/confounding context'], ['Table 2', 'Dataset architecture', 'Justifies sparse modelling'], ['Table 3', 'Five-model comparison', 'Select best classifier'], ['Table 4', 'Initial best model', 'Best development parameters/features'], ['Table 5', 'Backward ablation', 'Remove negative/non-inferior features'], ['Table 6', 'Locked final model', 'Analysis plan freezes here'], ['Table 7', 'Information sets', 'Primary internal-validation results'], ['Table 8', 'Incremental value', 'Does cerebellum add beyond demographics?'], ['Table 9', 'Feature stability', 'Reproducible candidate signal'], ['Table 10', 'Feature associations', 'Effect sizes / biological interpretation'], ['Table 11', 'Subgroups', 'Sex/age heterogeneity'], ['Tables 12-13', 'Imbalance/outlier robustness', 'Sensitivity analyses'], ['Table 14', 'Permutation', 'Locked signal vs chance'], ['Table 15', 'Literature context', 'Appropriate external context']]
    df = pd.DataFrame(rows, columns=['Manuscript item', 'Content', 'Role'])
    df.to_csv(os.path.join(OUT, 'paper_output_map.csv'), index=False)
    return df

def main(argv=None):
    global DATA_PATH, OUT
    parser = argparse.ArgumentParser(description='Run the legacy ADHD cerebellum analysis.')
    parser.add_argument('--data', default=DATA_PATH, help='Input CSV path (default: %(default)s).')
    parser.add_argument('--output', default=OUT, help='Output directory (default: %(default)s).')
    args = parser.parse_args(argv)
    DATA_PATH = args.data
    OUT = args.output
    os.makedirs(OUT, exist_ok=True)
    print()
    print('ADHD cerebellum analysis')
    print()
    print(f'Publication mode: {PUBLICATION_MODE}')
    print(f'Model selection: {OUTER_FOLDS}x{MODEL_SELECT_REPEATS}; final evaluation: {OUTER_FOLDS}x{FINAL_REPEATS}')
    print(f'Models: {available_models()}')
    D = pd.read_csv(DATA_PATH)
    D = D.dropna(subset=['DX', 'Age', 'Gender', 'ICVcm3']).reset_index(drop=True)
    D['DX'] = D.DX.astype(int)
    D['Gender'] = D.Gender.astype(int)
    y = D.DX.copy()
    X = D.copy()
    candidates = candidate_features(D.columns)
    print(f'N={len(D)} | ADHD={(y == 1).sum()} | Control={(y == 0).sum()} | candidate features={len(candidates)}')
    baseline_table(D)
    architecture_table(D, candidates)
    study_design_figure(len(D), len(candidates))
    output_map()
    best_name, model_results, _ = choose_best_model(X, y, candidates)
    initial_params_all, initial_features = tune_initial_model(X, y, candidates, best_name)
    initial_model_params = model_params(initial_params_all)
    locked_features, _ = backward_ablation(X, y, initial_features, best_name, initial_model_params)
    final_params, final_estimator = tune_locked_model(X, y, locked_features, best_name)
    R, primary_table, inc = information_set_tables(X, y, locked_features, best_name, final_params)
    stability = stability_table(model_results[best_name], candidates, locked_features)
    associations, residualized = association_table(D, y, locked_features, stability)
    prediction_figures(y, R, best_name)
    subgroup_table(D, y, R)
    imbalance_test(X, y, locked_features, best_name, final_params)
    outlier_test(X, y, locked_features, best_name, final_params)
    feature_count_sensitivity(X, y, candidates, best_name, final_params)
    family_ablation(X, y, candidates, best_name, final_params)
    perm = permutation_test(X, y, locked_features, best_name, final_params)
    interpretability(D, y, locked_features, best_name, final_params, residualized)
    literature_table(R)
    summary = pd.DataFrame([{'Best_Model': best_name, 'Model_Selection_Mean_AUC': model_results[best_name]['summary']['AUC'], 'N_Locked_Features': len(locked_features), 'Final_Params': json.dumps(final_params), 'Demographics_OOF_AUC': R['Demographics']['summary']['OOF_AUC'], 'Cerebellum_OOF_AUC': R['Cerebellum']['summary']['OOF_AUC'], 'Combined_OOF_AUC': R['Combined']['summary']['OOF_AUC'], 'Permutation_AUC': perm.iloc[0]['True locked CV AUC'], 'Permutation_P': perm.iloc[0]['Permutation p'], 'Publication_Mode': PUBLICATION_MODE}])
    summary.to_csv(os.path.join(OUT, 'final_model_summary.csv'), index=False)
    with open(os.path.join(OUT, 'research_summary.txt'), 'w', encoding='utf-8') as f:
        f.write('PRIMARY ANALYSIS SUMMARY\n\n')
        f.write(f'Best classifier selected in development: {best_name}\n')
        f.write(f'Locked features ({len(locked_features)}): {locked_features}\n')
        f.write(f'Final hyperparameters: {final_params}\n\n')
        f.write(f"Demographics OOF AUC: {R['Demographics']['summary']['OOF_AUC']:.4f}\n")
        f.write(f"Cerebellum OOF AUC: {R['Cerebellum']['summary']['OOF_AUC']:.4f}\n")
        f.write(f"Combined OOF AUC: {R['Combined']['summary']['OOF_AUC']:.4f}\n\n")
        f.write('Incremental comparisons:\n')
        f.write(inc.to_string(index=False) + '\n\n')
        f.write(f"Locked-signature permutation p: {perm.iloc[0]['Permutation p']:.6f}\n\n")
        f.write('Interpretation: model-selection and ablation AUCs are development results. Use Table 7 for the main locked internal-validation performance. Because this dataset was also used to choose the model and feature set, external validation remains necessary. The pooled MRI AUC of 0.81 is a heterogeneous context value rather than a direct cerebellum-only benchmark.\n')
    print('\nFinished')
    print(f'Output folder: {OUT}/')
    print('Main tables: 3, 5, 6, 7, 8, 14')
if __name__ == '__main__':
    main()
