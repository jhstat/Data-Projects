# Full SECOM Analysis Script
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    f1_score, precision_recall_curve, auc,
    confusion_matrix, classification_report,
    average_precision_score, recall_score, precision_score
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks, RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN
import shap
import warnings
from itertools import product
from collections import defaultdict
import json

warnings.filterwarnings('ignore')
np.random.seed(42)

print("="*60)
print("UCI SECOM SEMICONDUCTOR DEFECT DETECTION ANALYSIS")
print("="*60)

# ============================================================
# CELL 2: Load Data
# ============================================================
print("\n[1] Loading data...")
df = pd.read_csv('uci-secom.csv')
df = df.rename(columns={'Pass/Fail': 'target'})
df['target'] = df['target'].map({-1: 0, 1: 1})

print(f"Dataset shape: {df.shape}")
print(f"Target distribution:\n{df['target'].value_counts()}")
print(f"Defect rate: {df['target'].mean():.2%}")

# ============================================================
# CELL 3: Train-Test Split
# ============================================================
print("\n[2] Train-Test Split...")
feature_cols = [col for col in df.columns if col not in ['target', 'Time']]
X = df[feature_cols]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Total features: {len(feature_cols)}")
print(f"Train set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Train defect rate: {y_train.mean():.2%}")
print(f"Test defect rate: {y_test.mean():.2%}")

# ============================================================
# CELL 4-8: Feature Engineering
# ============================================================
print("\n[3] Feature Engineering...")

# Calculate missingness on training set
missing_pct = X_train.isnull().mean() * 100

# Columns to drop (>=80% missing)
cols_drop_high_missing = missing_pct[missing_pct >= 80].index.tolist()
print(f"Columns with >=80% missing: {len(cols_drop_high_missing)}")

# Columns with 50-80% missing -> create indicators
cols_medium_missing = missing_pct[(missing_pct >= 50) & (missing_pct < 80)].index.tolist()
print(f"Columns with 50-80% missing: {len(cols_medium_missing)}")
missingness_indicators = [f"{col}_missing" for col in cols_medium_missing]

# Remove constant and near-constant columns
remaining_cols = [col for col in X_train.columns
                  if col not in cols_drop_high_missing and col not in cols_medium_missing]

cols_to_drop_constant = []
for col in remaining_cols:
    col_data = X_train[col].dropna()
    if len(col_data) == 0:
        cols_to_drop_constant.append(col)
        continue
    if col_data.nunique() == 1:
        cols_to_drop_constant.append(col)
        continue
    q1, q3 = col_data.quantile([0.25, 0.75])
    if q3 - q1 == 0:
        cols_to_drop_constant.append(col)
        continue
    top_freq = col_data.value_counts(normalize=True).iloc[0]
    if top_freq > 0.99:
        cols_to_drop_constant.append(col)
        continue
    mode_val = col_data.mode().iloc[0] if len(col_data.mode()) > 0 else None
    if mode_val is not None:
        count_nonmode = (col_data != mode_val).sum()
        if count_nonmode < 20:
            cols_to_drop_constant.append(col)
            continue

print(f"Columns dropped (constant/near-constant): {len(cols_to_drop_constant)}")

# Remove highly correlated columns
cols_for_corr = [col for col in remaining_cols if col not in cols_to_drop_constant]
print(f"Checking correlation among {len(cols_for_corr)} columns...")

corr_matrix = X_train[cols_for_corr].corr().abs()
cols_to_drop_corr = set()
checked_pairs = set()

for i, col1 in enumerate(cols_for_corr):
    for col2 in cols_for_corr[i+1:]:
        if (col1, col2) in checked_pairs or col1 in cols_to_drop_corr or col2 in cols_to_drop_corr:
            continue
        corr_val = corr_matrix.loc[col1, col2]
        if corr_val > 0.95:
            missing1 = X_train[col1].isnull().sum()
            missing2 = X_train[col2].isnull().sum()
            if missing1 != missing2:
                drop_col = col1 if missing1 > missing2 else col2
            else:
                std1 = X_train[col1].std()
                std2 = X_train[col2].std()
                drop_col = col1 if std1 < std2 else col2
            cols_to_drop_corr.add(drop_col)
        checked_pairs.add((col1, col2))

cols_to_drop_corr = list(cols_to_drop_corr)
print(f"Columns dropped (high correlation): {len(cols_to_drop_corr)}")

# Lock feature set
all_cols_to_drop = set(cols_drop_high_missing + cols_medium_missing +
                       cols_to_drop_constant + cols_to_drop_corr)
final_feature_cols = [col for col in feature_cols if col not in all_cols_to_drop]
final_feature_cols_with_indicators = final_feature_cols + missingness_indicators

feature_engineering_log = {
    'original_features': len(feature_cols),
    'dropped_high_missing': len(cols_drop_high_missing),
    'medium_missing_to_indicators': len(cols_medium_missing),
    'dropped_constant': len(cols_to_drop_constant),
    'dropped_correlated': len(cols_to_drop_corr),
    'final_numeric_features': len(final_feature_cols),
    'missingness_indicators': len(missingness_indicators),
    'total_final_features': len(final_feature_cols_with_indicators)
}

print("\n" + "="*60)
print("FEATURE ENGINEERING SUMMARY")
print("="*60)
for k, v in feature_engineering_log.items():
    print(f"  {k}: {v}")

FINAL_NUMERIC_COLS = final_feature_cols
MISSINGNESS_INDICATOR_SOURCE_COLS = cols_medium_missing
MISSINGNESS_INDICATOR_COLS = missingness_indicators

# ============================================================
# Define helper functions
# ============================================================
def prepare_features(X_data, numeric_cols, indicator_source_cols, indicator_cols):
    X_numeric = X_data[numeric_cols].copy()
    for src_col, ind_col in zip(indicator_source_cols, indicator_cols):
        if src_col in X_data.columns:
            X_numeric[ind_col] = X_data[src_col].isnull().astype(int)
        else:
            X_numeric[ind_col] = 0
    return X_numeric

def impute_and_scale(X_train_fold, X_val_fold):
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = pd.DataFrame(
        imputer.fit_transform(X_train_fold),
        columns=X_train_fold.columns,
        index=X_train_fold.index
    )
    X_val_imputed = pd.DataFrame(
        imputer.transform(X_val_fold),
        columns=X_val_fold.columns,
        index=X_val_fold.index
    )
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_imputed),
        columns=X_train_imputed.columns,
        index=X_train_imputed.index
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val_imputed),
        columns=X_val_imputed.columns,
        index=X_val_imputed.index
    )
    return X_train_scaled, X_val_scaled, imputer, scaler

def apply_resampling(X_train_fold, y_train_fold, strategy):
    if strategy == 'none':
        return X_train_fold, y_train_fold
    elif strategy == 'undersample':
        n_pos = int(y_train_fold.sum())
        n_neg_target = n_pos * 3
        rus = RandomUnderSampler(
            sampling_strategy={0: min(n_neg_target, int((y_train_fold == 0).sum())), 1: n_pos},
            random_state=42
        )
        X_resampled, y_resampled = rus.fit_resample(X_train_fold, y_train_fold)
        return X_resampled, y_resampled
    elif strategy == 'smote_tomek':
        n_pos = int(y_train_fold.sum())
        k_neighbors = min(3, n_pos - 1) if n_pos > 1 else 1
        smote_tomek = SMOTETomek(
            smote=SMOTE(k_neighbors=k_neighbors, random_state=42),
            random_state=42
        )
        X_resampled, y_resampled = smote_tomek.fit_resample(X_train_fold, y_train_fold)
        return X_resampled, y_resampled
    elif strategy == 'smote_enn':
        n_pos = int(y_train_fold.sum())
        k_neighbors = min(3, n_pos - 1) if n_pos > 1 else 1
        smote_enn = SMOTEENN(
            smote=SMOTE(k_neighbors=k_neighbors, random_state=42),
            random_state=42
        )
        X_resampled, y_resampled = smote_enn.fit_resample(X_train_fold, y_train_fold)
        return X_resampled, y_resampled
    else:
        raise ValueError(f"Unknown resampling strategy: {strategy}")

def recall_at_precision_threshold(y_true, y_proba, precision_threshold=0.2):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    valid_indices = np.where(precisions >= precision_threshold)[0]
    if len(valid_indices) == 0:
        return 0.0, None
    max_recall_idx = valid_indices[np.argmax(recalls[valid_indices])]
    best_recall = recalls[max_recall_idx]
    if max_recall_idx < len(thresholds):
        best_threshold = thresholds[max_recall_idx]
    else:
        best_threshold = thresholds[-1] if len(thresholds) > 0 else 0.5
    return best_recall, best_threshold

def find_optimal_threshold(y_true, y_proba, precision_threshold=0.2):
    recall_at_prec, threshold = recall_at_precision_threshold(y_true, y_proba, precision_threshold)
    if threshold is not None and recall_at_prec > 0:
        return threshold, 'recall@precision'
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_f1_idx = np.argmax(f1_scores[:-1])
    best_threshold = thresholds[best_f1_idx]
    return best_threshold, 'max_f1'

def calculate_metrics_cv(y_true, y_proba, threshold=0.5):
    """Calculate metrics for CV (includes threshold sweeping for recall@prec>0.2)."""
    y_pred = (y_proba >= threshold).astype(int)
    pr_auc = average_precision_score(y_true, y_proba)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall_at_prec, _ = recall_at_precision_threshold(y_true, y_proba, 0.2)

    return {
        'pr_auc': pr_auc,
        'f1': f1,
        'recall_at_prec_0.2': recall_at_prec,
        'recall': recall,
        'precision': precision,
        'threshold': threshold
    }

def calculate_metrics_test(y_true, y_proba, threshold):
    """Calculate metrics for test set (fixed threshold, no sweeping)."""
    y_pred = (y_proba >= threshold).astype(int)
    pr_auc = average_precision_score(y_true, y_proba)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)

    return {
        'pr_auc': pr_auc,
        'f1': f1,
        'recall': recall,
        'precision': precision,
        'threshold': threshold
    }

# ============================================================
# Model configs
# ============================================================
MODEL_CONFIGS = {
    'LogisticRegression': {
        'model_class': LogisticRegression,
        'base_params': {'solver': 'saga', 'penalty': 'elasticnet', 'max_iter': 1000, 'random_state': 42},
        'param_grid': {
            'C': [0.01, 0.1, 1.0],
            'l1_ratio': [0.3, 0.5, 0.7]
        }
    },
    'LightGBM': {
        'model_class': LGBMClassifier,
        'base_params': {'random_state': 42, 'verbose': -1, 'n_jobs': -1},
        'param_grid': {
            'num_leaves': [15, 31],
            'min_data_in_leaf': [10, 20],
            'learning_rate': [0.05, 0.1]
        }
    },
    'XGBoost': {
        'model_class': XGBClassifier,
        'base_params': {'random_state': 42, 'eval_metric': 'logloss', 'n_jobs': -1},
        'param_grid': {
            'max_depth': [3, 5],
            'min_child_weight': [1, 5],
            'learning_rate': [0.05, 0.1]
        }
    },
    'RandomForest': {
        'model_class': RandomForestClassifier,
        'base_params': {'random_state': 42, 'n_jobs': -1},
        'param_grid': {
            'n_estimators': [100, 200],
            'max_depth': [5, 10],
            'min_samples_leaf': [5, 10]
        }
    }
}

RESAMPLING_STRATEGIES = ['none', 'undersample', 'smote_tomek', 'smote_enn']

# ============================================================
# CV Training
# ============================================================
print("\n[4] Cross-Validation Training...")
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

def run_cv_experiment(model_name, model_config, resampling_strategy, X_train_full, y_train_full, skf):
    X_train_prepared = prepare_features(
        X_train_full, FINAL_NUMERIC_COLS,
        MISSINGNESS_INDICATOR_SOURCE_COLS, MISSINGNESS_INDICATOR_COLS
    )
    param_grid = model_config['param_grid']
    param_names = list(param_grid.keys())
    param_combinations = list(product(*param_grid.values()))

    best_score = -np.inf
    best_params = None
    best_oof_proba = None

    for param_values in param_combinations:
        params = dict(zip(param_names, param_values))
        full_params = {**model_config['base_params'], **params}

        if resampling_strategy == 'none' and model_name != 'XGBoost':
            full_params['class_weight'] = 'balanced'
        elif resampling_strategy == 'none' and model_name == 'XGBoost':
            n_neg = (y_train_full == 0).sum()
            n_pos = (y_train_full == 1).sum()
            full_params['scale_pos_weight'] = n_neg / n_pos

        oof_proba = np.zeros(len(y_train_full))
        fold_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train_prepared, y_train_full)):
            X_fold_train = X_train_prepared.iloc[train_idx]
            X_fold_val = X_train_prepared.iloc[val_idx]
            y_fold_train = y_train_full.iloc[train_idx]
            y_fold_val = y_train_full.iloc[val_idx]

            X_fold_train_scaled, X_fold_val_scaled, _, _ = impute_and_scale(X_fold_train, X_fold_val)
            X_fold_train_resampled, y_fold_train_resampled = apply_resampling(
                X_fold_train_scaled, y_fold_train, resampling_strategy
            )

            model = model_config['model_class'](**full_params)
            model.fit(X_fold_train_resampled, y_fold_train_resampled)

            fold_proba = model.predict_proba(X_fold_val_scaled)[:, 1]
            oof_proba[val_idx] = fold_proba

            fold_score = average_precision_score(y_fold_val, fold_proba)
            fold_scores.append(fold_score)

        mean_score = np.mean(fold_scores)
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
            best_oof_proba = oof_proba.copy()

    return best_oof_proba, best_params, best_score

results = []
oof_predictions = {}

total_experiments = len(MODEL_CONFIGS) * len(RESAMPLING_STRATEGIES)
exp_count = 0

for model_name, model_config in MODEL_CONFIGS.items():
    for resampling in RESAMPLING_STRATEGIES:
        exp_count += 1
        print(f"  [{exp_count}/{total_experiments}] {model_name} + {resampling}...", end=" ")

        try:
            oof_proba, best_params, cv_pr_auc = run_cv_experiment(
                model_name, model_config, resampling, X_train, y_train, skf
            )
            opt_threshold, threshold_method = find_optimal_threshold(y_train, oof_proba, 0.2)
            metrics = calculate_metrics_cv(y_train, oof_proba, opt_threshold)

            result = {
                'model': model_name,
                'resampling': resampling,
                'best_params': best_params,
                'cv_pr_auc': cv_pr_auc,
                'optimal_threshold': opt_threshold,
                'threshold_method': threshold_method,
                **metrics
            }
            results.append(result)

            key = f"{model_name}_{resampling}"
            oof_predictions[key] = {
                'proba': oof_proba,
                'best_params': best_params,
                'threshold': opt_threshold
            }

            print(f"PR-AUC={cv_pr_auc:.4f}, Recall@Prec>0.2={metrics['recall_at_prec_0.2']:.4f}")

        except Exception as e:
            print(f"ERROR: {str(e)}")
            continue

# ============================================================
# Results Summary
# ============================================================
print("\n[5] CV Results Summary...")
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('recall_at_prec_0.2', ascending=False)

display_cols = ['model', 'resampling', 'recall_at_prec_0.2', 'f1', 'pr_auc', 'optimal_threshold']
print("\nCV RESULTS (sorted by Recall@Precision>0.2):")
print(results_df[display_cols].to_string(index=False))

# ============================================================
# Test Set Evaluation
# ============================================================
print("\n[6] Test Set Evaluation...")

X_train_prepared = prepare_features(
    X_train, FINAL_NUMERIC_COLS,
    MISSINGNESS_INDICATOR_SOURCE_COLS, MISSINGNESS_INDICATOR_COLS
)
X_test_prepared = prepare_features(
    X_test, FINAL_NUMERIC_COLS,
    MISSINGNESS_INDICATOR_SOURCE_COLS, MISSINGNESS_INDICATOR_COLS
)
X_train_final, X_test_final, final_imputer, final_scaler = impute_and_scale(
    X_train_prepared, X_test_prepared
)

test_results = []
final_models = {}

for idx, row in results_df.iterrows():
    model_name = row['model']
    resampling = row['resampling']
    best_params = row['best_params']
    opt_threshold = row['optimal_threshold']

    model_config = MODEL_CONFIGS[model_name]

    try:
        full_params = {**model_config['base_params'], **best_params}

        if resampling == 'none' and model_name != 'XGBoost':
            full_params['class_weight'] = 'balanced'
        elif resampling == 'none' and model_name == 'XGBoost':
            n_neg = (y_train == 0).sum()
            n_pos = (y_train == 1).sum()
            full_params['scale_pos_weight'] = n_neg / n_pos

        X_train_resampled, y_train_resampled = apply_resampling(
            X_train_final, y_train, resampling
        )

        model = model_config['model_class'](**full_params)
        model.fit(X_train_resampled, y_train_resampled)

        test_proba = model.predict_proba(X_test_final)[:, 1]
        test_pred = (test_proba >= opt_threshold).astype(int)

        test_metrics = calculate_metrics_test(y_test, test_proba, opt_threshold)

        test_result = {
            'model': model_name,
            'resampling': resampling,
            'threshold': opt_threshold,
            **test_metrics
        }
        test_results.append(test_result)

        key = f"{model_name}_{resampling}"
        final_models[key] = {
            'model': model,
            'threshold': opt_threshold,
            'test_proba': test_proba,
            'test_pred': test_pred
        }

    except Exception as e:
        print(f"  ERROR {model_name}+{resampling}: {str(e)}")
        continue

test_results_df = pd.DataFrame(test_results)
test_results_df = test_results_df.sort_values('f1', ascending=False)

print("\nTEST SET RESULTS (sorted by F1):")
display_cols = ['model', 'resampling', 'f1', 'pr_auc', 'recall', 'precision', 'threshold']
print(test_results_df[display_cols].to_string(index=False))

# ============================================================
# Top 3 Analysis
# ============================================================
print("\n" + "="*60)
print("TOP 3 TEST SET PERFORMERS (by F1):")
print("="*60)
top_3 = test_results_df.head(3)
for i, (_, row) in enumerate(top_3.iterrows()):
    print(f"\n#{i+1}: {row['model']} + {row['resampling']}")
    print(f"     F1: {row['f1']:.4f}")
    print(f"     PR-AUC: {row['pr_auc']:.4f}")
    print(f"     Recall: {row['recall']:.4f}, Precision: {row['precision']:.4f}")
    print(f"     Threshold: {row['threshold']:.4f}")

# ============================================================
# Confusion Matrices
# ============================================================
print("\n[7] Generating confusion matrices...")
top_3_keys = [f"{row['model']}_{row['resampling']}" for _, row in top_3.iterrows()]

for key in top_3_keys:
    if key in final_models:
        test_pred = final_models[key]['test_pred']
        cm = confusion_matrix(y_test, test_pred)
        print(f"\nConfusion Matrix - {key}:")
        print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
        print(f"  FN={cm[1,0]}, TP={cm[1,1]}")

# ============================================================
# SHAP Analysis
# ============================================================
print("\n[8] SHAP Analysis...")
shap_results = {}

for i, (_, row) in enumerate(top_3.head(2).iterrows()):  # Top 2 for SHAP
    key = f"{row['model']}_{row['resampling']}"
    if key not in final_models:
        continue

    model = final_models[key]['model']
    print(f"\n  Computing SHAP for: {key}")

    try:
        if 'LightGBM' in key or 'XGBoost' in key or 'RandomForest' in key:
            explainer = shap.TreeExplainer(model)
        else:
            background = shap.sample(X_train_final, 100, random_state=42)
            explainer = shap.LinearExplainer(model, background)

        shap_values = explainer.shap_values(X_test_final)

        if isinstance(shap_values, list):
            shap_values_display = shap_values[1]
        else:
            shap_values_display = shap_values

        # Get top features
        mean_abs_shap = np.abs(shap_values_display).mean(axis=0)
        top_features_idx = np.argsort(mean_abs_shap)[-10:][::-1]
        top_features = [(X_test_final.columns[i], mean_abs_shap[i]) for i in top_features_idx]

        shap_results[key] = {
            'top_features': top_features
        }

        print(f"    Top 5 features:")
        for feat, imp in top_features[:5]:
            print(f"      - {feat}: {imp:.4f}")

    except Exception as e:
        print(f"    SHAP Error: {str(e)}")

# ============================================================
# Save results
# ============================================================
print("\n[9] Saving results...")

# Save to JSON for report generation
def convert_shap_value(v):
    try:
        if isinstance(v, np.ndarray):
            if v.size == 1:
                return float(v.flatten()[0])
            else:
                return float(np.mean(v))
        elif hasattr(v, 'item'):
            return v.item()
        else:
            return float(v)
    except:
        return 0.0

report_data = {
    'feature_engineering': feature_engineering_log,
    'cv_results': results_df[display_cols].to_dict('records'),
    'test_results': test_results_df[display_cols].to_dict('records'),
    'top_3': top_3.to_dict('records'),
    'shap_results': {k: {'top_features': [(f, convert_shap_value(v)) for f, v in v['top_features']]}
                    for k, v in shap_results.items()},
    'train_samples': len(y_train),
    'test_samples': len(y_test),
    'defect_rate_train': float(y_train.mean()),
    'defect_rate_test': float(y_test.mean())
}

with open('analysis_results.json', 'w') as f:
    json.dump(report_data, f, indent=2, default=str)

print("\nResults saved to analysis_results.json")

# ============================================================
# Final Summary
# ============================================================
print("\n" + "="*80)
print("FINAL ANALYSIS SUMMARY")
print("="*80)

print("\n1. DATA OVERVIEW:")
print(f"   - Original features: {feature_engineering_log['original_features']}")
print(f"   - Final features after engineering: {feature_engineering_log['total_final_features']}")
print(f"   - Training samples: {len(y_train)}")
print(f"   - Test samples: {len(y_test)}")
print(f"   - Defect rate: {y_train.mean():.2%}")

print("\n2. BEST MODEL CONFIGURATION:")
best_result = test_results_df.iloc[0]
print(f"   - Model: {best_result['model']}")
print(f"   - Resampling: {best_result['resampling']}")
print(f"   - Optimal threshold: {best_result['threshold']:.3f}")

print("\n3. TEST SET PERFORMANCE (using threshold from CV):")
print(f"   - F1 Score: {best_result['f1']:.4f}")
print(f"   - PR-AUC: {best_result['pr_auc']:.4f}")
print(f"   - Recall: {best_result['recall']:.4f}")
print(f"   - Precision: {best_result['precision']:.4f}")

if shap_results:
    print("\n4. KEY INSIGHTS FROM SHAP:")
    first_shap = list(shap_results.values())[0]
    print(f"   Top 5 most important features:")
    for i, (feat, imp) in enumerate(first_shap['top_features'][:5], 1):
        print(f"   {i}. {feat} (importance: {imp:.4f})")

print("\n" + "="*80)
print("Analysis complete!")
print("="*80)
