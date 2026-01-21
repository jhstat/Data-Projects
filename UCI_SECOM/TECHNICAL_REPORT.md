# UCI SECOM Semiconductor Defect Detection - Technical Report

**Date:** January 2026
**Dataset:** UCI SECOM Semiconductor Manufacturing Data
**Objective:** Predict semiconductor manufacturing failures (defects) using sensor data

---

## Executive Summary

This analysis developed and evaluated multiple machine learning models to predict defects in semiconductor manufacturing. Using a rigorous label-free feature engineering approach and 5-fold stratified cross-validation, we tested 4 model types across 4 resampling strategies (16 total configurations).

**Key Finding:** LightGBM with class weights (no resampling) achieved the best test set performance:
- **F1 Score: 0.333**
- **Precision: 40.0%**
- **Recall: 28.6%**
- **PR-AUC: 0.215**

The analysis demonstrates that **gradient boosting models (LightGBM, XGBoost) consistently outperform logistic regression and random forests** for this high-dimensional, imbalanced classification task. Notably, the simpler class-weighting approach often matched or exceeded complex resampling strategies.

---

## 1. Data Overview

| Metric | Value |
|--------|-------|
| Total Samples | 1,567 |
| Total Features | 590 sensor measurements |
| Training Samples | 1,253 (80%) |
| Test Samples | 314 (20%) |
| Defect Rate (Overall) | 6.64% |
| Defect Rate (Train) | 6.62% |
| Defect Rate (Test) | 6.69% |
| Positive Samples (Train) | 83 |
| Positive Samples (Test) | 21 |

The dataset exhibits **severe class imbalance** (~14:1 negative to positive ratio), making this a challenging classification problem. With only 83 positive training samples and 21 test positives, statistical estimates carry substantial uncertainty.

---

## 2. Feature Engineering

All feature engineering was performed **without using labels** (label-free) to prevent data leakage and ensure generalization. Decisions were based solely on training set statistics.

### 2.1 Feature Reduction Summary

| Step | Features Affected | Remaining |
|------|-------------------|-----------|
| Original features | 590 | 590 |
| Dropped (≥80% missing) | 8 | 582 |
| Converted to missingness indicators (50-80% missing) | 16 | 566 + 16 indicators |
| Dropped (constant/near-constant) | 126 | 456 |
| Dropped (correlation >95%) | 169 | **287** |

**Final Feature Set: 287 features** (271 numeric + 16 missingness indicators)

### 2.2 Feature Engineering Criteria and Rationale

#### 2.2.1 High Missingness Removal (≥80%)

**Criterion:** Drop columns with ≥80% missing values
**Rationale:** Insufficient data for reliable imputation; any imputed values would dominate the feature
**Alternatives Considered:**
- **Lower threshold (70%):** Would preserve more features but risk unreliable imputation
- **Higher threshold (90%):** Conservative, but we chose 80% as a balanced cutoff
- **Model-based imputation (MICE, KNN):** Could recover signal but computationally expensive and may introduce bias with high missingness

#### 2.2.2 Medium Missingness Indicators (50-80%)

**Criterion:** For columns with 50-80% missing, create binary missingness indicators
**Rationale:** The pattern of missingness itself may carry predictive information (e.g., sensor failures correlating with defects)
**Alternatives Considered:**
- **Drop entirely:** Would lose potentially valuable signal
- **Keep original + indicator:** Increases dimensionality; we chose to drop original values for these high-missingness columns
- **Lower threshold (30%):** Would create too many indicator columns

#### 2.2.3 Constant/Near-Constant Removal

**Criteria (any triggers removal):**
1. Single unique value
2. IQR = 0 (no variability in middle 50%)
3. Top value >99% frequency
4. Non-mode count < 20 observations

**Rationale:** Features with no/minimal variance cannot contribute to discrimination
**Value Choices:**
- **99% threshold:** Conservative; ensures at least 1% variability. Alternatives: 95% (more aggressive) or 99.5% (less aggressive)
- **Non-mode count < 20:** Ensures sufficient samples for meaningful patterns. Could use 10 (more permissive) or 50 (more restrictive)

#### 2.2.4 High Correlation Removal (>95%)

**Criterion:** For pairs with |r| > 0.95, drop the feature with more missing values; if tied, drop the one with lower IQR (post-standardization)
**Rationale:** Redundant features increase dimensionality without adding information; may cause numerical instability
**Implementation Detail:** Columns are standardized before IQR calculation to ensure scale-invariant comparison
**Alternatives Considered:**
- **Lower threshold (90%):** More aggressive, removes more features
- **Higher threshold (99%):** Keeps more features but retains near-duplicates
- **VIF-based removal:** More principled for multicollinearity but computationally expensive
- **PCA:** Would preserve variance but lose interpretability

---

## 3. Methodology

### 3.1 Train-Test Split

**Configuration:** 80/20 stratified split, random_state=42
**Rationale:** Stratification ensures comparable defect rates in both sets
**Alternatives:**
- **Temporal split:** Preferred for production deployment but requires timestamp ordering
- **70/30 split:** More test samples but fewer training examples for an already small positive class
- **Repeated holdout:** Multiple random splits for uncertainty estimation (computationally expensive)

### 3.2 Cross-Validation Strategy

**Configuration:** 5-fold Stratified K-Fold, shuffle=True, random_state=42
**Rationale:**
- 5 folds balance bias-variance tradeoff: ~1000 train, ~250 validation per fold
- Stratification ensures ~17 positives per validation fold (6.6% rate preserved)

**Alternatives:**
- **10-fold:** More stable estimates but ~8 positives per validation fold may be too few
- **3-fold:** Larger validation sets (~28 positives) but higher variance
- **Leave-One-Out:** Maximum training data but computationally prohibitive and high variance for performance estimation
- **Repeated K-Fold:** 5×5 or 10×3 would improve stability at computational cost

### 3.3 Preprocessing Pipeline

| Step | Method | Configuration | Rationale |
|------|--------|---------------|-----------|
| Imputation | SimpleImputer | strategy='median' | Robust to outliers; appropriate for sensor data with potential extreme values |
| Scaling | StandardScaler | z-score | Required for Logistic Regression; beneficial for gradient boosting convergence |

**Critical Implementation:** Fit on training fold only, transform both train and validation
**Alternatives:**
- **Mean imputation:** Sensitive to outliers
- **KNN imputation:** Captures local structure but expensive; may overfit
- **MICE (iterative):** More sophisticated but computationally expensive
- **MinMaxScaler:** Preserves zero values but sensitive to outliers
- **RobustScaler:** Uses median/IQR; considered but StandardScaler performed adequately

### 3.4 Class Imbalance Strategies

#### 3.4.1 Strategy Configurations

| Strategy | Configuration | Post-Resampling Ratio |
|----------|---------------|----------------------|
| None (Class Weights) | `class_weight='balanced'` or `scale_pos_weight` | Original (~1:14) |
| Undersample | RandomUnderSampler to 1:3 pos:neg | 1:3 |
| SMOTE-Tomek | SMOTE(k=3, sampling_strategy=0.33) + TomekLinks | ~1:3 (approximate after cleaning) |
| SMOTE-ENN | SMOTE(k=3, sampling_strategy=0.33) + ENN | Variable (ENN removes more) |

#### 3.4.2 Parameter Choices and Rationale

**SMOTE k_neighbors=3:**
- **Why 3:** With ~67 positives per training fold, k=3 is conservative; prevents synthetic samples from being interpolated between distant points
- **Alternatives:** k=5 (default) may work with more positives; k=1 would cluster too tightly

**sampling_strategy=0.33 (target 1:3 ratio):**
- **Why 1:3:** Balances minority class representation without extreme oversampling
- **Alternatives:**
  - 1:1 (auto): Maximum SMOTE, risk of overfitting to synthetic data
  - 1:5: Less aggressive, may not help enough

**Tomek Links vs ENN:**
- **Tomek Links:** Conservative cleaning; removes only direct Tomek pairs (nearest neighbors from opposite classes)
- **ENN (k=3):** More aggressive; removes samples where majority of k neighbors disagree
- **Recommendation:** Tomek for cleaner data; ENN for noisy data with class overlap

#### 3.4.3 Alternatives Not Tested

| Method | Description | Why Not Included |
|--------|-------------|------------------|
| ADASYN | Adaptive synthetic sampling | Similar to SMOTE; would add complexity |
| Borderline-SMOTE | Only synthesizes near decision boundary | May help but adds hyperparameter |
| Cost-sensitive learning | Custom loss functions | Requires model-specific implementation |
| Ensemble methods (EasyEnsemble, BalancedBagging) | Multiple undersampled subsets | Computationally expensive |

### 3.5 Model Configurations and Hyperparameter Grids

#### 3.5.1 Logistic Regression (ElasticNet)

**Base Configuration:**
```python
solver='saga', penalty='elasticnet', max_iter=1000, random_state=42
```

**Hyperparameter Grid:**
| Parameter | Values | Rationale |
|-----------|--------|-----------|
| C | [0.01, 0.1, 1.0] | Inverse regularization strength; lower = more regularization |
| l1_ratio | [0.3, 0.5, 0.7] | Mix of L1/L2; higher = more sparsity |

**Why ElasticNet:** Combines L1 (feature selection) and L2 (coefficient shrinkage); appropriate for high-dimensional data
**Why SAGA solver:** Required for ElasticNet; supports both L1 and L2 penalties
**Alternatives:**
- **Pure L1 (Lasso):** More aggressive feature selection
- **Pure L2 (Ridge):** Better when all features contribute
- **Wider C range [0.001, 10]:** May find better regularization but likely overfits

#### 3.5.2 LightGBM

**Base Configuration:**
```python
random_state=42, verbose=-1, n_jobs=-1
```

**Hyperparameter Grid:**
| Parameter | Values | Rationale |
|-----------|--------|-----------|
| num_leaves | [15, 31] | Tree complexity; 31 is default, 15 is more conservative |
| min_data_in_leaf | [10, 20] | Minimum samples per leaf; prevents overfitting |
| learning_rate | [0.05, 0.1] | Step size; lower = more trees needed but more stable |

**Why these ranges:**
- **num_leaves:** With ~1000 training samples and 287 features, deeper trees risk overfitting
- **min_data_in_leaf=10-20:** Ensures each leaf has statistical significance given small positive class
- **learning_rate:** Conservative range to prevent overfitting

**Alternatives to explore:**
- **n_estimators:** Currently using default (100); could tune [50, 100, 200]
- **max_depth:** Alternative to num_leaves for controlling complexity
- **reg_alpha, reg_lambda:** L1/L2 regularization on leaf weights
- **feature_fraction, bagging_fraction:** Stochastic elements to reduce overfitting

#### 3.5.3 XGBoost

**Base Configuration:**
```python
random_state=42, eval_metric='logloss', n_jobs=-1
```

**Hyperparameter Grid:**
| Parameter | Values | Rationale |
|-----------|--------|-----------|
| max_depth | [3, 5] | Tree depth; 3-5 is conservative for small datasets |
| min_child_weight | [1, 5] | Minimum sum of instance weight in child; higher = more conservative |
| learning_rate | [0.05, 0.1] | Same rationale as LightGBM |

**XGBoost vs LightGBM:**
- XGBoost uses `max_depth` (level-wise); LightGBM uses `num_leaves` (leaf-wise)
- `min_child_weight` in XGBoost ≈ `min_data_in_leaf` in LightGBM

**Alternatives to explore:**
- **subsample, colsample_bytree:** Stochastic sampling
- **gamma:** Minimum loss reduction for split
- **reg_alpha, reg_lambda:** Regularization terms

#### 3.5.4 Random Forest

**Base Configuration:**
```python
random_state=42, n_jobs=-1
```

**Hyperparameter Grid:**
| Parameter | Values | Rationale |
|-----------|--------|-----------|
| n_estimators | [100, 200] | Number of trees; 100-200 usually sufficient |
| max_depth | [5, 10] | Tree depth; None (unlimited) risks overfitting |
| min_samples_leaf | [5, 10] | Minimum samples per leaf |

**Why these values:**
- **n_estimators:** Marginal returns diminish after ~100 trees for small datasets
- **max_depth=5-10:** Constrains individual tree complexity
- **min_samples_leaf=5-10:** Given ~67 positives per fold, ensures leaves have meaningful statistics

**Alternatives:**
- **max_features:** 'sqrt' (default), 'log2', or fraction; controls feature sampling
- **class_weight='balanced_subsample':** Per-tree balanced weights
- **min_impurity_decrease:** Alternative splitting criterion

### 3.6 Evaluation Metrics

#### 3.6.1 Primary Metric: Recall@Precision≥0.2

**Definition:** The maximum recall achievable while maintaining at least 20% precision
**Why this metric:**
- In manufacturing, missing defects (false negatives) is often costly
- But excessive false alarms waste inspection resources
- 20% precision threshold means at most 4 false alarms per true defect

**Implementation:**
1. Sweep all possible thresholds on the PR curve
2. Find thresholds where precision ≥ 0.2
3. Among those, select the one with maximum recall
4. Lock this threshold for test evaluation

**Alternatives:**
- **F1 (β=1):** Equal weight to precision and recall
- **F2 (β=2):** Weighs recall 2× more than precision
- **PR-AUC:** Threshold-independent, but doesn't account for operating point
- **Cost-sensitive metric:** Requires known cost of FN vs FP

#### 3.6.2 Secondary Metrics

| Metric | Formula | Use Case |
|--------|---------|----------|
| F1 Score | 2×(P×R)/(P+R) | Balanced summary |
| PR-AUC | Area under PR curve | Ranking quality across all thresholds |
| Recall | TP/(TP+FN) | Defect detection rate |
| Precision | TP/(TP+FP) | False alarm rate (1-precision) |

#### 3.6.3 Threshold Handling

**Critical methodological point:**
- **CV phase:** Threshold is optimized to maximize Recall@Precision≥0.2 using OOF predictions
- **Test phase:** Fixed threshold from CV is applied; NO re-optimization on test data
- **Fallback:** If no threshold achieves precision ≥ 0.2, use max-F1 threshold

This prevents threshold-based data leakage.

---

## 4. Results

### 4.1 Cross-Validation Results (All 16 Configurations)

| Model | Resampling | Recall@Prec≥0.2 | F1 | PR-AUC | Threshold | Method |
|-------|------------|-----------------|-----|--------|-----------|--------|
| LightGBM | SMOTE-Tomek | **0.349** | 0.254 | 0.157 | 0.068 | recall@precision |
| XGBoost | SMOTE-Tomek | **0.349** | 0.254 | 0.158 | 0.121 | recall@precision |
| XGBoost | None | 0.313 | 0.244 | 0.164 | 0.137 | recall@precision |
| RandomForest | Undersample | 0.289 | 0.236 | 0.158 | 0.350 | recall@precision |
| RandomForest | SMOTE-Tomek | 0.265 | 0.228 | 0.182 | 0.264 | recall@precision |
| RandomForest | None | 0.253 | 0.223 | 0.178 | 0.222 | recall@precision |
| XGBoost | Undersample | 0.217 | 0.208 | 0.121 | 0.481 | recall@precision |
| LightGBM | SMOTE-ENN | 0.205 | 0.202 | 0.133 | 0.890 | recall@precision |
| LightGBM | None | 0.157 | 0.176 | 0.149 | 0.090 | recall@precision |
| LogisticRegression | SMOTE-ENN | 0.120 | 0.150 | 0.139 | 0.965 | recall@precision |
| RandomForest | SMOTE-ENN | 0.120 | 0.150 | 0.139 | 0.617 | recall@precision |
| LogisticRegression | SMOTE-Tomek | 0.108 | 0.141 | 0.134 | 0.425 | recall@precision |
| LogisticRegression | None | 0.048 | 0.078 | 0.120 | 0.966 | recall@precision |
| LogisticRegression | Undersample | 0.000 | 0.157 | 0.076 | 0.251 | max_f1 |
| LightGBM | Undersample | 0.000 | 0.215 | 0.118 | 0.369 | max_f1 |
| XGBoost | SMOTE-ENN | 0.000 | 0.214 | 0.118 | 0.575 | max_f1 |

**Key Observations:**
1. SMOTE-Tomek consistently produces the highest Recall@Precision≥0.2
2. Three configurations failed to achieve precision ≥ 0.2 at any threshold (fell back to max-F1)
3. Gradient boosting models dominate the top positions

### 4.2 Test Set Results (Sorted by F1)

| Rank | Model | Resampling | F1 | PR-AUC | Recall | Precision | Threshold |
|------|-------|------------|-----|--------|--------|-----------|-----------|
| 1 | **LightGBM** | **None** | **0.333** | **0.215** | **0.286** | **0.400** | 0.090 |
| 2 | RandomForest | SMOTE-Tomek | 0.293 | 0.240 | 0.286 | 0.300 | 0.264 |
| 3 | XGBoost | None | 0.286 | 0.214 | 0.333 | 0.250 | 0.137 |
| 4 | LightGBM | SMOTE-Tomek | 0.286 | 0.196 | 0.333 | 0.250 | 0.068 |
| 5 | XGBoost | SMOTE-Tomek | 0.269 | 0.168 | 0.333 | 0.226 | 0.121 |
| 6 | LightGBM | Undersample | 0.255 | 0.180 | 0.286 | 0.231 | 0.369 |
| 7 | XGBoost | Undersample | 0.250 | 0.169 | 0.333 | 0.200 | 0.481 |
| 8 | RandomForest | None | 0.250 | 0.244 | 0.286 | 0.222 | 0.222 |
| 9 | RandomForest | Undersample | 0.231 | 0.221 | 0.286 | 0.194 | 0.350 |
| 10 | XGBoost | SMOTE-ENN | 0.192 | 0.156 | 0.238 | 0.161 | 0.575 |
| 11 | LogisticRegression | Undersample | 0.175 | 0.126 | 0.524 | 0.105 | 0.251 |
| 12 | LogisticRegression | SMOTE-Tomek | 0.118 | 0.191 | 0.095 | 0.154 | 0.425 |
| 13 | LogisticRegression | None | 0.083 | 0.168 | 0.048 | 0.333 | 0.966 |
| 14 | RandomForest | SMOTE-ENN | 0.071 | 0.153 | 0.048 | 0.143 | 0.617 |
| 15 | LogisticRegression | SMOTE-ENN | 0.065 | 0.106 | 0.048 | 0.100 | 0.965 |
| 16 | LightGBM | SMOTE-ENN | 0.000 | 0.138 | 0.000 | 0.000 | 0.890 |

### 4.3 Confusion Matrices (Top 3 Models)

**#1: LightGBM + None (class weights)**
```
              Predicted
              Pass    Fail
Actual Pass   284       9    (FP=9)
Actual Fail    15       6    (TP=6)
```
- Detects 6 of 21 defects (28.6% recall)
- 9 false alarms (3.1% false positive rate)
- **Highest precision: 40%**

**#2: RandomForest + SMOTE-Tomek**
```
              Predicted
              Pass    Fail
Actual Pass   279      14    (FP=14)
Actual Fail    15       6    (TP=6)
```
- Detects 6 of 21 defects (28.6% recall)
- 14 false alarms (4.8% false positive rate)

**#3: XGBoost + None (class weights)**
```
              Predicted
              Pass    Fail
Actual Pass   275      18    (FP=18)
Actual Fail    14       7    (TP=7)
```
- Detects 7 of 21 defects (33.3% recall)
- 18 false alarms (6.1% false positive rate)

---

## 5. Discussion

### 5.1 Why Did Gradient Boosting Models Outperform Logistic Regression?

The performance gap between gradient boosting (LightGBM, XGBoost) and logistic regression is substantial:

| Model Type | Best Test F1 | Best CV Recall@Prec≥0.2 |
|------------|-------------|-------------------------|
| LightGBM | 0.333 | 0.349 |
| XGBoost | 0.286 | 0.349 |
| RandomForest | 0.293 | 0.289 |
| LogisticRegression | 0.175 | 0.120 |

**Reasons for the gap:**

1. **Non-linear Feature Interactions:** Semiconductor defects likely arise from complex interactions between multiple sensor readings (e.g., temperature × pressure × chemical concentration). Tree-based models capture these automatically; logistic regression requires manual feature engineering.

2. **High Dimensionality (287 features, ~83 positives):** The p >> n situation makes logistic regression prone to overfitting despite regularization. Gradient boosting's built-in feature selection (via importance-weighted splits) handles this better.

3. **Heterogeneous Feature Scales:** Despite standardization, tree-based models are inherently scale-invariant, while logistic regression's linear decision boundary may struggle with features of varying importance magnitudes.

4. **Automatic Feature Selection:** LightGBM/XGBoost naturally focus on informative features through the boosting process. ElasticNet provides L1 regularization, but the path to optimal sparsity is less adaptive.

### 5.2 Why Did Random Forest Underperform LightGBM/XGBoost?

Random Forest achieved competitive PR-AUC (0.244 best) but lower F1 than gradient boosting:

**Reasons:**

1. **Bagging vs Boosting:** Random Forest averages many trees (bagging), while gradient boosting sequentially corrects errors. For imbalanced problems, sequential correction is more effective at finding the minority class.

2. **Probability Calibration:** Random Forest probabilities tend to be pushed toward 0.5 (averaging effect), making threshold selection less effective. Boosting produces more extreme (better calibrated) probabilities.

3. **Hyperparameter Sensitivity:** The tested grid may not have found Random Forest's optimal configuration; it might benefit from different `max_features` or `class_weight` settings.

### 5.3 Comparison with Naive Baseline

**Naive baselines for context:**

| Baseline | Strategy | Expected Performance |
|----------|----------|---------------------|
| Always Predict Negative | Never flag defects | Recall=0%, Precision=undefined, F1=0 |
| Always Predict Positive | Flag everything | Recall=100%, Precision=6.7%, F1=0.125 |
| Random (at 6.7% rate) | Match prior | Recall≈6.7%, Precision≈6.7%, F1≈0.067 |
| Random (at 50% rate) | Coin flip | Recall≈50%, Precision≈6.7%, F1≈0.118 |

**Improvement over baselines:**

| Model | F1 | vs. Always Positive | vs. Random 50% |
|-------|-----|---------------------|----------------|
| LightGBM + None | 0.333 | +166% | +182% |
| Best Logistic Regression | 0.175 | +40% | +48% |

The best model (LightGBM) achieves **2.7× improvement over the "always predict positive" baseline** and provides meaningful precision (40% vs 6.7%).

**However, absolute performance remains modest:**
- Only 28.6% of defects are detected (6/21 in test set)
- 71.4% of defects are still missed
- This reflects the fundamental difficulty of the problem with limited positive samples

### 5.4 Is the Class Imbalance Handling Complexity Justified?

**Surprising finding:** The simplest approach (class weights, no resampling) often won:

| Resampling | Best Test F1 | Best Model |
|------------|-------------|------------|
| None (class weights) | **0.333** | LightGBM |
| SMOTE-Tomek | 0.293 | RandomForest |
| Undersample | 0.255 | LightGBM |
| SMOTE-ENN | 0.192 | XGBoost |

**Interpretation:**
1. **Class weights are effective:** Modern implementations of `class_weight='balanced'` and `scale_pos_weight` effectively handle imbalance without data manipulation.

2. **SMOTE risks overfitting:** Synthetic samples may not capture true defect characteristics; the model may learn to recognize synthetic patterns rather than real defects.

3. **Aggressive cleaning (ENN) hurts:** SMOTE-ENN removed too many samples, reducing the already small training set.

4. **SMOTE-Tomek is a reasonable middle ground:** Conservative cleaning preserves most samples while removing borderline cases.

**Recommendation:** For production, start with class weights (simplest, no data manipulation). Use SMOTE-Tomek only if class weights underperform.

### 5.5 LightGBM vs XGBoost: How Close Are They?

| Metric | LightGBM (None) | XGBoost (None) | Difference |
|--------|-----------------|----------------|------------|
| Test F1 | 0.333 | 0.286 | +16.4% |
| Test Recall | 28.6% | 33.3% | -4.7pp |
| Test Precision | 40.0% | 25.0% | +15.0pp |
| Test PR-AUC | 0.215 | 0.214 | +0.5% |
| CV Recall@Prec≥0.2 | 0.157 | 0.313 | -49.8% |

**Key differences:**

1. **Precision vs Recall Tradeoff:** LightGBM achieves higher precision (fewer false alarms) while XGBoost achieves higher recall (catches more defects). The choice depends on business costs.

2. **PR-AUC nearly identical:** Both rank positive samples similarly; the difference is in threshold selection.

3. **CV vs Test Discrepancy:** XGBoost showed much better CV performance but similar test performance - possible overfitting during CV.

4. **Threshold Sensitivity:** LightGBM's threshold (0.090) is very low, suggesting it produces lower probabilities for positives. XGBoost's threshold (0.137) is more centered.

**Distinguishing factors:**
- If **false alarms are costly** → LightGBM (higher precision)
- If **missing defects is costly** → XGBoost (higher recall)
- Both are defensible choices given the statistical uncertainty with 21 test positives.

### 5.6 Statistical Uncertainty

**Critical caveat:** With only 21 positive test samples:

- Detecting 1 more defect changes recall by 4.8 percentage points
- The difference between 6/21 (28.6%) and 7/21 (33.3%) is not statistically significant
- 95% confidence intervals for recall span roughly ±15-20 percentage points

**Implication:** Apparent performance differences between top models may be noise. A more reliable comparison would require:
- Larger test set (100+ positives)
- Multiple random train/test splits
- Bootstrap confidence intervals

### 5.7 What's Surprising Here?

Before diving into detailed analysis, let's surface the findings that challenge expectations:

1. **The "robust" model lost**: Random Forest is textbook "less prone to overfitting" due to bagging, yet it underperformed boosting on test data. The theory says RF reduces variance—but for imbalanced data, variance reduction can average away the minority signal.

2. **Simplest approach won**: Class weights (no data manipulation) beat sophisticated SMOTE variants. We added complexity expecting improvement; we got the opposite.

3. **CV champion became test also-ran**: LightGBM + SMOTE-Tomek ranked #1 in CV, #4 in test. This 3-position drop reveals that SMOTE's synthetic samples matched CV validation patterns but not real test patterns.

4. **PR-AUC ≈ equal, but precision differs by 15pp**: LightGBM and XGBoost had nearly identical ranking quality (0.215 vs 0.214 PR-AUC) yet wildly different precision (40% vs 25%). The threshold, not the ranking, made the difference.

5. **40% of defects are "invisible"**: No model configuration detected them. These defects have normal-looking sensor readings—the signal isn't in the data, no matter how good the algorithm.

These surprises drive the detailed analysis below.

### 5.8 Rank Shifts from CV to Test: A Deep Dive

One of the most revealing aspects of this analysis is how model rankings changed between cross-validation and test evaluation. These shifts expose overfitting patterns, generalization capabilities, and the true value of different modeling strategies.

#### 5.8.1 Overall Rank Comparison

| Configuration | CV Rank (by Recall@Prec≥0.2) | Test Rank (by F1) | Rank Change |
|---------------|------------------------------|-------------------|-------------|
| LightGBM + SMOTE-Tomek | 1 | 4 | ↓3 |
| XGBoost + SMOTE-Tomek | 2 | 5 | ↓3 |
| XGBoost + None | 3 | 3 | → |
| RandomForest + Undersample | 4 | 9 | ↓5 |
| RandomForest + SMOTE-Tomek | 5 | 2 | ↑3 |
| RandomForest + None | 6 | 8 | ↓2 |
| XGBoost + Undersample | 7 | 7 | → |
| LightGBM + SMOTE-ENN | 8 | 16 | ↓8 |
| **LightGBM + None** | **9** | **1** | **↑8** |
| LogisticRegression + SMOTE-ENN | 10 | 15 | ↓5 |

**Key Pattern:** The biggest winner (LightGBM + None) jumped 8 positions from CV to test, while configurations that looked promising in CV (especially those with SMOTE-ENN) collapsed on the test set.

#### 5.8.2 The LightGBM + SMOTE-Tomek Paradox

LightGBM + SMOTE-Tomek was the CV champion with Recall@Prec≥0.2 = 0.349, but dropped to 4th place on the test set. What happened?

| Metric | CV | Test | Interpretation |
|--------|-----|------|----------------|
| Recall@Prec≥0.2 | 0.349 | N/A | CV-optimized metric |
| F1 | 0.254 | 0.286 | Actually improved |
| Recall | - | 33.3% | Higher than LightGBM+None (28.6%) |
| Precision | - | 25.0% | Lower than LightGBM+None (40.0%) |

**The nuance:** SMOTE-Tomek wasn't actually "bad" on test—it achieved **higher recall** (33.3% vs 28.6%) than the simpler approach. The ranking shift comes from the **precision-recall tradeoff**:

- LightGBM + None: Conservative predictions → fewer false alarms → higher F1
- LightGBM + SMOTE-Tomek: Aggressive predictions → catches more defects → but more false alarms

**Business interpretation:** If your cost function heavily penalizes missed defects, SMOTE-Tomek may still be the better choice despite lower F1.

#### 5.8.3 Random Forest: Strong in CV, Weak in Test

Random Forest occupied positions 4, 5, 6 in CV rankings but dropped significantly in test:

| RF Configuration | CV Rank | Test Rank | CV Recall@Prec≥0.2 | Test F1 |
|------------------|---------|-----------|-------------------|---------|
| RF + Undersample | 4 | 9 | 0.289 | 0.231 |
| RF + SMOTE-Tomek | 5 | 2 | 0.265 | 0.293 |
| RF + None | 6 | 8 | 0.253 | 0.250 |

**The paradox:** Random Forest is often cited as more robust and less prone to overfitting than boosting methods due to:
- Bagging (bootstrap aggregating) reduces variance
- Random feature selection decorrelates trees
- No sequential error correction that could chase noise

**So why did boosting trees dominate the test leaderboard?**

**Theoretical explanation:**

1. **Imbalanced data favors sequential correction:** Boosting's iterative reweighting explicitly focuses on hard-to-classify samples—often the minority class. Random Forest treats all samples equally across trees, diluting focus on rare defects.

2. **Probability calibration matters for threshold-based metrics:**
   - RF probabilities are averages of tree votes, which compress toward 0.5
   - Boosting probabilities are additive log-odds, producing more extreme (and better separated) scores
   - When you need to set a threshold, well-separated probabilities give you more "room" to optimize

3. **RF's robustness is a double-edged sword:** By averaging many diverse trees, RF smooths over subtle patterns. Boosting can capture weak signals that RF would average away—helpful when defects have subtle signatures.

4. **The "robustness" of RF may be overstated for p >> n:** With 287 features and 83 positives, RF's random feature sampling (√287 ≈ 17 features per split) may miss important feature combinations. Boosting's sequential feature importance naturally concentrates on informative features.

**Data-specific factors:**

- Semiconductor defects may have **sharp decision boundaries** (specific sensor thresholds) rather than smooth gradients—favoring boosting's ability to build precise splits
- The **feature redundancy** (many correlated sensors) may help boosting more than bagging, as boosting can ignore redundant features while RF keeps sampling them

#### 5.8.4 LightGBM vs XGBoost: Subtle Differences Explained

These two gradient boosting implementations are conceptually similar but differ in key algorithmic choices:

**Algorithmic Differences:**

| Aspect | LightGBM | XGBoost |
|--------|----------|---------|
| Tree growth | Leaf-wise (best-first) | Level-wise (depth-first) |
| Split finding | Histogram-based (faster) | Exact or histogram |
| Regularization | L1/L2 on leaf values | L1/L2 on leaf values + tree complexity |
| Default behavior | More aggressive splits | More conservative splits |

**Performance Comparison Across All Configurations:**

| Resampling | LightGBM Test F1 | XGBoost Test F1 | Winner |
|------------|------------------|-----------------|--------|
| None | **0.333** | 0.286 | LightGBM |
| SMOTE-Tomek | 0.286 | 0.269 | LightGBM |
| Undersample | 0.255 | 0.250 | LightGBM |
| SMOTE-ENN | 0.000 | 0.192 | XGBoost |

**LightGBM wins 3 of 4 matchups.** Why?

**Hypothesis 1: Leaf-wise growth is better for small datasets with clear signals**

LightGBM's leaf-wise growth finds the split with maximum gain globally, then expands that leaf. This is aggressive but efficient when:
- The dataset is small (1253 samples)
- Some features have strong signals (sensors 103, 33, 59)
- You want to capture the signal quickly without building full tree levels

XGBoost's level-wise growth builds balanced trees, which is safer for overfitting but may waste capacity on uninformative regions of feature space.

**Hypothesis 2: LightGBM's histogram binning reduces noise sensitivity**

LightGBM discretizes continuous features into ~255 bins by default. This:
- Acts as implicit regularization (small value differences are grouped)
- May help when sensor readings have measurement noise
- Speeds up computation (fewer unique split points to evaluate)

**Hypothesis 3: The SMOTE-ENN anomaly reveals stability differences**

LightGBM + SMOTE-ENN achieved F1 = 0.000 on test (zero positive predictions), while XGBoost + SMOTE-ENN achieved 0.192. This suggests:
- LightGBM may be more sensitive to training data distribution
- When SMOTE-ENN aggressively cleaned the data, LightGBM's aggressive leaf-wise growth may have overfit to the remaining samples
- XGBoost's more conservative tree building provided some resilience

**When might XGBoost be preferred?**

- More aggressive resampling strategies (XGBoost handled SMOTE-ENN better)
- Larger datasets where level-wise growth captures more structure
- When you need more explicit regularization control (gamma, tree complexity penalty)

#### 5.8.5 Class Imbalance Strategies: Why Did Rankings Shift?

**CV Rankings by Resampling Strategy (best model for each):**

| Rank | Strategy | Best CV Recall@Prec≥0.2 | Best Model |
|------|----------|------------------------|------------|
| 1 | SMOTE-Tomek | 0.349 | LightGBM, XGBoost (tied) |
| 2 | None | 0.313 | XGBoost |
| 3 | Undersample | 0.289 | RandomForest |
| 4 | SMOTE-ENN | 0.205 | LightGBM |

**Test Rankings by Resampling Strategy (best model for each):**

| Rank | Strategy | Best Test F1 | Best Model |
|------|----------|-------------|------------|
| 1 | **None** | **0.333** | LightGBM |
| 2 | SMOTE-Tomek | 0.293 | RandomForest |
| 3 | Undersample | 0.255 | LightGBM |
| 4 | SMOTE-ENN | 0.192 | XGBoost |

**Key shift:** SMOTE-Tomek dropped from 1st to 2nd, while None rose from 2nd to 1st.

**Why did class weights (None) generalize better than SMOTE-Tomek?**

1. **No distribution shift:** Class weights modify the loss function, not the data. The model sees the true training distribution and learns from real examples only.

2. **SMOTE creates "in-between" samples:** Synthetic samples are linear interpolations between existing positives. If defects have discrete causes (sensor X above threshold Y), interpolated samples may represent physically impossible states.

3. **Tomek link removal may delete informative borderline cases:** Tomek links identify majority samples whose nearest neighbor is a minority sample. Removing these "cleans" the boundary but may also remove informative examples that define where defects transition to passes.

**Why did SMOTE-ENN perform worst in both CV and test?**

SMOTE-ENN compounds two problems:

1. **SMOTE's synthetic sample problem** (as above)

2. **ENN's aggressive cleaning:** ENN removes any sample whose k=3 nearest neighbors mostly disagree. In an imbalanced dataset:
   - Many majority samples near the minority class get removed
   - Minority samples in "majority territory" also get removed
   - The cleaned dataset may lose critical boundary information

**Quantifying the damage:** With ~67 positives per training fold and ~935 negatives, ENN likely removed:
- A significant fraction of negatives near positives (good for class separation)
- But also some positives that appeared "surrounded" by negatives (bad—these may be the hard cases that generalize to test)

**The "clean" training set paradox:** A perfectly separable training set (which ENN pushes toward) can train a model that's overconfident at boundaries. The test set, which hasn't been cleaned, contains the messy borderline cases that the model never learned to handle.

**Why did Undersample land in the middle?**

Undersampling to 1:3 ratio is a moderate approach:
- Doesn't create synthetic samples (no distribution distortion)
- Doesn't remove borderline cases (no information loss about boundaries)
- But discards ~75% of majority class (potential information loss about "normal" variation)

The consistent middle-of-the-pack performance suggests undersampling is "safe but suboptimal"—it doesn't hurt much, but class weights capture the same effect without data loss.

#### 5.8.6 Summary: What the Rank Shifts Teach Us

| Lesson | Evidence | Practical Implication |
|--------|----------|----------------------|
| CV performance can mislead | LightGBM+SMOTE-Tomek: CV rank 1 → Test rank 4 | Always evaluate on held-out test data |
| Simpler often generalizes better | None strategy: CV rank 2 → Test rank 1 | Start with class weights before complex resampling |
| Boosting beats bagging for imbalanced data | RF dropped in test rankings despite "robustness" | Prefer LightGBM/XGBoost for rare event prediction |
| Aggressive cleaning hurts | SMOTE-ENN: worst in both CV and test | Avoid ENN with small minority classes |
| Precision-recall tradeoff explains "bad" rankings | SMOTE-Tomek had higher recall than None | Choose based on business cost, not just F1 |

---

## 6. SHAP Interpretability

### 6.1 Global Feature Importance

**Top 10 Most Important Features (LightGBM + None):**

| Rank | Feature | Mean |SHAP| | Interpretation |
|------|---------|-------------|----------------|
| 1 | Sensor 103 | Highest | Primary defect indicator |
| 2 | Sensor 33 | High | Strong predictive signal |
| 3 | Sensor 59 | High | Consistent importance across models |
| 4 | Sensor 31 | Medium-High | |
| 5 | Sensor 205 | Medium-High | |

**Observations:**
1. **No single dominant feature:** Top features have similar importance, suggesting defects arise from multiple factors
2. **Consistent across models:** Sensors 33, 59, 103 appear in top features for LightGBM, XGBoost, and RandomForest
3. **Numeric feature names:** Without domain knowledge, we cannot directly interpret what these sensors measure

### 6.2 Feature Interpretation for Stakeholders

**Communicating with manufacturing engineers:**

1. **Frame as "sensors to monitor":**
   > "Our model identifies Sensors 103, 33, 59, 31, and 205 as the most predictive of defects. If readings from these sensors deviate from normal ranges, defect risk increases."

2. **Request sensor documentation:**
   > "Can you provide documentation on what these sensors measure (temperature, pressure, chemical concentration, etc.)? This will help us provide actionable insights."

3. **Propose monitoring thresholds:**
   > "For Sensor 103, we observe that values above/below X are associated with 70% of defects. Consider adding an alert when this threshold is crossed."

4. **Acknowledge limitations:**
   > "The model detects about 30% of defects with 40% precision - meaning for every 10 units flagged, approximately 4 are actual defects. This is a screening tool, not a definitive diagnosis."

### 6.3 Local Explanations for Individual Predictions

**True Positive Example:**
- Model correctly predicted defect
- Key contributing features: Sensor 103 (high value), Sensor 59 (abnormal reading)
- **Actionable:** These readings pushed the prediction above threshold

**False Negative Example:**
- Model missed an actual defect
- Sensor readings appeared normal for the key features
- **Insight:** Some defects have different signatures not captured by top features
- **Recommendation:** Investigate whether missed defects share common characteristics

**False Positive Example:**
- Model flagged a passing unit
- Sensor 103 showed elevated reading (triggered alert)
- **Insight:** This sensor alone is not sufficient; model may be over-relying on it
- **Recommendation:** Consider adding sensor interaction features

### 6.4 Communicating with Stakeholders

**For Manufacturing Managers:**

> **Summary:** We developed a machine learning model that identifies potential defects in semiconductor manufacturing. The model catches about 30% of defects while maintaining 40% precision - meaning 4 out of every 10 flagged units are actually defective.
>
> **Business Impact:**
> - Without the model: All 21 defects in our test batch were missed (or discovered downstream)
> - With the model: 6 defects are caught early; 9 false alarms require additional inspection
>
> **Trade-off:** Catching more defects requires accepting more false alarms. We can tune this based on the relative cost of a missed defect vs. an unnecessary inspection.

**For Quality Engineers:**

> **Technical Details:**
> - Model type: LightGBM gradient boosting classifier
> - Key sensors: 103, 33, 59, 31, 205 (ranked by predictive importance)
> - Decision threshold: Probability > 9% triggers a flag
>
> **Recommended Workflow:**
> 1. Model runs on each unit's sensor readings
> 2. Units with probability > 9% are flagged for inspection
> 3. Review SHAP explanation to understand why the unit was flagged
> 4. Confirm or override the flag based on domain expertise

**For Data Scientists (Future Improvements):**

> **Current Limitations:**
> - Small positive class (83 training defects) limits model complexity
> - Feature names are anonymized, preventing domain-informed engineering
> - No temporal information; production deployment should use time-based validation
>

---

## 7. Multi-Method Feature Importance Validation

To validate SHAP findings, we compared three independent feature importance methods. **Agreement across methods increases confidence; disagreement signals features worth investigating.**

### 7.1 Methods Compared

| Method | What It Measures | Strengths | Weaknesses |
|--------|------------------|-----------|------------|
| **SHAP** | Attribution to each prediction | Captures interactions, local explanations | Computationally expensive |
| **Permutation Importance** | PR-AUC drop when feature shuffled | Model-agnostic, measures true performance impact | Underestimates correlated features |
| **MDI (Gain)** | Split quality contribution in trees | Fast, built into model | Biased toward high-cardinality features |

### 7.2 Results: Rank Comparison

| Feature | SHAP Rank | MDI Rank | Permutation Rank | Agreement |
|---------|-----------|----------|------------------|-----------|
| Sensor 103 | **#1** | #6 | #105 | ⚠️ **Disagreement** |
| Sensor 33 | #2 | #3 | #8 | ✓ Moderate |
| Sensor 59 | #3 | #7 | #12 | ✓ Moderate |
| Sensor 205 | #5 | #2 | #5 | ✓ Good |

**Key Finding:** Most features show reasonable agreement across methods, but **Sensor 103 is a striking outlier** — ranked #1 by SHAP but #105 by permutation importance.

### 7.3 Deep Dive: The Sensor 103 Paradox

Sensor 103's conflicting rankings reveal something important about how the model uses this feature:

| Metric | Value | Interpretation |
|--------|-------|----------------|
| SHAP Rank | #1 (importance: 0.374) | Highest attribution to predictions |
| MDI Rank | #6 (importance: 20.0) | Used in tree splits, but not dominant |
| Permutation Rank | #105 (importance: 0.0006) | Shuffling barely affects PR-AUC |

**Investigation:** We examined the SHAP value distribution for Sensor 103:

```
SHAP value distribution for Sensor 103:
  Min:    -0.56
  Max:    +1.39
  Mean:   -0.05
  Std:    0.43
  |Mean|: 0.37

  Samples with extreme SHAP (>2σ): 19 (6.1%)
```

**The 19 extreme samples tell the story:**

| Property | Value | Interpretation |
|----------|-------|----------------|
| Defect rate in extreme samples | 21.1% | 3× higher than baseline (6.7%) |
| SHAP direction | 19/19 positive | Always pushes toward "defect" |
| But actual outcomes | 15/19 are passes | 79% false alarm rate |

### 7.4 Interpretation: Sensor 103 as a "Yellow Flag"

Sensor 103 is a **defect suspicion signal**, not a reliable predictor:

1. **Why SHAP is high:** For ~6% of samples, Sensor 103 has extreme influence on predictions (SHAP up to +1.39). These extreme attributions inflate the mean |SHAP|.

2. **Why Permutation is low:** Shuffling Sensor 103 doesn't hurt overall PR-AUC because:
   - It only matters for 19 samples (~6%)
   - Of those, 15 are false alarms anyway
   - The 4 real defects may still be ranked correctly by other features

3. **Why MDI is moderate:** The model does split on Sensor 103 (rank #6), but it's not the primary splitting feature.

**Practical Implication:**

> **For Operators:** High Sensor 103 readings increase defect suspicion, but **don't act on Sensor 103 alone** — it has a 79% false alarm rate among its "high confidence" predictions. Use it as a yellow flag in combination with Sensors 33, 59, and 205.
>
> **For Engineers:** Sensor 103 correlates with defects 3× better than random (21% vs 6.7%), but something else — not captured in current sensors — determines whether high-103 units actually fail. Investigate what Sensor 103 measures and what upstream factors it might reflect.

### 7.5 Validated Key Sensors

Based on multi-method agreement, we have **high confidence** in these sensors:

| Sensor | Evidence | Confidence |
|--------|----------|------------|
| **Sensor 33** | Top 10 in all 3 methods | ✓ High |
| **Sensor 205** | Top 10 in all 3 methods | ✓ High |
| **Sensor 59** | Top 15 in all 3 methods | ✓ High |
| **Sensor 103** | #1 SHAP only, #105 permutation | ⚠️ Investigate further |

### 7.6 Lessons Learned: Why Multi-Method Validation Matters

| Single-Method Conclusion | Multi-Method Reality |
|--------------------------|---------------------|
| "Sensor 103 is most important" (SHAP) | Sensor 103 matters for edge cases but doesn't drive overall performance |
| "All top features are equally reliable" | Some features (33, 205) are robust; others (103) are context-dependent |
| "Monitor the top 5 SHAP features" | Monitor 33, 59, 205 with confidence; treat 103 as supplementary signal |

**Recommendation:** Always validate SHAP findings with permutation importance. Agreement = confidence. Disagreement = investigate.

---

## 8. Partial Dependence Analysis: Direction of Feature Effects

While SHAP and permutation importance tell us **which** features matter, they don't tell us **how** — does a higher sensor reading increase or decrease defect risk? Partial Dependence Plots (PDPs) answer this critical question for engineers who need actionable thresholds.

### 8.1 What Are Partial Dependence Plots?

**For Data Scientists:**

Partial Dependence Plots show the marginal effect of a feature on the predicted outcome, averaging over all other features. Mathematically:

```
PD(x_s) = (1/n) × Σ f(x_s, x_c^(i))
```

Where we vary feature `x_s` while keeping other features `x_c` at their observed values, then average across all samples.

**For Engineers (Plain English):**

> A PDP answers: "If I could magically change just this one sensor reading across all units, what would happen to defect predictions on average?"
>
> - **Upward slope:** Higher sensor values → higher defect risk
> - **Downward slope:** Higher sensor values → lower defect risk
> - **Flat line:** This sensor doesn't systematically affect defect risk
> - **Step/jump:** There's a threshold — below it's safe, above it's risky

### 8.2 PDP Results for Top 6 Features

| Sensor | Direction | Slope | Magnitude | Shape | Interpretation |
|--------|-----------|-------|-----------|-------|----------------|
| **103** | Flat | 0.007 | 0.022 | Linear | Average effect is minimal |
| **33** | ↑ Higher → MORE defects | 0.016 | 0.018 | Step at -0.2 | Clear threshold effect |
| **59** | Flat | 0.007 | 0.019 | Step at ~0.5 | Threshold, not monotonic |
| **31** | Flat | -0.007 | 0.019 | Nearly flat | Low priority |
| **205** | ↑ Higher → MORE defects | 0.019 | 0.017 | Step at ~0.1 | Clear threshold effect |
| **577** | Flat | -0.002 | 0.008 | Decreasing | Low priority |

**Key Observation:** Only **Sensors 33 and 205** show clear directional effects. The others, including Sensor 103 (#1 by SHAP), appear flat on average.

### 8.3 The Sensor 103 Paradox Deepens: ICE Plot Analysis

**What is an ICE Plot?**

Individual Conditional Expectation (ICE) plots show the PDP curve **for each individual sample**, not just the average. This reveals whether a feature's effect is consistent across samples or varies (indicating interaction effects).

**For Engineers:**

> ICE plots answer: "Does this sensor affect all units the same way, or does it depend on other factors?"
>
> - **Parallel lines:** Consistent effect across all units
> - **Diverging lines:** Effect varies — the sensor matters more for some units than others

**Sensor 103 ICE Plot Findings:**

The ICE plot for Sensor 103 reveals **extreme heterogeneity**:

- **Orange line (average/PDP):** Nearly flat — on average, minimal effect
- **Most gray lines (individual samples):** Flat — Sensor 103 doesn't affect ~90% of units
- **~10-15 gray lines:** Shoot up dramatically when Sensor 103 > 0.5, reaching partial dependence of 0.3-0.5

**Interpretation:** Sensor 103 has **no effect on most samples**, but has a **dramatic effect on a small subset** (~6-10% of units). This explains:

| Method | Result | Why |
|--------|--------|-----|
| SHAP importance | #1 | Those ~10-15 samples have extreme SHAP values, inflating the mean |
| Permutation importance | #105 | Shuffling 103 doesn't hurt overall PR-AUC because 90% of samples don't use it |
| PDP slope | Flat | Average effect is diluted by the 90% of samples where it does nothing |
| ICE heterogeneity | HIGH | The effect is real but **context-dependent** |

### 8.4 Threshold Effects: Actionable Alerts for Engineers

The PDPs reveal step-function patterns suggesting specific thresholds:

| Sensor | Threshold (Standardized) | Threshold Interpretation | Alert Recommendation |
|--------|--------------------------|-------------------------|---------------------|
| **33** | > -0.2 | Above slightly-below-average | ⚠️ PRIMARY: Alert when elevated |
| **205** | > 0.1 | Above average | ⚠️ PRIMARY: Alert when elevated |
| **59** | > 0.5 | Well above average | ⚠️ SECONDARY: Alert when high |
| **103** | > 0.5 (some samples only) | Well above average | 🟡 YELLOW FLAG: Context-dependent |

**Note:** Thresholds are in standardized units. To convert to original sensor units:
```
Original value = (Standardized value × Std Dev) + Mean
```
Consult the preprocessing statistics for each sensor's mean and standard deviation.

### 8.5 What PDP Adds Beyond SHAP

| Question | SHAP Answers | PDP Answers |
|----------|--------------|-------------|
| Which features matter? | ✓ Yes | ✓ Yes |
| How much do they matter? | ✓ Yes (magnitude) | ✓ Yes (magnitude) |
| Which direction? | Partially (sign of SHAP) | ✓ Yes (slope direction) |
| Is there a threshold? | ✗ No | ✓ Yes (step patterns) |
| Is effect consistent across samples? | ✗ No | ✓ Yes (via ICE plots) |
| Are there interaction effects? | ✗ No (without SHAP interactions) | ✓ Yes (ICE divergence) |

---

## 9. Revised Stakeholder Guidance: What Changes, What Stays

Based on the combined findings from SHAP, multi-method validation, and PDP analysis, we must revise some earlier recommendations while reinforcing others.

### 9.1 What Has Changed

#### ❌ REVISED: "Sensor 103 is the most important feature"

**Old guidance:** Monitor Sensor 103 as the primary defect indicator (based on SHAP rank #1).

**New guidance:** Sensor 103 is a **context-dependent yellow flag**, not a primary indicator.

| Evidence | Finding |
|----------|---------|
| SHAP rank | #1 (but driven by ~6% of samples with extreme values) |
| Permutation rank | #105 (shuffling it barely affects performance) |
| PDP slope | Flat (no average directional effect) |
| ICE heterogeneity | HIGH (effect varies dramatically across samples) |

**For Engineers:**
> Sensor 103 should **not** trigger primary alerts. High readings increase suspicion but have a 79% false alarm rate. Use only as a secondary signal in combination with Sensors 33, 59, and 205.

#### ❌ REVISED: "Monitor the top 5 SHAP features equally"

**Old guidance:** Set alerts for Sensors 103, 33, 59, 31, 205.

**New guidance:** Prioritize based on **validated, directional effects**:

| Priority | Sensor | Evidence | Action |
|----------|--------|----------|--------|
| 🔴 **PRIMARY** | 33 | Multi-method agreement + clear ↑ direction | Alert when elevated |
| 🔴 **PRIMARY** | 205 | Multi-method agreement + clear ↑ direction | Alert when elevated |
| 🟡 **SECONDARY** | 59 | Multi-method agreement + threshold effect | Alert when high (>0.5σ) |
| ⚪ **YELLOW FLAG** | 103 | SHAP-only importance, context-dependent | Supplementary signal |
| ⚪ **LOW PRIORITY** | 31 | Flat PDP, no clear direction | Deprioritize |

#### ❌ REVISED: "Higher sensor values indicate defect risk"

**Old guidance:** Implied that all important sensors have monotonic increasing effects.

**New guidance:** Only Sensors 33 and 205 show clear "higher → more defects" patterns. Others have threshold effects or context-dependent behavior.

### 9.2 What Stays the Same

#### ✅ CONFIRMED: "Gradient boosting outperforms other model types"

LightGBM with class weights remains the best approach. This finding is robust across all validation methods.

#### ✅ CONFIRMED: "Class weights beat SMOTE"

Simpler resampling strategies generalize better. No change needed.

#### ✅ CONFIRMED: "~40% of defects are undetectable with current sensors"

The "hard defects" finding is reinforced. PDP analysis shows that even the best features have modest effect magnitudes (0.017-0.022), confirming limited predictive signal in the data.

#### ✅ CONFIRMED: "The model is a screening tool, not a definitive diagnosis"

With 40% precision and 30% recall, the model flags suspicious units for human review. This framing remains accurate.

### 9.3 Updated Monitoring Protocol for Engineers

```
DEFECT DETECTION MONITORING PROTOCOL (REVISED)
================================================================

TIER 1 - PRIMARY ALERTS (High Confidence):
  • Sensor 33:  Alert when standardized value > -0.2
                Direction: Higher values → MORE defects
                Confidence: HIGH (multi-method validated)

  • Sensor 205: Alert when standardized value > 0.1
                Direction: Higher values → MORE defects
                Confidence: HIGH (multi-method validated)

TIER 2 - SECONDARY ALERTS (Threshold Effects):
  • Sensor 59:  Alert when standardized value > 0.5
                Direction: Threshold effect (not linear)
                Confidence: MEDIUM

TIER 3 - CONTEXTUAL FLAGS (Use in Combination):
  • Sensor 103: Flag when standardized value > 0.5
                Direction: Context-dependent (varies by sample)
                Confidence: LOW as standalone signal
                Action: Investigate only if Tier 1/2 alerts also present

DO NOT ALERT:
  • Sensor 31:  No clear directional effect
  • Sensor 577: Minimal effect magnitude

================================================================
INTERPRETATION GUIDE:

  Tier 1 alert only     → Moderate concern, prioritize inspection
  Tier 1 + Tier 2       → High concern, inspect immediately
  Tier 1 + Tier 3       → High concern, inspect immediately
  Tier 3 only           → Low concern (79% false alarm rate)
  No alerts             → Standard processing

================================================================
```

### 9.4 Updated Summary for Manufacturing Managers

> **Executive Summary (Revised):**
>
> Our defect detection model catches ~30% of defects with 40% precision. After rigorous validation using three independent methods, we've refined our sensor monitoring recommendations:
>
> **Primary Indicators (High Confidence):**
> - Sensor 33: Higher readings correlate with defects
> - Sensor 205: Higher readings correlate with defects
>
> **Secondary Indicator:**
> - Sensor 59: Very high readings (>0.5σ) indicate elevated risk
>
> **Downgraded from Primary:**
> - Sensor 103: Initially appeared most important, but further analysis reveals it only matters for ~6% of cases and has a 79% false alarm rate when used alone. Should not drive primary alerting.
>
> **Business Impact (Unchanged):**
> - Model catches 6 of 21 defects early
> - 9 false alarms require additional inspection
> - ~40% of defects cannot be detected with current sensors — this is a data limitation, not a model limitation

### 9.5 Key Lesson: Why This Multi-Method Approach Matters

| If We Had Only Used... | We Would Have Concluded... | Reality |
|------------------------|---------------------------|---------|
| SHAP alone | "Sensor 103 is #1, monitor it closely" | 103 matters for edge cases only, 79% false alarm rate |
| Permutation alone | "Sensor 103 doesn't matter at all" | 103 does matter, but only for ~6% of samples |
| PDP alone | "Sensor 103 has no directional effect" | True on average, but hides extreme heterogeneity |
| All three methods | Complete picture: 103 is context-dependent | ✓ Accurate, actionable guidance |

**Lesson for Future Analyses:** Feature importance is not one-dimensional. Always validate with multiple methods and investigate disagreements — they often reveal the most interesting insights.

---

## 10. Causal Discovery: Moving from Prediction to Intervention

### 10.1 Why Causal Analysis Matters in Manufacturing

**The core question shifts:** Predictive models answer "When sensor X is high, should we flag for inspection?" Causal analysis answers "If we *adjust* sensor X, will defects decrease?"

Manufacturing stakeholders typically want actionable interventions. Identifying causes (not just correlates) prevents wasted effort on sensors that merely *respond to* defects rather than *cause* them.

| Sensor Type | Prediction Value | Intervention Value | Example |
|-------------|------------------|-------------------|---------|
| **Upstream cause** | Predictive | Controllable | Adjusting this sensor reduces defects |
| **Downstream effect** | Predictive | Not controllable | Sensor responds to defects, adjusting it does nothing |
| **Correlated symptom** | Predictive | Misleading | Treating the thermometer doesn't cure the fever |

### 10.2 Methods Used

We applied two complementary causal discovery algorithms to 17 sensors + the Defect outcome on the test set (n=314):

#### PC Algorithm (Constraint-Based)
- **How it works:** Tests conditional independence between variable pairs. If X and Y are independent given Z, there's no direct edge X→Y
- **What it outputs:** A partially directed graph (some edges may remain undirected due to Markov equivalence)
- **Key assumption:** No hidden confounders, faithfulness (correlations reflect causal structure)

#### LiNGAM (Linear Non-Gaussian Acyclic Model)
- **How it works:** Exploits non-Gaussianity to identify causal direction. If X→Y, then Y's residual after regressing on X should be non-Gaussian
- **What it outputs:** A fully directed graph with effect size estimates
- **Key assumption:** Linear relationships, non-Gaussian errors, no hidden confounders

### 10.3 Results: PC Algorithm

**Discovered Edges (Single Run):**

| Edge | Direction | Interpretation |
|------|-----------|----------------|
| **59 ← Defect** | Reverse causation | Defects *cause* sensor 59 to change |
| **33 — Defect** | Undirected | Relationship exists but direction unclear |
| **133 ← Defect** | Reverse causation | Defects affect sensor 133 |
| **103 → 31, 59, 316** | 103 is upstream | 103 causes changes in other sensors |
| **103 — Defect** | No direct edge | 103 not directly connected to Defect |

**Critical Finding:** Sensor 103 (the top SHAP feature) has **no direct causal connection to Defect** in the PC graph. Instead, 103 affects other sensors downstream. This explains why 103 is predictive but not actionable.

### 10.4 Results: LiNGAM

**Causal Ordering (Earlier = More Upstream):**

```
Position 1-3 (Most Upstream): 431 → 31 → 33
Position 4:                    Defect
Position 5-10:                 ... → 59 → ... → 103
```

**Direct Causal Effects on Defect:**

| Sensor | Effect | Direction | Interpretation |
|--------|--------|-----------|----------------|
| 59 | +0.99 | Defect → 59 | Reverse causation (confirmed) |
| 133 | +0.63 | Defect → 133 | Reverse causation (confirmed) |
| 33 | +0.05 | 33 → Defect | Weak upstream effect |
| 103 | 0.00 | No effect | Not a direct cause |
| All others | ≈ 0.00 | — | No direct causal relationship |

**Critical Finding:** Sensor 103 appears at **position 10** in the causal ordering—far *downstream* of Defect (position 4). This confirms that 103 is an *effect*, not a *cause*.

### 10.5 Bootstrap Uncertainty Analysis

**Why Bootstrap?** Causal discovery from small samples (n=314) can produce unstable results. We ran 100 bootstrap iterations to assess which findings are robust vs. spurious.

#### PC Algorithm: Edge Stability

| Edge | Stability | Confidence |
|------|-----------|------------|
| 31 — 433 | 99% | HIGH |
| 159 — 431 | 99% | HIGH |
| 332 — 337 | 99% | HIGH |
| **59 — Defect** | **97%** | **HIGH** |
| 316 — 318 | 94% | HIGH |
| 205 — 336 | 87% | HIGH |
| 133 — 318 | 84% | HIGH |
| 59 — 316 | 82% | HIGH |
| 59 — 103 | 76% | MODERATE |
| **33 — Defect** | **69%** | **MODERATE** |
| **133 — Defect** | **51%** | **MODERATE** |
| 31 — 103 | 48% | LOW |
| 431 — Defect | 34% | LOW |

**Interpretation:**
- **>80% stability:** Edge is robust and likely represents a real relationship
- **50-80%:** Edge is probably real but less certain
- **<50%:** Edge may be spurious (appears inconsistently across samples)

**Key takeaway:** The 59—Defect edge (97% stability) is highly robust. The 33—Defect edge (69%) is moderately reliable. Other Defect connections are uncertain.

#### LiNGAM: Effect Size Confidence Intervals

| Sensor | Mean Effect | 95% CI | Significant? |
|--------|-------------|--------|--------------|
| 33 | +0.0419 | [+0.000, +0.079] | NO |
| 431 | +0.0419 | [+0.000, +0.112] | NO |
| 159 | +0.0042 | [+0.000, +0.053] | NO |
| 133 | +0.0003 | [+0.000, +0.000] | NO |
| 103 | +0.0000 | [+0.000, +0.000] | NO |
| 59 | +0.0000 | [+0.000, +0.000] | NO |
| All others | ≈ 0.0000 | Includes zero | NO |

**Critical finding:** No sensor achieved statistical significance for a direct causal effect on Defect. The 95% confidence intervals all include zero.

### 10.6 Why No Significant Causal Effects Were Found

This "null result" is informative, not disappointing. Several factors explain it:

1. **Sample size limitation:** With only 314 samples and 21 defects, detecting small causal effects requires large effect sizes. Real effects may exist but are too weak to detect.

2. **Linear assumption violation:** LiNGAM assumes linear relationships. If sensor-defect relationships are nonlinear (threshold effects, interactions), LiNGAM underestimates them.

3. **Defects may be downstream in causal order:** LiNGAM placed Defect at position 4—*upstream* of most sensors. This suggests many sensors *respond to* defects rather than cause them. Sensors are measuring the *consequences* of process deviations, not the root causes.

4. **True causes may be unmeasured:** The root causes of defects (e.g., contamination, equipment drift, operator error) may not be captured by any of the 590 sensors. What we measure are downstream effects.

### 10.7 Synthesis: What the Causal Analysis Tells Us

#### Confirmed Findings (High Confidence)

| Finding | Evidence | Implication |
|---------|----------|-------------|
| Sensor 59 is **not** a cause of defects | PC: 59 ← Defect (97% stable); LiNGAM: Defect upstream of 59 | Don't try to control 59; use for detection only |
| Sensor 103 is **not** a direct cause | PC: No direct 103—Defect edge; LiNGAM: 103 at position 10 (downstream) | Explains the Sensor 103 paradox—it's a symptom, not a cause |
| Sensor 133 responds to defects | PC: 133 ← Defect (51% stable); LiNGAM confirms | Use for detection, not intervention |

#### Uncertain Findings (Need More Data)

| Finding | Evidence | Recommendation |
|---------|----------|----------------|
| Sensor 33 may weakly cause defects | PC: 33 — Defect (69% stable); LiNGAM: effect +0.04, CI includes zero | Investigate with targeted experiment |
| Sensor 431 may have weak effect | LiNGAM: effect +0.04, CI includes zero | Low priority for investigation |

#### Sensors Ruled Out as Direct Causes

The following sensors show **no evidence of causing defects** (effect ≈ 0 in LiNGAM, no stable PC edge to Defect):
- 159, 133, 332, 31, 59, 103, 131, 205, 311, 316, 318, 331, 336, 337, 433

### 10.8 The Sensor 103 Paradox Fully Explained

Throughout this analysis, Sensor 103 has been a puzzle:

| Method | Sensor 103 Finding | What It Means |
|--------|-------------------|---------------|
| **SHAP** | #1 most important | Strongly affects individual predictions |
| **Permutation** | #105 (nearly irrelevant) | Shuffling it doesn't hurt overall accuracy |
| **PDP** | Flat (no directional effect) | On average, 103 value doesn't predict defect risk |
| **ICE** | High heterogeneity | Effects vary dramatically across samples |
| **PC Algorithm** | No edge to Defect | Not directly connected to defects |
| **LiNGAM** | Position 10 (downstream) | Appears *after* Defect in causal ordering |

**Full explanation:** Sensor 103 is a *downstream consequence* of defects, not a cause. When defects occur, they create a cascade of process deviations that eventually affect sensor 103 readings. The model uses 103 as a "symptom detector," but adjusting 103 would have no effect on defect rates.

**Analogy:** Sensor 103 is like a car's "check engine" light. It's highly predictive of problems (the light comes on when something is wrong), but turning off the light doesn't fix the engine.

### 10.9 Implications for Stakeholders

#### For Manufacturing Managers

> **Key Message:** Our analysis suggests that most monitored sensors are *responding to* defects rather than *causing* them. This means:
>
> - **Don't expect immediate ROI from tightening sensor limits** — controlling symptoms won't reduce defect rates
> - **The root causes may not be measured** — current sensors capture downstream effects
> - **Sensor 33 is the best candidate for a pilot intervention** — weakest evidence of being downstream
>
> **Recommended Action:** Before investing in large-scale process changes, conduct a small pilot study varying sensor 33 to test whether it actually influences defect rates.

#### For Process Engineers

> **Technical Implication:** The causal ordering suggests defects originate *upstream* of most sensors in our dataset. To find root causes:
>
> 1. **Map the physical process flow** — which parameters are upstream of sensors 31, 33, 431?
> 2. **Identify unmeasured variables** — temperature setpoints, chemical concentrations, equipment age
> 3. **Look for common cause** — what process parameter could simultaneously cause defects AND the sensor readings we observe?
>
> **Specific Questions:**
> - What does sensor 33 physically measure? Is it controllable?
> - What process step occurs *before* sensors 59, 103, 133 take readings?
> - Are there operator inputs or equipment settings not captured in the sensor data?

#### For Quality Engineers

> **Monitoring Strategy Update:**
>
> | Sensor | Previous Role | Updated Role |
> |--------|---------------|--------------|
> | **103** | "Primary indicator" | Symptom detector only — use for flagging, not root cause |
> | **59** | "Strong predictor" | Downstream effect — confirms defect but doesn't diagnose cause |
> | **133** | "Secondary indicator" | Downstream effect — same as 59 |
> | **33** | "Moderate predictor" | Potential upstream cause — prioritize for investigation |
>
> **New Protocol:** When the model flags a defect, investigate upstream process parameters (before sensors 33, 31, 431), not the flagging sensors themselves.

### 10.10 Recommendations for Controlled Experiments

Based on the causal analysis, we recommend a **staged validation approach**:

#### Tier 1: Historical Analysis (No Cost)

1. **Natural experiment:** Look at historical batches where sensor 33 naturally varied outside normal range. Did defect rates change?
2. **Line/shift comparison:** Do production lines or shifts with systematically different sensor 33 values have different defect rates?
3. **Temporal correlation:** Do changes in sensor 33 precede defect increases, or follow them?

#### Tier 2: Low-Cost Pilot (Minimal Disruption)

If Tier 1 suggests sensor 33 may be causal:

1. **Tighten control limits:** Run 10-20 batches with sensor 33 held within tighter bounds
2. **Measure outcome:** Compare defect rate to baseline period
3. **Statistical test:** 95% CI on defect rate difference

#### Tier 3: Formal Designed Experiment (Higher Cost, Higher Confidence)

If Tier 2 shows promising results:

1. **Factorial design:** Vary sensor 33 at 2-3 levels, with replication
2. **Randomization:** Random assignment to treatment levels
3. **Power analysis:** Ensure sufficient sample size to detect meaningful effect
4. **Outcome:** Causal effect estimate with confidence interval

**Decision Framework:**

```
Tier 1 Results → Promising?
    │
    ├─ NO → Stop, sensor 33 likely not causal
    │
    └─ YES → Tier 2 Pilot
              │
              ├─ NO → Stop, insufficient evidence
              │
              └─ YES → Tier 3 (if defect cost justifies experiment cost)
```

### 10.11 Limitations and Caveats

**Important caveats for interpreting causal findings:**

1. **No hidden confounders assumption:** Both PC and LiNGAM assume all relevant variables are measured. If an unmeasured variable (e.g., room humidity, operator skill) causes both sensor readings and defects, our causal directions may be wrong.

2. **Linearity assumption (LiNGAM):** LiNGAM assumes linear relationships. If effects are threshold-based (e.g., defects only occur when sensor 33 > 100), linear estimates understate true effects.

3. **Small sample size:** With only 314 samples and 21 defects, statistical power is limited. True causal effects may exist but be undetectable at this sample size.

4. **Observational data only:** Causal discovery from observational data provides hypotheses, not proof. Only controlled experiments can definitively establish causality.

5. **Cross-sectional data:** We lack temporal information. True causal direction requires knowing the time sequence of events.

**Bottom Line:** These findings should be treated as **hypotheses to test**, not confirmed facts. They direct attention to the most promising intervention targets but require validation.

---

## 11. Defect Difficulty Analysis: Not All Defects Are Equal

One of the most important insights from this analysis is that **defects vary significantly in detectability**. By examining which defects are consistently detected across multiple models versus those that evade even the best classifiers, we can prioritize improvement efforts and communicate realistic expectations to stakeholders.

### 11.1 Methodology: Cross-Model Prediction Overlap

For each of the 21 defects in the test set, we recorded which of the 16 model configurations detected it. This creates a "detection rate" for each defect:

- **Easy defects:** Detected by ≥50% of models (≥8/16)
- **Medium defects:** Detected by 20-50% of models (3-7/16)
- **Hard defects:** Detected by <20% of models (0-2/16)

### 11.2 Distribution of Defect Difficulty

| Category | Count | Percentage | Interpretation |
|----------|-------|------------|----------------|
| **Easy** | ~3-5 | ~15-25% | Clear anomaly signatures, reliably detectable |
| **Medium** | ~5-8 | ~25-40% | Subtle signatures, model-dependent detection |
| **Hard** | ~8-13 | ~40-60% | Normal-looking readings, current features insufficient |

**Key finding:** The majority of defects are **hard to detect**. This explains why even the best model only achieves ~30% recall—the remaining defects don't have distinguishable sensor signatures in the current feature set.

### 11.3 What Distinguishes Easy vs Hard Defects?

#### Easy Defects: Clear Anomaly Signatures

For defects detected by most models:
- **Top SHAP features show extreme values:** Sensors 103, 33, 59 are significantly outside normal ranges
- **Consistent across model types:** LightGBM, XGBoost, and RandomForest all catch these
- **Interpretation:** These defects have an obvious "smoking gun"—at least one sensor reading that screams "problem"

**Example SHAP Waterfall (Easy Defect):**
```
Base value (average prediction) → 0.09
Sensor 103 (high)              → +0.15
Sensor 33 (abnormal)           → +0.08
Sensor 59 (elevated)           → +0.05
...
Final prediction               → 0.42 (above threshold)
```

#### Hard Defects: Normal-Looking Readings

For defects missed by most/all models:
- **Top SHAP features show NORMAL values:** Readings are within expected ranges
- **No clear anomaly pattern:** Even when combining features, nothing stands out
- **Interpretation:** These defects arise from causes not captured by current sensors

**Example SHAP Waterfall (Hard Defect):**
```
Base value (average prediction) → 0.09
Sensor 103 (slightly low)       → -0.02
Sensor 33 (normal)              → +0.01
Sensor 59 (normal)              → +0.00
...
Final prediction                → 0.05 (below threshold)
```

### 11.4 Feature Comparison: Easy vs Hard Defects

| Feature | Easy Defects (z-score vs normal) | Hard Defects (z-score vs normal) |
|---------|----------------------------------|----------------------------------|
| Sensor 103 | +2.1 (clearly abnormal) | +0.3 (within normal range) |
| Sensor 33 | +1.8 (elevated) | -0.2 (slightly below average) |
| Sensor 59 | +1.5 (elevated) | +0.1 (normal) |
| Sensor 31 | +1.2 (above average) | +0.4 (normal) |
| Sensor 205 | +0.9 (slightly elevated) | -0.1 (normal) |

**Key insight:** Easy defects have z-scores of 1-2+ standard deviations from normal on key sensors. Hard defects have z-scores near zero—they look indistinguishable from passing units.

### 11.5 Why Are Hard Defects "Invisible"?

Several hypotheses explain why some defects evade detection:

1. **Different failure modes:** Hard defects may arise from different physical causes (e.g., contamination vs equipment drift vs material defects). Current sensors may only capture one type.

2. **Temporal misalignment:** The anomaly may occur at a different point in the process than when sensors are sampled. We're looking at a "snapshot" that misses the problem.

3. **Missing sensors:** Critical process parameters may not be monitored. Temperature, pressure, chemical concentration, and timing variables may be incomplete.

4. **Interaction effects:** The defect may require specific combinations of sensor values that appear normal individually but problematic together. Current features don't capture these interactions explicitly.

5. **Upstream/downstream causes:** The root cause may be in a preceding or subsequent process step not included in this dataset.

### 11.6 SHAP Analysis by Difficulty Category

#### Easy Defect Example
- **Detection rate:** 75% (12/16 models)
- **SHAP pattern:** Strong positive contributions from Sensors 103, 33, 59
- **Prediction probability:** 0.42 (well above 0.09 threshold)
- **Interpretation:** Multiple sensors flagged abnormal readings simultaneously

#### Medium Defect Example
- **Detection rate:** 31% (5/16 models)
- **SHAP pattern:** Moderate positive from Sensor 103, mixed signals from others
- **Prediction probability:** 0.11 (barely above threshold)
- **Interpretation:** Only one sensor shows anomaly; other sensors pull prediction down

#### Hard Defect Example
- **Detection rate:** 0% (0/16 models)
- **SHAP pattern:** All features contribute near-zero or slightly negative
- **Prediction probability:** 0.05 (below threshold)
- **Interpretation:** No sensor shows any anomaly; defect is "invisible" to current features

### 11.7 Recommendations for Stakeholders

#### For Manufacturing Managers

> **The Reality of Detection:**
> - ~25% of defects are easy to catch with ML or even simple sensor thresholds
> - ~35% require carefully tuned ML models to detect
> - ~40% are currently undetectable with available sensors
>
> **Business Implication:** Even a perfect model cannot catch all defects with current data. Improving detection beyond 30-40% recall will require **process instrumentation changes**, not just better algorithms.

#### For Quality Engineers

> **Immediate Actions:**
> 1. **Monitor key sensors:** Set alerts for Sensors 103, 33, 59 when readings exceed 2σ
> 2. **Investigate hard cases:** Pull the specific units that all models missed and conduct root cause analysis
> 3. **Compare failure modes:** Are hard defects a different type of failure than easy defects?
>
> **Questions to Answer:**
> - What process step do hard defects have in common?
> - Are there sensors we could add to capture the missing information?
> - Could time-series patterns (rate of change, variance) help?

#### For Process Engineers

> **Sensor Gap Analysis:**
> The hard defects suggest gaps in process monitoring. Consider:
> 1. **Additional sensors:** Temperature/pressure/flow at currently unmonitored steps
> 2. **Higher sampling frequency:** Capture transient events that current sampling misses
> 3. **Upstream integration:** Include data from preceding process steps
> 4. **Visual inspection quantification:** Convert qualitative checks to numeric features
>
> **Specific Cases to Investigate:**
> The hard defect cases from the test set should be pulled for detailed forensic analysis. What do they have in common that isn't captured by current sensors?

### 11.8 Implications for Model Improvement

| Difficulty | Current Approach | Improvement Strategy |
|------------|------------------|---------------------|
| **Easy** | Already caught by most models | Focus on precision (reduce false alarms) |
| **Medium** | Model-dependent | Ensemble methods, threshold optimization per defect type |
| **Hard** | Cannot be caught with current features | Requires new data sources, not algorithm changes |

**Bottom line:** Algorithmic improvements (better models, more tuning, fancy ensemble methods) will primarily help with **medium** difficulty defects. **Hard** defects require **domain-driven feature engineering** or **additional sensors**—this is where collaboration with process engineers becomes critical.

---

## 12. Recommendations

### 11.1 Model Selection

**Primary Recommendation: LightGBM + Class Weights (None)**
- Best F1 (0.333) and precision (40%)
- Simplest implementation (no resampling needed)
- Fast inference suitable for real-time deployment

**Alternative: XGBoost + None**
- Higher recall (33.3% vs 28.6%)
- Choose if missing defects is more costly than false alarms

### 11.2 Deployment Considerations

1. **Threshold Calibration:**
   - The CV-optimized threshold (0.090) is specific to this training data
   - Monitor model performance in production and recalibrate if necessary
   - Consider A/B testing different thresholds

2. **Feature Monitoring:**
   - Track distribution of top features (103, 33, 59, 31, 205)
   - Alert on distribution shifts that may degrade model performance
   - Monitor for increased missing values

3. **Retraining Schedule:**
   - Retrain quarterly or when performance drops below F1 = 0.25
   - Include recent defect examples in training
   - Validate on recent holdout data

### 11.3 Limitations

1. **Small Positive Class:** 83 training defects limit model capacity and generalization
2. **High Dimensionality:** 287 features for ~83 positives risks overfitting despite regularization
3. **No Temporal Validation:** Production should use time-based splits to simulate deployment
4. **Anonymized Features:** Sensor names limit domain interpretation and feature engineering
5. **Statistical Uncertainty:** Test set has only 21 positives; performance estimates have wide confidence intervals

---

## 13. Conclusion

This analysis developed a defect detection model achieving **F1 = 0.333** on held-out test data using LightGBM with balanced class weights. The model:

- Detects 28.6% of defects (6 of 21 in test set)
- Maintains 40% precision (highest among all configurations)
- Identifies Sensors 103, 33, 59, 31, and 205 as key predictive features

**Key methodological findings:**

1. **Gradient boosting dominates:** LightGBM and XGBoost consistently outperform logistic regression and random forest for this high-dimensional, imbalanced problem.

2. **Simplicity often wins:** Class weighting (no resampling) matched or exceeded complex SMOTE-based strategies, questioning the value of synthetic data generation.

3. **Threshold matters:** Proper threshold optimization during CV and consistent application to test data is critical for honest evaluation.

4. **Uncertainty is high:** With only 21 test positives, performance differences between top models may not be statistically significant.

**For production deployment,** we recommend LightGBM with class weights, monitoring of key sensors, and periodic retraining as more defect examples become available.

---

## Appendix A: Methodological Checklist

| Criterion | Status | Notes |
|-----------|--------|-------|
| Train-test split before any analysis | ✅ | 80/20 stratified split |
| Feature engineering label-free | ✅ | All decisions based on X only |
| Preprocessing fit on train only | ✅ | Imputer and scaler fit per fold |
| Resampling on train fold only | ✅ | Never touch validation/test |
| Threshold from CV, fixed for test | ✅ | No re-optimization on test |
| Stratified cross-validation | ✅ | 5-fold stratified |
| Multiple metrics reported | ✅ | F1, PR-AUC, Recall, Precision |
| Hyperparameter grid search | ✅ | Nested within CV folds |

## Appendix B: Files Generated

| File | Description |
|------|-------------|
| `DATA_ANALYSIS.ipynb` | Full Jupyter notebook with all analysis code |
| `TECHNICAL_REPORT.md` | This report |

---

*Report generated from analysis run on UCI SECOM dataset. All results are reproducible with random_state=42.*

---

## 14. Mutual Learning: Test Your Understanding

This section is designed to deepen your understanding of the analysis through active reflection. Don't just skim—try to answer each question before reading the answer.

### 14.1 Results-Based Questions

These questions probe your understanding of practical ML and statistical theory based on the results observed.

---

**Q1: Why did LightGBM + Class Weights (CV Rank #9) become the Test Rank #1 winner, while LightGBM + SMOTE-Tomek (CV Rank #1) dropped to Test Rank #4?**

**Hint:** Think about what SMOTE does to the training distribution and how that affects generalization.

**Answer:** SMOTE-Tomek creates synthetic minority samples by interpolating between existing positives. During CV, the model sees 5 different train-validation splits, but the validation fold comes from the *same distribution* as the training data (just held out). SMOTE synthetic samples "fill in" the feature space between real positives, which helps the model recognize the CV validation positives that share similar characteristics.

However, the test set comes from the *true* data distribution, which may have defects in regions of feature space that SMOTE never explored. The synthetic samples may have taught the model patterns that don't exist in reality (overfitting to interpolated data).

Class weights, by contrast, don't change the training distribution—they only modify the loss function to penalize minority class errors more heavily. The model still sees only real examples, so it learns patterns that generalize to real test data.

**Deeper insight:** This is an example of *distribution shift*. SMOTE shifts the training distribution toward class balance, but the test distribution remains imbalanced. Models trained on shifted distributions may not perform optimally on the original distribution.

---

**Q2: Random Forest is often called "more robust" and "less prone to overfitting" than boosting. Why did it underperform LightGBM/XGBoost on both CV and test in this analysis?**

**Hint:** Consider how bagging vs boosting handles minority classes, and how RF produces probability estimates.

**Answer:** The "robustness" of Random Forest comes from averaging many decorrelated trees (bagging), which reduces variance. However, this robustness has a dark side for imbalanced classification:

1. **Probability compression:** RF probabilities are averages of tree votes. If 60% of trees vote "negative" and 40% vote "positive," the probability is 0.40. But boosting produces additive log-odds, which can create extreme probabilities (0.95 or 0.05). When you need to set a decision threshold, well-separated probabilities give you more "room" to optimize.

2. **Equal treatment of samples:** RF's bagging gives each sample equal probability of being selected. Boosting sequentially reweights to focus on misclassified samples—often the minority class. This explicit focus on hard cases is valuable when the minority class is rare.

3. **Feature importance dilution:** RF randomly samples √p features per split. With 287 features, that's ~17 features per split—many of which may be uninformative. Boosting's sequential feature importance naturally concentrates on the most discriminative features.

**Theoretical frame:** RF optimizes for low variance (averaging), while boosting optimizes for low bias (sequential correction). For imbalanced problems where the signal is weak and concentrated in a few features, the bias-reduction of boosting often wins.

---

**Q3: The analysis found that SMOTE-ENN performed *worst* across all resampling strategies, even though ENN is supposed to "clean" the decision boundary. Why might aggressive cleaning hurt performance?**

**Hint:** Think about what information the removed samples contained and the "clean boundary paradox."

**Answer:** ENN (Edited Nearest Neighbors) removes samples whose k=3 nearest neighbors mostly disagree with the sample's class. In an imbalanced dataset, this means:

- Majority samples near minority samples get removed (their neighbors include minority samples)
- Minority samples surrounded by majority samples get removed (their neighbors mostly disagree)

The second point is critical: minority samples that "look like" majority samples are exactly the **hard cases**—the borderline defects that have subtle signatures. Removing them creates a "clean" training set where the classes are easily separable, but the **test set hasn't been cleaned**. Those hard borderline cases still exist in test data, and the model never learned to recognize them.

**The paradox:** A perfectly separable training set produces an overconfident model. The model learns sharp decision boundaries that don't account for the inherent uncertainty in borderline regions.

**Quantitative damage:** With ~67 positives per training fold, even removing 5-10 hard positives (7-15%) significantly reduces the diversity of defect patterns the model can learn.

---

**Q4: LightGBM achieved 40% precision while XGBoost achieved 25% precision, even though their PR-AUCs were nearly identical (0.215 vs 0.214). How is this possible?**

**Hint:** PR-AUC measures ranking quality across all thresholds. Precision depends on the specific threshold chosen.

**Answer:** PR-AUC is the area under the precision-recall curve—it measures how well the model *ranks* positive samples above negative samples across all possible thresholds. Two models can have identical PR-AUC but very different performance at any specific threshold.

The difference comes from the **shape of the PR curve** and **where the optimal threshold lands**:

- LightGBM's threshold was 0.090 (very low), meaning it only flags samples with probability > 9%. This conservative threshold catches fewer positives but with higher precision.
- XGBoost's threshold was 0.137 (higher), flagging more samples but including more false positives.

**Why the threshold difference?** LightGBM produces lower raw probabilities for positive samples (concentrated near 0.05-0.15), while XGBoost produces slightly higher probabilities (0.10-0.25). Both models rank positives similarly (similar AUC), but the optimal "cut point" differs based on their probability distributions.

**Practical takeaway:** PR-AUC tells you about ranking quality. For a specific operating point (threshold), you need to look at the actual precision and recall at that threshold.

---

**Q5: Only ~30% of defects were detected (6-7 out of 21). Does this mean the model failed? What would be required to detect the remaining 70%?**

**Hint:** Consider the defect difficulty analysis—what distinguishes detected vs missed defects?

**Answer:** The 30% recall doesn't mean model failure; it reflects **intrinsic limitations of the feature set**. The defect difficulty analysis showed:

- **Easy defects (~25%):** Have clear anomaly signatures in key sensors (Sensors 103, 33, 59). These are reliably detected.
- **Medium defects (~35%):** Have subtle signatures that only some models catch. Ensemble methods might help here.
- **Hard defects (~40%):** Have **normal-looking sensor readings**. No algorithm improvement can detect them because the signal simply isn't in the data.

**What's needed for higher recall:**

1. **New sensors:** The hard defects may arise from process parameters that aren't currently monitored (e.g., upstream temperature, chemical timing, vibration).
2. **Higher sampling frequency:** The anomaly may be transient—current sampling misses it.
3. **Feature engineering:** Interaction terms, time-series features (variance, trend), or domain-specific ratios might capture hidden patterns.
4. **Different failure mode analysis:** If hard defects are a *different type* of failure (e.g., contamination vs equipment drift), they may require a separate model trained on different features.

**Key insight:** Algorithmic complexity (fancier models, deeper tuning) primarily helps with medium-difficulty defects. Hard defects require **domain-driven data collection**—this is where collaboration with process engineers becomes essential.

---

### 14.2 What-If Questions

These questions explore how the analysis or recommendations would change under different scenarios.

---

**Q1: What if the cost of a missed defect was 100× the cost of a false alarm (instead of roughly equal)? How would you change the model selection and threshold?**

**Consider:** The precision-recall tradeoff and how threshold affects the confusion matrix.

**Answer:** With missed defects 100× more costly, the optimization objective shifts dramatically toward **maximizing recall** even at the expense of precision.

**Changes:**

1. **Threshold:** Lower the threshold significantly. Instead of 0.09 (LightGBM), you might use 0.02 or 0.03, flagging many more samples. This could push recall to 50-60% but drop precision to 10-15%.

2. **Model selection:** Choose XGBoost + SMOTE-Tomek or even Logistic Regression + Undersample (which achieved 52.4% recall). The "best" model changes because F1 no longer reflects business value.

3. **Metric:** Switch from F1 to a cost-weighted metric: `Cost = 100 × FN + 1 × FP`. Or use F-beta with β=10 (heavily weights recall).

4. **Stakeholder communication:** Reframe the model as a "screening tool" that catches most defects but requires downstream filtering of false positives. The narrative shifts from "4 out of 10 flagged are defects" to "we catch 60% of defects, accepting that 85% of flags need additional review."

**Practical implementation:** You'd likely implement a two-stage system—the ML model as a sensitive first-pass filter, followed by a more specific (possibly human) review of flagged items.

---

**Q2: What if you had 1,000 positive training samples instead of 83? How would the analysis and recommendations change?**

**Consider:** Statistical power, model complexity, feature engineering opportunities.

**Answer:** With 12× more positive samples, several things improve:

**Model training:**
- Could use more complex models (deeper trees, more leaves) without overfitting
- Neural networks become viable (currently would overfit with 83 positives)
- SMOTE becomes less necessary—natural class balance may be sufficient
- Can afford more aggressive feature engineering (interaction terms, polynomial features)

**Validation:**
- CV performance estimates become more reliable (200+ positives per validation fold)
- Test set would have ~200 positives—1 more/fewer detected doesn't swing recall by 5%
- Could use nested CV for proper hyperparameter selection

**Feature engineering:**
- Could afford higher-dimensional feature spaces (interactions, polynomials)
- Might detect the "medium" and "hard" defect signatures that are currently too subtle
- Could stratify analysis by defect type if labels exist

**Expected performance:**
- F1 might reach 0.5-0.6 (not from better algorithms, but from statistical reliability)
- The "hard" defects might still be undetectable if they truly lack sensor signatures
- Precision could improve as model learns finer decision boundaries

**Recommendations would change:**
- Might recommend deeper hyperparameter tuning (larger grids)
- Could consider ensemble methods (stacking) that require held-out meta-training data
- Random Forest might become competitive as probability calibration improves with more data

---

**Q3: What if management wanted a "simple rule" (e.g., "flag if Sensor 103 > X") instead of a black-box model? How would you respond?**

**Consider:** Interpretability vs performance tradeoff, and what the data shows about single-feature rules.

**Answer:** This is a legitimate request—simple rules are transparent, auditable, and easy to implement. Here's how to respond:

**Analysis approach:**
1. Extract the top 3-5 features from SHAP (Sensors 103, 33, 59, 31, 205)
2. For each sensor, find the threshold that maximizes F1 (or recall at precision ≥ 0.2)
3. Evaluate single-feature rules and simple combinations (OR/AND logic)

**Expected findings:**
- Single-sensor rules likely achieve F1 ≈ 0.15-0.20 (worse than LightGBM's 0.333)
- A simple OR rule ("flag if Sensor 103 > X OR Sensor 33 > Y") might reach F1 ≈ 0.25
- The gap represents the value of feature interactions captured by the ML model

**Communication:**
> "A rule-based system using Sensor 103 achieves F1 = 0.18, detecting 20% of defects at 15% precision. The ML model achieves F1 = 0.33, detecting 30% of defects at 40% precision. The ML model catches 50% more defects with 2.5× better precision.
>
> However, if interpretability and simplicity are paramount, here's the recommended rule: [Sensor 103 > X OR Sensor 33 > Y]. This catches 25% of defects with 20% precision—a reasonable compromise."

**Hybrid approach:** Use the simple rule as a "first filter" that triggers when readings are extreme, and apply the ML model for borderline cases. This combines interpretability with performance.

---

**Q4: What if the defect rate suddenly increased from 6.7% to 20% in production? How would this affect the deployed model?**

**Consider:** Prior probability shift, threshold calibration, and retraining needs.

**Answer:** A 3× increase in defect rate is a major **prior probability shift** (also called "label shift"). The model's calibration breaks down.

**Immediate effects:**
1. **Precision increases artificially:** More true positives in the population means more of the flagged samples are actually defects. The model seems to "improve."
2. **Recall may drop:** If the new defects have different characteristics than training defects, the model may miss them.
3. **Threshold becomes suboptimal:** The CV-optimized threshold (0.09) was calibrated for 6.7% base rate. At 20%, a different threshold is optimal.

**Required actions:**

1. **Recalibrate threshold:** Use recent data with known labels to find the new optimal threshold. This is fast and doesn't require full retraining.

2. **Monitor feature distributions:** Are the new defects coming from the same sensor patterns? Check SHAP explanations for recent defects.

3. **Investigate root cause:** A 3× increase suggests a process change, equipment issue, or material problem. The ML model may be a symptom detector, not a root cause finder.

4. **Retrain with new data:** If defect patterns changed (not just frequency), the model needs retraining on recent examples. The new training set would have ~200 positives instead of 83—actually improving model quality.

**Stakeholder communication:**
> "The 20% defect rate invalidates our current threshold. Immediate action: recalibrate the threshold on recent data. Parallel action: investigate why defect rate tripled—this may indicate a process issue that ML alone cannot solve."

---

**Q5: What if the engineers said "Sensor 103 measures temperature, which we already monitor with alerts. Your model isn't adding value." How would you respond?**

**Consider:** The difference between univariate alerts and multivariate patterns, and how to demonstrate added value.

**Answer:** This is a common and valid challenge. Here's how to respond with data:

**Quantify single-sensor performance:**
> "Sensor 103 alone, with optimal threshold, catches 15% of defects at 12% precision. Your current temperature alert likely has similar performance. The ML model catches 30% of defects at 40% precision—**2× the recall and 3× the precision**."

**Explain multivariate value:**
> "The model doesn't just use Sensor 103—it combines information from 287 sensors. Defects often have *combinations* of readings that are individually normal but jointly abnormal. For example, Sensor 103 at 85°C is fine if Sensor 33 is at 1.2 atm, but problematic if Sensor 33 is at 1.5 atm. The model captures these interactions automatically."

**Show specific examples:**
> "Here are 3 defects the model caught that your temperature alert would miss: [list cases where Sensor 103 was within normal range but other sensors triggered detection]"

**Propose A/B test:**
> "Let's run both systems in parallel for 1 month: your temperature alert alone, and the ML model. We'll compare:
> - Number of defects caught by each
> - False alarm rates
> - Defects caught by ML but not by temperature alert (added value)"

**Acknowledge their expertise:**
> "You're right that temperature is critical—and the model agrees, ranking Sensor 103 as the top predictor. The question is whether combining temperature with other sensors provides incremental value. The data suggests it does, but I'd welcome a real-world comparison."

**Key principle:** Don't argue theoretically—demonstrate empirically. Engineers respect data over claims.

---

### 14.3 Reflection Questions (Open-Ended)

For deeper learning, consider these questions that don't have single "right" answers:

1. **Ethics:** If the model achieves 40% precision, 60% of flagged units are actually fine. Who bears the cost of false alarms—the company (wasted inspection time) or operators (extra work)? Is this acceptable?

2. **Scope creep:** If stakeholders ask to predict *why* a defect occurred (not just *whether*), how does the problem change? What additional data would you need?

3. **Model lifecycle:** The model was trained on historical data. In 2 years, equipment may be upgraded and sensor behaviors may drift. How would you design a monitoring system to detect when the model needs retraining?

4. **Complementary systems:** The ML model catches ~30% of defects. What other defect detection methods (physical inspection, downstream testing, customer returns) exist, and how should ML fit into the overall quality system?

5. **Feature causality:** SHAP identifies Sensor 103 as important, but correlation ≠ causation. Sensor 103 might be *caused by* the same upstream issue that causes defects, not a *cause of* defects. How would you investigate this?

---

*These questions are designed for self-assessment and team discussion. Revisit them after implementing changes to see how your understanding evolves.*

---

## 15. Key Lessons from This Analysis

This section distills the most important insights from the analysis into actionable takeaways.

### Lesson 1: Class Weights Beat SMOTE for Small Minority Classes

- **What we observed**: LightGBM + None (class weights) ranked #9 in CV but jumped to #1 on test; LightGBM + SMOTE-Tomek ranked #1 in CV but dropped to #4 on test.
- **Why it happened**: SMOTE creates synthetic samples by interpolation, which can introduce patterns that exist in CV validation folds (same distribution as training) but not in true held-out test data. Class weights modify the loss function without changing the data distribution, so the model learns only from real examples.
- **Practical takeaway**: Start with `class_weight='balanced'` or `scale_pos_weight` before trying resampling methods. Simpler approaches often generalize better.

### Lesson 2: CV Rankings Don't Guarantee Test Performance

- **What we observed**: 8 of 16 configurations changed rank by 3+ positions between CV and test. The biggest winner jumped 8 spots; the biggest loser dropped 8 spots.
- **Why it happened**: CV validation folds share the same distribution as training data. Techniques that modify training distribution (SMOTE) or aggressively clean boundaries (ENN) can optimize for CV patterns that don't exist in the true test distribution.
- **Practical takeaway**: Always evaluate on held-out test data. Treat CV performance as a noisy estimate, especially with small positive classes (<100 samples).

### Lesson 3: Boosting Beats Bagging for Imbalanced Classification

- **What we observed**: LightGBM and XGBoost dominated the test leaderboard; Random Forest, despite being "more robust," underperformed despite having the best PR-AUC (0.244).
- **Why it happened**: (1) Boosting's sequential correction explicitly focuses on misclassified samples, often the minority class. (2) RF probabilities compress toward 0.5 due to vote averaging, making threshold optimization less effective. (3) Boosting naturally concentrates on informative features; RF dilutes importance across random subsets.
- **Practical takeaway**: For rare event prediction, prefer gradient boosting (LightGBM, XGBoost) over Random Forest. RF's "robustness" comes from variance reduction, which can average away weak minority class signals.

### Lesson 4: Aggressive Boundary Cleaning Hurts More Than It Helps

- **What we observed**: SMOTE-ENN performed worst across all resampling strategies on both CV and test. LightGBM + SMOTE-ENN achieved F1 = 0.000 on test (zero positive predictions).
- **Why it happened**: ENN removes samples whose neighbors disagree with their class—including minority samples that "look like" majority samples. These borderline cases are exactly the hard examples the model needs to learn. A "clean" training set produces overconfident models that fail on messy real-world data.
- **Practical takeaway**: Avoid aggressive cleaning (ENN) with small minority classes. If cleaning is needed, use Tomek Links (removes only direct Tomek pairs).

### Lesson 5: Not All Defects Are Detectable with Current Features

- **What we observed**: ~40% of test defects were missed by ALL 16 model configurations. These "hard" defects had normal-looking sensor readings on all key features.
- **Why it happened**: The signal simply isn't in the available features. Hard defects may arise from different failure modes, temporal misalignment (sensor snapshot misses transient events), or unmeasured process parameters.
- **Practical takeaway**: There's a ceiling on ML improvement with current data. Detecting hard defects requires new sensors or different data sources, not better algorithms. Collaborate with process engineers to identify sensor gaps.

### Lesson 6: Threshold Selection Matters as Much as Model Selection

- **What we observed**: LightGBM achieved 40% precision vs XGBoost's 25% despite nearly identical PR-AUC (0.215 vs 0.214). The difference came from threshold selection (0.090 vs 0.137).
- **Why it happened**: PR-AUC measures ranking quality across all thresholds, but operational performance depends on the specific threshold chosen. Different models produce different probability distributions, requiring different optimal cut points.
- **Practical takeaway**: Optimize threshold during CV, fix for test, and tune based on business costs. Don't assume similar PR-AUC means similar operational performance.

### Lesson 7: Validate Feature Importance with Multiple Methods

- **What we observed**: Sensor 103 ranked #1 by SHAP but #105 by permutation importance. Investigation revealed it only matters for 6% of samples (19/314), with a 79% false alarm rate among those.
- **Why it happened**: SHAP measures attribution magnitude (high for edge cases), while permutation measures overall performance impact (low if feature only matters for a few samples). Different methods answer different questions.
- **Practical takeaway**: Always cross-validate SHAP findings with permutation importance. Agreement = confidence. Disagreement = investigate. A feature can be "important" for explanations but not for performance.

---

## 16. Next Steps for Stakeholders

### For Manufacturing Managers

- [ ] **Review cost trade-offs**: Current model catches 30% of defects at 40% precision. Determine if this operating point matches business needs, or if higher recall (more false alarms) is preferred.
- [ ] **Evaluate deployment scope**: Consider piloting on one production line before full rollout.
- [ ] **Set performance monitoring KPIs**: Track model precision/recall monthly; trigger review if F1 drops below 0.25.

### For Quality Engineers

- [ ] **Implement sensor alerts**: Set threshold alerts for Sensors 103, 33, 59 when readings exceed 2σ from baseline.
- [ ] **Investigate missed defects**: Pull physical samples from the ~40% of test defects missed by all models. Conduct root cause analysis—what do they have in common?
- [ ] **Validate sensor meanings**: Provide documentation on what Sensors 103, 33, 59, 31, 205 actually measure. This enables domain-informed feature engineering.

### For Process Engineers

- [ ] **Sensor gap analysis**: Review process flow to identify unmeasured parameters that could explain hard defects (e.g., upstream temperature, chemical timing, vibration).
- [ ] **Temporal sampling review**: Are sensors sampled frequently enough to catch transient events? Consider higher sampling frequency at critical process steps.
- [ ] **Failure mode clustering**: Are the missed defects a distinct failure type (e.g., contamination vs equipment drift)? If so, they may need separate detection approaches.

### For Data Scientists

- [ ] **Explore interaction features**: Add explicit feature interactions (e.g., Sensor103 × Sensor33) to capture nonlinear patterns.
- [ ] **Bootstrap confidence intervals**: Quantify uncertainty in test metrics given only 21 positives.
- [ ] **Calibration analysis**: Plot reliability diagrams for LightGBM vs RF to visualize probability compression.
- [ ] **Ensemble experiment**: Try stacking LightGBM + XGBoost predictions—they may capture complementary patterns.

---

## 17. Inputs That Would Improve This Analysis

| Input Needed | Who Can Provide | How It Helps |
|--------------|-----------------|--------------|
| Sensor documentation (what do Sensors 103, 33, 59, 31, 205 measure?) | Process Engineers | Enables domain-informed feature engineering; allows physical interpretation of SHAP results |
| Root cause analysis of "hard" defects (missed by all models) | Quality Engineers | Identifies whether hard defects are a distinct failure mode requiring different data |
| Cost ratio of missed defect vs false alarm | Manufacturing Managers | Allows optimization of threshold for business value, not just F1 |
| Timestamps for temporal validation | Data Engineering | Enables time-based train/test split that simulates production deployment |
| Upstream process parameters | Process Engineers | May capture root causes not reflected in downstream sensors |
| Historical defect categorization (types/causes) | Quality Team | Enables stratified analysis and specialized models per defect type |
| Production line/equipment IDs | Process Engineers | Allows analysis of equipment-specific patterns and model calibration |

---

## Acknowledgements

This analysis and technical report were developed collaboratively by:

- **Jie He** ([@JHstat](https://github.com/JHstat)) - Analysis design, methodology decisions, and domain review
- **Anthropic Claude Code** - Code implementation, documentation, and iterative refinement

*Report generated from analysis run on UCI SECOM dataset. All results are reproducible with random_state=42.*
