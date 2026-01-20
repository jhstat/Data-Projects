# UCI SECOM Project Notes

**Dataset:** UCI SECOM Semiconductor Manufacturing Data
**Objective:** Predict semiconductor defects from sensor readings
**Best Model:** LightGBM + Class Weights (F1 = 0.333, Precision = 40%, Recall = 28.6%)

---

## Key Lessons Learned

### 1. Class Weights Beat SMOTE for Small Minorities
- **Observed:** LightGBM+None outperformed all SMOTE variants on test set
- **Why:** SMOTE creates synthetic samples by interpolation—these may not represent real defect signatures. Class weights modify the loss function without changing the data distribution.
- **Takeaway:** Start with `class_weight='balanced'` or `scale_pos_weight` before trying resampling methods.

### 2. CV Rankings Don't Always Transfer to Test
- **Observed:** Top CV performer (LightGBM+SMOTE-Tomek, Rank #1) dropped to Rank #4 on test; LightGBM+None jumped from Rank #9 to Rank #1
- **Why:** CV validation folds come from the same distribution as training. SMOTE-augmented models can overfit to patterns in that distribution that don't exist in the true test distribution.
- **Takeaway:** Always validate on held-out test data. Be skeptical of CV-only results, especially with small samples.

### 3. Not All Defects Are Equal
- **Observed:** ~40% of defects were missed by ALL 16 model configurations
- **Why:** "Hard" defects have normal-looking sensor readings—the signal simply isn't in the available features
- **Takeaway:** There's a ceiling on ML improvement with current features. Detecting hard defects requires new sensors or different data, not better algorithms.

### 4. Boosting Beats Bagging for Imbalanced Data
- **Observed:** LightGBM/XGBoost consistently outperformed RandomForest across all resampling strategies
- **Why:**
  - Boosting's sequential correction focuses on misclassified samples (often minority class)
  - RF probability compression (averaging tree votes) makes threshold optimization harder
  - Boosting naturally concentrates on informative features; RF dilutes across random subsets
- **Takeaway:** For rare event prediction, prefer gradient boosting over random forests.

### 5. Threshold Selection Matters as Much as Model Choice
- **Observed:** LightGBM achieved 40% precision vs XGBoost's 25% despite nearly identical PR-AUC
- **Why:** The optimal threshold depends on the model's probability distribution. LightGBM's lower threshold (0.09 vs 0.137) produced more conservative predictions.
- **Takeaway:** Optimize threshold on CV, fix for test, and tune based on business costs (FN vs FP tradeoff).

### 6. Aggressive Boundary Cleaning Hurts Performance
- **Observed:** SMOTE-ENN performed worst across all strategies
- **Why:** ENN removes borderline samples—but these "hard cases" are exactly what the model needs to learn. A "clean" training set produces overconfident models that fail on messy real-world data.
- **Takeaway:** Avoid aggressive cleaning (ENN) with small minority classes. Tomek Links is safer.

---

## Data Science Knowledge Tidbits

### Distribution Shift from SMOTE
SMOTE shifts the training distribution toward class balance, but test data remains imbalanced. Models trained on shifted distributions may not perform optimally on the original distribution. This is a form of **covariate shift**.

### RF Probability Compression
Random Forest probabilities are averages of tree votes, compressing toward 0.5. Boosting produces additive log-odds, creating more extreme (better separated) probabilities. For threshold-based metrics, well-separated probabilities provide more "room" to optimize.

### The Clean Boundary Paradox
Cleaning decision boundaries (via ENN or similar) removes uncertainty from training. But uncertainty exists in real data—models that never see borderline cases during training become overconfident and fail on test data.

### PR-AUC vs Point Metrics
Two models can have identical PR-AUC (ranking quality) but very different precision/recall at specific thresholds. PR-AUC tells you about overall ranking; point metrics depend on threshold selection and probability calibration.

### Bias-Variance in Imbalanced Settings
- **Bagging (RF):** Reduces variance through averaging, but may average away weak minority class signals
- **Boosting:** Reduces bias through sequential correction, explicitly focusing on hard-to-classify (often minority) samples

---

## Dataset Characteristics

| Property | Value |
|----------|-------|
| Total samples | 1,567 |
| Training samples | 1,253 (80%) |
| Test samples | 314 (20%) |
| Original features | 590 |
| Final features | 287 (after engineering) |
| Defect rate | 6.6% (~1:14 imbalance) |
| Training positives | 83 |
| Test positives | 21 |

**Challenges:**
- Severe class imbalance limits model complexity
- High dimensionality (590 features) with many correlated sensors
- Anonymous sensor names prevent domain-informed feature engineering
- Small test set (21 positives) means high variance in performance estimates

**Key sensors identified:** 103, 33, 59, 31, 205

---

## Future Exploration Ideas

### Methodological
- [ ] **Nested CV:** Use outer CV for model selection, inner CV for hyperparameter tuning—would give more honest performance estimates
- [ ] **Bootstrap confidence intervals:** Quantify uncertainty in test metrics given only 21 positives
- [ ] **Calibration analysis:** Plot reliability diagrams for LightGBM vs RF to visualize probability compression
- [ ] **Stacking/blending:** Combine LightGBM + XGBoost predictions—might capture complementary patterns

### Feature Engineering
- [ ] **Interaction terms:** Explicitly model sensor combinations (e.g., Sensor103 × Sensor33)
- [ ] **Ratio features:** Domain-informed ratios if sensor meanings were known
- [ ] **Missingness patterns:** More sophisticated use of missing value indicators

### Model Variants
- [ ] **CatBoost:** Alternative gradient boosting with built-in handling of categoricals
- [ ] **Isolation Forest for anomaly pre-filtering:** Flag extreme sensor readings before classification
- [ ] **Cost-sensitive learning:** Custom loss function with explicit FN/FP cost ratio

### Analysis Extensions
- [ ] **Temporal patterns:** If timestamps available, analyze defect clustering in time
- [ ] **Defect subtyping:** Cluster the 21 test defects—are there distinct failure modes?
- [ ] **Simple rule comparison:** Quantify F1 of single-sensor threshold rules vs ML model

### Stakeholder Questions
- What do Sensors 103, 33, 59 actually measure?
- Are the "hard" defects (missed by all models) a different failure mode?
- What is the actual cost ratio of missed defects vs false alarms?
- Are there upstream process parameters not captured in current sensors?

---

## Acknowledgements

This analysis was developed collaboratively by:
- **Jie He** ([@JHstat](https://github.com/JHstat)) - Analysis design, methodology decisions, and domain review
- **Anthropic Claude Code** - Code implementation, documentation, and iterative refinement

*See `TECHNICAL_REPORT.md` for full methodology and results.*
