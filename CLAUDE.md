# CLAUDE.md - Data Science Project Guidelines

This file provides context and standards for Claude when working on data science, machine learning, and statistical analysis projects in this repository.

## Project Context

This repository contains data science projects focusing on:
- Predictive modeling and classification
- Statistical analysis and hypothesis testing
- Feature engineering and selection
- Model evaluation and interpretation

## Pre-Analysis: The ANALYSIS_PLAN.md Requirement

**CRITICAL: Do NOT write analysis code until a written plan is approved by the user.**

### Why Plans Matter

Data analysis involves hundreds of micro-decisions that compound. Without explicit planning:
- Assumptions go unchallenged
- Alternatives go unexplored
- Methodological errors get baked in
- Results become harder to interpret or reproduce

### Workflow

1. **User supplies or requests a plan** — Either provide an existing plan or ask Claude to draft one
2. **Claude drafts `ANALYSIS_PLAN.md`** — Comprehensive, end-to-end, zero ambiguity
3. **Discussion phase** — Weigh pros/cons of alternatives at critical decision points
4. **User approves** — Explicit "OK" or "Approved" before any analysis code runs
5. **Mark the plan** — Add approval timestamp and proceed

### What "Zero Ambiguity" Means

The plan must be detailed enough that **anyone with the manual can execute it without asking questions**. Every decision must be explicit:

| Level | Bad Example | Good Example |
|-------|-------------|--------------|
| Model choice | "Run XGBoost" | "XGBoost with `tree_method='hist'`, `objective='binary:logistic'`" |
| Hyperparameters | "Tune the usual params" | "`max_depth`: [3, 5, 7], `learning_rate`: [0.01, 0.05, 0.1], `min_child_weight`: [1, 5, 10]" |
| Data split | "Use cross-validation" | "5-fold stratified CV, `shuffle=True`, `random_state=42`; 80/20 train-test split first" |
| Preprocessing | "Handle missing values" | "Median imputation via `SimpleImputer`; fit on train fold only, transform both" |
| Threshold | "Optimize threshold" | "Maximize Recall @ Precision ≥ 0.2 on OOF predictions; fix threshold for test" |

### ANALYSIS_PLAN.md Template

```markdown
# Analysis Plan: [Project Name]

**Status:** 🔴 DRAFT | 🟡 UNDER REVIEW | 🟢 APPROVED
**Approved by:** [User name/initials]
**Approval date:** [Date]

---

## 1. Problem Definition

**Objective:** [What are we predicting/analyzing?]
**Success criteria:** [What metric matters? What's "good enough"?]
**Constraints:** [Time, compute, interpretability requirements]

---

## 2. Data Overview

**Source:** [Where does the data come from?]
**Size:** [Rows, columns, file size]
**Target variable:** [Name, type, distribution]
**Known issues:** [Missing data, class imbalance, leakage risks]

---

## 3. Train-Test Split Strategy

**Method:** [Random, temporal, stratified]
**Ratios:** [e.g., 80/20 train/test]
**Stratification:** [On which variable?]
**Random seed:** [e.g., 42]

**Rationale:** [Why this split strategy?]
**Alternative considered:** [What else could we do?]

---

## 4. Feature Engineering

### 4.1 Missing Value Handling
| Condition | Action | Rationale |
|-----------|--------|-----------|
| ≥X% missing | Drop column | Insufficient data for imputation |
| Y-X% missing | Create indicator + impute | Missingness may be informative |
| <Y% missing | Impute only | Standard handling |

**Imputation method:** [Mean/median/mode/KNN/etc.]
**Fit on:** [Train only — NEVER validation/test]

### 4.2 Feature Removal
- **Constant/near-constant:** [Threshold, e.g., >99% single value]
- **High correlation:** [Threshold, e.g., |r| > 0.95; tiebreaker rule]
- **Low variance:** [Threshold if applicable]

### 4.3 Feature Transformations
- **Scaling:** [StandardScaler/MinMaxScaler/None]
- **Encoding:** [For categoricals if any]
- **New features:** [Interactions, polynomials, domain-specific]

---

## 5. Class Imbalance Strategy

**Imbalance ratio:** [e.g., 1:14]

**Strategies to test:**
| Strategy | Configuration | Rationale |
|----------|---------------|-----------|
| Class weights | `class_weight='balanced'` | Baseline, no data manipulation |
| Undersampling | `RandomUnderSampler`, target ratio 1:3 | Reduces majority class |
| SMOTE-Tomek | `k_neighbors=3`, `sampling_strategy=0.33` | Synthetic oversampling + cleaning |
| SMOTE-ENN | `k_neighbors=3`, `kind_sel='all'` | More aggressive cleaning |

**Apply resampling:** [Train folds only — NEVER validation/test]

---

## 6. Model Selection

### 6.1 Models to Test
| Model | Base Configuration | Why Include |
|-------|-------------------|-------------|
| LogisticRegression | `solver='saga'`, `penalty='elasticnet'`, `max_iter=1000` | Linear baseline |
| LightGBM | `verbose=-1`, `n_jobs=-1` | Fast, leaf-wise boosting |
| XGBoost | `eval_metric='logloss'`, `n_jobs=-1` | Level-wise boosting comparison |
| RandomForest | `n_jobs=-1` | Bagging baseline |

### 6.2 Hyperparameter Grids

**LightGBM:**
```python
{
    'num_leaves': [15, 31],
    'min_data_in_leaf': [10, 20],
    'learning_rate': [0.05, 0.1]
}
```
**Rationale:** Conservative complexity for small positive class (~80 samples)

**XGBoost:**
```python
{
    'max_depth': [3, 5],
    'min_child_weight': [1, 5],
    'learning_rate': [0.05, 0.1]
}
```

**RandomForest:**
```python
{
    'n_estimators': [100, 200],
    'max_depth': [5, 10],
    'min_samples_leaf': [5, 10]
}
```

**LogisticRegression:**
```python
{
    'C': [0.01, 0.1, 1.0],
    'l1_ratio': [0.3, 0.5, 0.7]
}
```

---

## 7. Cross-Validation Strategy

**Method:** Stratified K-Fold
**Folds:** 5
**Shuffle:** True
**Random seed:** 42

**What happens in each fold:**
1. Split into train/validation
2. Apply resampling to train only (if applicable)
3. Fit imputer/scaler on train, transform both
4. Fit model on train
5. Predict probabilities on validation
6. Collect OOF predictions

---

## 8. Evaluation Metrics

**Primary metric:** [e.g., Recall @ Precision ≥ 0.2]
**Secondary metrics:** F1, PR-AUC, Recall, Precision

**Threshold selection:**
- Method: [e.g., find threshold maximizing recall where precision ≥ 0.2]
- Fallback: [e.g., if no threshold achieves precision ≥ 0.2, use max-F1 threshold]
- Apply to: OOF predictions during CV
- Fix for test: YES — no re-optimization on test data

---

## 9. Test Set Evaluation

**Process:**
1. Retrain best model on full training set
2. Apply preprocessing (fit on full train)
3. Apply fixed threshold from CV
4. Report metrics with confidence context (note: N=21 positives → high variance)

---

## 10. Interpretability

**Methods:**
- SHAP TreeExplainer for tree-based models
- Global: feature importance bar plot
- Local: waterfall plots for selected predictions

**Analyses:**
- Compare feature importance across models
- Identify disagreement between models
- Characterize false negatives vs true positives

---

## 11. Deliverables

| Deliverable | Format | Location |
|-------------|--------|----------|
| Analysis notebook | `.ipynb` | `DATA_ANALYSIS.ipynb` |
| Technical report | `.md` | `TECHNICAL_REPORT.md` |
| Project notes | `.md` | `PROJECT_NOTES.md` |
| Trained models | `.pkl` | `outputs/models/` |
| Results summary | `.csv` | `outputs/results/` |

---

## 12. Known Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Small positive class (N=83) | Conservative hyperparameters, wide confidence intervals |
| High dimensionality (590 features) | Aggressive correlation removal, regularization |
| Threshold overfitting | Fix threshold from CV, don't re-optimize on test |
| SMOTE distribution shift | Compare with class weights (no data manipulation) |

---

## 13. Decision Log

| Decision Point | Options Considered | Choice | Rationale |
|----------------|-------------------|--------|-----------|
| Train-test split | Random vs temporal | Random stratified | No timestamps; preserve class ratio |
| Imputation | Mean vs median vs KNN | Median | Robust to outliers in sensor data |
| Correlation threshold | 0.90 vs 0.95 vs 0.99 | 0.95 | Balance redundancy removal vs info loss |
| SMOTE k_neighbors | 3 vs 5 | 3 | Conservative for small positive class |

---

## Approval

- [ ] Plan reviewed by user
- [ ] Alternatives discussed at critical decision points
- [ ] Data handling verified (no leakage risks)
- [ ] Ready to proceed

**User approval:** _______________
**Date:** _______________
```

### Discussion Checklist

Before approval, ensure these are discussed:

**Data handling:**
- [ ] Is train-test split done BEFORE any feature engineering?
- [ ] Is preprocessing fit on train only?
- [ ] Is resampling applied to train folds only?
- [ ] Is threshold fixed from CV (not re-optimized on test)?

**Methodological choices:**
- [ ] Are hyperparameter ranges justified for this dataset size?
- [ ] Are there simpler baselines to compare against?
- [ ] What's the fallback if primary metric can't be achieved?

**Alternatives:**
- [ ] Did we consider at least 2 options at each critical decision?
- [ ] Are pros/cons of alternatives documented?
- [ ] Is the rationale for final choice clear?

---

## Core Principles

### 1. Observe, Hypothesize, Verify

**Always ask "why?"** when observing results:
- Why did a model perform well in CV but poorly on test?
- Why did two similar approaches produce different outcomes?
- Why did a particular feature become important?

**Generate hypotheses** grounded in both theory and data:
- What mechanism could explain this pattern?
- Is this consistent with domain knowledge?
- What would we expect to see if this hypothesis is true?

**Verify with evidence**:
- Can we test the hypothesis with additional analysis?
- Does the pattern hold across different subsets?
- Are there confounding factors we haven't considered?

### 2. Curiosity-Driven Exploration (THINK HARD)

**This is critical.** Don't just report results—actively hunt for insights. Spend extra effort examining model fitting results, comparisons, and interpretations from multiple angles.

#### Angles to Examine Model Results

**Performance comparisons:**
- Which models agree/disagree on predictions? What characterizes the disagreement cases?
- Did rankings shift between CV and test? What does that reveal about overfitting or distribution shift?
- Are there models that excel on easy cases but fail on hard ones (or vice versa)?
- Do ensemble components capture different signals, or are they redundant?

**Feature importance:**
- Do different models identify the same important features? If not, why?
- Are important features correlated with each other? Could one be a proxy for another?
- Does feature importance change across subgroups (e.g., easy vs hard cases)?
- Are any "important" features suspiciously perfect (potential leakage)?

**Error analysis:**
- What do false positives have in common? False negatives?
- Are errors random or systematic (clustered in feature space)?
- Do certain models make complementary errors (good for ensembling)?
- Are there "hard" cases that ALL models miss? What makes them hard?

**Calibration and thresholds:**
- How do probability distributions differ across models?
- Is the optimal threshold stable across CV folds, or does it vary wildly?
- Would a different threshold serve business needs better?

**Unexpected findings:**
- Did a "worse" method outperform expectations? Why might that be?
- Did adding complexity (features, hyperparameters) hurt performance?
- Are there counterintuitive patterns that challenge assumptions?

#### Probing Questions to Always Ask

After every major result, ask:
1. **"What's surprising here?"** — If nothing surprises you, you haven't looked hard enough
2. **"What would change my conclusion?"** — Identify the assumptions your interpretation depends on
3. **"Who would disagree, and why?"** — Steel-man alternative explanations
4. **"What's missing from this picture?"** — What data or analysis would strengthen the conclusion?
5. **"So what?"** — Connect findings to actionable decisions or deeper understanding

#### Examples of Curiosity-Driven Findings

- "Model A and B have similar accuracy, but they disagree on 30% of predictions—why? The disagreement cases cluster near the decision boundary and have higher missingness rates."
- "This feature is important globally but irrelevant for the hardest cases—suggesting the hard cases represent a different failure mode not captured by this sensor."
- "Performance improved after removing a feature—investigating further, this feature had future information leakage through timestamp encoding."
- "SMOTE-Tomek won CV but lost on test. The rank shift suggests SMOTE created synthetic samples that matched CV validation patterns but not real-world test patterns."
- "Random Forest had the best PR-AUC but worst F1—its probability compression made threshold optimization ineffective."

**Bottom line:** Treat every result as a mystery to be solved, not a box to be checked.

### 3. Dual Foundations: Theory + Data

Explanations must be grounded in **both**:

**Theoretical foundation:**
- How does the algorithm work? (e.g., leaf-wise vs level-wise tree growth)
- What are the known properties? (e.g., RF reduces variance through bagging)
- What does statistical theory predict? (e.g., bias-variance tradeoff)

**Data-specific context:**
- How does the data structure interact with the algorithm?
- What characteristics of this dataset favor/disfavor certain approaches?
- Are there domain-specific patterns at play?

**Include references** for deeper learning:
- Link to papers, documentation, or tutorials when introducing concepts
- Explain *why* a technique works, not just *that* it works

### 4. Stakeholder-Centric Communication

**Always close the loop with domain experts:**

- What findings should engineers investigate?
- What business decisions does this analysis inform?
- What additional data or context would improve the model?
- What domain knowledge would help interpret the results?

**Tailor communication to the audience:**
- **Executives**: Impact metrics, trade-offs, resource implications
- **Engineers**: Actionable thresholds, features to monitor, root causes to investigate
- **Data scientists**: Methodology details, reproducibility, improvement opportunities

**Ask for help explicitly:**
- "To improve detection of hard defects, we need sensor documentation for features 103, 33, 59"
- "Root cause analysis of cases X, Y, Z would help us understand model blind spots"
- "Domain expert review of our feature engineering assumptions would validate our approach"

### 5. Learning-Oriented Summaries

Every project should include **take-home lessons** (max 10):

Format:
```
**Lesson**: [Concise statement]
**Context**: [Where in the analysis this emerged]
**Why it matters**: [Practical implication]
```

Example:
```
**Lesson**: Class weights often outperform SMOTE for small minority classes
**Context**: LightGBM+None beat LightGBM+SMOTE-Tomek on test set despite worse CV
**Why it matters**: Simpler approaches generalize better; try them first
```

### 6. Enable Mutual Learning

Don't just give summaries—**ask questions** to help readers deepen their understanding. Include two types:

#### Results-Based Questions (5)
Probe understanding of practical ML and statistical theory based on what was observed.

Format:
```
**Q**: [Question about why something happened]
**Hint**: [What to consider]
**Answer**: [Explanation with theory + data context]
```

#### What-If Questions (5)
Explore counterfactuals and how conclusions would change under different scenarios.

Format:
```
**Q**: What if [alternative scenario]?
**Consider**: [Key factors that would change]
**Answer**: [How analysis/recommendations would differ]
```

This section turns passive reading into active learning and prepares readers for similar situations in future projects.

---

## Methodological Standards

### Data Handling

1. **Train-Test Split First**: Always split data before any analysis or feature engineering to prevent data leakage.

2. **Label-Free Feature Engineering**: Feature selection decisions (missing value thresholds, correlation removal, variance filtering) must be based only on X, never on y.

3. **Preprocessing Fit on Train Only**:
   - Imputers, scalers, encoders: `fit_transform()` on train, `transform()` on validation/test
   - Never use validation/test statistics (mean, median, std) for preprocessing

4. **Cross-Validation Hygiene**:
   - All preprocessing steps must be inside the CV loop
   - Resampling (SMOTE, undersampling) applies only to training folds
   - Threshold optimization happens on OOF predictions, then fixed for test

### Class Imbalance

1. **Start Simple**: Try `class_weight='balanced'` or `scale_pos_weight` before resampling methods.

2. **SMOTE Considerations**:
   - Use conservative `k_neighbors` (3-5) for small minority classes
   - `sampling_strategy=0.33` (1:3 ratio) is often better than 1:1
   - SMOTE-Tomek is less aggressive than SMOTE-ENN

3. **Avoid Aggressive Cleaning**: ENN can remove too many samples from small datasets, causing overfitting to "clean" boundaries.

### Model Evaluation

1. **Metrics for Imbalanced Data**:
   - Primary: PR-AUC, F1, Recall@Precision threshold
   - Avoid: Accuracy, ROC-AUC (can be misleading with severe imbalance)

2. **Threshold Handling**:
   - Optimize threshold during CV only
   - Apply fixed threshold to test set (no re-optimization)
   - Document the threshold selection method

3. **Statistical Uncertainty**:
   - With small test sets (<50 positives), confidence intervals are wide
   - Report sample sizes alongside metrics
   - Consider bootstrap CIs for critical comparisons

### Interpretability

1. **SHAP Analysis**:
   - Use `TreeExplainer` for tree-based models
   - Report both global (bar/beeswarm) and local (waterfall/force) explanations
   - Compare feature importance across model types for robustness

2. **Stakeholder Communication**:
   - Translate technical metrics to business impact
   - Provide actionable recommendations, not just model outputs
   - Acknowledge limitations honestly

## Code Standards

### Jupyter Notebooks

1. **Cell Organization**:
   - Number cells in comments (e.g., `# Cell 1: Setup and Imports`)
   - Use markdown cells for section headers
   - Keep cells focused on single tasks

2. **Reproducibility**:
   - Set `random_state=42` (or document the seed used)
   - Print dataset shapes and class distributions
   - Log feature engineering decisions

3. **Output Management**:
   - Print intermediate results for verification
   - Save important results to files (JSON, CSV)
   - Generate visualizations for key findings

### Documentation

1. **Technical Reports** should include:
   - Executive summary with key findings
   - Methodology with parameter justifications
   - Results with appropriate uncertainty quantification
   - Discussion of why certain approaches worked/failed
   - Actionable recommendations

2. **Parameter Documentation**:
   - Explain why each parameter value was chosen
   - List alternatives considered and their trade-offs
   - Note which parameters are sensitive vs robust

## Analysis Patterns

### When Comparing Models

1. Compare on the same held-out test set
2. Use the same preprocessing pipeline
3. Report multiple metrics (not just the optimization target)
4. Analyze rank shifts between CV and test (they reveal overfitting)
5. Check prediction overlap (which cases do models agree/disagree on?)

### When Analyzing Failures

1. Categorize errors by difficulty (easy/medium/hard)
2. Compare feature distributions for correct vs incorrect predictions
3. Use SHAP to explain individual false negatives and false positives
4. Identify whether failures are random or systematic

### When Communicating Results

1. **For Technical Audiences**: Full methodology, parameter grids, statistical caveats
2. **For Business Audiences**: Impact metrics, trade-offs, deployment considerations
3. **For Engineering Audiences**: Actionable thresholds, features to monitor, integration points

## Common Pitfalls to Avoid

1. **Data Leakage**: Using test data statistics, future information, or target-derived features
2. **Threshold Gaming**: Re-optimizing thresholds on test data
3. **Cherry-Picking**: Reporting only the best metric while hiding others
4. **Overconfidence**: Small sample sizes mean high variance; report uncertainty
5. **Over-Engineering**: Start simple; complexity should be justified by improvement
6. **Ignoring Domain**: Statistical patterns without domain validation may be spurious

## Report Structure Template

Every technical report should end with:

### Take-Home Lessons Section

```markdown
## Key Lessons from This Analysis

1. **[Lesson title]**
   - *What we observed*: [Brief description]
   - *Why it happened*: [Theoretical + data explanation]
   - *Practical takeaway*: [What to do differently next time]

2. ...
```

### Stakeholder Action Items Section

```markdown
## Next Steps for Stakeholders

### For [Role 1]:
- [ ] Action item with specific details
- [ ] Question to investigate

### For [Role 2]:
- [ ] Action item with specific details
- [ ] Data/input needed from them
```

### What Would Make This Better Section

```markdown
## Inputs That Would Improve This Analysis

| Input Needed | Who Can Provide | How It Helps |
|--------------|-----------------|--------------|
| [Specific data/knowledge] | [Role/team] | [Impact on analysis] |
```

---

## Project Structure

Each data project should follow a consistent folder structure:

```
Project_Name/
├── ANALYSIS_PLAN.md         # Approved plan before coding (required)
├── DATA_ANALYSIS.ipynb      # Main analysis notebook (required)
├── TECHNICAL_REPORT.md      # Comprehensive methodology & results (required)
├── PROJECT_NOTES.md         # Lessons learned & future ideas (required)
├── data/                    # Raw and processed data
│   ├── raw/                 # Original, unmodified data files
│   └── processed/           # Cleaned/transformed data (if needed)
├── outputs/                 # Generated artifacts
│   ├── figures/             # Saved plots and visualizations
│   ├── models/              # Serialized models (.pkl, .joblib)
│   └── results/             # CSV/JSON exports of key results
└── scripts/                 # Helper scripts (optional)
    └── utils.py             # Reusable functions extracted from notebook
```

### Required Files

| File | Purpose |
|------|---------|
| `ANALYSIS_PLAN.md` | Approved plan with all decisions documented; must have user sign-off |
| `DATA_ANALYSIS.ipynb` | Main notebook with all analysis code, clearly sectioned |
| `TECHNICAL_REPORT.md` | Detailed methodology, parameter choices, results, discussion |
| `PROJECT_NOTES.md` | Key lessons, knowledge tidbits, future exploration ideas |

### PROJECT_NOTES.md Template

Each `PROJECT_NOTES.md` should include:
- **Key Lessons Learned** (what worked, what didn't, why)
- **Data Science Knowledge Tidbits** (theoretical insights gained)
- **Dataset Characteristics** (summary stats, quirks, challenges)
- **Future Exploration Ideas** (questions to investigate, improvements to try)

### Naming Conventions

- **Folders:** `Snake_Case` with underscores (e.g., `UCI_SECOM`, `Kaggle_Titanic`)
- **Notebooks:** `UPPER_CASE.ipynb` for main files, `lowercase_descriptive.ipynb` for exploratory
- **Data files:** Preserve original names in `raw/`; use descriptive names in `processed/`
- **Outputs:** Include date or version if iterating (e.g., `model_v2.pkl`, `results_2026-01.csv`)

### What Goes Where

| Content | Location |
|---------|----------|
| Exploratory code, experiments | `DATA_ANALYSIS.ipynb` |
| Methodology justification, full results | `TECHNICAL_REPORT.md` |
| Takeaways, lessons, future work | `PROJECT_NOTES.md` |
| Reusable helper functions | `scripts/utils.py` |
| Figures for report/presentation | `outputs/figures/` |
| Trained model artifacts | `outputs/models/` |

### Data Management

- **Never commit large data files** to git (use `.gitignore`)
- **Document data sources** in `TECHNICAL_REPORT.md` Section 1
- **Include data dictionary** if feature names are non-obvious
- **Note preprocessing steps** that transform raw → processed

---

*This file guides Claude's behavior for data science work. Update as new patterns emerge.*

---

## Acknowledgements

This repository and its guidelines were developed collaboratively by:

- **Jie He** ([@JHstat](https://github.com/JHstat)) - Analysis design, domain expertise, and methodology review
- **Anthropic Claude Code** - Code generation, technical documentation, and iterative refinement
