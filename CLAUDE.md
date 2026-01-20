# CLAUDE.md - Data Science Project Guidelines

This file provides context and standards for Claude when working on data science, machine learning, and statistical analysis projects in this repository.

## Project Context

This repository contains data science projects focusing on:
- Predictive modeling and classification
- Statistical analysis and hypothesis testing
- Feature engineering and selection
- Model evaluation and interpretation

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

### 2. Curiosity-Driven Exploration

- **Look for interesting patterns** beyond the primary objective
- **Generate options** for users to explore further
- **Flag anomalies** that might reveal deeper insights
- **Ask probing questions** that lead to better understanding

Examples of curiosity-driven findings:
- "Model A and B have similar accuracy, but they disagree on 30% of predictions—why?"
- "This feature is important globally but irrelevant for the hardest cases—what does that mean?"
- "Performance improved after removing a feature—could it have been causing leakage?"

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
