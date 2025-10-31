# Seller Group Segmentation Analysis - Documentation

## Overview

This template provides a complete implementation for identifying customer characteristics that differentiate performance between two seller groups (A and B). The analysis answers three key questions:

1. **Which customer types does Group A serve better?**
2. **Which customer types does Group B serve better?**
3. **Which customer types show no meaningful difference?**

## Data Requirements

Your dataset should have the following structure:

| Column | Type | Description |
|--------|------|-------------|
| `client_id` | int | Unique client identifier |
| `group` | str | Seller group ('A' or 'B') |
| `sale_success` | int/bool | Sale outcome (0 = fail, 1 = success) |
| `feature_*` | mixed | Customer characteristics (numeric, categorical, binary) |

**Important:**
- Each client is served by exactly ONE seller (from one group only)
- Mix of feature types is supported:
  - **Numerical**: Most values are one dominant value (80%+ concentration)
  - **Categorical**: Multiple distinct categories
  - **Binary**: 0/1 or True/False values

## Three Methods Implemented

### 1. T-Learner with Shallow Decision Trees

**How it works:**
- Trains separate shallow decision trees (max_depth=2) for each seller group
- Extracts interpretable rules from leaf nodes
- Each rule represents a customer segment with specific characteristics

**Advantages:**
- Highly interpretable results
- Simple, actionable rules
- Handles feature interactions automatically

**Output:** Segments like "num_feature > 102.5 AND binary_feature ≤ 0.5"

---

### 2. Information Value (IV) Segmentation

**How it works:**
- Calculates IV for each feature in each group separately
- Identifies bins (ranges) with strongest discriminative power
- Ranks features by IV difference between groups

**Formula:**
```
WOE = ln(% of Events / % of Non-Events)
IV = (% of Events - % of Non-Events) × WOE
```

**Advantages:**
- Quantifies feature importance objectively
- Identifies "which bins matter most"
- Recursive approach finds nested segments

**Output:** Single-feature segments with bin ranges

---

### 3. Logistic Regression with L1 (LASSO)

**How it works:**
- Fits L1-regularized logistic regression for each group
- Automatic feature selection (some coefficients become zero)
- Creates segments based on median splits of top features

**Advantages:**
- Penalizes complex models
- Natural feature importance from coefficients
- Fast and scalable

**Output:** Segments based on threshold splits

---

## Output Table Structure

The final results table contains top 10 segments with these columns:

| Column | Description |
|--------|-------------|
| `Feature_1` | Primary segmentation feature |
| `Range_1` | Value range for Feature_1 (e.g., ">100.5" or "[2.1, 3.5]") |
| `Feature_2` | Secondary feature (if 2-way segment) |
| `Range_2` | Range for Feature_2 |
| `Pct_Train_%` | % of training data in this segment |
| `Pct_Test_%` | % of test data in this segment |
| `Success_Train_%` | Sales success rate in training set |
| `Success_Test_%` | Sales success rate in test set |
| `Success_Diff_%` | Test success - Train success (generalization measure) |
| `Method` | Which algorithm identified this segment |

---

## How to Use with Your Data

### Step 1: Load Your Data

```python
import pandas as pd

df = pd.read_csv('your_data.csv')
```

### Step 2: Modify Data Generation (Optional)

If using example data, skip this. Otherwise, replace the `generate_example_data()` call:

```python
# Instead of:
df = generate_example_data(n_samples=1500)

# Use:
df = pd.read_csv('your_data.csv')
```

### Step 3: Update Feature Classifications

In the `main()` function, specify your categorical features:

```python
categorical_features = ['cat_feature_1', 'cat_feature_2', 'cat_feature_3']
```

All features not in this list will be treated as numerical.

### Step 4: Run Analysis

```python
python seller_segmentation_template.py
```

This will output:
1. Console report with summary statistics
2. `top_10_segments_results.csv` - The main results table

---

## Interpretation Guide

### Example Result 1: Group B performs better
```
Feature_1: num_feature_2
Range_1: > 102.50
Feature_2: binary_feature_1
Range_2: <= 0.50
Success_Train_%: 71.83%
Success_Test_%: 74.51%
Success_Diff_%: 2.68%
```

**Interpretation:** 
- Customers with `num_feature_2 > 102.50` AND `binary_feature_1 ≤ 0.50` 
- Group B's success rate: ~74%
- Represents 33.6% of test population
- Stable performance (small train-test difference)

### Example Result 2: Group A performs better
```
Feature_1: num_feature_1
Range_1: [7.20, 9.00]
Success_Train_%: 7.14%
Success_Test_%: 25.0%
Success_Diff_%: 17.86%
```

**Interpretation:**
- Customers with `num_feature_1` between 7.2 and 9.0
- Group A's success rate: ~25%
- Small segment (2.7% of data)
- ⚠️ Note: Large train-test difference suggests small sample size

---

## Key Parameters to Adjust

### Tree Depth
```python
tlearner_method(..., max_depth=2, min_samples_leaf=10)
```
- Increase for more complex segments (less interpretable)
- Decrease for simpler segments (more robust)

### IV Binning
```python
calculate_iv(..., n_bins=5)
```
- More bins = finer granularity but potentially unstable
- Fewer bins = broader segments but may miss details

### Minimum Sample Size
Change throughout code:
```python
if n_train >= 5 and n_test >= 3:  # Adjust these thresholds
```

---

## Common Issues & Solutions

### Issue: Few segments found
**Solution:** Reduce minimum sample size requirements or increase max_depth in trees

### Issue: Different results between methods
**Solution:** This is expected! Different methods find different patterns. Use all three for comprehensive view.

### Issue: Very small segments in results
**Solution:** Filter results by `Pct_Train_%` > threshold to focus on larger populations

### Issue: Large train-test difference in success rates
**Solution:** Watch for overfitting indicators. May indicate small population size or high variability.

---

## Advanced Customization

### Add Feature Interactions
Modify `tlearner_method` to use `max_depth=3` for interaction terms.

### Different Success Criteria
Change target variable generation to use different success definitions.

### Weighted Samples
Add `sample_weight` parameter to fit methods if certain customers are more valuable.

### Group-Specific Analysis
Modify code to analyze more than 2 groups (currently fixed at A vs B).

---

## Output Files Generated

1. **top_10_segments_results.csv** - Main results table (numeric format)
2. **Console output** - Summary statistics and method distribution

To export to Excel for presentation:
```python
df = pd.read_csv('top_10_segments_results.csv')
df.to_excel('results.xlsx', index=False)
```

---

## Statistical Validation

The template includes built-in validation:

- **Train/Test Split**: 80/20 stratified split
- **Minimum Samples**: Only segments with n≥5 (train) and n≥3 (test) are reported
- **Stability Check**: Shows train vs test success rates
  - Small differences = stable pattern
  - Large differences = potential overfitting

---

## Literature References

**T-Learner / Causal Forests:**
- Athey, S., & Wager, S. (2019). "Generalized random forests"

**Information Value:**
- Naeem Siddiqi (2006). "Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring"

**L1 Regularization:**
- Tibshirani, R. (1996). "Regression shrinkage and selection via the lasso"

---

## Support

To modify this template:

1. Understand each method section (4-6 in code)
2. Adjust parameters for your business context
3. Run on your full dataset after testing on sample
4. Validate segments through A/B testing in production

For questions about interpretation, refer to the "Interpretation Guide" section above.
