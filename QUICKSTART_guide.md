# Quick Start Guide - Seller Group Segmentation

## 5-Minute Setup

### 1. Install Requirements
```bash
pip install pandas numpy scikit-learn
```

### 2. Load Your Data

Replace the data generation with your actual data:

```python
import pandas as pd
from seller_segmentation_template import *

# Load your data
df = pd.read_csv('your_sales_data.csv')

# Make sure columns match: client_id, group, sale_success, and features
print(df.head())
print(df.columns)
```

### 3. Update Categorical Features

```python
# List all categorical feature columns from your data
categorical_features = [
    'region', 'product_type', 'customer_segment'
]
# All other features will be treated as numerical
```

### 4. Run Analysis

```python
# Prepare data
data_dict = prepare_data(df, test_size=0.2)
feature_names = data_dict['feature_names']

# Run all methods
tree_segments = tlearner_method(data_dict, categorical_features, feature_names)
iv_segments = iv_segmentation_method(data_dict, categorical_features, feature_names)
lr_segments = lr_l1_segmentation(data_dict, categorical_features, feature_names)

# Combine and get top 10
all_results = []
for seg_list in [tree_segments, iv_segments, lr_segments]:
    all_results.extend(seg_list)

results_df = pd.DataFrame(all_results)
results_df_unique = results_df.drop_duplicates(
    subset=['feature_1', 'range_1', 'feature_2', 'range_2'], keep='first'
)
results_df_unique['abs_success_diff'] = results_df_unique['success_diff'].abs()
top_10 = results_df_unique.nlargest(10, 'abs_success_diff')

# Save results
top_10.to_csv('results.csv', index=False)
```

---

## Real-World Example: SaaS Sales Team

### Your Data Structure
```
client_id | group | sale_success | company_size | industry | region | num_contacts | budget_range | contacted_before
----------|-------|--------------|--------------|----------|--------|--------------|--------------|------------------
1001      | A     | 1            | 50           | tech     | US     | 3            | high         | 1
1002      | B     | 0            | 2            | retail   | EU     | 1            | low          | 0
1003      | A     | 1            | 100          | finance  | US     | 5            | high         | 1
...
```

### Modified Code
```python
import pandas as pd
from seller_segmentation_template import *

# Load data
df = pd.read_csv('sales_team_data.csv')

# Specify categorical features
categorical_features = ['industry', 'region', 'budget_range', 'contacted_before']

# Run analysis
data_dict = prepare_data(df, test_size=0.2)
main()

# Review results
results = pd.read_csv('top_10_segments_results.csv')
print(results)
```

### Expected Output Example

**Segment 1 (Group A much better):**
- company_size > 50 AND num_contacts ≥ 3
- Success rate: 65% (vs 35% for Group B)
- Represents: 12% of data
- → **Recommendation:** Route large companies with multiple decision-makers to Group A

**Segment 2 (Group B much better):**
- company_size ≤ 10 AND budget_range = "low"
- Success rate: 42% (vs 15% for Group A)
- Represents: 18% of data
- → **Recommendation:** Route small startups to Group B (better at closing quick deals)

**Segment 3 (No difference):**
- contacted_before = 1 AND region = "EU"
- Success rate: 58% (Group A) vs 57% (Group B)
- Represents: 9% of data
- → **Recommendation:** Either group OK; use other criteria

---

## Interpreting Results Table

### Column-by-Column Guide

```
Feature_1: "company_size"
Range_1: "> 50"
  → Customers where company_size > 50

Feature_2: "num_contacts"
Range_2: "<= 1"
  → Additional filter: and num_contacts ≤ 1

Pct_Train_%: 12.3%
Pct_Test_%: 11.8%
  → This segment is ~12% of your customer base
  → Numbers are similar, good generalization

Success_Train_%: 65.4%
Success_Test_%: 64.2%
  → Both train and test are close → stable pattern
  → Reliable for real-world deployment

Success_Diff_%: -1.2%
  → Small difference between train and test
  → ✓ Low risk of overfitting

Method: "tree"
  → Decision tree identified this pattern
```

---

## Customization Examples

### Example 1: Focus on Large Segments Only
```python
# Filter for segments representing >5% of data
top_10_filtered = top_10[top_10['pct_test'] > 5]
```

### Example 2: Find Group A Specialties
```python
# Filter results to only Group A's best segments
group_a_specialty = top_10[top_10['success_test'] > 0.6]
```

### Example 3: Stable vs Volatile Segments
```python
top_10['stability'] = abs(
    top_10['success_train'] - top_10['success_test']
)
# Smaller = more stable
stable_segs = top_10[top_10['stability'] < 0.05]
volatile_segs = top_10[top_10['stability'] > 0.10]
```

### Example 4: Compare Methods
```python
tree_results = top_10[top_10['method'] == 'tree']
iv_results = top_10[top_10['method'] == 'IV']
lr_results = top_10[top_10['method'] == 'LogReg']

print(f"Tree found {len(tree_results)} segments")
print(f"IV found {len(iv_results)} segments")
print(f"LogReg found {len(lr_results)} segments")
```

---

## Production Deployment Checklist

- [ ] Data validation (check for missing values, data types)
- [ ] Run on training data: `df_train = df[df['date'] < '2024-01-01']`
- [ ] Validate on holdout test: `df_test = df[df['date'] >= '2024-01-01']`
- [ ] Manual review: Do segments make business sense?
- [ ] A/B test before full rollout
- [ ] Monitor segment stability over time (quarterly refresh recommended)
- [ ] Set up alerts if segment success rates deviate by >5%

---

## Common Adaptations

### Scenario 1: More than 2 groups (A, B, C, D)
```python
# Run analysis for each pair:
# A vs B, A vs C, B vs D, etc.
# Store results separately
```

### Scenario 2: Different success metrics
```python
# Instead of binary (0/1), use continuous:
df['sale_success'] = df['revenue']  # Use revenue as target

# Or classification with multiple classes:
df['sale_success'] = (df['revenue'] > threshold).astype(int)
```

### Scenario 3: Time-based analysis
```python
# Analyze Q1 data only:
df_q1 = df[df['quarter'] == 'Q1']

# Or track how segments change over time:
for quarter in ['Q1', 'Q2', 'Q3', 'Q4']:
    df_quarter = df[df['quarter'] == quarter]
    # Run analysis and store results
```

### Scenario 4: Weighted importance
```python
# Some customers are more valuable than others
df['customer_weight'] = df['annual_revenue'] / df['annual_revenue'].max()

# Pass to tree:
tree_A.fit(X_A_train_enc, y_A_train, 
           sample_weight=df_A['customer_weight'])
```

---

## Troubleshooting

**Q: Getting "ValueError: too many values to unpack"**
A: Check that categorical_features list matches column names exactly

**Q: Very few segments found**
A: Reduce minimum sample size:
```python
if n_train >= 2 and n_test >= 1:  # Lower thresholds
```

**Q: Results don't make business sense**
A: Check if features are correctly classified (categorical vs numerical)
```python
df.info()  # Verify data types
df.describe()  # Check value distributions
```

**Q: Want simpler segments (1 feature only)**
A: Modify feature extraction to force feature_2 = '-'

**Q: Want 3-way segments (3 features)**
A: Increase tree depth:
```python
tree_A = DecisionTreeClassifier(max_depth=3, ...)
```

---

## Next Steps

1. **Implement**: Deploy segment-based routing to sales teams
2. **Monitor**: Track success rate by segment monthly
3. **Iterate**: Retrain model quarterly or when business changes
4. **Expand**: Add more features as data becomes available
5. **Combine**: Layer results with geography, seasonality, etc.

---

## Key Metrics to Track

After implementing recommendations:

- **Overall win rate**: Should increase
- **Win rate by segment**: Should stabilize
- **Win rate by team**: Group A and B should specialize
- **Deal cycle time**: May decrease if routing is effective
- **Rep productivity**: Monitor individual performance by segment

Set targets:
- Increase Group A's specialty segments by 10% win rate
- Increase Group B's specialty segments by 8% win rate
- Maintain baseline segments at current rates
