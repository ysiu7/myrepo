# ============================================================================
# SELLER GROUP SEGMENTATION ANALYSIS
# Identifying customer characteristics that differ between seller groups
# ============================================================================
# This template implements three methods:
# 1. T-Learner with shallow decision trees
# 2. Information Value (IV) based segmentation
# 3. Logistic Regression with L1 (LASSO) penalty
# ============================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# SECTION 1: GENERATE EXAMPLE DATA
# ============================================================================

def generate_example_data(n_samples=1500, random_state=42):
    """
    Generate synthetic data with the following characteristics:
    - Two seller groups (A and B)
    - Multiple feature types (numerical with dominant values, categorical, binary)
    - Sales success as target variable (0/1)
    - Treatment heterogeneity (different groups perform better on different segments)
    """
    np.random.seed(random_state)
    
    data = {
        'client_id': range(n_samples),
        'group': np.random.choice(['A', 'B'], n_samples, p=[0.5, 0.5]),
        
        # Numerical features with one dominant value (>80% of cases)
        'num_feature_1': np.where(np.random.rand(n_samples) < 0.85, 0, np.random.randint(1, 10, n_samples)),
        'num_feature_2': np.where(np.random.rand(n_samples) < 0.90, 100, np.random.randint(0, 500, n_samples)),
        'num_feature_3': np.where(np.random.rand(n_samples) < 0.80, 1, np.random.randint(2, 20, n_samples)),
        'num_feature_4': np.where(np.random.rand(n_samples) < 0.88, 50, np.random.randint(10, 100, n_samples)),
        'num_feature_5': np.where(np.random.rand(n_samples) < 0.92, 0, np.random.exponential(5, n_samples).astype(int)),
        
        # Categorical features
        'cat_feature_1': np.random.choice(['X', 'Y', 'Z'], n_samples, p=[0.5, 0.3, 0.2]),
        'cat_feature_2': np.random.choice(['Small', 'Medium', 'Large'], n_samples),
        'cat_feature_3': np.random.choice(['Type_A', 'Type_B', 'Type_C', 'Type_D'], n_samples),
        
        # Binary features
        'binary_feature_1': np.random.binomial(1, 0.6, n_samples),
        'binary_feature_2': np.random.binomial(1, 0.4, n_samples),
        'binary_feature_3': np.random.binomial(1, 0.7, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Create target variable with treatment heterogeneity
    success_probs = []
    for idx, row in df.iterrows():
        base_prob = 0.3 if row['group'] == 'A' else 0.35
        
        # Different features have different effects by group
        if row['num_feature_1'] > 0 and row['group'] == 'A':
            base_prob += 0.15
        if row['num_feature_2'] > 100 and row['group'] == 'B':
            base_prob += 0.20
        if row['cat_feature_1'] == 'X' and row['group'] == 'A':
            base_prob += 0.10
        if row['binary_feature_1'] == 1 and row['group'] == 'B':
            base_prob += 0.12
        if row['num_feature_3'] <= 1 and row['group'] == 'A':
            base_prob -= 0.10
        
        success_probs.append(np.clip(base_prob, 0.1, 0.8))
    
    df['sale_success'] = [np.random.binomial(1, p) for p in success_probs]
    
    return df


# ============================================================================
# SECTION 2: DATA PREPARATION
# ============================================================================

def prepare_data(df, test_size=0.2, random_state=42):
    """
    Split data by group and prepare train/test sets
    """
    df_A = df[df['group'] == 'A'].copy()
    df_B = df[df['group'] == 'B'].copy()
    
    X_A = df_A.drop(['client_id', 'group', 'sale_success'], axis=1)
    y_A = df_A['sale_success']
    X_B = df_B.drop(['client_id', 'group', 'sale_success'], axis=1)
    y_B = df_B['sale_success']
    
    X_A_train, X_A_test, y_A_train, y_A_test = train_test_split(
        X_A, y_A, test_size=test_size, random_state=random_state, stratify=y_A
    )
    X_B_train, X_B_test, y_B_train, y_B_test = train_test_split(
        X_B, y_B, test_size=test_size, random_state=random_state, stratify=y_B
    )
    
    return {
        'X_A_train': X_A_train, 'X_A_test': X_A_test, 'y_A_train': y_A_train, 'y_A_test': y_A_test,
        'X_B_train': X_B_train, 'X_B_test': X_B_test, 'y_B_train': y_B_train, 'y_B_test': y_B_test,
        'feature_names': X_A.columns.tolist()
    }


def encode_features_for_modeling(X, categorical_features):
    """Encode categorical features for tree-based models"""
    X_encoded = X.copy()
    label_encoders = {}
    
    for cat_feat in categorical_features:
        if cat_feat in X_encoded.columns:
            le = LabelEncoder()
            X_encoded[cat_feat] = le.fit_transform(X_encoded[cat_feat].astype(str))
            label_encoders[cat_feat] = le
    
    return X_encoded, label_encoders


# ============================================================================
# SECTION 3: METHOD 1 - T-LEARNER WITH SHALLOW DECISION TREES
# ============================================================================

def tlearner_method(data_dict, categorical_features, feature_names, max_depth=2, min_samples_leaf=10):
    """
    Fit shallow decision trees for each group and extract interpretable rules
    """
    X_A_train_enc, _ = encode_features_for_modeling(data_dict['X_A_train'], categorical_features)
    X_A_test_enc, _ = encode_features_for_modeling(data_dict['X_A_test'], categorical_features)
    X_B_train_enc, _ = encode_features_for_modeling(data_dict['X_B_train'], categorical_features)
    X_B_test_enc, _ = encode_features_for_modeling(data_dict['X_B_test'], categorical_features)
    
    # Train trees
    tree_A = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=42)
    tree_A.fit(X_A_train_enc, data_dict['y_A_train'])
    
    tree_B = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=42)
    tree_B.fit(X_B_train_enc, data_dict['y_B_train'])
    
    # Extract rules from trees
    rules_A = extract_tree_rules_simple(
        tree_A, X_A_train_enc, X_A_test_enc, 
        data_dict['y_A_train'], data_dict['y_A_test'],
        feature_names, categorical_features, 'A',
        data_dict['X_A_train'], data_dict['X_A_test']
    )
    
    rules_B = extract_tree_rules_simple(
        tree_B, X_B_train_enc, X_B_test_enc, 
        data_dict['y_B_train'], data_dict['y_B_test'],
        feature_names, categorical_features, 'B',
        data_dict['X_B_train'], data_dict['X_B_test']
    )
    
    return rules_A + rules_B


def extract_tree_rules_simple(tree, X_train_enc, X_test_enc, y_train, y_test, 
                              feature_names, categorical_features, group_name, 
                              X_train_orig, X_test_orig):
    """Extract simple rules from decision tree leaves"""
    from sklearn.tree import _tree
    
    rules = []
    tree_structure = tree.tree_
    
    def recurse(node, depth, conditions):
        if depth > 2:
            return
        
        if tree_structure.feature[node] != _tree.TREE_UNDEFINED:
            feat_idx = tree_structure.feature[node]
            feat_name = feature_names[feat_idx]
            threshold = tree_structure.threshold[node]
            
            left_conds = conditions + [(feat_name, '<=', threshold)]
            recurse(tree_structure.children_left[node], depth + 1, left_conds)
            
            right_conds = conditions + [(feat_name, '>', threshold)]
            recurse(tree_structure.children_right[node], depth + 1, right_conds)
        else:
            # Leaf node
            mask_train = np.ones(len(X_train_enc), dtype=bool)
            mask_test = np.ones(len(X_test_enc), dtype=bool)
            
            for feat_name, op, thresh in conditions:
                feat_idx = feature_names.index(feat_name)
                
                if op == '<=':
                    mask_train &= X_train_enc.iloc[:, feat_idx] <= thresh
                    mask_test &= X_test_enc.iloc[:, feat_idx] <= thresh
                else:
                    mask_train &= X_train_enc.iloc[:, feat_idx] > thresh
                    mask_test &= X_test_enc.iloc[:, feat_idx] > thresh
            
            n_train = mask_train.sum()
            n_test = mask_test.sum()
            
            if n_train >= 5 and n_test >= 3:
                success_train = y_train[mask_train].mean()
                success_test = y_test[mask_test].mean()
                
                rule_parts = []
                for feat_name, op, thresh in conditions:
                    if feat_name in categorical_features:
                        rule_parts.append(f"{feat_name} encoded {op} {thresh:.1f}")
                    else:
                        rule_parts.append(f"{feat_name} {op} {thresh:.2f}")
                
                rule_str = " AND ".join(rule_parts) if rule_parts else "All"
                
                rules.append({
                    'rule': rule_str,
                    'feature_1': conditions[0][0] if conditions else 'All',
                    'range_1': f"{conditions[0][1]} {conditions[0][2]:.2f}" if conditions else 'All',
                    'feature_2': conditions[1][0] if len(conditions) > 1 else '-',
                    'range_2': f"{conditions[1][1]} {conditions[1][2]:.2f}" if len(conditions) > 1 else '-',
                    'n_train': n_train,
                    'n_test': n_test,
                    'pct_train': n_train / len(X_train_enc) * 100,
                    'pct_test': n_test / len(X_test_enc) * 100,
                    'success_train': success_train,
                    'success_test': success_test,
                    'success_diff': success_test - success_train,
                    'group': group_name,
                    'method': 'tree'
                })
    
    recurse(0, 0, [])
    return rules


# ============================================================================
# SECTION 4: METHOD 2 - INFORMATION VALUE (IV) SEGMENTATION
# ============================================================================

def calculate_iv(X, y, feature_names, categorical_features, n_bins=5):
    """Calculate Information Value for each feature"""
    iv_results = []
    
    for feat_name in feature_names:
        if feat_name in categorical_features:
            bins = X[feat_name].unique()
        else:
            _, bins = pd.cut(X[feat_name], bins=n_bins, retbins=True, duplicates='drop')
        
        if feat_name in categorical_features:
            binned = X[feat_name]
        else:
            binned = pd.cut(X[feat_name], bins=bins, duplicates='drop')
        
        bin_data = pd.DataFrame({'bin': binned, 'target': y})
        
        iv_total = 0
        bin_info = []
        
        for bin_val in bin_data['bin'].unique():
            mask = bin_data['bin'] == bin_val
            total_in_bin = mask.sum()
            events_in_bin = bin_data.loc[mask, 'target'].sum()
            non_events_in_bin = total_in_bin - events_in_bin
            
            total_events = y.sum()
            total_non_events = len(y) - total_events
            
            pct_events = events_in_bin / total_events if total_events > 0 else 0
            pct_non_events = non_events_in_bin / total_non_events if total_non_events > 0 else 0
            
            woe = np.log(pct_events / pct_non_events) if (pct_events > 0 and pct_non_events > 0) else 0
            iv_bin = (pct_events - pct_non_events) * woe
            
            bin_info.append({
                'bin': bin_val, 'count': total_in_bin, 'events': events_in_bin,
                'pct_events': pct_events, 'pct_non_events': pct_non_events, 'woe': woe, 'iv': iv_bin
            })
            
            iv_total += iv_bin
        
        iv_results.append({'feature': feat_name, 'iv': iv_total, 'bins_info': bin_info})
    
    return sorted(iv_results, key=lambda x: x['iv'], reverse=True)


def iv_segmentation_method(data_dict, categorical_features, feature_names):
    """Extract segments based on Information Value"""
    iv_A = calculate_iv(data_dict['X_A_train'], data_dict['y_A_train'], 
                        feature_names, categorical_features)
    iv_B = calculate_iv(data_dict['X_B_train'], data_dict['y_B_train'], 
                        feature_names, categorical_features)
    
    segments = []
    
    for group_letter, X_train, y_train, X_test, y_test, iv_list in [
        ('A', data_dict['X_A_train'], data_dict['y_A_train'], 
         data_dict['X_A_test'], data_dict['y_A_test'], iv_A),
        ('B', data_dict['X_B_train'], data_dict['y_B_train'], 
         data_dict['X_B_test'], data_dict['y_B_test'], iv_B)
    ]:
        
        for iv_item in iv_list[:3]:
            feat_name = iv_item['feature']
            bins_info = iv_item['bins_info']
            top_bins = sorted(bins_info, key=lambda x: abs(x['iv']), reverse=True)[:2]
            
            for bin_info in top_bins:
                bin_val = bin_info['bin']
                
                if feat_name in categorical_features:
                    mask_train = X_train[feat_name] == bin_val
                    mask_test = X_test[feat_name] == bin_val
                else:
                    try:
                        mask_train = X_train[feat_name].apply(lambda x: x in bin_val)
                        mask_test = X_test[feat_name].apply(lambda x: x in bin_val)
                    except:
                        mask_train = (X_train[feat_name] >= bin_val.left) & (X_train[feat_name] <= bin_val.right)
                        mask_test = (X_test[feat_name] >= bin_val.left) & (X_test[feat_name] <= bin_val.right)
                
                n_train, n_test = mask_train.sum(), mask_test.sum()
                
                if n_train >= 5 and n_test >= 3:
                    success_train = y_train[mask_train].mean()
                    success_test = y_test[mask_test].mean()
                    
                    bin_str = str(bin_val)
                    if not feat_name in categorical_features:
                        try:
                            bin_str = f"[{bin_val.left:.2f}, {bin_val.right:.2f}]"
                        except:
                            pass
                    
                    segments.append({
                        'feature_1': feat_name, 'range_1': bin_str,
                        'feature_2': '-', 'range_2': '-',
                        'n_train': n_train, 'n_test': n_test,
                        'pct_train': n_train / len(X_train) * 100,
                        'pct_test': n_test / len(X_test) * 100,
                        'success_train': success_train, 'success_test': success_test,
                        'success_diff': success_test - success_train,
                        'group': group_letter, 'method': 'IV'
                    })
    
    return segments


# ============================================================================
# SECTION 5: METHOD 3 - LOGISTIC REGRESSION WITH L1 (LASSO)
# ============================================================================

def lr_l1_segmentation(data_dict, categorical_features, feature_names):
    """Extract segments using L1-regularized logistic regression"""
    
    X_A_train_enc, _ = encode_features_for_modeling(data_dict['X_A_train'], categorical_features)
    X_B_train_enc, _ = encode_features_for_modeling(data_dict['X_B_train'], categorical_features)
    X_A_test_enc, _ = encode_features_for_modeling(data_dict['X_A_test'], categorical_features)
    X_B_test_enc, _ = encode_features_for_modeling(data_dict['X_B_test'], categorical_features)
    
    lr_A = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, random_state=42)
    lr_A.fit(X_A_train_enc, data_dict['y_A_train'])
    
    lr_B = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, random_state=42)
    lr_B.fit(X_B_train_enc, data_dict['y_B_train'])
    
    segments = []
    
    for group_letter, X_train, y_train, X_test, y_test, lr_model in [
        ('A', data_dict['X_A_train'], data_dict['y_A_train'], 
         data_dict['X_A_test'], data_dict['y_A_test'], lr_A),
        ('B', data_dict['X_B_train'], data_dict['y_B_train'], 
         data_dict['X_B_test'], data_dict['y_B_test'], lr_B)
    ]:
        
        # Get top features
        coef_abs = pd.DataFrame({'feature': feature_names, 'coef_abs': np.abs(lr_model.coef_[0])})
        top_features = coef_abs.nlargest(2, 'coef_abs')['feature'].tolist()
        
        for feat1 in top_features[:1]:
            if feat1 in categorical_features:
                for cat_val in X_train[feat1].unique():
                    mask_train = X_train[feat1] == cat_val
                    mask_test = X_test[feat1] == cat_val
                    
                    n_train, n_test = mask_train.sum(), mask_test.sum()
                    if n_train >= 5 and n_test >= 3:
                        segments.append({
                            'feature_1': feat1, 'range_1': str(cat_val),
                            'feature_2': '-', 'range_2': '-',
                            'n_train': n_train, 'n_test': n_test,
                            'pct_train': n_train / len(X_train) * 100,
                            'pct_test': n_test / len(X_test) * 100,
                            'success_train': y_train[mask_train].mean(),
                            'success_test': y_test[mask_test].mean(),
                            'success_diff': y_test[mask_test].mean() - y_train[mask_train].mean(),
                            'group': group_letter, 'method': 'LogReg'
                        })
            else:
                median_val = X_train[feat1].median()
                for op, op_name in [('<=', '<='), ('>', '>')]:
                    mask_train = X_train[feat1] <= median_val if op == '<=' else X_train[feat1] > median_val
                    mask_test = X_test[feat1] <= median_val if op == '<=' else X_test[feat1] > median_val
                    
                    n_train, n_test = mask_train.sum(), mask_test.sum()
                    if n_train >= 5 and n_test >= 3:
                        segments.append({
                            'feature_1': feat1, 'range_1': f"{op_name} {median_val:.2f}",
                            'feature_2': '-', 'range_2': '-',
                            'n_train': n_train, 'n_test': n_test,
                            'pct_train': n_train / len(X_train) * 100,
                            'pct_test': n_test / len(X_test) * 100,
                            'success_train': y_train[mask_train].mean(),
                            'success_test': y_test[mask_test].mean(),
                            'success_diff': y_test[mask_test].mean() - y_train[mask_train].mean(),
                            'group': group_letter, 'method': 'LogReg'
                        })
    
    return segments


# ============================================================================
# SECTION 6: MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 80)
    print("SELLER GROUP SEGMENTATION ANALYSIS")
    print("=" * 80)
    
    # Generate data
    print("\n1. Generating example data...")
    df = generate_example_data(n_samples=1500)
    print(f"   Data shape: {df.shape}")
    print(f"   Group distribution:\n{df['group'].value_counts()}")
    print(f"   Success rate by group:\n{df.groupby('group')['sale_success'].mean()}")
    
    # Prepare data
    print("\n2. Preparing train/test split...")
    categorical_features = ['cat_feature_1', 'cat_feature_2', 'cat_feature_3']
    data_dict = prepare_data(df, test_size=0.2)
    feature_names = data_dict['feature_names']
    print(f"   Features: {len(feature_names)}")
    print(f"   Categorical: {categorical_features}")
    
    # Run methods
    print("\n3. Running segmentation methods...")
    
    print("   - T-Learner with shallow trees...")
    tree_segments = tlearner_method(data_dict, categorical_features, feature_names)
    print(f"     Found {len(tree_segments)} segments")
    
    print("   - Information Value (IV) segmentation...")
    iv_segments = iv_segmentation_method(data_dict, categorical_features, feature_names)
    print(f"     Found {len(iv_segments)} segments")
    
    print("   - Logistic Regression (L1)...")
    lr_segments = lr_l1_segmentation(data_dict, categorical_features, feature_names)
    print(f"     Found {len(lr_segments)} segments")
    
    # Combine and rank
    print("\n4. Combining and ranking results...")
    all_results = []
    
    for seg_list in [tree_segments, iv_segments, lr_segments]:
        for seg in seg_list:
            if seg['n_train'] >= 5 and seg['n_test'] >= 3:
                all_results.append({
                    'feature_1': seg['feature_1'],
                    'range_1': seg['range_1'],
                    'feature_2': seg['feature_2'],
                    'range_2': seg['range_2'],
                    'pct_train': seg['pct_train'],
                    'pct_test': seg['pct_test'],
                    'success_train': seg['success_train'],
                    'success_test': seg['success_test'],
                    'success_diff': seg['success_diff'],
                    'method': seg['method']
                })
    
    results_df = pd.DataFrame(all_results)
    results_df_unique = results_df.drop_duplicates(
        subset=['feature_1', 'range_1', 'feature_2', 'range_2'], keep='first'
    )
    results_df_unique['abs_success_diff'] = results_df_unique['success_diff'].abs()
    
    # Get top 10
    top_10 = results_df_unique.nlargest(10, 'abs_success_diff')
    
    # Format output
    print("\n5. Top 10 results:")
    final_output = top_10[[
        'feature_1', 'range_1', 'feature_2', 'range_2',
        'pct_train', 'pct_test', 'success_train', 'success_test', 'success_diff', 'method'
    ]].copy()
    
    final_output.columns = [
        'Feature_1', 'Range_1', 'Feature_2', 'Range_2',
        'Pct_Train_%', 'Pct_Test_%', 'Success_Train_%', 'Success_Test_%', 'Success_Diff_%', 'Method'
    ]
    
    print("\n" + final_output.to_string())
    
    # Save results
    final_output.to_csv('top_10_segments_results.csv')
    print("\n✓ Results saved to 'top_10_segments_results.csv'")
    
    return top_10, results_df_unique


if __name__ == "__main__":
    top_10_results, all_segments = main()
