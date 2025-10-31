
# ===================================================================
# KOMPLETNY PRZYKŁAD: Analiza uplift dla sprzedawców A vs B
# ===================================================================

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy.stats import chi2_contingency, bootstrap
import matplotlib.pyplot as plt
import seaborn as sns

# Ustawienia
np.random.seed(42)
sns.set_style('whitegrid')

# ===================================================================
# KROK 1: GENEROWANIE DANYCH PRZYKŁADOWYCH
# ===================================================================

def generate_sales_data(n_customers=2000):
    """
    Generuje syntetyczne dane sprzedażowe z heterogenicznymi efektami.

    Zasymulowane efekty:
    - Młodzi klienci (<30) lepiej reagują na grupę A
    - Starsi klienci (>=50) lepiej reagują na grupę B
    - Klienci o wysokich dochodach lepiej reagują na A
    - W miastach: A jest lepsza, na wsi: B jest lepsza
    """

    data = {
        'customer_id': range(n_customers),
        'age': np.random.randint(18, 75, n_customers),
        'income': np.random.choice(['low', 'medium', 'high'], n_customers, p=[0.3, 0.5, 0.2]),
        'location': np.random.choice(['urban', 'rural'], n_customers, p=[0.6, 0.4]),
        'previous_purchases': np.random.randint(0, 20, n_customers),
        'product_interest': np.random.choice(['electronics', 'fashion', 'home'], n_customers),
        'group': np.random.choice(['A', 'B'], n_customers, p=[0.55, 0.45])  # Lekko niezbalansowane
    }

    df = pd.DataFrame(data)

    # Symulacja sukcesu sprzedaży z interakcjami
    df['sale'] = 0

    for idx in df.index:
        base_prob = 0.25  # Bazowe prawdopodobieństwo

        age = df.loc[idx, 'age']
        income = df.loc[idx, 'income']
        location = df.loc[idx, 'location']
        group = df.loc[idx, 'group']
        prev_purch = df.loc[idx, 'previous_purchases']

        # Efekt wieku - młodzi preferują A, starsi B
        if age < 30 and group == 'A':
            base_prob += 0.25
        elif age >= 50 and group == 'B':
            base_prob += 0.20

        # Efekt dochodu - wysokie dochody preferują A
        if income == 'high' and group == 'A':
            base_prob += 0.20
        elif income == 'low' and group == 'B':
            base_prob += 0.10

        # Efekt lokacji
        if location == 'urban' and group == 'A':
            base_prob += 0.15
        elif location == 'rural' and group == 'B':
            base_prob += 0.12

        # Efekt poprzednich zakupów
        if prev_purch > 10 and group == 'A':
            base_prob += 0.10

        # Losowanie wyniku
        df.loc[idx, 'sale'] = np.random.binomial(1, min(base_prob, 0.95))

    return df

# Generuj dane
print("="*70)
print("GENEROWANIE DANYCH PRZYKŁADOWYCH")
print("="*70)

df = generate_sales_data(n_customers=2000)

print(f"\nWygenerowano {len(df)} klientów")
print(f"Grupa A: {(df['group']=='A').sum()} ({(df['group']=='A').mean():.1%})")
print(f"Grupa B: {(df['group']=='B').sum()} ({(df['group']=='B').mean():.1%})")
print(f"\nSuccess rate Grupa A: {df[df['group']=='A']['sale'].mean():.2%}")
print(f"Success rate Grupa B: {df[df['group']=='B']['sale'].mean():.2%}")

# ===================================================================
# KROK 2: EKSPLORACYJNA ANALIZA DANYCH (EDA)
# ===================================================================

print("\n" + "="*70)
print("EKSPLORACYJNA ANALIZA DANYCH")
print("="*70)

# Test czy jest ogólny efekt grupy
contingency = pd.crosstab(df['group'], df['sale'])
chi2, pval, _, _ = chi2_contingency(contingency)
print(f"\nTest chi-kwadrat dla głównego efektu grupy:")
print(f"Chi2 = {chi2:.2f}, p-value = {pval:.4f}")
if pval < 0.05:
    print("✓ Istnieje statystycznie istotny główny efekt grupy")
else:
    print("✗ Brak głównego efektu grupy - trudniej będzie znaleźć heterogeniczne efekty")

# Analiza po cechach
print("\n--- Analiza success rate po cechach ---")

for feature in ['age', 'income', 'location']:
    if feature == 'age':
        # Dla wieku: podziel na kategorie
        df_temp = df.copy()
        df_temp['age_cat'] = pd.cut(df_temp['age'], bins=[0, 30, 50, 100], 
                                     labels=['<30', '30-50', '50+'])
        pivot = pd.crosstab([df_temp['age_cat'], df_temp['group']], 
                           df_temp['sale'], normalize='index')
        print(f"\n{feature}:")
        print(pivot[1].unstack())
    else:
        pivot = pd.crosstab([df[feature], df['group']], 
                           df['sale'], normalize='index')
        print(f"\n{feature}:")
        print(pivot[1].unstack())

# ===================================================================
# KROK 3: METODA PROSTA - Drzewo decyzyjne
# ===================================================================

print("\n" + "="*70)
print("METODA 1: DRZEWO DECYZYJNE (INTERPRETOWALNE)")
print("="*70)

# Przygotowanie danych
feature_cols = ['age', 'income', 'location', 'previous_purchases', 'product_interest']
X = df[feature_cols].copy()

# Konwersja kategorycznych na dummy
X_encoded = pd.get_dummies(X, drop_first=False)

# Target: pozytywny uplift dla A, negatywny dla B
y_uplift = df['sale'].copy()
y_uplift[df['group'] == 'B'] = -y_uplift[df['group'] == 'B']

# Trenuj drzewo
tree_model = DecisionTreeClassifier(
    max_depth=3,              # Max 3 cechy
    min_samples_leaf=80,      # Min 80 obserwacji na liść
    min_samples_split=160,
    random_state=42
)

tree_model.fit(X_encoded, y_uplift)

# Wyświetl reguły
print("\nReguły drzewa decyzyjnego:")
rules = export_text(tree_model, feature_names=list(X_encoded.columns))
print(rules[:1000], "...")  # Pokaż pierwsze 1000 znaków

# Przypisz segmenty
df['tree_segment'] = tree_model.apply(X_encoded)

# Analiza segmentów
print("\n--- Analiza segmentów z drzewa ---")
segments_analysis = []

for seg in df['tree_segment'].unique():
    seg_data = df[df['tree_segment'] == seg]
    seg_a = seg_data[seg_data['group'] == 'A']
    seg_b = seg_data[seg_data['group'] == 'B']

    if len(seg_a) >= 30 and len(seg_b) >= 30:  # Min 30 obs
        sr_a = seg_a['sale'].mean()
        sr_b = seg_b['sale'].mean()
        diff = sr_a - sr_b

        segments_analysis.append({
            'segment': seg,
            'n_total': len(seg_data),
            'n_A': len(seg_a),
            'n_B': len(seg_b),
            'SR_A': f"{sr_a:.1%}",
            'SR_B': f"{sr_b:.1%}",
            'Różnica': f"{diff:+.1%}",
            'Interpretacja': 'A lepsze' if diff > 0.10 else ('B lepsze' if diff < -0.10 else 'Brak różnicy')
        })

seg_df = pd.DataFrame(segments_analysis).sort_values('Różnica', ascending=False)
print(seg_df.to_string(index=False))

# ===================================================================
# KROK 4: METODA ZAAWANSOWANA - T-Learner
# ===================================================================

print("\n" + "="*70)
print("METODA 2: T-LEARNER (MACHINE LEARNING)")
print("="*70)

# Podział na train/test
train_df, test_df = train_test_split(df, test_size=0.3, 
                                     stratify=df['group'].astype(str) + '_' + df['sale'].astype(str),
                                     random_state=42)

print(f"\nRozmiar train: {len(train_df)}, test: {len(test_df)}")

# Przygotuj dane
X_train = pd.get_dummies(train_df[feature_cols], drop_first=True)
X_test = pd.get_dummies(test_df[feature_cols], drop_first=True)

# Upewnij się że kolumny są takie same
for col in X_train.columns:
    if col not in X_test.columns:
        X_test[col] = 0
X_test = X_test[X_train.columns]

y_train = train_df['sale']
treatment_train = (train_df['group'] == 'A').astype(int)

# Model dla grupy A
model_a = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_leaf=30,
    random_state=42
)
model_a.fit(X_train[treatment_train == 1], y_train[treatment_train == 1])

# Model dla grupy B
model_b = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_leaf=30,
    random_state=42
)
model_b.fit(X_train[treatment_train == 0], y_train[treatment_train == 0])

# Predykcje uplift na test
pred_a = model_a.predict_proba(X_test)[:, 1]
pred_b = model_b.predict_proba(X_test)[:, 1]
uplift = pred_a - pred_b

test_df = test_df.copy()
test_df['uplift'] = uplift

# Segmentacja na podstawie uplift
test_df['uplift_segment'] = pd.qcut(test_df['uplift'], q=5, 
                                     labels=['B_dużo_lepsze', 'B_lepsze', 'Neutralne', 
                                            'A_lepsze', 'A_dużo_lepsze'],
                                     duplicates='drop')

print("\n--- Analiza segmentów uplift (T-Learner) ---")
uplift_analysis = []

for seg in test_df['uplift_segment'].unique():
    seg_data = test_df[test_df['uplift_segment'] == seg]
    seg_a = seg_data[seg_data['group'] == 'A']
    seg_b = seg_data[seg_data['group'] == 'B']

    if len(seg_a) > 0 and len(seg_b) > 0:
        uplift_analysis.append({
            'Segment': seg,
            'n_total': len(seg_data),
            'Śr. uplift': f"{seg_data['uplift'].mean():+.3f}",
            'SR_A': f"{seg_a['sale'].mean():.1%}" if len(seg_a) > 0 else 'N/A',
            'SR_B': f"{seg_b['sale'].mean():.1%}" if len(seg_b) > 0 else 'N/A',
            'Obserwowana różnica': f"{seg_a['sale'].mean() - seg_b['sale'].mean():+.1%}" 
                                  if len(seg_a) > 0 and len(seg_b) > 0 else 'N/A'
        })

uplift_df = pd.DataFrame(uplift_analysis)
print(uplift_df.to_string(index=False))

# Feature importance
print("\n--- Ważność cech (Feature Importance) ---")
feature_imp_a = pd.DataFrame({
    'feature': X_train.columns,
    'importance_A': model_a.feature_importances_,
    'importance_B': model_b.feature_importances_
})
feature_imp_a['importance_diff'] = feature_imp_a['importance_A'] - feature_imp_a['importance_B']
feature_imp_a = feature_imp_a.sort_values('importance_diff', ascending=False)
print(feature_imp_a.head(10).to_string(index=False))

# ===================================================================
# KROK 5: ANALIZA Z PRZEDZIAŁAMI UFNOŚCI
# ===================================================================

print("\n" + "="*70)
print("ANALIZA Z PRZEDZIAŁAMI UFNOŚCI (BOOTSTRAP)")
print("="*70)

def bootstrap_uplift(group_a_outcomes, group_b_outcomes, n_bootstrap=1000):
    """Oblicza uplift z bootstrap CI"""

    def uplift_stat(data_a, data_b):
        return data_a.mean() - data_b.mean()

    # Oblicz punkt estimate
    point_est = uplift_stat(group_a_outcomes, group_b_outcomes)

    # Bootstrap
    bootstrap_uplifts = []
    rng = np.random.default_rng(42)

    for _ in range(n_bootstrap):
        # Resample z replacement
        boot_a = rng.choice(group_a_outcomes, size=len(group_a_outcomes), replace=True)
        boot_b = rng.choice(group_b_outcomes, size=len(group_b_outcomes), replace=True)
        bootstrap_uplifts.append(uplift_stat(boot_a, boot_b))

    # CI
    ci_lower = np.percentile(bootstrap_uplifts, 2.5)
    ci_upper = np.percentile(bootstrap_uplifts, 97.5)

    return point_est, ci_lower, ci_upper

# Przykładowa analiza dla segmentów wieku
print("\nBootstrap CI dla segmentów wieku:")

age_bins = [0, 30, 50, 100]
age_labels = ['<30', '30-50', '50+']
test_df['age_group'] = pd.cut(test_df['age'], bins=age_bins, labels=age_labels)

for age_group in age_labels:
    seg_data = test_df[test_df['age_group'] == age_group]

    outcomes_a = seg_data[seg_data['group'] == 'A']['sale'].values
    outcomes_b = seg_data[seg_data['group'] == 'B']['sale'].values

    if len(outcomes_a) >= 30 and len(outcomes_b) >= 30:
        point, ci_low, ci_high = bootstrap_uplift(outcomes_a, outcomes_b, n_bootstrap=1000)

        sig = "✓ Istotne" if (ci_low > 0 or ci_high < 0) else "✗ Nieistotne"
        print(f"{age_group}: {point:+.3f} (95% CI: [{ci_low:+.3f}, {ci_high:+.3f}]) {sig}")

# ===================================================================
# KROK 6: PODSUMOWANIE I REKOMENDACJE
# ===================================================================

print("\n" + "="*70)
print("PODSUMOWANIE I REKOMENDACJE")
print("="*70)

print("""
ZNALEZIONE WZORCE:

1. **Wiek**:
   - Klienci <30 lat: Grupa A osiąga wyższy success rate
   - Klienci 50+: Grupa B osiąga wyższy success rate
   - Grupa 30-50: Brak wyraźnej różnicy

2. **Lokalizacja**:
   - Miasta (urban): Grupa A lepsza
   - Wieś (rural): Grupa B lepsza

3. **Dochód**:
   - Wysokie dochody: Grupa A zdecydowanie lepsza
   - Niskie dochody: Lekka przewaga grupy B

REKOMENDACJE BIZNESOWE:

→ Przypisuj do grupy A:
  - Młodych klientów (<30)
  - Klientów z miast
  - Klientów o wysokich dochodach

→ Przypisuj do grupy B:
  - Starszych klientów (50+)
  - Klientów z obszarów wiejskich
  - Klientów o niskich dochodach

→ Dowolna grupa (brak różnicy):
  - Klienci 30-50 lat bez wyraźnych preferencji

POTENCJALNY WZROST KONWERSJI:
Przy optymalnym przypisaniu klientów do grup, można zwiększyć
ogólny success rate o około 5-10 punktów procentowych.
"""
)

print("\n" + "="*70)
print("ANALIZA ZAKOŃCZONA")
print("="*70)

# Zapisz wyniki
test_df.to_csv('uplift_analysis_results.csv', index=False)
print("\nWyniki zapisane do: uplift_analysis_results.csv")
