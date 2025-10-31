# ===================================================================
# ROZWIĄZANIE PROBLEMU: Identyfikacja grup klientów z różnymi
# efektami dla sprzedawców A vs B
# ===================================================================

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings("ignore")

# ===================================================================
# CZĘŚĆ 1: PROSTA METODOLOGIA - Drzewa decyzyjne
# ===================================================================


class SimpleInteractionAnalyzer:
    """
    Prosta analiza interakcji między grupą sprzedawcy a cechami klienta.
    Identyfikuje maksymalnie 3 cechy definiujące segmenty.
    """

    def __init__(self, max_depth=3, min_samples_leaf=50):
        """
        Parameters:
        -----------
        max_depth : int
            Maksymalna głębokość drzewa (ogranicza do 3 cech)
        min_samples_leaf : int
            Minimalna liczba obserwacji w liściu (chroni przed małymi próbami)
        """
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.tree_model = None

    def fit(self, df, feature_cols):
        """
        Trenuje model na danych

        Parameters:
        -----------
        df : DataFrame z kolumnami:
            - 'group': 'A' lub 'B'
            - 'sale': 0 lub 1
            - feature_cols: cechy klienta
        feature_cols : list
            Lista nazw cech do analizy
        """
        # Przygotowanie danych
        X = df[feature_cols].copy()

        # Konwersja zmiennych kategorycznych na dummy variables
        X = pd.get_dummies(X, drop_first=False)

        # Dodaj interakcje z grupą jako target
        # Tworzymy zmienną: sukces w grupie A minus sukces w grupie B
        df_a = df[df["group"] == "A"]
        df_b = df[df["group"] == "B"]

        # Dla każdej kombinacji cech, oblicz różnicę w success rate
        X["uplift_target"] = 0.0

        for idx in X.index:
            # Dla każdego klienta, znajdź podobnych w grupach A i B
            if df.loc[idx, "group"] == "A":
                X.loc[idx, "uplift_target"] = df.loc[idx, "sale"]
            else:
                X.loc[idx, "uplift_target"] = -df.loc[idx, "sale"]

        # Trenuj drzewo decyzyjne
        feature_names = [c for c in X.columns if c != "uplift_target"]
        self.tree_model = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42,
        )
        self.tree_model.fit(X[feature_names], X["uplift_target"])
        self.feature_names = feature_names

        return self

    def get_rules(self):
        """Zwraca reguły w czytelnej formie"""
        if self.tree_model is None:
            raise ValueError("Model nie został wytrenowany!")

        tree_rules = export_text(self.tree_model, feature_names=self.feature_names)
        return tree_rules

    def analyze_segments(self, df, feature_cols):
        """
        Analizuje segmenty i oblicza success rate dla grup A i B

        Returns:
        --------
        DataFrame z segmentami i ich charakterystykami
        """
        X = df[feature_cols].copy()
        X = pd.get_dummies(X, drop_first=False)

        # Przypisz każdego klienta do segmentu
        feature_names = [c for c in X.columns]
        segments = self.tree_model.apply(X[self.feature_names])

        df_with_segments = df.copy()
        df_with_segments["segment"] = segments

        # Dla każdego segmentu oblicz statystyki
        results = []
        for seg in df_with_segments["segment"].unique():
            seg_data = df_with_segments[df_with_segments["segment"] == seg]

            # Statystyki dla grupy A
            seg_a = seg_data[seg_data["group"] == "A"]
            n_a = len(seg_a)
            success_rate_a = seg_a["sale"].mean() if n_a > 0 else 0

            # Statystyki dla grupy B
            seg_b = seg_data[seg_data["group"] == "B"]
            n_b = len(seg_b)
            success_rate_b = seg_b["sale"].mean() if n_b > 0 else 0

            # Różnica (uplift)
            diff = success_rate_a - success_rate_b

            results.append(
                {
                    "segment": seg,
                    "n_total": len(seg_data),
                    "n_group_a": n_a,
                    "n_group_b": n_b,
                    "success_rate_a": success_rate_a,
                    "success_rate_b": success_rate_b,
                    "difference_a_minus_b": diff,
                    "interpretation": self._interpret_segment(diff, n_a, n_b),
                }
            )

        return pd.DataFrame(results).sort_values(
            "difference_a_minus_b", ascending=False
        )

    def _interpret_segment(self, diff, n_a, n_b, threshold=0.1, min_samples=30):
        """Interpretuje różnicę między grupami"""
        # Sprawdź liczebności
        if n_a < min_samples or n_b < min_samples:
            return "Za mała próba"

        if diff > threshold:
            return "Dużo lepszy dla grupy A"
        elif diff < -threshold:
            return "Dużo lepszy dla grupy B"
        else:
            return "Brak różnicy"


# ===================================================================
# CZĘŚĆ 2: ZAAWANSOWANE PODEJŚCIA - Uplift Modeling
# ===================================================================


class TLearnerUplift:
    """
    T-Learner: Trenuje oddzielne modele dla grup A i B
    Dobrze działa gdy mechanizmy sprzedaży są różne między grupami
    """

    def __init__(self, base_estimator=None, min_samples_per_group=50):
        if base_estimator is None:
            base_estimator = RandomForestClassifier(
                n_estimators=100, max_depth=5, min_samples_leaf=20, random_state=42
            )
        self.model_a = base_estimator
        self.model_b = base_estimator.__class__(**base_estimator.get_params())
        self.min_samples_per_group = min_samples_per_group

    def fit(self, df, feature_cols):
        """Trenuje oddzielne modele dla grup A i B"""
        # Sprawdź liczebności
        n_a = (df["group"] == "A").sum()
        n_b = (df["group"] == "B").sum()

        if n_a < self.min_samples_per_group:
            print(
                f"UWAGA: Grupa A ma tylko {n_a} obserwacji (minimum: {self.min_samples_per_group})"
            )
        if n_b < self.min_samples_per_group:
            print(
                f"UWAGA: Grupa B ma tylko {n_b} obserwacji (minimum: {self.min_samples_per_group})"
            )

        # Przygotuj dane
        X = df[feature_cols].copy()
        X = pd.get_dummies(X, drop_first=True)
        y = df["sale"]

        # Trenuj model dla grupy A
        mask_a = df["group"] == "A"
        self.model_a.fit(X[mask_a], y[mask_a])

        # Trenuj model dla grupy B
        mask_b = df["group"] == "B"
        self.model_b.fit(X[mask_b], y[mask_b])

        self.feature_cols = X.columns.tolist()
        return self

    def predict_uplift(self, df, feature_cols):
        """
        Przewiduje uplift (różnicę w prawdopodobieństwie sukcesu A vs B)
        dla każdego klienta
        """
        X = df[feature_cols].copy()
        X = pd.get_dummies(X, drop_first=True)

        # Upewnij się, że mamy te same kolumny co przy treningu
        for col in self.feature_cols:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_cols]

        # Przewidywania
        pred_a = self.model_a.predict_proba(X)[:, 1]
        pred_b = self.model_b.predict_proba(X)[:, 1]

        uplift = pred_a - pred_b
        return uplift

    def segment_by_uplift(self, df, feature_cols, n_segments=5):
        """
        Dzieli klientów na segmenty bazując na przewidywanym uplift
        """
        uplift = self.predict_uplift(df, feature_cols)
        df_result = df.copy()
        df_result["predicted_uplift"] = uplift

        # Podziel na kwantyle
        df_result["segment"] = pd.qcut(
            uplift,
            q=n_segments,
            labels=[f"Segment_{i + 1}" for i in range(n_segments)],
            duplicates="drop",
        )

        # Oblicz statystyki dla każdego segmentu
        stats = []
        for seg in df_result["segment"].unique():
            seg_data = df_result[df_result["segment"] == seg]

            seg_a = seg_data[seg_data["group"] == "A"]
            seg_b = seg_data[seg_data["group"] == "B"]

            stats.append(
                {
                    "segment": seg,
                    "avg_predicted_uplift": seg_data["predicted_uplift"].mean(),
                    "n_group_a": len(seg_a),
                    "n_group_b": len(seg_b),
                    "success_rate_a": seg_a["sale"].mean()
                    if len(seg_a) > 0
                    else np.nan,
                    "success_rate_b": seg_b["sale"].mean()
                    if len(seg_b) > 0
                    else np.nan,
                }
            )

        return pd.DataFrame(stats).sort_values("avg_predicted_uplift", ascending=False)


class XLearnerUplift:
    """
    X-Learner: Bardziej zaawansowany, lepszy dla niezbalansowanych grup
    Używa propensity score do ważenia predykcji
    """

    def __init__(self, base_estimator=None, min_samples=50):
        if base_estimator is None:
            base_estimator = RandomForestClassifier(
                n_estimators=100, max_depth=5, min_samples_leaf=20, random_state=42
            )

        # Stage 1: modele dla Y
        self.model_a_y = base_estimator
        self.model_b_y = base_estimator.__class__(**base_estimator.get_params())

        # Stage 2: modele dla uplift
        self.model_a_uplift = base_estimator.__class__(**base_estimator.get_params())
        self.model_b_uplift = base_estimator.__class__(**base_estimator.get_params())

        # Propensity score model
        self.propensity_model = base_estimator.__class__(**base_estimator.get_params())

        self.min_samples = min_samples

    def fit(self, df, feature_cols):
        """Trenuje X-Learner w trzech etapach"""

        # Sprawdź liczebności
        n_a = (df["group"] == "A").sum()
        n_b = (df["group"] == "B").sum()

        print(f"Liczebności: Grupa A = {n_a}, Grupa B = {n_b}")

        if n_a < self.min_samples or n_b < self.min_samples:
            print(f"UWAGA: Grupy są małe. X-Learner może nie działać optymalnie.")

        # Przygotuj dane
        X = df[feature_cols].copy()
        X = pd.get_dummies(X, drop_first=True)
        y = df["sale"]
        group = (df["group"] == "A").astype(int)

        # Stage 1: Trenuj modele dla Y
        mask_a = group == 1
        mask_b = group == 0

        self.model_a_y.fit(X[mask_a], y[mask_a])
        self.model_b_y.fit(X[mask_b], y[mask_b])

        # Stage 2: Oblicz imputed treatment effects
        # Dla grupy A: D_A = Y_A - pred_B(X_A)
        pred_b_on_a = self.model_b_y.predict_proba(X[mask_a])[:, 1]
        d_a = y[mask_a].values - pred_b_on_a

        # Dla grupy B: D_B = pred_A(X_B) - Y_B
        pred_a_on_b = self.model_a_y.predict_proba(X[mask_b])[:, 1]
        d_b = pred_a_on_b - y[mask_b].values

        # Trenuj modele dla uplift
        self.model_a_uplift.fit(X[mask_a], d_a)
        self.model_b_uplift.fit(X[mask_b], d_b)

        # Stage 3: Propensity score
        self.propensity_model.fit(X, group)

        self.feature_cols = X.columns.tolist()
        return self

    def predict_uplift(self, df, feature_cols):
        """Przewiduje uplift używając ważonej kombinacji modeli"""
        X = df[feature_cols].copy()
        X = pd.get_dummies(X, drop_first=True)

        # Upewnij się, że mamy odpowiednie kolumny
        for col in self.feature_cols:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_cols]

        # Przewidywania z obu modeli
        tau_a = self.model_a_uplift.predict(X)
        tau_b = self.model_b_uplift.predict(X)

        # Propensity scores (prawdopodobieństwo bycia w grupie A)
        g = self.propensity_model.predict_proba(X)[:, 1]

        # Ważona kombinacja
        uplift = g * tau_b + (1 - g) * tau_a

        return uplift


# ===================================================================
# PRZYKŁAD UŻYCIA
# ===================================================================


def main_example():
    # Stwórz przykładowe dane
    np.random.seed(42)
    n_customers = 1000

    data = {
        "customer_id": range(n_customers),
        "group": np.random.choice(["A", "B"], n_customers),
        "age": np.random.randint(18, 70, n_customers),
        "income": np.random.choice(["low", "medium", "high"], n_customers),
        "location": np.random.choice(["urban", "rural"], n_customers),
        "previous_purchases": np.random.randint(0, 10, n_customers),
    }

    df = pd.DataFrame(data)

    # Symuluj sukces sprzedaży z interakcjami
    # Młodzi klienci lepiej reagują na grupę A
    # Starsi klienci lepiej reagują na grupę B
    success_prob = 0.3
    for idx in df.index:
        if df.loc[idx, "age"] < 40 and df.loc[idx, "group"] == "A":
            success_prob = 0.6  # A jest lepsze dla młodych
        elif df.loc[idx, "age"] >= 40 and df.loc[idx, "group"] == "B":
            success_prob = 0.65  # B jest lepsze dla starszych
        else:
            success_prob = 0.35

        df.loc[idx, "sale"] = np.random.binomial(1, success_prob)

    feature_cols = ["age", "income", "location", "previous_purchases"]

    print("=" * 70)
    print("PRZYKŁAD 1: PROSTA ANALIZA Z DRZEWEM DECYZYJNYM")
    print("=" * 70)

    analyzer = SimpleInteractionAnalyzer(max_depth=3, min_samples_leaf=30)
    analyzer.fit(df, feature_cols)

    print("\nReguły drzewa:")
    print(analyzer.get_rules())

    print("\nAnaliza segmentów:")
    segments = analyzer.analyze_segments(df, feature_cols)
    print(segments.to_string())

    print("\n" + "=" * 70)
    print("PRZYKŁAD 2: T-LEARNER")
    print("=" * 70)

    t_learner = TLearnerUplift(min_samples_per_group=50)
    t_learner.fit(df, feature_cols)

    print("\nSegmentacja bazująca na uplift:")
    t_segments = t_learner.segment_by_uplift(df, feature_cols, n_segments=5)
    print(t_segments.to_string())

    print("\n" + "=" * 70)
    print("PRZYKŁAD 3: X-LEARNER")
    print("=" * 70)

    x_learner = XLearnerUplift(min_samples=50)
    x_learner.fit(df, feature_cols)

    uplift_predictions = x_learner.predict_uplift(df, feature_cols)
    print(f"\nŚredni przewidywany uplift: {uplift_predictions.mean():.4f}")
    print(f"Mediana: {np.median(uplift_predictions):.4f}")
    print(f"Zakres: [{uplift_predictions.min():.4f}, {uplift_predictions.max():.4f}]")


if __name__ == "__main__":
    main_example()
