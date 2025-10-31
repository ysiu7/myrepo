# Instalacja (jeśli nie masz)
# pip install econml scikit-learn numpy pandas matplotlib

import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ======================================================================
# 1️⃣ Tworzymy dane
# ======================================================================
np.random.seed(42)
n = 2000

X = np.random.normal(0, 1, (n, 2))  # cechy klientów
T = np.random.binomial(1, 0.5, n)  # losowy treatment (np. reklama)
# efekt treatmentu zależy od X[:,0]
tau = 2 * (X[:, 0] > 0) + 0.5 * X[:, 1]  # prawdziwy efekt
Y = 5 + X[:, 0] + 0.5 * X[:, 1] + T * tau + np.random.normal(0, 1, n)

# ======================================================================
# 2️⃣ Uczenie modelu Causal Forest
# ======================================================================
est = CausalForestDML(
    model_t=RandomForestRegressor(),
    model_y=RandomForestRegressor(),
    n_estimators=100,
    min_samples_leaf=20,
    random_state=42,
)

est.fit(Y, T, X=X)

# ======================================================================
# 3️⃣ Predykcja efektu przyczynowego
# ======================================================================
te_pred = est.effect(X)  # HTE: efekt dla każdego klienta

# ======================================================================
# 4️⃣ Analiza wyników
# ======================================================================
print("Średni szacowany efekt (ATE):", np.mean(te_pred))
print("Pierwsze 5 efektów indywidualnych:", te_pred[:5])

# Wykres: prawdziwy vs oszacowany efekt
plt.scatter(tau, te_pred, alpha=0.5)
plt.xlabel("Prawdziwy efekt (tau)")
plt.ylabel("Szacowany efekt (Causal Forest)")
plt.title("Causal Forest: rzeczywisty vs przewidziany efekt")
plt.axline((0, 0), slope=1, color="red", linestyle="--")
plt.show()
