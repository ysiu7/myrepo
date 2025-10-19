from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd


def evaluate_models(X, y, models):
    results = []
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
        results.append(
            {"model": name, "mean_acc": scores.mean(), "std_acc": scores.std()}
        )
    return pd.DataFrame(results)


if __name__ == "__main__":
    # Załaduj dane
    X = pd.read_csv("data/processed/X_train.csv")
    y = pd.read_csv("data/processed/y_train.csv")

    models = {
        "LogReg": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100),
    }

    df_results = evaluate_models(X, y, models)
    df_results.to_csv("results/metrics.csv", index=False)
    print(df_results)
