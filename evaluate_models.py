import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import json
from datetime import datetime
import uuid


def get_scorer(scoring_name):
    """
    Zwraca funkcję scoringu na podstawie nazwy w configu.
    Obsługuje m.in. gini.
    """
    if scoring_name == "gini":
        from sklearn.metrics import make_scorer, roc_auc_score

        def gini_score(y_true, y_pred_proba):
            auc = roc_auc_score(y_true, y_pred_proba)
            return 2 * auc - 1

        return make_scorer(gini_score, needs_proba=True)
    else:
        return scoring_name


def evaluate_models(X, y, models, scoring, models_config):
    results = []

    categorical_cols = X.select_dtypes(include=["object", "category"]).columns
    numeric_cols = X.select_dtypes(exclude=["object", "category"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    scorer = get_scorer(scoring)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    experiment_id = uuid.uuid4().hex[:8]

    for name, model in models.items():
        print(f"🔄 Ewaluacja modelu: {name}")

        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

        scores = cross_val_score(pipeline, X, y, cv=5, scoring=scorer)

        # Tylko te parametry, które są w config.yaml
        defined_params = models_config[name].get("params", {})

        results.append(
            {
                "experiment_id": experiment_id,
                "timestamp": timestamp,
                "model": name,
                "params": json.dumps(defined_params),
                "mean_score": scores.mean(),
                "std_score": scores.std(),
            }
        )

    return pd.DataFrame(results)
