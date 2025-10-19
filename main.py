from pathlib import Path
import pandas as pd
import yaml
from src.split_dataset import split_and_save_dataset
from src.evaluate_models import evaluate_models, get_scorer

BASE_DIR = Path(__file__).resolve().parent


def main():
    # 1️⃣ Wczytanie configu
    with open(BASE_DIR / "config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 2️⃣ Jeśli processed nie istnieje → podziel dane
    processed_dir = BASE_DIR / "data" / "processed"
    if not processed_dir.exists():
        print("📦 Brak danych w processed — dzielę dataset...")
        raw_path = BASE_DIR / "data" / "raw" / "dataset.csv"
        split_and_save_dataset(raw_path, target_col="czy_PL")
    else:
        print("✅ Dane w processed już istnieją")

    # 3️⃣ Wczytaj dane treningowe
    X = pd.read_csv(processed_dir / "X_train.csv")
    y = pd.read_csv(processed_dir / "y_train.csv").squeeze()

    # 4️⃣ Przygotuj scorer
    scoring = get_scorer(config["scoring"])

    # 5️⃣ Zbuduj modele z config.yaml
    models = {}
    for model_name, model_info in config["models"].items():
        module_name, class_name = model_info["type"].rsplit(".", 1)
        ModelClass = __import__(module_name, fromlist=[class_name]).__dict__[class_name]
        models[model_name] = ModelClass(**model_info.get("params", {}))

    # 6️⃣ Ewaluacja modeli — przekazujemy też config params
    df_results = evaluate_models(X, y, models, scoring, config["models"])

    # 7️⃣ Zapis wyników
    results_file = BASE_DIR / "results" / "metrics.csv"
    results_file.parent.mkdir(exist_ok=True)

    # Dopisywanie bez nadpisywania
    df_results.to_csv(
        results_file, mode="a", header=not results_file.exists(), index=False
    )

    # 8️⃣ Wyświetlenie
    print("\n📊 Wyniki modeli:")
    print(df_results)


if __name__ == "__main__":
    main()
