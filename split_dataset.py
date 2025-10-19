from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# 📌 Ustawienie folderu nadrzędnego względem tego pliku
BASE_DIR = Path(__file__).resolve().parent.parent


def split_and_save_dataset(input_path, target_col, test_size=0.2, random_state=42):
    df = pd.read_csv(input_path)

    X = df.drop(columns=[target_col])
    y = df["czy_PL"].map({"PL": 1, "Inne": 0})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    processed_dir = BASE_DIR / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(processed_dir / "X_train.csv", index=False)
    X_test.to_csv(processed_dir / "X_test.csv", index=False)
    y_train.to_csv(processed_dir / "y_train.csv", index=False, header=[target_col])
    y_test.to_csv(processed_dir / "y_test.csv", index=False, header=[target_col])

    print(f"✅ Dataset został zapisany w {processed_dir}")


if __name__ == "__main__":
    raw_path = BASE_DIR / "data" / "raw" / "dataset.csv"
    split_and_save_dataset(raw_path, target_col="czy_PL")
