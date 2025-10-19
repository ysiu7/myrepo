from IPython.display import display
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def dataset_overview(df, target):
    """
    Wyświetla podstawowe informacje o zbiorze danych.
    """
    print("🔎 Rozmiar zbioru:", df.shape)
    print("\n📊 Typy danych:\n", df.dtypes)
    print("\n❓ Braki danych:\n", df.isna().sum())
    print("\n⚖️ Rozkład klasy celu:\n", df[target].value_counts(normalize=True))
    display(df.head())


def plot_target_distribution(df, target):
    """
    Rysuje wykres słupkowy zmiennej celu.
    """
    df[target].value_counts().plot(kind="bar")
    plt.title("Rozkład zmiennej celu")
    plt.xlabel("Klasa")
    plt.ylabel("Liczba próbek")
    plt.show()


def plot_numeric_distributions(df, target):
    """
    Rysuje histogramy dla wszystkich zmiennych numerycznych.
    """
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.drop(target)
    df[num_cols].hist(figsize=(15, 10), bins=30)
    plt.suptitle("Rozkład zmiennych numerycznych")
    plt.show()


def boxplots_by_target(df, target):
    """
    Rysuje boxploty dla zmiennych numerycznych względem klasy celu.
    """
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.drop(target)
    for col in num_cols:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=target, y=col, data=df)
        plt.title(f"{col} vs {target}")
        plt.show()


def catplot_by_target(df, target):
    """
    Rysuje rozkład zmiennych kategorycznych względem klasy celu.
    """
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        plt.figure(figsize=(6, 4))
        sns.countplot(x=col, hue=target, data=df)
        plt.title(f"{col} vs {target}")
        plt.xticks(rotation=45)
        plt.show()


def correlation_heatmap(df, target):
    """
    Rysuje macierz korelacji dla zmiennych numerycznych.
    """
    num_cols = df.select_dtypes(include=["int64", "float64"])
    corr = num_cols.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Macierz korelacji")
    plt.show()


from sklearn.ensemble import RandomForestClassifier


def feature_importance(df, target):
    """
    Oblicza i rysuje prostą ważność cech przy pomocy RandomForest.
    """
    X = df.drop(columns=[target])
    y = df[target]
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    importances = pd.Series(model.feature_importances_, index=X.columns)
    importances.sort_values().plot(kind="barh", figsize=(8, 6))
    plt.title("Ważność cech (RandomForest)")
    plt.show()
