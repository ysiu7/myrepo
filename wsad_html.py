import pandas as pd
import seaborn as sns
from io import BytesIO
import base64
import matplotlib.pyplot as plt
import numpy as np
from jinja2 import Environment, FileSystemLoader
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px


def wykres_do_base64(fig):
    """
    Konwertuje wykres matplotlib na string base64
    """
    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", dpi=300)
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()

    graphic = base64.b64encode(image_png)
    graphic = graphic.decode("utf-8")

    return graphic


def summarize_dataframe(df):
    stats = []

    for column in df.columns:
        # Podstawowe informacje
        col_stats = {
            "Kolumna": column,
            "Typ_danych": df[column].dtype,
            "Liczba_wierszy": len(df),
            "Liczba_unikalnych": df[column].nunique(),
            "Liczba_brakow": df[column].isna().sum(),
            "Procent_brakow": round(df[column].isna().mean() * 100, 2),
            "Najczęstsza_wartość": df[column].mode().iloc[0]
            if not df[column].empty
            else np.nan,
            "Procent_najczęstszej_wartości": round(
                (df[column].value_counts().iloc[0] / len(df)) * 100, 2
            )
            if not df[column].empty
            else np.nan,
        }

        # Statystyki dla danych numerycznych
        if pd.api.types.is_numeric_dtype(df[column]):
            col_stats.update(
                {
                    "Średnia": round(df[column].mean(), 2),
                    "Mediana": round(df[column].median(), 2),
                    "Min": round(df[column].min(), 2),
                    "Max": round(df[column].max(), 2),
                    "Odchylenie_std": round(df[column].std(), 2),
                }
            )

        stats.append(col_stats)

    # Tworzenie DataFrame z wynikami
    summary_df = pd.DataFrame(stats)

    # Dodanie procentowego udziału unikalnych wartości
    summary_df["Procent_unikalnych"] = round(
        (summary_df["Liczba_unikalnych"] / summary_df["Liczba_wierszy"]) * 100, 2
    )

    # Ustawienie kolumny jako indeks
    summary_df.set_index("Kolumna", inplace=True)

    return summary_df


df = pd.read_csv("ESS11.csv")

l_wierszy, l_kolumn = df.shape

df = df.drop(["name", "essround", "edition", "proddate"], axis=1)

podsum = summarize_dataframe(df)

y = [podsum[podsum["Procent_brakow"] > k].shape[0] for k in range(0, 101)]
x = [100 - xx for xx in range(0, 101)]


plt.figure(figsize=(8, 3))
plt.plot(x, y)
plt.xlabel("% z niepustymi wartościami w danej kolumnie")
plt.ylabel("Liczba kolumn wyciętych")
plt.title(
    "Ile kolumn zostałoby usuniętych przy wymagananym danym procencie niepustych wierszy?"
)
plt.axvline(x=91.5, color="r", linestyle="--", label="Linia x=91.5")

wykres_zaleznosci_brakow_kolumn = wykres_do_base64(plt)


l_kolumn_z_brakami = [
    podsum[podsum["Procent_brakow"] > k].shape[0] for k in range(0, 101)
]

do_usuniecia = podsum[podsum["Procent_brakow"] > 8.5].index

df = df.drop(do_usuniecia, axis=1)


podsum = podsum.reset_index()  # Zamienia indeks na zwykłą kolumnę

kolumny = podsum.columns.tolist()
dane_dict = podsum.to_dict("records")

podsum_po_usunieciach = summarize_dataframe(df)

typy_danych = (
    podsum_po_usunieciach.groupby("Typ_danych").size().reset_index(name="Liczba_kolumn")
)

kolumny_typy_danych = typy_danych.columns.tolist()
dict_typy_danych = typy_danych.to_dict("records")

lista_id_inne = ["idno"]


# Lista zmiennych do wizualizacji
zmienne = sorted(
    podsum_po_usunieciach[podsum_po_usunieciach["Typ_danych"] == "object"].index
)

# Przygotowanie danych dla wszystkich zmiennych
dane = {}
for zmienna in zmienne:
    dane_temp = (
        df[zmienna]
        .astype(str)
        .value_counts(normalize=True)
        .nlargest(10)
        .sort_values(ascending=True)
    )

    dane[zmienna] = {"x": dane_temp.values.tolist(), "y": dane_temp.index.tolist()}

# Tworzenie figury
fig = go.Figure()

# Dodanie wszystkich wykresów
for i, (zmienna, dane_z) in enumerate(dane.items()):
    fig.add_trace(
        go.Bar(
            x=dane_z["x"],
            y=dane_z["y"],
            orientation="h",
            name=zmienna,
            hovertemplate="<b>%{y}</b><br>Udział: %{x:.1%}<extra></extra>",
            visible=True if i == 0 else False,  # Tylko pierwszy widoczny
        )
    )

# Tworzenie przycisków dropdown
buttons = []
for i, zmienna in enumerate(zmienne):
    # Lista widoczności - wszystkie False oprócz aktualnej
    visibility = [False] * len(zmienne)
    visibility[i] = True

    buttons.append(
        {
            "label": zmienna,
            "method": "update",
            "args": [{"visible": visibility}, {"title": f"Top 10 dla: {zmienna}"}],
        }
    )


# Konfiguracja layoutu
fig.update_layout(
    updatemenus=[
        {
            "buttons": buttons,
            "direction": "down",
            "showactive": True,
            "x": 0.1,
            "xanchor": "left",
            "y": 1.15,
            "yanchor": "top",
        }
    ],
    title=f"Top 10 dla: {zmienne[0]}",
    xaxis_title="Udział",
    yaxis_title="Kategoria",
    xaxis_tickformat=".1%",
    height=600,
    margin=dict(l=150, r=50, t=100, b=50),  # Więcej marginesu dla długich etykiet
    showlegend=False,
)

wykres_html = fig.to_html(full_html=False, include_plotlyjs="cdn")


# A teraz ZMIENNE NUMERYCZNE gdzie jest mniej niż 15 wartości

# Lista zmiennych do wizualizacji
zmienne_num = sorted(
    podsum_po_usunieciach[
        (podsum_po_usunieciach["Typ_danych"].isin(["int64", "float64"]))
        & (podsum_po_usunieciach["Liczba_unikalnych"] <= 15)
    ].index
)


print(zmienne_num)
# Przygotowanie danych dla wszystkich zmiennych
dane = {}
for zmienna in zmienne_num:
    dane_temp = df[zmienna].value_counts(normalize=True)

    dane[zmienna] = {"x": dane_temp.values.tolist(), "y": dane_temp.index.tolist()}

# Tworzenie figury
fig = go.Figure()

# Dodanie wszystkich wykresów
for i, (zmienna, dane_z) in enumerate(dane.items()):
    fig.add_trace(
        go.Bar(
            x=dane_z["x"],
            y=dane_z["y"],
            orientation="h",
            name=zmienna,
            hovertemplate="<b>%{y}</b><br>Udział: %{x:.1%}<extra></extra>",
            visible=True if i == 0 else False,  # Tylko pierwszy widoczny
        )
    )

# Tworzenie przycisków dropdown
buttons = []
for i, zmienna in enumerate(zmienne_num):
    # Lista widoczności - wszystkie False oprócz aktualnej
    visibility = [False] * len(zmienne_num)
    visibility[i] = True

    buttons.append(
        {
            "label": zmienna,
            "method": "update",
            "args": [{"visible": visibility}, {"title": f"Top 10 dla: {zmienna}"}],
        }
    )

print(zmienne_num)
# Konfiguracja layoutu
fig.update_layout(
    updatemenus=[
        {
            "buttons": buttons,
            "direction": "down",
            "showactive": True,
            "x": 0.1,
            "xanchor": "left",
            "y": 1.15,
            "yanchor": "top",
        }
    ],
    title=f"Top 10 dla: {zmienne_num[0]}",
    xaxis_title="Udział",
    yaxis_title="Kategoria",
    xaxis_tickformat=".1%",
    height=600,
    margin=dict(l=150, r=50, t=100, b=50),  # Więcej marginesu dla długich etykiet
    showlegend=False,
)

wykres_html_num = fig.to_html(full_html=False, include_plotlyjs="cdn")


# Konfiguracja Jinja2
env = Environment(loader=FileSystemLoader("templates"))

# Ładowanie szablonu
template = env.get_template("prezentacja_danych_2.html")


# Dane do przekazania do szablonu
dane = {
    "nazwa_danych": "Tajemnicze dane",
    "l_wierszy": l_wierszy,
    "l_kolumn": l_kolumn,
    "kolumny": kolumny,
    "dane_dict": dane_dict,
    "lista": ["name", "essround", "edition", "proddate"],
    "wykres_zaleznosci_brakow_kolumn": wykres_zaleznosci_brakow_kolumn,
    "ile_kolumn_z_brakami": len(do_usuniecia),
    "kolumny_typy_danych": kolumny_typy_danych,
    "dict_typy_danych": dict_typy_danych,
    "lista_id_inne": lista_id_inne,
    "wykres_html": wykres_html,
    "wykres_html_num": wykres_html_num,
}

# Renderowanie szablonu z danymi
html_output = template.render(**dane)

# Zapisanie do pliku lub wyświetlenie
with open("output.html", "w", encoding="utf-8") as f:
    f.write(html_output)

print("Szablon został wyrenderowany i zapisany jako 'output.html'")
