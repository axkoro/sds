from os.path import realpath

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages


def teilaufgabe_a():
    """
    Rückgabewerte:
    sides: numpy array (integer, 12 Elemente), die Werte der Würfelseiten
    side_probabilities: numpy array (float, 12 Elemente), die Wahrscheinlichkeit jeder Würfelseite
    expected_value: float, der Erwartungswert der Zufallsvariable "Würfelergebnis"
    cdf: Kumulative Wahrscheinlichkeitsverteilungsfunktion
    """

    side_probability = 1/12 # assuming fair dice with 12 sides

    sides = np.asarray([1,1,1,1,2,2,2,4,4,8,16,32])
    side_probabilities = np.full(12, side_probability)
    expected_value = np.sum(np.multiply(sides, side_probabilities))

    unique_side_values, unique_sides_counts = np.unique(sides, return_counts=True)
    value_probabilities = np.multiply(unique_sides_counts, side_probability) # probabilites of X being a certain value
    cumulative_probabilites = np.cumsum(value_probabilities)

    def cdf(X: float):
        # index = np.where(unique_sides == X)[0][0]
        if X < unique_side_values[0]:
            return 0
        
        index = 0
        while index <= unique_side_values.size-2 and X >= unique_side_values[index+1]:
            index += 1

        return cumulative_probabilites[index]

    return sides, side_probabilities, expected_value, cdf


def teilaufgabe_b():
    """
    Rückgabewerte:
    fig: die matplotlib figure
    sample_mean: float, der gesuchte mean der Würfelergebnisse im Datensatz
    """
    fig, ax = plt.subplots()
    
    casino_data = pd.read_csv("casino.csv")
    casino_data["zeit"] = pd.to_datetime(casino_data["zeit"], format="%Y-%m-%d %H:%M:%S")

    sus_results = casino_data[(casino_data["tisch"] == "B") & 
                              (casino_data["spieler"] == 1) & 
                              (casino_data["zeit"] >= pd.Timestamp("2024-03-27T21:00:00"))]
    
    sample_mean = sus_results["ergebnis"].mean()

    dice_values = [1,2,4,8,16,32]
    frequencies = sus_results["ergebnis"].value_counts(normalize=True).sort_index()

    ax.bar(range(len(dice_values)), np.asarray(frequencies))
    ax.set_xlabel("Würfelergebnis")
    ax.set_ylabel("Relative Häufigkeit")
    ax.set_xticks(range(len(dice_values)))
    ax.set_xticklabels(dice_values)

    return fig, sample_mean


def teilaufgabe_c(expected_value_fair, spieler_name=1, tisch_name="B"):
    """
    Rückgabewert:
    fig: die matplotlib figure
    """

    casino_data = pd.read_csv("casino.csv")
    player_results = casino_data[(casino_data["spieler"] == spieler_name) & (casino_data["tisch"] == tisch_name)]
    player_results = player_results.reset_index(drop=True)

    mean_over_time = np.zeros(len(player_results), dtype=float)
    mean_over_time[0] = player_results["ergebnis"][0]
    for i in range(1,len(player_results)):
        mean_over_time[i] = (1/(i+1)) * player_results["ergebnis"][i] + (i/(i+1)) * mean_over_time[i-1]

    fig, ax = plt.subplots()
    player_results["zeit"] = pd.to_datetime(player_results["zeit"], format="%Y-%m-%d %H:%M:%S")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    ax.plot(player_results["zeit"], mean_over_time, label="Sample Mean (im Laufe der Zeit)")
    ax.axhline(expected_value_fair, linestyle="--", color="grey", label="Erwartungswert")
    ax.axvline(pd.to_datetime("2024-03-27 21:00:00"), linestyle=":", color="grey", label="Vermuteter Würfeltausch")

    ax.legend()
    ax.set_xlabel("Uhrzeit (hh:mm)")
    ax.set_ylabel("Sample Mean")

    return fig

if __name__ == "__main__": # given by the tutors
    figures = []

    sides, side_probabilities, expected_mean, cdf = teilaufgabe_a()

    # Test der Ergebnisse aus Teilaufgabe (a)
    print("Teilaufgabe (a) :")
    assert len(sides) == 12, "'sides' muss 12 Elemente haben."
    assert len(side_probabilities) == 12, "'side_probabilities' muss 12 Elemente haben."
    assert np.isclose(
        np.sum(side_probabilities), 1.0
    ), "Die Summe aller Wahrscheinlichkeiten muss 1 sein."
    print(f"{expected_mean=}")  # ~ 6.167
    print(f"{cdf(0)=}")  # ~ 0.0
    print(f"{cdf(1)=}")  # ~ 0.333
    print(f"{cdf(2)=}")  # ~ 0.583
    print(f"{cdf(42)=}")  # ~ 1.0

    # Visualisierung der kumulativen Wahrscheinlichkeitsfunktion
    fig, ax = plt.subplots()
    X = np.arange(0, 33, 0.01)
    ax.plot(X, [cdf(x) for x in X])
    ax.set_xlabel("Mögliches Würfelergebnis")
    ax.set_ylabel("Kumulative Wahrscheinlichkeit")
    ax.set_xlim(0, 33)
    ax.grid()

    figures.append(fig)

    fig, sample_mean = teilaufgabe_b()

    figures.append(fig)

    print()
    print("Teilaufgabe (b) :")
    print(f"{sample_mean=}")  # ~ 10.754

    fig = teilaufgabe_c(expected_mean)
    figures.append(fig)

    fig = teilaufgabe_c(expected_mean, spieler_name=2, tisch_name="B")
    figures[-1].axes[0].sharey(fig.axes[0])
    figures.append(fig)

    # Save the figures to a multi-page PDF
    pdf_path = "aufgabe3_plots.pdf"
    with PdfPages(pdf_path) as pdf:
        for fig in figures:
            pdf.savefig(fig)
            plt.close(fig)

    print()
    print(f"Figures saved to {realpath(pdf_path)}")
