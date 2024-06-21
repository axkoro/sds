from os.path import realpath

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def teilaufgabe_a():
    """
    Rückgabewerte:
    fig: die matplotlib figure
    expected_mean: float
    """

    sample_size = 200
    num_samples = 10000

    sample_means = np.zeros(num_samples)
    rng = np.random.default_rng()

    for index, _ in enumerate(sample_means):
        sample = rng.integers(1, high=6, size=sample_size, endpoint=True)
        sample_means[index] = sample.mean()

    fig, ax = plt.subplots()
    ax.hist(sample_means, 20)
    ax.set_title('Aufgabe 3a')
    ax.set_xlabel('Sample Mean')
    ax.set_ylabel('Häufigkeit') # eher 'Häufigkeit des Sample-Mean-Intervalls'
    ax.set_xlim([3.0, 4.0])
    ax.grid()

    expected_mean = 3.5
    ax.axvline(expected_mean, color='grey', linestyle='--')

    '''
    Interpretation:
    Da jedes Sample eine Menge von unabhängigen Würfelwürfen zusammenfasst (die alle derselben Wahrscheinlichkeitsverteilung folgen),
    ist nach dem zentralen Grenzwertsatz zu erwarten, dass sich die Wahrscheinlichkeitsverteilung aller dieser Samples (mit steigender
    Samplezahl) einer Normalverteilung annähert.
    Tatsächlich ähnelt unser Histogramm für 10.000 Samples bereits sehr einer Glockenkurve mit einem Erwartungswert von 3,5.
    '''
    
    return fig, expected_mean


def teilaufgabe_b():
    """
    Rückgabewerte:
    figures: Eine Liste aller matplotlib Figures
    """
    sample_size = 50 # samples pro wiederholung
    num_samples = 10000 # wiederholungen
    num_bins = 20
    
    figures = []

    # Simulation hypergeometrischer Verteilung
    sample_sums = np.zeros(num_samples)
    sample_means = np.zeros(num_samples)

    rng = np.random.default_rng()
    ngood, nbad, nsample = 6, 43, 6

    for i in range(num_samples): # eine Wiederholung
        sample = rng.hypergeometric(ngood, nbad, nsample, sample_size) # equivalent zu h(x|49,6,6)
        sample_sums[i] = sample.sum()
        sample_means[i] = sample.mean()

    # fig, ax = plt.subplots()
    # ax.hist(sample_sums, 20)
    # ax.set_xlabel('Summe der \'Richtigen\' in einer Wiederholung')
    # ax.set_ylabel('Häufigkeit')
    # figures.append(fig)

    fig, ax = plt.subplots()
    ax.hist(sample_means, num_bins)
    ax.set_title('Aufgabe 3b - hypergeometrische Verteilung')
    ax.set_xlabel('Mittelwert der \'Richtigen\' in 50 Lottospielen')
    ax.set_ylabel('Häufigkeit')
    figures.append(fig)

    # Simulation Poisson-Verteilung
    sample_sums = np.zeros(num_samples)
    sample_means = np.zeros(num_samples)

    rng = np.random.default_rng()
    lam = 3.17

    for i in range(num_samples): # eine Wiederholung
        sample = rng.poisson(lam, sample_size)
        sample_sums[i] = sample.sum()
        sample_means[i] = sample.mean()

    # fig, ax = plt.subplots()
    # ax.hist(sample_sums, 20)
    # ax.set_xlabel('Summe der \'Richtigen\' in einer Wiederholung')
    # ax.set_ylabel('Häufigkeit')
    # figures.append(fig)

    fig, ax = plt.subplots()
    ax.hist(sample_means, num_bins)
    ax.set_title('Aufgabe 3b - Poisson-Verteilung')
    ax.set_xlabel('Mittelwert der Tore in 50 Bundesligaspielen')
    ax.set_ylabel('Häufigkeit')
    figures.append(fig)

    '''
    Interpretation:
    Man sieht, dass die Summen (und somit die Mittelwerte) der einzelnen Wiederholungen annähernd normalverteilt sind, wobei teilweise
    noch eine leichte Schiefe vorhanden ist. Es ist nach dem zentralen Grenzwertsatz zu erwarten, dass sich mit steigender Größe der
    einzelnen Wiederholungen (hier "sample_size") die Verteilung noch weiter an eine Normalverteilung annähert, da die 50 Zufallsvariablen
    innerhalb einer Wiederholung stochastisch unabhängig sind und derselben Verteilung folgen.
    '''
    return figures


def teilaufgabe_c():
    """
    Rückgabewerte:
    figures: Eine Liste aller matplotlib Figures
    """
    figures = []

    # Implementieren Sie hier Ihre Lösung

    '''
    Interpretation:
    '''
    return figures


if __name__ == "__main__":
    figures = []

    fig, expected_mean = teilaufgabe_a()
    figures.append(fig)
    print(f"{expected_mean=}")  # ~7.0

    figures_b = teilaufgabe_b()
    figures.extend(figures_b)

    figures_c = teilaufgabe_c()
    figures.extend(figures_c)

    # Save the figures to a multi-page PDF
    pdf_path = "aufgabe3_plots.pdf"
    with PdfPages(pdf_path) as pdf:
        for fig in figures:
            if fig is None:
                continue

            pdf.savefig(fig)
            plt.close(fig)

    print()
    print(f"Figures saved to {realpath(pdf_path)}")
