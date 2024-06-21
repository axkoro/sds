from os.path import realpath

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def normal_pdf(range=10, resolution=0.1):
    x = np.arange(-range/2, range/2, resolution)
    y = (1/math.sqrt(2 * math.pi)) * np.exp(-0.5 * x**2)
    return x, y

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
    csv_path = "Pokemon.csv"
    sample_size = 6
    num_samples_list = [10, 100, 1000, 10000]
    figures = []

    data = pd.read_csv(csv_path, usecols=["HP", "Attack", "Speed"])
    hp_data = data["HP"].to_numpy()
    attack_data = data["Attack"].to_numpy()
    speed_data = data["Speed"].to_numpy()

    # HP figure
    hp_fig = plt.figure()
    hp_fig.suptitle('Verteilungen von m Sample Means (HP-Werte)')
    hp_fig.subplots_adjust(hspace=0.4)
    for index, num_samples in enumerate(num_samples_list):
        sample_means = np.zeros(num_samples)

        # Samples nehmen
        for i in range(num_samples):
            sample = np.random.choice(hp_data, sample_size)
            sample_means[i] = sample.mean()

        # Z-Transformation der Sample Means
        mean = sample_means.mean()
        sigma = sample_means.std()
        z_sample_means = np.zeros(num_samples)
        z_sample_means = (sample_means - mean) / sigma

        ax = hp_fig.add_subplot(2, 2, index+1)
        ax.hist(z_sample_means, weights=np.ones(num_samples)/num_samples) 
        ax.set_xlim(-5, 5)
        ax.set_title('m=' + str(num_samples))

        # Standardnormalverteilung plotten
        x, y = normal_pdf()
        ax.plot(x, y)

    figures.append(hp_fig)

    # Attack figure
    attack_fig = plt.figure()
    attack_fig.suptitle('Verteilungen von m Sample Means (Attack-Werte)')
    attack_fig.subplots_adjust(hspace=0.4)
    for index, num_samples in enumerate(num_samples_list):
        sample_means = np.zeros(num_samples)

        # Samples nehmen
        for i in range(num_samples):
            sample = np.random.choice(attack_data, sample_size)
            sample_means[i] = sample.mean()

        # Z-Transformation der Sample Means
        mean = sample_means.mean()
        sigma = sample_means.std()
        z_sample_means = np.zeros(num_samples)
        z_sample_means = (sample_means - mean) / sigma

        ax = attack_fig.add_subplot(2, 2, index+1)
        ax.hist(z_sample_means, weights=np.ones(num_samples)/num_samples)
        ax.set_xlim(-5, 5)
        ax.set_title('m=' + str(num_samples))

        # Standardnormalverteilung plotten
        x, y = normal_pdf()
        ax.plot(x, y)

    figures.append(attack_fig)

    # Speed figure
    speed_fig = plt.figure()
    speed_fig.suptitle('Verteilungen von m Sample Means (Speed-Werte)')
    speed_fig.subplots_adjust(hspace=0.4)
    for index, num_samples in enumerate(num_samples_list):
        sample_means = np.zeros(num_samples)

        # Samples nehmen
        for i in range(num_samples):
            sample = np.random.choice(speed_data, sample_size)
            sample_means[i] = sample.mean()

        # Z-Transformation der Sample Means
        mean = sample_means.mean()
        sigma = sample_means.std()
        z_sample_means = np.zeros(num_samples)
        z_sample_means = (sample_means - mean) / sigma

        ax = speed_fig.add_subplot(2, 2, index+1)
        ax.hist(z_sample_means, weights=np.ones(num_samples)/num_samples)
        ax.set_xlim(-5, 5)
        ax.set_title('m=' + str(num_samples))

        # Standardnormalverteilung plotten
        x, y = normal_pdf()
        ax.plot(x, y)

    figures.append(speed_fig)


    '''
    Interpretation:
    Die Verteilungen nähern sich zwar scheinbar an die Normalverteilung an, dies ist aber ein anderer Effekt als der des zentralen
    Grenzwertsatzes. Dieser beschreibt nämlich eine Annäherung für n -> unendlich, wobei n nicht etwa der Anzahl der Samples entspricht,
    sondern der Größe der einzelnen Samples.
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
