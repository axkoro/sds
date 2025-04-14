import math
import numpy as np
from scipy.stats import t, mannwhitneyu

def teilaufgabe_b(samples_a, samples_b):
    """
    Führen Sie einen T-Test für die beiden Stichproben mit einem Signifikanzniveau von 5% durch.
    Kann die Nullhypothese zugunsten der Alternativhypothese verworfen werden?
    Geben Sie die Differenz der Mittelwerte, den p-value und das Testergebnis (boolean) zurück.
    """

    sig_level = 0.05

    a_size = len(samples_a)
    b_size = len(samples_b)
    a_var = np.var(samples_a, ddof=1)
    b_var = np.var(samples_b, ddof=1)
    dof = a_size + b_size - 2 # degrees of freedom (see t-distribution)
    
    weighted_var = ((a_size - 1) * a_var + (b_size - 1) * b_var) / dof
    mean_diff = abs(np.mean(samples_a) - np.mean(samples_b))

    t_value = mean_diff / math.sqrt(weighted_var*((1/a_size)+(1/b_size)))

    p_value = t.cdf(-t_value, dof) + 1 - t.cdf(t_value, dof)
    decision = p_value <= sig_level

    """
    Interpretation: Es ist zwar eher unwahrscheinlich, dass die Differenz der Stichprobenmittelwerte
    rein durch Zufall entstanden sind, jedoch ist der Unterschied nicht signifikant genug, um die Nullhypothese
    zu verwerfen.
    """

    return mean_diff, p_value, decision


if __name__ == "__main__":
    samples_a = np.array([0.24, 0.22, 0.20, 0.25], dtype=np.float64)
    samples_b = np.array([0.2, 0.19, 0.22, 0.18], dtype=np.float64)

    print("Teilaufgabe b)")

    mean_diff, p_value, decision = teilaufgabe_b(samples_a, samples_b)
    print(f"{mean_diff=}")  # ~ 0.03
    print(f"{p_value=}")  # ~ 0.038
    print(f"{decision=}")  # ~ True
