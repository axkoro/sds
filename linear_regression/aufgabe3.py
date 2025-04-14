import math
import numpy as np


def load_honda_city_dataset():
    """
    Liest den Honda City Datensatz als NumPy Arrays ein. 

    Output:

    year_built = np.array[int],
    km_driven = np.array[int],
    selling_price = np.array[float]
    """
    X = np.loadtxt("honda_city.csv", delimiter=",", skiprows=1, dtype=float)

    year_built = np.asarray(X[:,0], dtype=int)
    km_driven = np.asarray(X[:,1], dtype=int)
    selling_price = X[:,2]

    return year_built, km_driven, selling_price


def teilaufgabe_a():
    """
    Nutzen Sie den Datensatz, um eine multivariate lineare Regression zu implementieren, die mithilfe des Baujahres und des Kilometerstandes den Verkaufspreis schätzt.
    Implementieren Sie die algebraische Lösung zur Berechnung der Regression. Geben Sie die Schätzwerte für den Datensatz zurück.

    Output:

    y_pred = np.array[float]
    """
    year_built, km_driven, selling_price = load_honda_city_dataset()
    
    num_instances = year_built.size
    data_matrix = np.column_stack((np.full(num_instances, 1), year_built, km_driven)) # holds "learning data"
    dependent_variable_vector = np.array(selling_price)

    data_matrix_transposed = np.transpose(data_matrix)

    weights = np.matmul(
        np.matmul(
            np.linalg.inv(
                np.matmul(data_matrix_transposed, data_matrix)), data_matrix_transposed), dependent_variable_vector)
    
    return weights


def teilaufgabe_b():
    """
    Berechnen Sie den ''Root Mean Square Error'' der Schätzwerte (bezüglich der echten Verkaufspreise) aus Aufgabe 3 (a).
    Notieren Sie sitchhaltig, als Kommentar, was dieser Fehlerwert im Kontext dieser Aufgabe bedeutet.

    Output:

    rmse = float
    """
    year_built, km_driven, selling_price = load_honda_city_dataset()
    num_instances = year_built.size
    weights = teilaufgabe_a()
    
    squared_error = 0
    for i in range(num_instances):
        prediction = weights[0] + weights[1]*year_built[i] + weights[2]*km_driven[i]
        squared_error += (prediction - selling_price[i])**2
    rmse = math.sqrt((1/num_instances)*squared_error)
    
    return rmse

    '''
    Was bedeutet der RMSE im Kontext dieser Aufgabe?
    Bedeutung: Der RMSE bemisst die Abweichung zwischen den Vorhersagen unserer Regressionsfunktion und den tatsächlichen Verkaufspreisen. Aufgrund des algebraischen Ermittlungsverfahrens der Gewichte ist dieser Fehler minimal.
    '''

if __name__ == "__main__":
    print(f"Teilaufgabe a:\n{teilaufgabe_a()}")
    print(f"Teilaufgabe b: {teilaufgabe_b()}")
