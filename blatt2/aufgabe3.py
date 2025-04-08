import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

def teilaufgabe_a(documents_train, documents_test):
    """
    Implementieren Sie ein Bag of Word Modell zur Umwandlung der 'Lyrics' Spalten im Trainings- und Testdatensatz.
    Nutzen Sie die CountVectorizer Klasse aus der sklearn Bibliothek.

    Der Rückgabewert ist ein 3-tupel bestehend aus:
      vectorizer: Das Bag-of-words Modell (CountVectorizer Instanz aus sklearn)
       X_train:  ein numpy array (integer) in der Struktur ( n x m ). Hier steht n für die Anzahl der Elemente im Trainingsdatensatz
         und m für die Anzahl der abgeleiteten Features. Der Wert 1 (0) bedeutet, dass ein Wort im Dokument (nicht) vorkommt.
       X_test: ein numpy array (integer)  in der Struktur ( n x m ). Hier steht n für die Anzahl der Elemente im Testdatensatz
         und m für die Anzahl der abgeleiteten Features. Der Wert 1 (0) bedeutet, dass ein Wort im Dokument (nicht) vorkommt.
    """

    vectorizer = CountVectorizer(binary=True, lowercase=True, strip_accents='unicode')
    X_train = vectorizer.fit_transform(documents_train).toarray()
    X_test = vectorizer.transform(documents_test).toarray()

    return vectorizer, X_train, X_test

def teilaufgabe_b(X: np.ndarray, y):
    """
    Nutzen Sie den Trainingsdatensatz, um einen Naive Bayes Classifier zu trainieren.
    Rückabe: ein 2-tuple aus

    priors: ein numpy array mit den a-priori Wahrscheinlichkeiten jeder Klasse
    conds: ein numpy array mit den Wahrscheinlichkeiten jedes Worts in jeder Klasse
           numpy array shape: ( Klassen x Worte )
    """

    num_documents = X.shape[0]
    num_words = X.shape[1]

    metal_prior = sum(y) / num_documents
    priors = np.asarray([1-metal_prior, metal_prior])

    conds = np.zeros((2, num_words))
    for word in range(num_words):
        occurrences_in_metal = 0
        occurrences_in_other = 0

        occurrences = X[:, word]
        for document in range(num_documents):
            if y[document]: # document is metal
                occurrences_in_metal += occurrences[document]
            else:
                occurrences_in_other += occurrences[document]

        delta = 1 # Delta fuer additive Glaettung
        conds[1][word] = (occurrences_in_metal + delta) / (priors[1] * num_documents + delta * num_documents)
        conds[0][word] = (occurrences_in_other + delta) / (priors[0] * num_documents + delta * num_documents)

    return priors, conds

def teilaufgabe_c(X: np.ndarray, classes: np.ndarray, priors: np.ndarray, conds: np.ndarray):
    """
    Nutzen Sie den zuvor trainierte Naive Bayes Klassifikator, um Vorhersagen für einen Datensatz zu treffen.

    Der Rückgabewert ist ein 2-tupel bestehend aus:
       prediction: Ein numpy array mit der binären Klassifikation. (enthält ein boolean je Zeile in X).
       prediction_log_probs: Ein 2D numpy array mit den berechneten Klassenzugehörigkeiten in natürlicher logarithmischer Skala.
                                shape: (Zeilen im Datensatz x mögliche Klassen)
    """

    num_documents = X.shape[0]
    prediction = np.zeros(num_documents, dtype=bool)
    prediction_log_probs = np.zeros((num_documents, len(classes)))

    for document in range(num_documents): # TODO: (nicht möglich mit vorgegebenen Eingabeparametern?) Wörter, die nicht in Trainingsdaten vorkamen (konstante Prob -> 1/num_documents)
        words_in_document = X[document]

        log_prob_metal = np.log(priors[1])
        log_prob_other = np.log(priors[0])
        for word in words_in_document:
            if word:
                log_prob_metal += np.log(conds[1][word]) # TODO: Overflow errors handlen
                log_prob_other += np.log(conds[0][word])
                
        prediction_log_probs[document][1] = log_prob_metal
        prediction_log_probs[document][0] = log_prob_other

        if (log_prob_metal >= log_prob_other):
            prediction[document] = True
        else:
            prediction[document] = False

    return prediction, prediction_log_probs


if __name__ == "__main__":
    # Laden des Datensatzes
    df_train = pd.read_csv("song_lyrics/train.csv")
    df_test = pd.read_csv("song_lyrics/test.csv")

    # Erstellen einer neuen Spalte mit einem binären Klassifikationslabel
    df_train["Label"] = (df_train["Genre"] == "Metal").astype(int)
    df_test["Label"] = (df_test["Genre"] == "Metal").astype(int)

    # Definition der Klassifikationsziels
    y_train = df_train["Label"].values
    y_test = df_test["Label"].values
    classes = np.unique(y_train)

    # Erstellen eines Bag of Word Modells und Transformation von Training und Testdatensatz
    vectorizer, X_train, X_test = teilaufgabe_a(
        df_train["Lyrics"].values, df_test["Lyrics"].values
    )

    # Trainieren eines Naive Bayes Klassifikators
    priors, conds = teilaufgabe_b(X_train, y_train)

    # Klassifikation eines Datensatzes mit Hilfe des trainierten Modells
    y_pred_test, _ = teilaufgabe_c(X_test, classes, priors, conds)  # Trainingsdatensatz
    y_pred_train, _ = teilaufgabe_c(X_train, classes, priors, conds)  # Testdatensatz

    # Evaluation mittels Metriken
    print(
        "A-priori Wahrscheinlichkeit je Klasse: ", priors
    )  # zu erwarten: ~ [0.6833 0.3166]

    train_accuracy = np.mean(y_pred_train == y_train)
    print(f"Train Accuracy: {train_accuracy:.2f}")  # zu erwarten: >= 0.83

    test_accuracy = np.mean(y_pred_test == y_test)
    print(f"Test Accuracy: {test_accuracy:.2f}")  # zu erwarten: >= 0.81

    # Weitere manuelle Evaluation des Klassifikators mit Texten aus Nutzereingaben
    while True:
        user_input = input("Enter some text (or press Enter to exit): ")
        if user_input == "":
            break
        else:
            X_user = vectorizer.transform(pd.Series([user_input])).toarray()
            y_user, y_log_probs_user = teilaufgabe_c(X_user, classes, priors, conds)

            # Umrechnung von logarithmischen Klassenzugehörigkeiten in Wahrscheinlichkeiten
            max_log_probs = np.max(y_log_probs_user, axis=1, keepdims=True)
            # Vermeiden von overflow, relative Größenordnung behalten
            probs = np.exp(y_log_probs_user - max_log_probs)
            # Normalisierung zu Wahrscheinlichkeiten
            probs = probs / np.sum(probs, axis=1, keepdims=True)
            probability = np.max(probs)

            is_metal = bool(y_user[0])
            str_klassenzugehoerigkeit = f"Logar. Klassenzugehörigkeiten = {y_log_probs_user[0][0]:.5} {y_log_probs_user[0][1]:.5}"

            print(
                f"{'' if is_metal else 'Kein '}Metal (Konfidenz: {probability:.0%}, {str_klassenzugehoerigkeit})"
            )
