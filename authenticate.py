########
# Importing libraries needed for the program
########
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def classification(data):
    data2 = data.copy()
    for i in range(len(data2)):
        data2[i] = np.round(data2[i], decimals = 0)
    return data2

def accuracy(groundtruth, predictions):
    """
    Calculate the accuracy for the prediction of an estimator
    :param groundtruth: true values
    :param predictions: predicted values
    :return: accuracy (0.0 - 1.0)
    """
    tn, fp, fn, tp = confusion_matrix(groundtruth, predictions).ravel()
    obs = len(groundtruth)
    result = (tp+tn)/obs
    return result

def precision(groundtruth, predictions):
    """
    Calculate the precision for the prediction of an estimator
    :param groundtruth: true values
    :param predictions: predicted values
    :return: precision (0.0 - 1.0)
    """
    # true positives / true positives + false positives
    tn, fp, fn, tp = confusion_matrix(groundtruth, predictions).ravel()
    result = tp / (tp + fp)
    return result

def recall(groundtruth, predictions):
    """
    Calculate the recall for the prediction of an estimator
    :param groundtruth: true values
    :param predictions: predicted values
    :return: recall (0.0 - 1.0)
    """
    tn, fp, fn, tp = confusion_matrix(groundtruth, predictions).ravel()
    result = tp / (tp + fn)
    return result

def F1(groundtruth, predictions):
    """
    Calculate the f1 score for the prediction of an estimator
    :param groundtruth: true values
    :param predictions: predicted values
    :return: F1 score (0.0 - 1.0)
    """
    numerator = precision(groundtruth, predictions) * recall(groundtruth, predictions)
    denominator = precision(groundtruth, predictions) + recall(groundtruth, predictions)
    result = 2 * (numerator/denominator)
    return result

def create_metrics(groundtruth, predictions):
    """
    Calculate and returns the overall performance for a given estimator
    :param groundtruth: True values
    :param predictions: Predicted values
    :return: a dictionary containing accuracy, precision, recall, and f1 score
    """
    dic = {"accuracy": accuracy(groundtruth, predictions),
           "precision" : precision(groundtruth, predictions),
           "recall" : recall(groundtruth, predictions),
           "Fvalue" : F1(groundtruth, predictions)}
    return dic


def roc_curve(groundtruth, probabilities, predictions, estimator_name=str):
    """
    Function for plotting the ROC curve and calculating the AUC and display ROC

    Parameters
    ----------
    groundtruth : List or 1D array
        The real values of the dataset.
    predictions : List or 1D array
        The predicted values by the classifier.
    estimator_name : string
        Name of the classifier, it will be printed in the Figure

    Returns
    -------
    Figure.
    AUC.
    """
    from sklearn.metrics import roc_auc_score
    sensitivities = []
    especificities = []

    sensitivities.append(1)
    especificities.append(1)

    thresholds = [i * 0.05 for i in range(1, 10, 1)]
    for t in thresholds:
        prob = probabilities[:, :]
        prob = np.where(prob >= t, 1, 0)
        recall_data = recall(groundtruth, prob)
        precision_data = precision(groundtruth, prob)
        sensitivities.append(recall_data)
        espc = 1 - precision_data
        especificities.append(espc)
    sensitivities.append(0)
    especificities.append(0)

    plt.figure(figsize=(8, 5), dpi=300)
    plt.plot(especificities, sensitivities, marker='o', linestyle='--', color='r')
    plt.plot([i * 0.01 for i in range(100)], [i * 0.01 for i in range(100)])
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.title(f'{estimator_name} ROC curve')
    plt.savefig(f'{estimator_name} ROC curve.jpg', dpi=300)

    AUC = roc_auc_score(groundtruth, predictions)
    return AUC


def plot_curves(model_hist, name=str):
    """
    Displays and save the training of an ANN
    :param model_hist: history of an ANN after training (Keras ANN)
    :param name: Name assigned to the model, it is used to store an unique figure for this model
    :return: stored figure
    """
    pd.DataFrame(model_hist.history).plot(figsize=(10, 6))
    fig = plt.gca().set_ylim(0, 1)
    fig = plt.ylabel('Accuracy and Loss')
    fig = plt.xlabel('Epochs')
    fig = plt.title('name')
    fig = plt.savefig('learning_curve_{}.png'.format(name), dpi=300)
    return


def snv(input_data):
    """
    Standard Normal Variate (SNV) is used to smooth a signal, in this case is used to smooth the signal of FTIR
    :param input_data: array 2D of sequence data
    :return: array 2D with normalized data
    """
    # Define a new array and populate it with the corrected data
    spectra = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        # Apply correction
        spectra[i, :] = (input_data.iloc[i, :] - np.mean(input_data.iloc[i, :])) / np.std(input_data.iloc[i, :])

    return spectra

def augmentation(df, labels = list, data_iloc = int):
    """
    Function for increase the size of dataset. It computes an array of noise with a normal distribution, and concatenate
    it with the original dataframe with 2x observations.
    :param df: original dataframe
    :param labels: name of the labels
    :param data_iloc:  index location of the data to add noise
    :return: df with 2x observations
    """
    data = df.copy()
    targets = df.loc[:, labels]
    augmented_data = data.iloc[:, data_iloc:]

    noise = np.random.normal(0, 0.00005, augmented_data.shape)
    augmented_data = augmented_data + noise

    augmented_data = pd.concat([targets, augmented_data], axis=1)
    augmented_data = pd.concat([df, augmented_data], axis=0)

    return augmented_data

def evaluate_set(X_test, scaler1, scaler2, regressor, clf):
    """
    This function is used for evaluate (make a prediction) of a given batch of coconut milk
    :param X_test: data; please do not include labels
    :param scaler1: regressor PCA to transform the data
    :param scaler2: classifier PCA to transform the data
    :param regressor: estimator to predict water addition
    :param clf: classifier to predict origin
    :return: dataframe containing the prediction and a stored csv with the date of analysis
    """
    day = int(input("Insert day of evaluation: "))
    month = int(input("Insert month of evaluation: "))
    year = int(input("Insert year of evaluation: "))

    water = scaler1.transform(X_test)
    origin = scaler2.transform(X_test)

    water_pred = regressor.predict(water).reshape(-1,1)
    origin_pred = clf.predict(origin).reshape(-1,1)

    data = np.concatenate((water_pred,origin_pred), axis = 1)
    data = pd.DataFrame(data = data, columns = ['Predicted water', 'Predicted origin'])

    data.to_csv(f"authenticated_set_{day}_{month}_{year}.csv")

    for i in range(data.shape[0]):
        print(f"Sample {i + 1} has {data.iloc[i,0]:0.2} % of added water "
              f"and is {'organic' if data.iloc[i,1] == 1 else 'industrialized'} ")

    return data

def trainANN(estimator, df, batch_size, n_epochs):
    """
    Train!: this function is used for training the regressor. Note that the PCA is set for
    the ANN regressor. You can fix the number of components to fit your regressor
    :param estimator: regressor to be trained
    :param df: dataframe; include labels
    :return: trained regressor
    """
    early_stopping_cb = keras.callbacks.EarlyStopping(
        patience=10,
        restore_best_weights=True
    )

    X = df.drop(['Add water (%)', 'Coconut milk type'], axis = 1)
    y = df.loc[:, 'Add water (%)']

    pca = PCA(n_components=35)
    pca.fit(X)

    scaler = pca

    X = pca.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    history = estimator.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=n_epochs,
        validation_data=[X_test, y_test],
        callbacks=early_stopping_cb
    )
    return scaler, estimator

def trainClf(clf, df):
    """
    Train!: this function is used for training the classifier. Note that the PCA is set for
    the Decision Tree Classifier. You can fix the number of components to fit your regressor
    :param clf: classifier to be trained
    :param df: dataframe; please include labels
    :return: trained classifier
    """
    dummies = pd.get_dummies(df['Coconut milk type'], prefix='type', prefix_sep=' ', drop_first=True)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(['Coconut milk type', 'Add water (%)'], axis=1)

    y = df.loc[:,'type Organic']
    X = df.drop(['type Organic'], axis = 1)

    pca = PCA(n_components=22)
    pca = pca.fit(X)

    scaler = pca

    X = pca.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    clf = clf.fit(X_train, y_train)

    return scaler, clf

