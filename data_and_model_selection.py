# virutal assistant: metri ojeda J.C
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold


def classification(data):
    data2 = data.copy()
    for i in range(len(data2)):
        data2[i] = np.round(data2[i], decimals = 0)
    return data2

def accuracy(groundtruth, predictions):
    tn, fp, fn, tp = confusion_matrix(groundtruth, predictions).ravel()
    obs = len(groundtruth)
    result = (tp+tn)/obs
    return result

def precision(groundtruth, predictions):
    # true positives / true positives + false positives
    tn, fp, fn, tp = confusion_matrix(groundtruth, predictions).ravel()
    result = tp / (tp + fp)
    return result

def recall(groundtruth, predictions):
    tn, fp, fn, tp = confusion_matrix(groundtruth, predictions).ravel()
    result = tp / (tp + fn)
    return result

def F1(groundtruth, predictions):
    numerator = precision(groundtruth, predictions) * recall(groundtruth, predictions)
    denominator = precision(groundtruth, predictions) + recall(groundtruth, predictions)
    result = 2 * (numerator/denominator)
    return result

def create_metrics(groundtruth, predictions):
    dic = {"accuracy": accuracy(groundtruth, predictions),
           "precision" : precision(groundtruth, predictions),
           "recall" : recall(groundtruth, predictions),
           "Fvalue" : F1(groundtruth, predictions)}
    return dic


def roc_curve(groundtruth, probabilities, predictions, estimator_name=str):
    """
    Function for plotting the ROC curve and calculating the AUC

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


def roc_curve(groundtruth, probabilities, predictions, estimator_name=str):
    """
    Function for plotting the ROC curve and calculating the AUC

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
    pd.DataFrame(model_hist.history).plot(figsize=(10, 6))
    fig = plt.gca().set_ylim(0, 1)
    fig = plt.ylabel('Accuracy and Loss')
    fig = plt.xlabel('Epochs')
    fig = plt.title('name')
    fig = plt.savefig('learning_curve_{}.png'.format(name), dpi=300)
    return


def snv(input_data):
    # Define a new array and populate it with the corrected data
    spectra = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        # Apply correction
        spectra[i, :] = (input_data.iloc[i, :] - np.mean(input_data.iloc[i, :])) / np.std(input_data.iloc[i, :])

    return spectra

def augmentation(df, labels = list, data_iloc = int):
    data = df.copy()
    targets = df.loc[:, labels]
    augmented_data = data.iloc[:, data_iloc:]

    noise = np.random.normal(0, 0.00005, augmented_data.shape)
    augmented_data = augmented_data + noise

    augmented_data = pd.concat([targets, augmented_data], axis=1)
    augmented_data = pd.concat([df, augmented_data], axis=0)

    return augmented_data

# %%
# open data file
data = pd.read_csv(r'coconut_ftir_data.csv')

# Data augmentation
augmented_data = augmentation(data, labels = ['Coconut milk type', 'Add water (%)'], data_iloc = 2)
augmented_data.to_csv('augmented_data.csv', index = False)
#%%
###################################################################
#### plotting FTIR data #####

# subset of organics and industrialized coconut milks for plotting#

#organic = augmented_data[augmented_data['Coconut milk type'] == 'Organic']
#industrial = augmented_data[augmented_data['Coconut milk type'] == 'Industrialized']

# creating the means of each subset for plotting
#organic_means = organic.groupby(by=['Add water (%)']).mean()
#industrial_means = industrial.groupby(by=['Add water (%)']).mean()

#colormap = plt.get_cmap('inferno')
#count = 0
#x_ticks = np.linspace(2500, 4000, num=15)
#xlabels = np.linspace(2500, 4000, num=15, dtype=np.int16)

#plt.figure(figsize=(10, 5), dpi=300)
#for i in range(3):
#    axis = np.linspace(2501, 4000, num=729)
#    plt.plot(axis, organic_means.iloc[i, :], color=colormap(count),
#             label=f'Organic {organic_means.index[i]} % added water')
#    count += 0.1
#for i in range(3):
#    axis = np.linspace(2501, 4000, num=729)
#    plt.plot(axis, industrial_means.iloc[i, :], color=colormap(count),
#             linestyle='--',
#             label=f'Industrial {industrial_means.index[i]} % added water')
#    count += 0.1
#plt.ylabel('Absorbance')
#plt.xlabel('Wavenumber (cm $^{-1}$)')
#plt.xticks(ticks=x_ticks, labels=xlabels, rotation=90)
#plt.legend()
#plt.title('FTIR spectra for coconut milks with water addition (augmented)')
#plt.tight_layout()
#plt.show()
# plt.savefig('FTIR_spectra (augmented).jpg', dpi=300)


# %%
############## Prediction of water content ################################

for i in range(2, 50):
    X = augmented_data.iloc[:, 2:]
    y = augmented_data.iloc[:, 1]
    testingpca = PCA(n_components=i)
    testingpca.fit(X)
    X = testingpca.transform(X)
    scores = cross_val_score(LinearRegression(), X, y)
    print(f"{np.mean(scores)} r2 with {i} components")

X = augmented_data.iloc[:, 2:]
y = augmented_data.iloc[:, 1]

pca = PCA(n_components=35)
pca.fit(X)
sum(pca.explained_variance_)

X = pca.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

####################### Linear Regressor #############
reg_scores = cross_val_score(LinearRegression(), X, y)
reg = LinearRegression().fit(X_train, y_train)
reg_y_pred = reg.predict(X_test)
reg_score = r2_score(y_test, reg_y_pred)
reg_error = mean_squared_error(y_test, reg_y_pred)

####################### ANN #########################

early_stopping_cb = keras.callbacks.EarlyStopping(
    patience=10,
    restore_best_weights=True
)
epochs = 800
bs = 8

mlp = Sequential()
mlp.add(Dense(36, input_shape=(35,), activation='relu'))
mlp.add(Dense(24, input_shape=(36,), activation='relu'))
mlp.add(Dense(6, input_shape=(24,), activation='relu'))
mlp.add(Dense(1, input_shape=(6,)))

mlp.compile(loss='mean_squared_error', optimizer='adam')

# estimator = [KerasRegressor(mlp, epochs = epochs,  batch_size = bs, verbose = 1)]
# kfold = KFold(10)
# mpl_scores = cross_val_score(estimator, X, y, cv = kfold, scoring = 'mean_squared_error')

history = mlp.fit(
    X_train, y_train,
    batch_size=bs,
    epochs=epochs,
    validation_data=[X_test, y_test],
    callbacks=early_stopping_cb
)

plt.figure(figsize=(8, 5))
plt.plot(mlp.history.history['loss'], color='orange', label='loss')
plt.plot(mlp.history.history['val_loss'], color='blue', label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.title('MLP-R-relu Learning curve')
plt.show()
#plt.savefig('MLP-relu raw data learning curve.jpg', dpi=300)

mlp_pred = mlp.predict(X_test)
mlp_score = r2_score(y_test, mlp_pred)
mlp_error = mean_squared_error(y_test, mlp_pred)

#mlp.save('mlp_relu_rawdata.h5')
my_cmap = plt.get_cmap("inferno")


fig, ax = plt.subplots(1,2, figsize=(8,5), dpi = 300)

ax[0].bar(x = np.arange(0,2), height = [reg_score, mlp_score], color = my_cmap([0.6, 0.1]))
ax[0].set_ylabel("r$^{2}$ score")
ax[0].set_xticks(np.arange(0,2), ['Linear Regressor', 'MLP'])

ax[1].bar(x = np.arange(0,2), height = [reg_error, mlp_error], color = my_cmap([0.6, 0.1]))
ax[1].set_ylabel("Mean Squared Error")
ax[1].set_xticks(np.arange(0,2), ['Linear Regressor', 'MLP'])
plt.savefig('Models performance.jpg', dpi = 300)

#%%
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

############## Organic - Industrialized Classification ################################

dummies = pd.get_dummies(augmented_data['Coconut milk type'], prefix= 'type', prefix_sep= ' ', drop_first= True)
augmented_data = pd.concat([augmented_data, dummies], axis = 1)
augmented_data = augmented_data.drop(['Coconut milk type', 'Add water (%)'], axis = 1)


X = augmented_data.iloc[:,:-1]
y = augmented_data.iloc[:,-1]

for i in range(2, 50):
    X = augmented_data.iloc[:, :-1]
    y = augmented_data.iloc[:, -1]
    testingpca = PCA(n_components=i)
    testingpca.fit(X)
    X = testingpca.transform(X)
    scores = cross_val_score(DecisionTreeClassifier(), X, y, scoring= 'precision')
    print(f"{np.mean(scores)} precision with {i} components")

X = augmented_data.iloc[:,:-1]
y = augmented_data.iloc[:,-1]

pca = PCA(n_components=22)
pca = pca.fit(X)
sum(pca.explained_variance_)

X = pca.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

parameters = {'min_samples_leaf':[1,2,3,4,5,6,7,8,9],
              'min_samples_split': [2,3,4,5,6,7,8,9,10],
              'max_depth': [None, 2,3,4,5,6,7,8,9,10]}
DT = DecisionTreeClassifier()
DT = GridSearchCV(DT, parameters)
DT.fit(X,y)
print(DT.best_params_)

clf = DecisionTreeClassifier(criterion = 'gini', max_depth= 4, min_samples_leaf=1, min_samples_split= 3).fit(X_train, y_train)

clf_pred = clf.predict(X_test)
clf_performance= create_metrics(y_test, clf_pred)

