########
# Importing libraries needed for the program
########
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.tree import DecisionTreeClassifier
import authenticate as at

#%%
data = pd.read_csv(r'augmented_data.csv')

############ Create the regressor for water adulteration ############

mlp = Sequential()
mlp.add(Dense(36, input_shape=(35,), activation='relu'))
mlp.add(Dense(24, input_shape=(36,), activation='relu'))
mlp.add(Dense(6, input_shape=(24,), activation='relu'))
mlp.add(Dense(1, input_shape=(6,)))

mlp.compile(loss='mean_squared_error', optimizer='adam')

############ Create the classifier for origin ############
dt = DecisionTreeClassifier(criterion = 'gini', max_depth= 4, min_samples_leaf=1, min_samples_split= 3)

#### Training #######
scaler_reg, mlp = at.trainANN(mlp, data, 8, 800)
scaler_cf, dt = at.trainClf(dt, data)

#### Prediction #####
indexes = np.random.randint(0,84,25).tolist()
evaluation = data.iloc[indexes,2:]

prediction= at.evaluate_set(evaluation, scaler_reg, scaler_cf, mlp, dt)