



###################################################################################
###################################################################################
######################### Smoothing signal ########################################
###################################################################################
###################################################################################

x = augmented_data.iloc[:, 2:]
y = augmented_data.iloc[:, 1]  ## selecting water addition as label, switch to 0 for organic/industrialized

x = savgol_filter(x, window_length=15, polyorder=2, deriv=1, axis=-1)
experiment_name = '1st Der 2nd Poly'

filtered = pd.DataFrame(x, columns=augmented_data.columns[2:].tolist())

labels = augmented_data.loc[:, ['Coconut milk type', 'Add water (%)']]
filtered = labels.join(filtered)

# subset of organics and industrialized coconut milks
organic = filtered[filtered['Coconut milk type'] == 'Organic']
industrial = filtered[filtered['Coconut milk type'] == 'Industrialized']

# creating the means of each subset for plotting
organic_means = organic.groupby(by=['Add water (%)']).mean()
industrial_means = industrial.groupby(by=['Add water (%)']).mean()

colormap = plt.get_cmap('inferno')
count = 0
x_ticks = np.linspace(2500, 4000, num=15)
xlabels = np.linspace(2500, 4000, num=15, dtype=np.int16)

plt.figure(figsize=(10, 5), dpi=300)
for i in range(3):
    axis = np.linspace(2501, 4000, num=729)
    plt.plot(axis, organic_means.iloc[i, :], color=colormap(count),
             label=f'Organic {organic_means.index[i]} % added water')
    count += 0.15
for i in range(3):
    axis = np.linspace(2501, 4000, num=729)
    plt.plot(axis, industrial_means.iloc[i, :], color=colormap(count),
             linestyle='--',
             label=f'Industrial {industrial_means.index[i]} % added water')
    count += 0.15
plt.ylabel('Signal magnitude')
plt.xlabel('Wavenumber (cm $^{-1}$)')
plt.xticks(ticks=x_ticks, labels=xlabels, rotation=90)
plt.legend()
plt.title(f'Sav-Gol Filter {experiment_name} (FTIR)')
plt.tight_layout()
plt.show()
# plt.savefig(f'Sav-Gol Filter {experiment_name} (FTIR)', dpi=300)

for i in range(2, 50):
    X = filtered.iloc[:, 2:]
    y = filtered.iloc[:, 1]
    testingpca = PCA(n_components=i)
    testingpca.fit(X)
    X = testingpca.transform(X)
    scores = cross_val_score(LinearRegression(), X, y)
    print(f"{np.mean(scores)} r2 with {i} components")

X = filtered.iloc[:, 2:]
y = filtered.iloc[:, 1]

pca = PCA(n_components=22)
pca.fit(X)
X = pca.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

####################### Linear Regressor #############
scores = cross_val_score(LinearRegression(), X, y)

reg = LinearRegression().fit(X_train, y_train)

y_pred = reg.predict(X_test)
score = r2_score(y_test, y_pred)
error = mean_squared_error(y_test, y_pred)

####################### ANN #########################
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold

early_stopping_cb = keras.callbacks.EarlyStopping(
    patience=10,
    restore_best_weights=True
)
epochs = 800
bs = 8

mlp = Sequential()
mlp.add(Dense(24, input_shape=(24,), activation='relu'))
mlp.add(Dense(36, input_shape=(24,), activation='relu'))
mlp.add(Dense(64, input_shape=(36,), activation='relu'))
mlp.add(Dense(36, input_shape=(64,), activation='relu'))
mlp.add(Dense(12, input_shape=(36,), activation='relu'))
mlp.add(Dense(1, input_shape=(12,)))

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
plt.title(f'MLP-{experiment_name} Learning curve')
plt.savefig(f'MLP-{experiment_name} learning curve.jpg', dpi=300)

mlp_pred = mlp.predict(X_test)
mlp_score = r2_score(y_test, mlp_pred)
mlp_error = mean_squared_error(y_test, mlp_pred)

mlp.save(f'mlp_{experiment_name}.h5')