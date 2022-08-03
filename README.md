# FTIR_FoodAnalysis
This is a repository where you can find Machine Learning models for authentication of foods. The data was retrieved from: 
Sitorus, A., Muslih, M., Cebro, I. S., & Bulan, R. (2021). Dataset of adulteration with water in coconut milk using FTIR spectroscopy. Data in Brief, 36, 107058.

The model selection and tuning of hyperparameters, data manipulation, and all testings are foun in `data_and_model_selection.py`. The functions as script are found in `authenticate.py`. The final testing like a "new dataset" is found in `prediction.py`.

## Data augmentation and selection

To make a little bit of context, the FTIR (Fourier Transform min-infrared spectroscopy) is a spectroscopy technique which measure the vibrations, bends, and stretching of functional groups within a molecule in the range of the infrared spectrum of light. Hence, it is widely used for study food composition. Moreover, this technique is usually coupled with chemometrics to describe food differences. Herein, I coupled a chemometric approach with machine learning to perform 2 tasks: create an authentication model to predict the quantity of water added to the product (adulteration) and another model for authentication between organic and industrialized samples.

Note that the original article used "market sample" and "instantized". I renamed the market samples as organic and instantized as industrialized.

The first step was to plot the data.
![FTIR_spectra](https://user-images.githubusercontent.com/87657676/182725562-36eae6f8-729e-405f-8d3c-055fbe3da64f.jpg)

Then, note that the dataset is very small (44 observations). I added twice the observations using normalized noise, resulting in 88 observation. I plotted the data to confirm that the data was not affected (compared to the original plot) after the augmentation. For this, I used the following lines:

```
# open data file
data = pd.read_csv(r'coconut_ftir_data.csv')

# Data augmentation
augmented_data = augmentation(data, labels = ['Coconut milk type', 'Add water (%)'], data_iloc = 2)
augmented_data.to_csv('augmented_data.csv', index = False)
```
![FTIR_spectra (augmented)](https://user-images.githubusercontent.com/87657676/182725854-031eb232-cbcb-46c9-bb5e-d473fd567528.jpg)

Sometimes, FTIR data is pretreated sith SNV or Savitzky Golay 1st or 2nd Derivative to smooth the curves. However, I noticed a great underperformance when this techniques were applied. You can try it by running the script `savitzky_golay_test.py`

# Data preprocessing and model selection

As I mentioned before, chemometrics usually perform unsupervised Machine Learning models such as PCA or PLS to reduce dimensionality. In this test, I decided to use PCA. The maximum variance explained was 0.11 (11 %); hence, I used Cross Validation Score from SKlearn to decide the number of components to be used. I tried SVR, Linear Regressor a Multilayer Perceptron

```
for i in range(2, 50):
    X = augmented_data.iloc[:, 2:]
    y = augmented_data.iloc[:, 1]
    testingpca = PCA(n_components=i)
    testingpca.fit(X)
    X = testingpca.transform(X)
    scores = cross_val_score(LinearRegression(), X, y)
    print(f"{np.mean(scores)} r2 with {i} components")
```

The highest CV score (0.97) and the highest r2 in the test set was obtained with 35 components. Therefore, I used 35 components for make the predictions. I also developed a MLP. The next table summarizes the models performance (including the classifier).


| Model             |      R2       |     MSE       |   Accuracy    |
| -------------     | ------------- |---------------|---------------|
| Linear Regressor  | 0.98          |        0.90   |       N/A     |
| MLP               | 0.99          |   0.12        |   N/A         |     
| Decisstion Tree   |  N/A          |       N/A     |      0.88     |

Due to the models' performance, I decided to use a MLP with the following architecture:
``` 
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

```

Finally, I tried with a SVM because it is powerful for binary classification; but the overal performace was around 60 % for all metrics. Then, I decided to use a Decission Tree Classifier due to their versatility. The configuration of the model was selected using the Grid Search Function of Sklearn.

```
parameters = {'min_samples_leaf':[1,2,3,4,5,6,7,8,9],
              'min_samples_split': [2,3,4,5,6,7,8,9,10],
              'max_depth': [None, 2,3,4,5,6,7,8,9,10]}
DT = DecisionTreeClassifier()
DT = GridSearchCV(DT, parameters)
DT.fit(X,y)
print(DT.best_params_)

clf = DecisionTreeClassifier(criterion = 'gini', max_depth= 4, min_samples_leaf=1, min_samples_split= 3).fit(X_train, y_train)

```

The model showed good accuracy (88 %), moderate precision (76 %), and high recall (100 %).
