import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics
from tensorflow import keras
from keras import optimizers
from keras.models import Model
from keras.layers import Dense, LSTM, Input, Activation
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

backcandles = 30
trained = 0# you can change this flag to 1 if you have already trained the model with the same data and trying to test it again

def process_training_data(data):
    data.reset_index(inplace=True)
    data.drop(['mkt_id', 'output_date'], axis=1, inplace=True)
    data = data[
        ['output_own_profits', 'output_own_cost', 'output_comp_price', 'output_X', 'output_own_share', 'output_own_sales',
         'output_own_price']]
    data_set = data.iloc[:, 0:6]  # .values without the price
    sc = MinMaxScaler(feature_range=(0, 1))
    data_set_scaled = sc.fit_transform(data_set)
    data['output_own_price'] .shift(-1)
    data.dropna(inplace= True  )
    data.reset_index(inplace= True)
    target = data.iloc[:, 7]
    X = []
    for j in range(6):
        X.append([])
        for i in range(backcandles, data_set_scaled.shape[0]):
            X[j].append(data_set_scaled[i - backcandles:i, j])

    X = np.moveaxis(X, [0], [2])
    X, yi = np.array(X), np.array(target[backcandles:])
    y = np.reshape(yi, (len(yi), 1))

    return X, y


def split_data_train_test(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    return X_train, y_train, X_test, y_test


def train_model(X_train, y_train):
    lstm_input = Input(shape=(backcandles, 6), name='lstm_input')
    inputs = LSTM(150, name='first_layer')(lstm_input)
    inputs = Dense(100, name='Second_dense_layer')(inputs)
    inputs = Dense(1, name='dense_layer')(inputs)
    output = Activation('linear', name='output')(inputs)
    model = Model(inputs=lstm_input, outputs=output)
    adam = optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=adam, loss='mse')
    model.fit(x=X_train, y=y_train, batch_size=10, epochs=30, shuffle=True, validation_split=0.1)
    model.save("pricing_optimization_model")
    return model

def predict_price(X_test, y_test, model):
    y_pred = model.predict(X_test)
    pd.DataFrame(y_pred).to_csv('predicted_prices.csv')
    Score = sklearn.metrics.r2_score(y_test, y_pred)
    plt.figure(figsize=(10, 5))
    plt.plot(y_test, color='black', label='true price')
    plt.plot(y_pred, color='green', label='predicted price')
    plt.title("predicted price on true price")
    plt.legend()
    plt.show()
    plt.plot(y_pred-y_test, color='black', label='y_pred-y_test')
    plt.ylim([-1, 1])
    plt.title("difference between predicted price and true price")
    plt.show()
    print("the score is : ") 
    print(Score)


data = pd.read_csv(r'E:\2onn\output_data.csv') # please place here the path of data on your pc

x, y = process_training_data(data)
X_train, y_train, X_test, y_test = split_data_train_test(x, y)

if trained == 0:
    model = train_model(X_train, y_train)
else:
    model = keras.models.load_model("pricing_optimization_model")
predict_price(X_test, y_test, model)
