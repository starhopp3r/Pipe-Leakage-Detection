import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def prep_data():
    data = pd.read_csv("fake_data.csv", header=None)
    sensor_readings = []
    [sensor_readings.append(data.iloc[:, i]) for i in range(data.shape[1]-1)]
    X = np.array(sensor_readings).transpose()
    y = np.array(data.iloc[:, 10]).reshape(X.shape[0], 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = prep_data()
