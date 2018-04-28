"""
Authored by:
Nikhil Raghavendra on 8/4/2018.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

# Hyperparameters
input_size = 10
num_classes = 2
num_epochs = 200
batch_size = 100
learning_rate = 1e-3

# File name of saved model
model_name = 'savedmodel.pt'
# Test size
t_size = 0.20


class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        y_hat = self.linear(x)
        return y_hat


def prep_data():
    data = pd.read_csv("fake_data.csv", header=None)  # Read CSV
    sensor_readings = []  # Initialize sensor readings
    # Append sensor readings to list
    [sensor_readings.append(data.iloc[:, i]) for i in range(data.shape[1]-1)]
    X = np.array(sensor_readings).transpose()
    y = np.array(data.iloc[:, 10]).reshape(X.shape[0], 1)
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=t_size)
    return X_train, X_test, y_train, y_test


def tt_split():
    X_train, X_test, y_train, y_test = prep_data()  # Prepare dataset
    train = list(zip(X_train, y_train))  # Zip taining data
    test = list(zip(X_test, y_test))  # Zip testing data
    train = DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
    test = DataLoader(dataset=test, batch_size=batch_size, shuffle=True)
    return train, test


def train_model(training_data, save=True):
    model = LogisticRegression(input_size, num_classes)  # Model
    criterion = nn.CrossEntropyLoss()  # Loss function
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Training the model
    for epoch in range(num_epochs):
        for i, (values, labels) in enumerate(training_data):
            values = Variable(values).float()
            labels = torch.max(Variable(labels), 1)[0]
            # Forward -> Backprop -> Optimize
            optimizer.zero_grad()  # Manually zero the gradients
            outputs = model(values)  # Predict new labels given value
            loss = criterion(outputs, labels)

            loss.backward()  # Compute the error gradients
            optimizer.step()  # Optimize the model

            if i % 100 == 0:
                print("Epoch {}, loss :{}".format(epoch + 1, loss.data[0]))
    # Save model
    if save:
        torch.save(model, model_name)


def evaluate_model(testing_data):
    model = torch.load(model_name)  # Load saved model
    model.eval()
    pred = []
    actual = []
    for values, labels in testing_data:
        values = Variable(values).float()
        outputs = model(values)
        _, predicted = torch.max(outputs.data, 1)
        for i in range(predicted.numpy().shape[0]):
            pred.append(predicted.numpy().reshape(predicted.size(0), 1)[i][0])
        for i in range(labels.numpy().shape[0]):
            actual.append(labels.numpy().reshape(labels.size(0), 1)[i][0])
    mean = np.mean(np.array(pred) == np.array(actual)) * 100
    print("Accuracy of model is {}%".format(mean))


def predict(values):
    model = torch.load(model_name)
    values = Variable(torch.from_numpy(values)).float()
    output = model(values)
    _, predicted = torch.max(output.data, 1)
    return predicted.numpy()


if __name__ == '__main__':
    # Get the training and testing data
    train_loader, test_loader = tt_split()
    train_model(train_loader)  # Train model
    evaluate_model(test_loader)  # Accuracy of model
    # Test readings
    test_readings = np.array([[179, 93, 113, 144, 55, 124, 43, 64, 50, 51]])
    # Predicted values
    predicted = predict(test_readings)
    print(predicted)
