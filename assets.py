import numpy as np
import matplotlib.pyplot as plt
import random

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def createData(n=1000):
    random.seed(42)
    np.random.seed(42)

    #randomize ages
    ages = []
    for ii in range(n):
        ages.append(random.randint(20,65))

    #randomize net worths
    net_worths = [6.25*ii + np.random.normal(scale=40.) for ii in ages]

    #reshape in np array
    ages = np.reshape(np.array(ages),(n,1))
    net_worths = np.reshape(np.array(net_worths),(n,1))

    #split train and test dataset
    ages_train, ages_test, net_worths_train, net_worths_test = train_test_split(ages, net_worths)

    return ages_train, ages_test, net_worths_train, net_worths_test

def createReg(X,y):
    reg = LinearRegression()
    reg.fit(X,y)
    return reg

def plotReg(reg, X_train, X_test, y_train, y_test):
    plt.clf()
    plt.scatter(X_train, y_train, color="b", label="train data")
    plt.scatter(X_test, y_test, color="r", label="test data")
    plt.plot(X_test, reg.predict(X_test), color="black")
    plt.legend(loc=2)
    plt.xlabel("ages")
    plt.ylabel("net worth")
    plt.show()


