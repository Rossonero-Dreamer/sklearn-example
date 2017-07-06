from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import random

def clfNB(X_train, y_train):
    """
    Obtain a Gaussian classifier based on training data
    Return: classifier for prediction
    """
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    return clf

def clfSVM(X_train, y_train):
    clf = svm.SVC(kernel="rbf",gamma=1.0,C=1.0)
    clf.fit(X_train, y_train)
    return clf

def clfTree(X_train, y_train):
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    return clf

def clfBoost(X_train, y_train):
    clf = AdaBoostClassifier()
    clf.fit(X_train, y_train)
    return clf

def clfKNN(X_train, y_train):
    clf = KNeighborsClassifier()
    clf.fit(X_train, y_train)
    return clf

def createData(n=1000):
    """
    Create training and test data
    Return: X_train, y_train, X_test, y_test
    """
    random.seed(42)

    #generate x values
    grade = [random.random() for ii in range(n)]
    bumpy = [random.random() for ii in range(n)]
    error = [random.random() for ii in range(n)]

    #generate y values
    #0 for slow, 1 for fast
    y = [round(grade[ii]*bumpy[ii]+0.3+0.1*error[ii]) for ii in range(n)]
    for ii in range(n):
        if grade[ii]>0.8 or bumpy[ii]>0.8:
            y[ii] = 1.0

    #split data into training and test sets
    #X[i][0]: grade, X[i][1]: bumpy
    X = [[gg,bb] for gg,bb in zip(grade,bumpy)]
    split = int(n*0.75)
    X_train = X[:split]
    X_test = X[split:]
    y_train = y[:split]
    y_test = y[split:]

    return X_train, y_train, X_test, y_test

def scoreClf(clf, X_test, y_test):
    pred = clf.predict(X_test)
    return accuracy_score(pred, y_test)

def plotClf(clf, X_test, y_test):
    """
    Plot the classifier with regard to test data
    """
    #x is grade and y is bumpy
    x_min = 0.0; x_max = 1.0
    y_min = 0.0; y_max = 1.0

    #create a meshgrid and predict every grid point in it
    h = 0.01 #step size of the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    #plot the result
    Z = Z.reshape(xx.shape)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.pcolormesh(xx, yy, Z, cmap=pl.cm.seismic)

    #plot the test labels
    n = len(X_test)
    grade_sig = [X_test[ii][0] for ii in range(n) if y_test[ii]==0]
    bumpy_sig = [X_test[ii][1] for ii in range(n) if y_test[ii]==0]
    grade_bkg = [X_test[ii][0] for ii in range(n) if y_test[ii]==1]
    bumpy_bkg = [X_test[ii][1] for ii in range(n) if y_test[ii]==1]

    plt.scatter(grade_sig, bumpy_sig, color='b', label='fast')
    plt.scatter(grade_bkg, bumpy_bkg, color='r', label='slow')
    plt.legend()
    plt.xlabel('grade')
    plt.ylabel('bumpiness')

    plt.show()


