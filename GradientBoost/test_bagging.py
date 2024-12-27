import numpy as np
import os

from bagging import Bagging
from decision_tree_reg import RegressionTree

from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse


def main():
    np.random.seed(42)
    X = np.random.rand(1000, 1) * 10
    y = 2 * X.flatten() + 3 + np.random.randn(1000) * 2 
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    lin_reg = LinearRegression()
    blr = Bagging(lin_reg)
    sklearn_blr = BaggingRegressor(lin_reg)
    
    blr.fit(X_train, y_train)
    sklearn_blr.fit(X_train, y_train)
    
    y_pred = blr.predict(X_test)
    sklearn_y_pred = sklearn_blr.predict(X_test)

    
    print("mse blr: {}".format(mse(y_test, y_pred)))
    print("mse sklearn_blr: {}".format(mse(y_test, sklearn_y_pred)))
    
    dtr = RegressionTree(max_depth=10, min_size=5)
    sklearn_dtr = DecisionTreeRegressor(max_depth=10, min_samples_split=5)
    
    bdt = Bagging(sklearn_dtr)
    sklearn_bdt = Bagging(sklearn_dtr)
    
    bdt.fit(X_train, y_train)
    sklearn_bdt.fit(X_train, y_train)
    
    y_pred = bdt.predict(X_train)
    sklearn_y_pred = sklearn_bdt.predict(X_train)
    
    print("mse bdt: {}".format(mse(y_train, y_pred)))
    print("mse sklearn_bdt: {}".format(mse(y_train, sklearn_y_pred)))
    
    
main()