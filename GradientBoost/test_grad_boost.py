import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from gradient_boost import GradientBoosting
from decision_tree_reg import RegressionTree

def mse(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()


def overfitting(model, X_train, X_test, y_train, y_test):
    loss_train = mse(y_train, model.predict(X_train))
    loss_test = mse(y_test, model.predict(X_test))
    
    print("mse train: {}".format(loss_train))
    print("mse test: {}".format(loss_test))
    
    value = 1 - (loss_train / loss_test)
    
    return value


def main():
    np.random.seed(42)
    X = np.random.rand(1000, 1) * 10
    y = 2 * X.flatten() + 3 + np.random.randn(1000) * 2 
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    GBM = GradientBoosting()
    GBM.fit(X_train, y_train)
    print(overfitting(GBM, X_train, X_test, y_train, y_test))
    
    RT = RegressionTree(max_depth=3, min_size=5)
    RT.fit(X_train, y_train)
    print(overfitting(RT, X_train, X_test, y_train, y_test))
    
main()