import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from decision_tree_reg import RegressionTree
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def main():

    np.random.seed(42)
    X = np.random.rand(100, 1) * 10
    y = 2 * X.flatten() + 3 + np.random.randn(100) * 2 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    

    dtr = DecisionTreeRegressor(max_depth=10, min_samples_split=2)
    dtr.fit(X_train, y_train)
    dtr_pred = dtr.predict(X_test)
    print(mean_squared_error(y_test, dtr_pred))

    tree = RegressionTree(10, 2)
    tree.fit(X_train, y_train)
    tree_pred = tree.predict(X_test)
    print(mean_squared_error(y_test, tree_pred))
    

main()