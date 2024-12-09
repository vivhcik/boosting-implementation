import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from decision_tree_reg import RegressionTree
from sklearn.metrics import mean_squared_error

def main():

    np.random.seed(42)
    X = np.random.rand(100, 1) * 10  # Признаки
    y = 2 * X.flatten() + 3 + np.random.randn(100) * 2  # Цель с шумом

    

    # Разделим на train/test
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]

    dtr = DecisionTreeRegressor()
    dtr.fit(X_train, y_train)
    dtr_pred = dtr.predict(X_test)
    print(mean_squared_error(y_test, dtr_pred))
    
    # Обучение
    tree = RegressionTree(10, 2)
    tree.fit(X_train, y_train)
    tree_pred = tree.predict(X_test)
    print(mean_squared_error(y_test, tree_pred))
    print(mean_squared_error(y_test, [y_test.mean() for _ in range(len(y_test))]))
    
    plt.scatter(X_test, y_test)
    plt.scatter(X_test, tree_pred)
    plt.scatter(X_test, dtr_pred)
    plt.show()
    

main()