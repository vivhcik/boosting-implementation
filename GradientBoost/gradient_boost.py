import numpy as np
from decision_tree_reg import RegressionTree


class GradientBoosting():
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, 
                 random_state=17, n_samples=15, min_size=5, base_tree='Tree'):
        """
        Инициализация класса GradientBoosting.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.initialization = lambda y: np.mean(y) * np.ones([y.shape[0]])
        self.min_size = min_size
        self.loss_by_iter = []
        self.trees_ = []
        self.n_samples = n_samples
        self.base_tree = base_tree

    def fit(self, X, y):
        """
        Обучение ансамбля деревьев на данных (X, y).
        """
        self.X = X
        self.y = y
        b = self.initialization(y)
        
        prediction = b.copy()  
        
        for t in range(self.n_estimators):  
            
            if t == 0:
                resid = y
            else:
                resid = y - prediction
            
            
            if self.base_tree == 'Tree':
                tree = RegressionTree(self.max_depth, self.min_size)
            else:
                raise ValueError("Unsupported base_tree type")

            
            tree.fit(X, resid)

            
            b = tree.predict(X)
            self.trees_.append(tree)
            prediction += self.learning_rate * b
            
            
            mse = ((y - prediction) ** 2).mean()
            self.loss_by_iter.append(mse)

        return self

    def predict(self, X):
        """
        Генерация предсказаний для новых данных X.
        """
        pred = np.mean(self.y) * np.ones(X.shape[0])
        
        for tree in self.trees_:
            pred += self.learning_rate * tree.predict(X)
            
        return pred
