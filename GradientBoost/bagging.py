import numpy as np


class Bagging():
    
    def __init__(self, esitmator=None, n_estimator=10):
        """
        Инициализация класса Bagging.
        """
        self.esitmator = esitmator
        self.n_esitmator = n_estimator
        self.models = []

    def get_bootstrap_samples(self, data, target, N):
        """
        Генерация бутстрап-выборок из обучающих данных.
        """
        n_sample = data.shape[0]
        bootstrap = []
        
        for _ in range(N):
            indexs = np.random.randint(0, n_sample, size=n_sample)
            sample_data = data[indexs]
            sample_target = target[indexs]
            
            bootstrap.append((sample_data, sample_target))
    
        return bootstrap
        
        
    def fit(self, X, y):
        """
        Обучение ансамбля деревьев на бутстрап-выборках.
        """
        bootstrap_data = self.get_bootstrap_samples(X, y, self.n_esitmator)
        
        for i in range(self.n_esitmator):
            model = self.esitmator
            X_sample = bootstrap_data[i][0]
            y_sample = bootstrap_data[i][1]
            self.models.append(model.fit(X_sample, y_sample))
        
        return self
        
        
    def predict(self, X):
        """
        Генерация предсказаний для тестовых данных.
        """
        pred_models = []
        for i in range(self.n_esitmator):
            pred_models.append(self.models[i].predict(X))
        pred_models = np.array(pred_models).T
        
        return np.array([np.mean(pred) for pred in pred_models])
        
