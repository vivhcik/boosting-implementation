import numpy as np


class RegressionTree:
    '''
    Класс RegressionTree решает задачу регрессии.
    деление на поддеревья на основе уменьшения ошибки.
    '''

    def __init__(self, max_depth=3, min_size=8):
        '''
        Инициализация параметров дерева.
        '''
        self.max_depth = max_depth # максимальная глубина дерева
        self.min_size = min_size   # минимальный размер поддерева min_samples_split
        self.value = 0             # значение в листе дерева
        self.feature_idx = -1      # индекс признака для разбиения
        self.feature_threshold = 0 # значение признака для разбиения
        self.left = None           # левое поддерево
        self.right = None          # правое поддерево

    def __mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred)**2)
    
    def fit(self, X, y):
        '''
        Обучение дерева на данных X (матрица признаков) и y (целевая переменная).
        '''
        self.value = np.mean(y) 
        base_error = self.__mse(self.value, y)

        error = base_error  
        flag = 0            # найдено ли хорошее разбиение

        prev_error_left = base_error
        prev_error_right = 0

        if self.max_depth <= 1:  # достигнута максимальная глубина
            return

        # Перебор признаков
        dim_shape = X.shape[1]
        for feat in range(dim_shape):
            idxs = np.argsort(X[:, feat])

            # количество сэмплов в левом и правом поддереве
            N = X.shape[0]
            N1, N2 = N, 0  # Левое и правое поддеревья
            thres = 1

            while thres < N - 1:
                N1 -= 1
                N2 += 1

                idx = idxs[thres]
                x = X[idx, feat]

                # пропуск одинаковых значений признаков
                if thres < N - 1 and x == X[idxs[thres + 1], feat]:
                    thres += 1
                    continue

                target_right = y[idxs][:thres]
                target_left = y[idxs][thres:]

                mean_right = np.mean(target_right)
                mean_left = np.mean(target_left)

                prev_error_left = N1 / N * self.__mse(target_left, mean_left)
                prev_error_right = N2 / N * self.__mse(target_right, target_right)
                loss_func = prev_error_left + prev_error_right
                
                # обновление параметров, если нашли лучшее
                if (loss_func < error) and (min(N1, N2) > self.min_size):
                    self.feature_idx = feat
                    self.feature_threshold = x
                    left_value = mean_left
                    right_value = mean_right 
                    flag = 1
                    error = loss_func

                thres += 1

        # если разбиение не найдено
        if self.feature_idx == -1:
            return

        self.left = RegressionTree(self.max_depth - 1)
        self.left.value = left_value
        self.right = RegressionTree(self.max_depth - 1)
        self.right.value = right_value

        #  разделите данные по индексу и обучите потомков
        idxs_l = (X[:, self.feature_idx] > self.feature_threshold)
        idxs_r = (X[:, self.feature_idx] <= self.feature_threshold)
        self.left.fit(X[idxs_l, :], y[idxs_l])
        self.right.fit(X[idxs_r, :], y[idxs_r])

    def __predict(self, x):
        '''
        Рекурсивное предсказание для одного примера.
        '''
        # если нет потомков, вернуть значение листа
        if self.feature_idx == -1:
            return self.value

        if x[self.feature_idx] > self.feature_threshold:
            return self.left.__predict(x)
        else:
            return self.right.__predict(x)

    def predict(self, X):
        '''
        Предсказание для всей матрицы данных.
        '''
        y = np.zeros(X.shape[0])

        for i in range(X.shape[0]):
            y[i] = self.__predict(X[i])

        return y


