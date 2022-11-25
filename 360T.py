import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier


class NeuralNetwork(object):
    
    def __init__(self, *args, **kwargs):
        self._data = pd.read_csv("360T.csv")

    def data_analyzer(self):
        """
            Разбиваем данные на предикторы и отклики
            На тестовую и тренировочную выборки (0.25/0.75)
        """
        predictors =self._data.drop(['class'], axis = 1)
        callbacks =self._data['class']
        stratify = self._data.iloc[:,-1]
        print(stratify)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(predictors, 
                                                                                stratify, 
                                                                                test_size=0.25, 
                                                                                random_state=1)

        print(self.X_train)
        print(self.X_test)
    
    def sc_processing(self):
        """
            Обучаем скейлер и применяем преобразование.
            Строим классификатор
        """
        self.sc = StandardScaler()
        self.X_train_scaled = self.sc.fit_transform(self.X_train)
        self.X_test_scaled = self.sc.transform(self.X_test)

        self.mlp = MLPClassifier(random_state=1, 
                    hidden_layer_sizes=(31,10), # Используем два скрытых слоя и укажем число нейронов в каждом
                    activation='logistic', # Определим функцию активации
                    max_iter=1000, # Максимальное число эпох обучения
                   )

        self.mlp.fit(self.X_train_scaled, self.y_train)

    def plot(self):
        mlp = self.mlp
        plt.plot(mlp.loss_curve_)
        plt.show()

    def report(self):
        y_mlp_pred = self.mlp.predict(self.X_test_scaled)
        print(classification_report(self.y_test, y_mlp_pred, digits=3))

    def predicted(self):
        """
            Берём сторонние данные и выполняем прогноз для каждого объёкта
        """
        test_obj_1_data = pd.read_csv('test_obj_1.csv')    
        test_obj_2_data = pd.read_csv('test_obj_2.csv')    
        test_obj_3_data = pd.read_csv('test_obj_3.csv')   
        test_obj_4_data = pd.read_csv('test_obj_4.csv')
        
        test_obj_1 = test_obj_1_data.loc[[0]]
        test_obj_2 = test_obj_2_data.loc[[0]]
        test_obj_3 = test_obj_3_data.loc[[0]]
        test_obj_4 = test_obj_4_data.loc[[0]]

        test_obj_1_transform = self.sc.transform(test_obj_1)
        test_obj_2_transform = self.sc.transform(test_obj_2)
        test_obj_3_transform = self.sc.transform(test_obj_3)
        test_obj_4_transform = self.sc.transform(test_obj_4)

        print("Assigned class 1 object: ",self.mlp.predict(test_obj_1_transform))
        print("Assigned class 2 object: ",self.mlp.predict(test_obj_2_transform))
        print("Assigned class 3 object: ",self.mlp.predict(test_obj_3_transform))
        print("Assigned class 4 object: ",self.mlp.predict(test_obj_4_transform))


if __name__ == '__main__':
    NN = NeuralNetwork()
    NN.data_analyzer()
    NN.sc_processing()
    NN.report()
    NN.predicted()
    NN.plot()