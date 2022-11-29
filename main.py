import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris


def kernel(r):  # Функция ядра Епанечникова
    return ((3 / 4) * (1 - r * r))


def distance(x1, x2):  # Нахождение расстояния между двумя точками в евклидовой метрике
    return np.sqrt(np.sum((x1 - x2) ** 2))


def get_neighbors(X, Y, X_i, k):  # Нахожу ближайших соседей для объекта X_test
    neighbors = np.array([kernel(distance(X[i], X_i)) for i in range(len(X))])
    neighbors = np.column_stack((neighbors, Y))  # соединяю массив расстояний и значения классов
    neighbors = neighbors[neighbors[:, 0].argsort()]  # сортирую по столбцу расстояний
    neighbors = np.flip(neighbors, axis=0)  # делаю реверс
    return np.array(neighbors[:k])  # возвращаю первые k строк


def predict(neighbors, Y):  # предсказываю класс для объекта
    cl_uniq = np.unique(Y)
    sum_weight = np.zeros(len(cl_uniq))
    for i in range(len(neighbors)):
        for j in range(len(cl_uniq)):
            if (cl_uniq[j] == neighbors[i, 1]):
                sum_weight[j] += neighbors[i, 0]
                break
    return sum_weight.argmax()  # возвращаю индекс элемента с максимальным значением. индекс означает класс


def getAccuracy(test, control):  # Возвращаю точность предсказанных ответов в общем
    acc = sum([test[i] == control[i] for i in range(len(test))])
    if (acc):
        acc = acc / len(test)
    return (acc * 100)


def LOO(X, Y):  # Проведение скользящего контроля
    mn = 10
    optimal = 0
    LOO_array = []
    for k in range(1, len(X) + 1):
        print("k = ", k)
        Y_control = []
        for i in range(len(X)):
            Y_control.append(predict(get_neighbors(np.delete(X, i, 0), np.delete(Y, i), X[i], k), obj.target))
        LOO = (sum([Y[i] != Y_control[i] for i in range(len(X))]))
        LOO_array.append(LOO)
        if LOO < mn:
            mn = LOO
            optimal = k
        print(LOO)
    k = optimal
    print("optimal k = :", k)
    return LOO_array, k


###################################
obj = load_iris()
X = obj.data
Y = obj.target
LOO, k = LOO(X, Y)

plt.title("Leave-one-out")
plt.xlabel("Количество ошибокЗначение k")
plt.ylabel("Значение k")
plt.plot(LOO)
plt.plot(k, LOO[k], marker="o")

plt.show()