# LAB1
В данной лабораторной работе используем пример полносвязной нейронной сети из 5 слоёв. Сначала добавляем Flatten-слой, разворачивающий нашу матрицу изображений в одномерную для дальнейшего поступления на Dense-слой. Далее используем пять слоёв Dense: 2 слоя на 128 нейронов, еще 2 слоя на 64 нейрона и слой на 10 нейронов. Первые 4 Dense слоя используют активацию 'relu', последний - 'softmax'.
При обучении нейронной сети происходит 10 этапов обучения, при этом в цикле обучения используются:
оптимизатор 'adam'
функция потерь 'sparse_categorical_crossentropy'
метрика 'accuracy'
Графики метрики точности и функции потерь:
 
 
Графики метрики точности и функции потерь на валидационной выборке:

