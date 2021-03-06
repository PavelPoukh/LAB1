# LAB1
В данной лабораторной работе используем пример полносвязной нейронной сети из 5 слоёв. 
Сначала добавляем Flatten-слой, разворачивающий нашу матрицу изображений в одномерную для дальнейшего поступления на Dense-слой.
Далее используем пять слоёв Dense: 2 слоя на 128 нейронов, еще 2 слоя на 64 нейрона и слой на 10 нейронов. 
Первые 4 Dense слоя используют активацию 'relu', последний - 'softmax'.

При обучении нейронной сети происходит 10 этапов обучения, при этом в цикле обучения используются:
-оптимизатор 'adam'
-функция потерь 'sparse_categorical_crossentropy'
-метрика 'accuracy'

Как ReLU сравнивает: ReLU является линейным (тождественным) для всех положительных значений и нулем для всех отрицательных значений.

Обучение нейронной сети происходит за 40 эпох

Значение итоговой точности: 0,3766

Графики метрики точности и функции потерь:
![Image alt](https://github.com/PavelPoukh/LAB1/blob/master/epoch_categorical_accuracy.PNG)
![Image alt](https://github.com/PavelPoukh/LAB1/blob/master/epoch_loss.PNG)

Графики метрики точности и функции потерь на валидационной выборке:

![Image alt](https://github.com/PavelPoukh/LAB1/blob/master/epoch_var_categorical_accuracy.PNG)
![Image alt](https://github.com/PavelPoukh/LAB1/blob/master/epoch_var_loss.PNG)
