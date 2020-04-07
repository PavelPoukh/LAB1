import tensorflow as tf
import datetime

#from tensorflow.keras import datasets, layers, models
from keras import layers, models, datasets, losses
from keras.callbacks import TensorBoard

def create_model():
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(32, 32, 3)))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='relu'))
    model.compile(optimizer='adam',
                  loss=losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    print(model.summary())
    return model

def main():
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    train_images, test_images = train_images / 255.0, test_images / 255.0

    model = create_model()

    log_dir = 'logs/{}'.format(datetime.datetime.now())
    tensor_board = TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = model.fit(train_images, train_labels, epochs=10, validation_split=0.3, callbacks=[tensor_board])
    test = model.evaluate(train_images, test_labels)

main()
