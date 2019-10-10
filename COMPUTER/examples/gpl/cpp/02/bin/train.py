#!/usr/bin/python

from keras.datasets import mnist
from keras.utils    import to_categorical

from keras import models
from keras import layers

def main():
    (images_train, labels_train), (images_test, labels_test) = mnist.load_data()
    images_train = images_train.reshape((images_train.shape[0], 28*28))
    images_test  = images_test .reshape((images_test .shape[0], 28*28))
    images_train = images_train.astype('float32') / 255
    images_test  = images_test .astype('float32') / 255
    labels_train = to_categorical(labels_train)
    labels_test  = to_categorical(labels_test )

    nn = models.Sequential()
    nn.add(layers.Dense(512, activation='relu', input_shape=(28*28, )))
    nn.add(layers.Dense( 10, activation='softmax'))
    nn.compile(optimizer='rmsprop',
               loss     ='categorical_crossentropy',
               metrics  =['accuracy'])

    nn.fit(images_train, labels_train, epochs=5, batch_size=128)
    loss_test, accuracy_test = nn.evaluate(images_test, labels_test)
    print(f'ACCURACY: {accuracy_test * 100:3.2f}%')

    nn.save('mnist.model')

if __name__ == '__main__':
    main()
