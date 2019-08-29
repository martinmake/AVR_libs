#!/usr/bin/python

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras
sys.stderr = stderr

import signal

from keras.datasets import mnist
from keras.utils    import to_categorical

from keras import models
from keras import layers

import numpy

def SIGUSR1_handler(sig, frame):
    pass

def main():
    signal.signal(signal.SIGUSR1, SIGUSR1_handler)
    image = numpy.zeros(shape=(28*28));
    nn = models.load_model('./bin/mnist.model')

    while True:
        with open(sys.argv[1], 'r') as image_file:
            for i in range(28 * 28):
                state = image_file.read(1)
                if state == '1':
                    image[i] = 1.0
                else:
                    image[i] = 0.0

        prediction = nn.predict_classes(numpy.array([image]))[0]
        os.system(f'notify-send -t 60 "PREDICTION IS: {prediction}"')
        print(f'PREDICTION IS: {prediction}', file=sys.stderr)

if __name__ == '__main__':
    main()
