from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import keras

import copy
import math
from keras.models import Sequential, Model
from keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Activation
from keras.callbacks import LearningRateScheduler
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
import tensorflow as tf
import numpy as np
import sys
import os
from six.moves import cPickle



#from keras.applications import ResNet50
from keras_model import ResNet50

acc_thresh = 0.749

from numpy.random import seed
seed(1234)
#from tensorflow import set_random_seed
tf.random.set_seed(1234)


class EarlyExitTrainAccuracy(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        actual_accuracy = float(logs.get('accuracy'))
        print("\nCallback: We have reached %2.2f%% accuracy." % (actual_accuracy * 100))
        if actual_accuracy >= acc_thresh:
            print("\nWe have reached %2.2f%% accuracy, so we will stopping training." %(acc_thresh*100))
            self.model.stop_training = True


def keras_resnet50(dataset="cifar10", number_classes=10, learning_rate=1e-2, momentum=0.9):

    base_model = ResNet50(include_top=False, weights=None, dataset=dataset)

    # add a global spatial average pooling layer
    #x = base_model.output
    #x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    #x = Dense(512)(x)
    #x = BatchNormalization()(x)
    #x = Activation('relu')(x)
    # and a logistic layer -- 10 classes for CIFAR10
    #predictions = Dense(number_classes, activation='softmax')(x)

    # this is the model we will train
    #model = Model(inputs=base_model.input, outputs=predictions)

    model = base_model
    opt = keras.optimizers.SGD(lr=learning_rate, momentum=momentum, nesterov=False)
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=opt, metrics=['accuracy'])
    model.summary()
    return model


def keras_train(model,  x_train, y_train, x_test, y_test, epochs=10, batch_size=128,  dynamic_lr=False, time2acc=False):
    callback_list = []

    if dynamic_lr:
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)
        callback_list.append(lr_scheduler)

    if time2acc:
        early_exit_callback = EarlyExitTrainAccuracy()
        callback_list.append(early_exit_callback)

    if len(callback_list) > 0:
        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            validation_data=(x_test, y_test), callbacks=callback_list)
        return model, history

    else:
        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            validation_data=(x_test, y_test))
        return model, history


def keras_train_flow(model,  train, test, epochs=10, dynamic_lr=False, time2acc=False):
    callback_list = []

    if dynamic_lr:
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)
        callback_list.append(lr_scheduler)

    if time2acc:
        early_exit_callback = EarlyExitTrainAccuracy()
        callback_list.append(early_exit_callback)

    if len(callback_list) > 0:
        if (tf.__version__).startswith("1"):
            history = model.fit_generator(train,
                                          steps_per_epoch=len(train),
                                          epochs=epochs,
                                          verbose=1,
                                          validation_data=test, callbacks=callback_list)
        else:
            history = model.fit(train,
                                epochs=epochs,
                                verbose=1,
                                validation_data=test,  callbacks=callback_list)
        return model, history

    else:
        if (tf.__version__).startswith("1"):
            history = model.fit_generator(train,
                                          steps_per_epoch=len(train),
                                          epochs=epochs,
                                          verbose=1,
                                          validation_data=test)
        else:
            history = model.fit(train,
                                epochs=epochs,
                                verbose=1,
                                validation_data=test)
        return model, history


def mn4_keras_train_flow(model,  train, test, epochs=10, dynamic_lr=False, time2acc=False):
    callback_list = []

    if dynamic_lr:
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)
        callback_list.append(lr_scheduler)

    if time2acc:
        early_exit_callback = EarlyExitTrainAccuracy()
        callback_list.append(early_exit_callback)

    if len(callback_list) > 0:
        history = model.fit_generator(train,
                                      steps_per_epoch=len(train),
                                      epochs=epochs,
                                      verbose=1,
                                      validation_data=test, callbacks=callback_list)
    else:
        history = model.fit_generator(train,
                                      steps_per_epoch = len(train),
                                      epochs=epochs,
                                      verbose=1,
                                      validation_data=test)
    return model, history


def keras_evaluate(model,  x_test, y_test):
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return score


def keras_evaluate_flow(model,  test):
    score = model.evaluate(test, verbose=0)
    print('Evaluation Test loss:', score[0])
    print('Evaluation Test accuracy:', score[1])
    return score


def scheduler(epoch, lr): #initial_lr=1e-2, epoch=90):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        iniital_lr (float): The initial learning rate
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    # initial_lr = 1e-2
    if epoch > 0 and epoch % 30 == 0:
        lr = lr * 0.1
    # lr = lr * (0.1 ** (epoch / 30))

    # print('Learning rate: ', lr)
    return lr


def keras_print_devices():
    print("Tensorflow version: ", tf.__version__)
    print("Keras version: ", keras.__version__)

    return
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(gpus))
    for gpu in gpus:
        print("gpu: ", gpu)
    cpus = tf.config.experimental.list_physical_devices('CPU')
    for cpu in gpus:
        print("cpu: ", cpu)
    return


def load_batch_helper(fpath, label_key='labels'):
    """Internal utility for parsing CIFAR data.

    # Arguments
        fpath: path the file to parse.
        label_key: key for label data in the retrieve
            dictionary.

    # Returns
        A tuple `(data, labels)`.
    """
    with open(fpath, 'rb') as f:
        if sys.version_info < (3,):
            d = cPickle.load(f)
        else:
            d = cPickle.load(f, encoding='bytes')
            # decode utf8
            d_decoded = {}
            for k, v in d.items():
                d_decoded[k.decode('utf8')] = v
            d = d_decoded
    data = d['data']
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels


def keras_load_data_cifar(path):
    """ Loads CIFAR10 dataset.
        ** this is based on load_data() from keras framework
        As we don't have internet connection on our HPC we should save a copy of the dataset under:
        ~/.keras/datasets/cifar-10-batches-py/

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    # dirname = 'cifar-10-batches-py'
    # origin = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    # path = keras.utils.data_utils.get_file(dirname, origin=origin, untar=True)

    num_train_samples = 50000

    x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.empty((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        (x_train[(i - 1) * 10000: i * 10000, :, :, :],
         y_train[(i - 1) * 10000: i * 10000]) = load_batch_helper(fpath)

    fpath = os.path.join(path, 'test_batch')
    x_test, y_test = load_batch_helper(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    return (x_train, y_train), (x_test, y_test)


def keras_prepare_dataset(path, dataset="cifar10", num_classes=10, debug=False, load_subset=False, subset_size=4096):

    if dataset == "cifar10":
        (x_train, y_train), (x_test, y_test) = keras_load_data_cifar(path)
        channels = 3
        img_rows = 32
        img_cols = 32
    else:
        sys.exit('The current implementation support cifar10 dataset only!')

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], channels, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], channels, img_rows, img_cols)
        input_shape = (channels, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)
        input_shape = (img_rows, img_cols, channels)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    if load_subset:
        x_train = copy.deepcopy(x_train[range(0, subset_size), :, :, :])
        y_train = copy.deepcopy(y_train[range(0, subset_size)])

        x_test = copy.deepcopy(x_test[range(0, math.ceil(subset_size / 4)), :, :, :])
        y_test = copy.deepcopy(y_test[range(0, math.ceil(subset_size / 4))])

    if debug:
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')
        print("input_shape: ", input_shape)

    return (x_train, y_train), (x_test, y_test)


def keras_prepare_isic_dataset(path, dataset="isic", batch_size=32, test=False):
    datagen_train = ImageDataGenerator(rescale=1./255)
    datagen_val = ImageDataGenerator(rescale=1. / 255)

    # prepare an iterators for each dataset
    train_generator = datagen_train.flow_from_directory(os.path.join(path, 'training/'), target_size=(224, 224),
                                                 class_mode='categorical', batch_size=batch_size)
    validation_generator = datagen_val.flow_from_directory(os.path.join(path,'validation/'),target_size=(224, 224),
                                             class_mode='categorical', batch_size=batch_size)
    if test:
        datagen_tests = ImageDataGenerator(rescale=1. / 255)
        test_generator = datagen_tests.flow_from_directory(os.path.join(path, 'tests/'), target_size=(224, 224),
                                                           class_mode='categorical', batch_size=batch_size)

    # test_it = datagen.flow_from_directory('data/test/', class_mode='binary')

    # confirm the iterator works
    print("len(train_generator) = {}".format(len(train_generator)))

    if test:
        return train_generator, validation_generator, test_generator
    return train_generator, validation_generator


# plot diagnostic learning curves
def summarize_diagnostics(history):
    # plot loss
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='test')
    # plot accuracy
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
    # save plot to file
    filename = sys.argv[0].split('/')[-1]
    pyplot.savefig(filename + '_plot.png')
    pyplot.close()
    return
