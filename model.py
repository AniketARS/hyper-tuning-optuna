from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense, \
    AveragePooling2D


def cnn_with_dense():
    model = Sequential()

    model.add(Conv2D(filters=44, kernel_size=(3, 3), padding='same', input_shape=(28, 28, 1), use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.18567221804314119))

    model.add(Conv2D(filters=56, kernel_size=(3, 3), padding='same', input_shape=(28, 28, 1), use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.05041324298614322))

    model.add(Flatten())

    model.add(Dense(units=212, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(rate=0.24997374242018003))

    model.add(Dense(units=78, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(rate=0.03918263915216241))

    model.add(Dense(units=10, use_bias=False))
    model.add(Activation('softmax'))

    return model

def full_cnn():
    model_full_cnn = Sequential()

    model_full_cnn.add(Conv2D(filters=53, kernel_size=(3, 3), padding='same', input_shape=(28, 28, 1), use_bias=False))
    model_full_cnn.add(BatchNormalization())
    model_full_cnn.add(Activation('relu'))
    model_full_cnn.add(MaxPool2D(pool_size=(2, 2)))
    model_full_cnn.add(Dropout(rate=0.24675468866323005))

    model_full_cnn.add(Conv2D(filters=50, kernel_size=(3, 3), padding='same', use_bias=False))
    model_full_cnn.add(BatchNormalization())
    model_full_cnn.add(Activation('relu'))
    model_full_cnn.add(MaxPool2D(pool_size=(2, 2)))
    model_full_cnn.add(Dropout(rate=0.24837435048210701))

    model_full_cnn.add(Conv2D(filters=108, kernel_size=(3, 3), padding='same'))
    model_full_cnn.add(Conv2D(filters=47, kernel_size=(3, 3), padding='same', use_bias=False))

    model_full_cnn.add(BatchNormalization())
    model_full_cnn.add(Activation('relu'))
    model_full_cnn.add(MaxPool2D(pool_size=(2, 2)))
    model_full_cnn.add(Dropout(rate=0.0257626138215777))

    model_full_cnn.add(Conv2D(filters=10, kernel_size=(1, 1), padding='same'))
    model_full_cnn.add(AveragePooling2D(pool_size=(3, 3)))
    model_full_cnn.add(Flatten())
    model_full_cnn.add(Activation('softmax'))

    return model_full_cnn
