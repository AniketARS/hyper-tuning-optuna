import tensorflow.keras as keras
from model import full_cnn
from tensorflow.keras.datasets import mnist
import os


def save_model(model_to_save):
    if not os.path.exists('saved_model'):
        os.mkdir('saved_model')
    model_to_save.save(os.path.join('saved_model', 'model.h5'))

batch_size = 128
num_classes = 10
epochs = 10

img_rows, img_cols = 28, 28
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255.0
x_test = x_test / 255.0

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = full_cnn()

optimizer = keras.optimizers.Adam(learning_rate=0.0018628251888028735)
callbacks = [
    keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
    keras.callbacks.CSVLogger(os.path.join('assets', 'history.csv'))
 ]

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test),
          callbacks=callbacks)

save_model(model)
