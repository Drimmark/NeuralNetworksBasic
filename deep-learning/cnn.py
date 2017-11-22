from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.losses import categorical_crossentropy
from keras.utils import to_categorical
from keras.optimizers import rmsprop

from keras.datasets import cifar10


class Cnn:
    num_classes = 10
    batch_size = 128
    # Image dimension
    img_rows, img_cols = 28, 28
    shape = (32, 32, 3)

    def __init__(self, n, hidden_layer, learning_rate, dropout_probability):
        self.n_iterations = n
        self.hidden_neurons = hidden_layer
        self.learning_rate = learning_rate
        self.dropout_probability = dropout_probability
        self.create_model()

    def create_model(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), padding='same',
                              input_shape=self.shape,
                              activation='relu'))
        self.model.add(Conv2D(32, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(self.dropout_probability))

        self.model.add(Flatten())
        self.model.add(Dense(self.hidden_neurons, activation='relu'))
        self.model.add(Dropout(self.dropout_probability))
        self.model.add(Dense(self.num_classes, activation='softmax'))

        self.model.compile(loss=categorical_crossentropy,
                           optimizer=rmsprop(lr=self.learning_rate, decay=1e-6),
                           metrics=['accuracy'])

    def train(self):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        y_train = to_categorical(y_train, self.num_classes)
        y_test = to_categorical(y_test, self.num_classes)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        datagen = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
        )
        datagen.fit(x_train)

        history = self.model.fit_generator(datagen.flow(x_train, y_train,
                                                        batch_size=self.batch_size),
                                           steps_per_epoch=x_train.shape[0] // self.batch_size,
                                           verbose=1,
                                           epochs=self.n_iterations,
                                           validation_data=(x_test, y_test))
        result = [(history.history.get('acc')[i], history.history.get('loss')[i],
                   history.history.get('val_acc')[i], history.history.get('val_loss')[i])
                  for i in range(len(history.history.get('acc')))]
        result_string = 'position,train_accuracy,train_loss,validation_accuracy,validation_loss'
        for position in range(len(result)):
            result_string += '\n{:d},{:1.5f},{:1.5f},{:1.5f},{:1.5f}'.format(
                position, result[position][0], result[position][1], result[position][2], result[position][3])

        return result_string
