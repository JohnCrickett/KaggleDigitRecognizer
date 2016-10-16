from keras import backend as K
K.set_image_dim_ordering('th')
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization

import pandas as pd
import numpy as np

NUM_CLASSES = 10

def complex_model():
    model = Sequential()
    model.add(Convolution2D(30, 5, 5, border_mode='valid', input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(15, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(20, activation='relu'))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

if __name__ == '__main__':
    # fix random seed for reproducibility
    seed = 171717
    np.random.seed(seed)

    # read in the training and test data
    train = pd.read_csv('./data/train.csv')
    labels = train.ix[:, 0].values.astype('int32')
    X_train = (train.ix[:, 1:].values).astype('float32')
    X_test = (pd.read_csv('./data/test.csv').values).astype('float32')

    scale = np.max(X_train)
    X_train /= scale
    X_test /= scale
    mean = np.std(X_train)
    X_train -= mean
    X_test -= mean
    input_dim = X_train.shape[1]

    # one hot encode outputs
    y_train = np_utils.to_categorical(labels)

    model = complex_model()

    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

    # Fit the model
    #model.fit(X_train, y_train, batch_size=200, nb_epoch=20, verbose=2)
    model.fit(X_train, y_train, batch_size=1000, nb_epoch=30, verbose=2)

    predictions = model.predict_classes(X_test, verbose=1)
    result = pd.DataFrame({"ImageId": list(range(1, len(predictions) + 1)), "Label": predictions})

    result.to_csv("./predictions.csv", columns=('ImageId', 'Label'), index=None)




