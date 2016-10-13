from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout

import pandas as pd
import numpy as np

NUM_PIXELS = 784
NUM_CLASSES = 10

def baseline_model():
    model = Sequential()
    model.add(Dense(NUM_PIXELS, input_shape=(NUM_PIXELS,), init='normal', activation='relu'))
    model.add(Dense(NUM_CLASSES, init='normal', activation='softmax'))
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

    model = baseline_model()

    # Fit the model
    model.fit(X_train, y_train, batch_size=200, nb_epoch=20, verbose=2)

    predictions = model.predict_classes(X_test, verbose=1)
    result = pd.DataFrame({"ImageId": list(range(1, len(predictions) + 1)), "Label": predictions})

    result.to_csv("./predictions.csv", columns=('ImageId', 'Label'), index=None)




