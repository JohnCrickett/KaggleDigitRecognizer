import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

NUM_PIXELS = 784

train = pd.read_csv('./data/train.csv')
labels = train.ix[:, 0].values.astype('int32')
images = (train.ix[:, 1:].values).astype('float32')

an_image = images[1]
an_image.shape = 28,28
print(an_image)
plt.imshow(an_image, cmap = 'gray')
plt.show()
