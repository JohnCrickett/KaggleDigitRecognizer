import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

NUM_PIXELS = 784


def move_image_up(passed_image):
    changed_image = np.copy(passed_image)
    new_row = [1.0] * 28
    changed_image = np.vstack([changed_image[1:], new_row])
    return changed_image


def move_image_down(passed_image):
    changed_image = np.copy(passed_image)
    new_row = [1.0] * 28
    changed_image = np.vstack([new_row, changed_image[:-1]])
    return changed_image


def move_image_left(passed_image):
    changed_image = np.copy(passed_image)
    changed_image = np.hstack((changed_image[:, 1:],
                               np.zeros((changed_image.shape[0], 1))))
    return changed_image


def move_image_right(passed_image):
    changed_image = np.copy(passed_image)
    changed_image = np.hstack((np.zeros((changed_image.shape[0], 1)),
                               changed_image[:, :-1]))
    return changed_image


if __name__ == '__main__':
    # load the images
    train = pd.read_csv('./data/train.csv')
    labels = train.ix[:, 0].values.astype('int32')
    images = train.ix[:, 1:].values.astype('float32')

    #  perform each of the manipulations:
    #   move by one pixel (u/d/l/r)
    #   shear l/r - limited based on label (i.e. less for 1/7)
    #   rotate l/r - limited based on label (i.e. less for 1/7)
    #  store new images in the training data with labels
    # save results to a new training set

    # for each image:
    for i, image in enumerate(images[:2]):
        # set the properties as needed for manipulation
        image.shape = 28, 28
        plt.imshow(image, cmap='gray')
        print("label {label}".format(label=labels[i]))
        plt.show()
        # move by one pixel (u/d/l/r)
        # moved_image = move_image_up(image)
        # # TODO append new image to train, with correct label
        # moved_image = move_image_down(image)
        # # TODO append new image to train, with correct label
        # moved_image = move_image_left(image)
        # # TODO append new image to train, with correct label
        # moved_image = move_image_right(image)
        # TODO append new image to train, with correct label
        # TODO could these all be moved by two without invalidating the image?

        # plt.imshow(moved_image, cmap='gray')
        # plt.show()
