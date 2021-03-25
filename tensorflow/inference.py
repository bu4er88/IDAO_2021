from parameters import *
from model import *
from data import get_images
import numpy as np


def predictions(model, x_test):
    # prediction of classes' probabilities and energies
    class_probabilities, energies = model.predict(np.array(x_test))
    # reshape energies
    energies = energies.reshape(-1)
    # transform probabilities to classes
    classes = []
    for prob in class_probabilities:
        '''This so because classes were confused'''
        if prob < 0.5:
            classes.append(1)
        else:
            classes.append(0)

    return classes, energies
