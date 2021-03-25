import numpy as np
import PIL
from parameters import *
from sklearn.model_selection import train_test_split

#train val test data
def energy_parsing(filenames):
    labels = []
    energies = []
    for i in range(len(filenames)):
        split_name = filenames[i].split('/')[-1]
        try:
            energies.append(int(split_name.split('_')[6]))
        except:
            energies.append(int(split_name.split('_')[7]))
    for i in range(len(filenames)):
        split_name = filenames[i].split('/')[-1]
        if split_name.split('_')[5] == 'ER':
            labels.append(split_name.split('_')[5])
        else:
            labels.append(split_name.split('_')[6])
    print('Parsing complete!')
    return labels, energies

def get_names_and_images(ER_path, NR_path):
    names = []
    images = []
    for filename in os.listdir(ER_path):
        names.append(ER_path + '/' + filename)
        images.append(np.expand_dims(np.asarray(PIL.Image.open(ER + '/' + filename)), -1))
    for filename in os.listdir(NR_path):
        names.append(NR_path + '/' + filename)
        images.append(np.expand_dims(np.asarray(PIL.Image.open(NR + '/' + filename)), -1))
    return names, images

def format_output(x, y1, y2):
    x_train, x_REST, y1_train, y1_REST = train_test_split(
        x, y1, test_size=0.2, shuffle=True, random_state=17)
    x_val, x_test, y1_val, y1_test = train_test_split(
        x_REST, y1_REST, test_size=0.5, shuffle=False, random_state=17)

    y2_train, y2_REST = train_test_split(
        y2, test_size=0.2, shuffle=True, random_state=17)
    y2_val, y2_test = train_test_split(
        y2_REST, test_size=0.5, shuffle=False, random_state=17)

    return x_train, x_val, x_test, y1_train, y1_val, y1_test, y2_train, y2_val, y2_test


#global test data - PUBLIC AND PRIVATE TESTS
def get_images(path):
    image_list = []
    names = []
    for filename in os.listdir(path):
        names.append(filename[:-4])
        image_list.append(
            np.expand_dims(
                np.asarray(PIL.Image.open(path + '/' + filename)), -1)
        )
    return image_list, names


