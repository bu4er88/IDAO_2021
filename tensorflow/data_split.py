import pandas as pd
from data import *
from parameters import *

#data
names, images = get_names_and_images(ER, NR)
labels, energies = energy_parsing(names)

#datasplit
encoded_labels = list(pd.Series(labels).replace(label_encoder))
x_train, x_val, x_test, y1_train, y1_val, y1_test, y2_train, y2_val, y2_test = format_output(
    images, encoded_labels, energies
)