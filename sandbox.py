import configparser
import gc
import logging
import pathlib as path
import sys
from collections import defaultdict
from itertools import chain
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikitplot as skplt
import torch
from more_itertools import bucket

from idao.data_module import IDAODataModule
from idao.model import SimpleConv
from idao.utils import delong_roc_variance

def main(cfg):
    config = configparser.ConfigParser()
    config.read("./config.ini")

    PATH = path.Path(config["DATA"]["DatasetPath"])

    dataset_dm = IDAODataModule(
        data_dir=PATH, batch_size=int(config["TRAINING"]["BatchSize"]), cfg=config
    )
    dataset_dm.prepare_data()
    
    dataset_dm.setup()
    dataset_dm = dataset_dm.train_dataloader()
    di = iter(dataset_dm)
    data = di.next()
    x_target, class_target, reg_target, _ = data
    print(x_target.shape)
    img = x_target.view(x_target.shape[2], x_target.shape[3], x_target.shape[1])
    plt.imshow(img)
    plt.show()



if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("./config.ini")
    main(cfg=config)
