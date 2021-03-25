import tensorflow as tf
from tensorflow.keras.layers import Input, ReLU, Conv2D, Lambda, MaxPooling2D, Flatten, Dense, BatchNormalization
from tensorflow.keras.models import Model
from parameters import IMSIZE


def one_shot_model():
    input = Input(shape=IMSIZE)
    norm = Lambda(lambda x: (x - tf.math.reduce_mean(x)) / tf.math.reduce_std(x))(input)
    x = Conv2D(8, 3, activation='relu')(norm)
    x = MaxPooling2D((3, 3))(x)
    x = Conv2D(16, 3, activation='relu')(x)
    x = MaxPooling2D((3, 3))(x)
    x = Conv2D(32, 3, activation='relu')(x)
    x = MaxPooling2D((3, 3))(x)
    x = Conv2D(32, 3, activation='relu')(x)
    x = MaxPooling2D((3, 3))(x)
    x = Flatten()(x)

    dense_1 = Dense(32, activation='relu')(x)  # to classifier

    dense_2 = Dense(512)(x)  # to regressor
    dense_2 = BatchNormalization()(dense_2)
    dense_2 = ReLU()(dense_2)
    dense_2 = Dense(512)(dense_2)
    dense_2 = BatchNormalization()(dense_2)
    dense_2 = ReLU()(dense_2)
    dense_2 = Dense(512)(dense_2)
    dense_2 = BatchNormalization()(dense_2)
    dense_2 = ReLU()(dense_2)
    dense_2 = Dense(512)(dense_2)
    dense_2 = BatchNormalization()(dense_2)
    dense_2 = ReLU()(dense_2)

    out_clf = Dense(1, activation='sigmoid', name='clf')(dense_1)
    out_reg = Dense(1, name='reg')(dense_2)
    model = Model(inputs=input, outputs=[out_clf, out_reg])

    return model
