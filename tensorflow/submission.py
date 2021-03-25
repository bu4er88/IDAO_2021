import pandas as pd
from inference import *
from model import one_shot_model
import time

#####################################
tic = time.time()  # for timing check
#####################################

model = one_shot_model()
model.load_weights('C:/MAIN FOLDER/idao dataset/weights_oneshotmodel_idao_2.h5')

# get data
public_test_images, public_names = get_images(PUBLIC_TEST_PATH)
private_test_images, private_names = get_images(PRIVATE_TEST_PATH)
final_test_names = public_names + private_names
X_TEST = public_test_images + private_test_images

'''############# MAKE BATCHED DATASET ################

ЗДЕСЬ ДОЛЖЕН БЫТЬ ЗАГРУЗЧИК ДАННЫХ!!!!
Типа такого: (но это не работате конечно же)))
dataset = tf.data.Dataset(X_TEST)
batched_dataset = dataset.batch(4)

СЕЙЧАС ДАННЫЕ НАХОДЯТСЯ В X_TEST - это list() состоящий 
из np.array
Когда я запускаю в таком виде, он жрет память (ему надо >5GB)

###################################################'''

classes, energies = predictions(model, batched_dataset)

# make submission
submission_df = pd.DataFrame({
    'id': final_test_names,
    'classification_predictions': classes,
    'regression_predictions': energies
})

submission_df.to_csv('submission.csv', index=False)

#########################################################
toc = time.time()  # for timing check
print(f'Running time {toc - tic} s.')  # for timing check
#########################################################



