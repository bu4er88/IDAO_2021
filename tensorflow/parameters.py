import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
BATCH_SIZE = 20
IMSIZE = (576, 576, 1)
LEARNING_RATE = 3e-4
EPOCHS = 500

label_encoder = {'ER': 0, 'NR': 1}

DATA_PATH = 'c:/MAIN FOLDER/idao dataset/idao_dataset/'
TRAIN_PATH = os.path.join(DATA_PATH, 'train')
PRIVATE_TEST_PATH = os.path.join(DATA_PATH, 'private_test')
PUBLIC_TEST_PATH = os.path.join(DATA_PATH, 'public_test')
ER = os.path.join(TRAIN_PATH, 'ER')
NR = os.path.join(TRAIN_PATH, 'NR')
