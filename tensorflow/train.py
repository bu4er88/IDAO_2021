#from model import one_shot_model
#from data_split import *

model = one_shot_model()

history = model.fit(
    x_train, [y1_train, y2_train],
    validation_data=(x_val, [y1_val, y2_val]),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)

model.save_weights('weights_oneshotmodel.h5')
