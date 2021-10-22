import numpy as np
from datetime import datetime
from tensorflow.keras.models import save_model
from model import *
from preprocess import *

IMG_SIZE = (256,256)

start_time = datetime.datetime.strptime('2021-05-01', '%Y-%m-%d')
stop_time = datetime.datetime.strptime('2021-06-10', '%Y-%m-%d')
ds = create_dataset(start_time, stop_time, img_size=IMG_SIZE, preprocess=True)
(x_train, y_train, x_val, y_val) = create_train_val_split(ds, train_history_len=1)

m = unet(input_size=IMG_SIZE + (1,))

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='checkpoints/unet_{}_{}x{}/cp.ckpt'.format(LOSS_FUNCTION, IMG_SIZE[0], IMG_SIZE[1]),
                                                 save_weights_only=True,
                                                 verbose=1)
early_stopping_callback = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)

hist = m.fit(x_train, y_train, epochs=100, batch_size=8, validation_data=(x_val, y_val), callbacks=[cp_callback, early_stopping_callback, reduce_lr_callback])

#hist = m.fit(train_ds, epochs=30, batch_size=32, callbacks=[cp_callback])

save_model(m, 'models/unet_{}_{}x{}'.format(LOSS_FUNCTION, IMG_SIZE[0], IMG_SIZE[1]))
