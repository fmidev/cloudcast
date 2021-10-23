from datetime import datetime
from tensorflow.keras.models import save_model
from model import *
from preprocess import *

IMG_SIZE = (256,256)
EPOCHS = 500

def with_full_dataset(m, callbacks):
    start_time = datetime.datetime.strptime('2021-05-01', '%Y-%m-%d')
    stop_time = datetime.datetime.strptime('2021-07-10', '%Y-%m-%d')

    ds = create_dataset(start_time, stop_time, img_size=IMG_SIZE, preprocess=True)
    (x_train, y_train, x_val, y_val) = create_train_val_split(ds, train_history_len=1)

    hist = m.fit(x_train, y_train, epochs=EPOCHS, batch_size=8, validation_data=(x_val, y_val), callbacks=callbacks)


def with_generator(m, callbacks):
    history_len = 1
    batch_size = 32

    start_time = datetime.datetime.strptime('2020-07-01', '%Y-%m-%d')
    stop_time = datetime.datetime.strptime('2021-09-01', '%Y-%m-%d')

    train_gen = EffectiveCloudinessGenerator(start_time, stop_time, history_len, batch_size=batch_size, img_size=IMG_SIZE)

    start_time = datetime.datetime.strptime('2021-09-01', '%Y-%m-%d')
    stop_time = datetime.datetime.strptime('2021-10-01', '%Y-%m-%d')

    val_gen = EffectiveCloudinessGenerator(start_time, stop_time, history_len, batch_size=batch_size, img_size=IMG_SIZE)

    hist = m.fit(train_gen, epochs = EPOCHS, validation_data = val_gen, callbacks=callbacks)


def callbacks():
    cp_cb = tf.keras.callbacks.ModelCheckpoint(filepath='checkpoints/unet_{}_{}x{}/cp.ckpt'.format(LOSS_FUNCTION, IMG_SIZE[0], IMG_SIZE[1]),
                                                 save_weights_only=True)
    early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
    reduce_lr_cb = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)

    return [cp_cb, early_stopping_cb, reduce_lr_cb]


m = unet(input_size=IMG_SIZE + (1,))

with_generator(m, callbacks())

save_model(m, 'models/unet_{}_{}x{}'.format(LOSS_FUNCTION, IMG_SIZE[0], IMG_SIZE[1]))
