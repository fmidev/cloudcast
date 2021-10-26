from datetime import datetime
from tensorflow.keras.models import save_model
from model import *
from preprocess import *
import argparse
import matplotlib.pyplot as plt

#IMG_SIZE = (256,256)
#IMG_SIZE = (128,128)
EPOCHS = 500
N_CHANNELS = 1

def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("img_size", type=int, nargs='+')

    args = parser.parse_args()

    return args

def with_full_dataset(m, args):
    start_time = datetime.datetime.strptime('2021-05-01', '%Y-%m-%d')
    stop_time = datetime.datetime.strptime('2021-07-10', '%Y-%m-%d')

    ds = create_dataset(start_time, stop_time, img_size=args.img_size, preprocess=True)
    (x_train, y_train, x_val, y_val) = create_train_val_split(ds, train_history_len=N_CHANNELS)

    hist = m.fit(x_train, y_train, epochs=EPOCHS, batch_size=8, validation_data=(x_val, y_val), callbacks=callbacks(args))

    return hist

def with_generator(m, args):
    batch_size = 16

    start_time = datetime.datetime.strptime('2020-04-01', '%Y-%m-%d')
    stop_time = datetime.datetime.strptime('2021-05-01', '%Y-%m-%d')

    train_gen = EffectiveCloudinessGenerator(start_time, stop_time, n_channels=N_CHANNELS, batch_size=batch_size, img_size=args.img_size)
    print("Number of train files: {}".format(len(train_gen.filenames)))

    start_time = datetime.datetime.strptime('2021-08-01', '%Y-%m-%d')
    stop_time = datetime.datetime.strptime('2021-10-01', '%Y-%m-%d')

    val_gen = EffectiveCloudinessGenerator(start_time, stop_time, n_channels=N_CHANNELS, batch_size=batch_size, img_size=args.img_size)
    print("Number of validation files: {}".format(len(val_gen.filenames)))

    hist = m.fit(train_gen, epochs = EPOCHS, validation_data = val_gen, callbacks=callbacks)

    return hist

def callbacks(args):
    cp_cb = tf.keras.callbacks.ModelCheckpoint(filepath='checkpoints/unet_{}_{}x{}/cp.ckpt'.format(LOSS_FUNCTION, args.img_size[0], args.img_size[1]),
                                                 save_weights_only=True)
    early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
    reduce_lr_cb = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)

    return [cp_cb, early_stopping_cb, reduce_lr_cb]


def plot_hist(hist, model_dir):
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('{}/accuracy.png'.format(model_dir))

    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('{}/loss.png'.format(model_dir))


def run_model(args):

    m = unet(input_size=args.img_size + (1,))

    start = datetime.datetime.now()

    hist = with_generator(m, callbacks())

    duration = datetime.datetime.now() - start

    model_dir = 'models/unet_{}_{}x{}_{}'.format(LOSS_FUNCTION, args.img_size[0], args.img_size[1], N_CHANNELS)
    save_model(m, model_dir)
    plot_hist(hist, model_dir)

    print(f"Model training finished in {duration}")


if __name__ == "__main__":
    args = parse_command_line()
    run_model(args)
