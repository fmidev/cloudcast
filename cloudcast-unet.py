from datetime import datetime
from tensorflow.keras.models import save_model
from model import *
from preprocess import *
import argparse
import matplotlib.pyplot as plt

EPOCHS = 500
N_CHANNELS = 1

def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_size", action='store', type=str, required=True)
    parser.add_argument("--start_date", action='store', type=str, required=True)
    parser.add_argument("--stop_date", action='store', type=str, required=True)

    args = parser.parse_args()

    args.img_size = tuple(map(int, args.img_size.split('x')))
    args.start_date = datetime.datetime.strptime(args.start_date, '%Y-%m-%d')
    args.stop_date = datetime.datetime.strptime(args.stop_date, '%Y-%m-%d')

    return args


def with_generator(m, args):
    batch_size = 64

    train_gen, val_gen = create_generators(args.start_date, args.stop_date, n_channels=N_CHANNELS, batch_size=batch_size, img_size=args.img_size)

    print("Number of train files: {}".format(len(train_gen.dataset)))
    print("Number of validation files: {}".format(len(val_gen.dataset)))

    hist = m.fit(train_gen, epochs = EPOCHS, validation_data = val_gen, callbacks=callbacks(args))

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
    model_dir = 'models/unet_{}_{}x{}_{}'.format(LOSS_FUNCTION, args.img_size[0], args.img_size[1], N_CHANNELS)

    m = unet(input_size=args.img_size + (1,))

    start = datetime.datetime.now()

    hist = with_generator(m, args)

    duration = datetime.datetime.now() - start

    save_model(m, model_dir)
    plot_hist(hist, model_dir)

    print(f"Model training finished in {duration}")


if __name__ == "__main__":
    args = parse_command_line()
    run_model(args)
