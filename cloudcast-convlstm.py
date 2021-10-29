import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
import datetime
from tensorflow.keras.models import save_model # import datasets, models, layers
from model import *
from preprocess import *

TIMESERIES_LENGTH = 10

def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_date", action='store', type=str, required=True)
    parser.add_argument("--stop_date", action='store', type=str, required=True)
    parser.add_argument("--n_channels", action='store', type=int)
    parser.add_argument("--cont", action='store_true')
    parser.add_argument("--loss_function", action='store', type=str, default='binary_crossentropy')
    parser.add_argument("--preprocess", action='store', type=str, default='area=Scandinavia,conv=3,classes=100,img_size=128x128')
    parser.add_argument("--label", action='store', type=str)

    args = parser.parse_args()

    if args.label is not None:
        args.model, args.loss_function, args.n_channels, args.preprocess = args.label.split('-')

    args.start_date = datetime.datetime.strptime(args.start_date, '%Y-%m-%d')
    args.stop_date = datetime.datetime.strptime(args.stop_date, '%Y-%m-%d')

    if args.n_channels is None:
        n_channels = TIMESERIES_LENGTH

    return args


def show_examples():

    # Construct a figure on which we will visualize the images.
    fig, axes = plt.subplots(2, 3, figsize=(10, 8))

    # Plot each of the sequential images for one random data example.
    data_choice = np.random.choice(range(len(dataset)), size=1)[0]

    for idx, ax in enumerate(axes.flat):
        ax.imshow(np.squeeze(dataset[data_choice][idx]), cmap="gray")
        ax.set_title(f"Frame {idx + 1}")
        ax.axis("off")

    # Print information and display the figure.
    print(f"Displaying frames for example {data_choice}.")
    plt.show()


def fit(m, args):
    batch_size = 8

    train_gen, val_gen = create_generators(args.start_date, args.stop_date, preprocess=args.preprocess, batch_size=batch_size, output_is_timeseries=True)

    print("Number of train dataset elements: {}".format(len(train_gen.dataset)))
    print("Number of validation dataset elements: {}".format(len(val_gen.dataset)))

    cp_cb = tf.keras.callbacks.ModelCheckpoint(filepath='checkpoints/{}/cp.ckpt'.format(model_name(args)),
                                                 save_weights_only=True,
                                                 verbose=1)

    early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_loss", patience=7, min_delta=0.001)
    reduce_lr_cb = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)

#    hist = m.fit(x_train, y_train, epochs=20, batch_size=3, validation_data=(x_val, y_val), callbacks=[cp_callback, early_stopping_callback, reduce_lr_callback])

    hist = m.fit(train_gen, epochs = 500, validation_data = val_gen, callbacks=[cp_cb, early_stopping_cb, reduce_lr_cb])

    return hist


def run_model(args):
    model_dir = 'models/{}'.format(model_name(args))

    pretrained_weights = 'checkpoints/{}/cp.ckpt'.format(model_name(args)) if args.cont else None

    img_size = None
    for x in args.preprocess.split(','):
        k,v = x.split('=')
        if k == 'img_size':
            img_size = tuple(map(int, v.split('x')))
            break

    m = convlstm(pretrained_weights=pretrained_weights, input_size=img_size + (1,), loss_function=args.loss_function)

    start = datetime.datetime.now()

    hist = fit(m, args)

    duration = datetime.datetime.now() - start

    save_model(m, model_dir)
    plot_hist(hist, model_dir)

    print(f"Model training finished in {duration}")


if __name__ == "__main__":
    args = parse_command_line()

    run_model(args)
