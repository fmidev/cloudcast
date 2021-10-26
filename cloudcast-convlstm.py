import numpy as np
import glob
import matplotlib.pyplot as plt
import sys
import argparser

from tensorflow.keras.models import save_model # import datasets, models, layers
from PIL import Image
from model import *

TRAIN_SERIES_LENGTH = 10

START_DATE = '20201001T0000'
STOP_DATE = '20210331T2345'

PREPROCESS = True

def parse_command_line():
    parser = argparser.ArgumentParser()
    parser.add_argument("--img_size", action='store', type=str, required=True)

    args.img_size = tuple(map(int, args.img_size.split('x')))


    return args


def create_train_val_split(dataset):

    # Split into train and validation sets using indexing to optimize memory.
    indexes = np.arange(dataset.shape[0])
    np.random.shuffle(indexes)
    train_index = indexes[: int(0.9 * dataset.shape[0])]
    val_index = indexes[int(0.9 * dataset.shape[0]) :]
    train_dataset = dataset[train_index]
    val_dataset = dataset[val_index]

    # We'll define a helper function to shift the frames, where
    # `x` is frames 0 to n - 1, and `y` is frames 1 to n.
    def create_shifted_frames(data):
        x = data[:, 0 : data.shape[1] - 1, :, :]
        y = data[:, 1 : data.shape[1], :, :]
        return x, y


    # Apply the processing function to the datasets.
    x_train, y_train = create_shifted_frames(train_dataset)
    x_val, y_val = create_shifted_frames(val_dataset)

    # Inspect the dataset.
    print("Training Dataset Shapes: " + str(x_train.shape) + ", " + str(y_train.shape))
    print("Validation Dataset Shapes: " + str(x_val.shape) + ", " + str(y_val.shape))

    return (x_train, y_train, x_val, y_val)

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

args = parse_command_line()

dataset = create_dataset()

# show_examples(dataset)

x_train, y_train, x_val, y_val = create_train_val_split(dataset)

print("train dataset size: {str(x_train)}")

m = convlstm(input_size=args.img_size + (1,))

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='checkpoints/convlstm_{}_{}x{}_{}/cp.ckpt'.format(LOSS_FUNCTION, args.img_size[0], args.img_size[1], TRAIN_SERIES_LENGTH),
                                                 save_weights_only=True,
                                                 verbose=1)

# Define some callbacks to improve training.
early_stopping_callback = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)

hist = m.fit(x_train, y_train, epochs=20, batch_size=3, validation_data=(x_val, y_val), callbacks=[cp_callback, early_stopping_callback, reduce_lr_callback])

print(hist)

save_model(m, 'models/convlstm_{}_{}x{}_{}/'.format(LOSS_FUNCTION, args.img_size[0], args.img_size[1], TRAIN_SERIES_LENGTH))

