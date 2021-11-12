from datetime import datetime
from tensorflow.keras.models import save_model
from model import *
from preprocess import *
from fileutils import *
from plotutils import *
import argparse
import matplotlib.pyplot as plt

EPOCHS = 500

def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_date", action='store', type=str, required=True)
    parser.add_argument("--stop_date", action='store', type=str, required=True)
    parser.add_argument("--cont", action='store_true')
    parser.add_argument("--n_channels", action='store', type=int, default=1)
    parser.add_argument("--loss_function", action='store', type=str, default='MeanSquaredError')
    parser.add_argument("--preprocess", action='store', type=str, default='img_size=128x128')
    parser.add_argument("--label", action='store', type=str)
    parser.add_argument("--include_datetime", action='store_true', default=False)
    parser.add_argument("--include_environment_data", action='store_true', default=False)
    parser.add_argument("--leadtime_conditioning", action='store', type=int, default=0)

    args = parser.parse_args()

    if args.label is not None:
        args.model, args.loss_function, args.n_channels, args.include_datetime, args.include_environment_data, args.leadtime_conditioning, args.preprocess = args.label.split('-')
        args.include_datetime = eval(args.include_datetime)
        args.include_environment_data = eval(args.include_environment_data)
        args.leadtime_conditioning = int(args.leadtime_conditioning)
        args.n_channels = int(args.n_channels)

    args.model = 'unet'

    args.start_date = datetime.datetime.strptime(args.start_date, '%Y-%m-%d')
    args.stop_date = datetime.datetime.strptime(args.stop_date, '%Y-%m-%d')

    return args


def with_generator(m, args):
    img_size = get_img_size(args.preprocess)

    if img_size[0] >= 384:
        batch_size = 3
    elif img_size[0] >= 256:
        batch_size = 8
    elif img_size[0] >= 128:
        batch_size = 16
    else:
        batch_size = 32

    train_gen, val_gen = create_generators(batch_size=batch_size, **vars(args))

    print("Number of train dataset elements: {}".format(len(train_gen.dataset)))
    print("Number of validation dataset elements: {}".format(len(val_gen.dataset)))

    hist = m.fit(train_gen, epochs = EPOCHS, validation_data = val_gen, callbacks=callbacks(args))

    return hist

def callbacks(args):
    cp_cb = tf.keras.callbacks.ModelCheckpoint(filepath='checkpoints/{}/cp.ckpt'.format(get_model_name(args)),
                                                 save_weights_only=True)
    early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_loss", patience=7, min_delta=0.001, verbose=1)
    reduce_lr_cb = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)

    return [cp_cb, early_stopping_cb, reduce_lr_cb]


def save_model_info(args, duration, hist, model_dir):
    with open('{}/info-{}.txt'.format(model_dir, datetime.datetime.now().strftime("%Y%m%dT%H%M%S")), 'w') as fp:
        fp.write(f'{args}\n')
        fp.write(f'duration: {duration}\n')
        fp.write(f'finished: {datetime.datetime.now()}\n')
    with open('{}/hist-{}.txt'.format(model_dir, datetime.datetime.now().strftime("%Y%m%dT%H%M%S")), 'w') as fp:
        fp.write(f'{hist}')

def run_model(args):
    model_dir = 'models/{}'.format(get_model_name(args))

    pretrained_weights = 'checkpoints/{}/cp.ckpt'.format(get_model_name(args)) if args.cont else None

    img_size = get_img_size(args.preprocess)
    n_channels = int(args.n_channels)

    if args.include_datetime:
        n_channels += 2
    if args.include_environment_data:
        n_channels += 2
    if args.leadtime_conditioning:
        n_channels += 1

    m = unet(pretrained_weights, input_size=img_size + (n_channels,), loss_function=args.loss_function, optimizer='SGD')

    start = datetime.datetime.now()

    hist = with_generator(m, args)

    duration = datetime.datetime.now() - start

    save_model(m, model_dir)
    save_model_info(args, duration, hist, model_dir)
    plot_hist(hist.history, model_dir)

    print(f"Model training finished in {duration}")


if __name__ == "__main__":
    args = parse_command_line()
    run_model(args)
