from datetime import datetime
from tensorflow.keras.models import save_model
from model import *
from base.preprocess import *
from base.fileutils import *
from base.plotutils import *
from base.generators import *
from base.opts import CloudCastOptions
from base.dataseries import LazyDataSeries
import math
import argparse

EPOCHS = 500

def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_date", action='store', type=str)
    parser.add_argument("--stop_date", action='store', type=str)
    parser.add_argument("--cont", action='store_true')
    parser.add_argument("--n_channels", action='store', type=int, default=4)
    parser.add_argument("--loss_function", action='store', type=str, default='MeanSquaredError')
    parser.add_argument("--preprocess", action='store', type=str, default='img_size=128x128')
    parser.add_argument("--label", action='store', type=str)
    parser.add_argument("--include_datetime", action='store_true', default=False)
    parser.add_argument("--include_topography", action='store_true', default=False)
    parser.add_argument("--include_terrain_type", action='store_true', default=False)
    parser.add_argument("--leadtime_conditioning", action='store', type=int, default=12)
    parser.add_argument("--dataseries_file", action='store', type=str, default='')

    args = parser.parse_args()

    if args.label is not None:
        opts = CloudCastOptions(label=args.label)
    else:
        vars_ = vars(args)
        vars_['model'] = 'unet'
        opts = CloudCastOptions(**vars_)

    if (not args.start_date and not args.stop_date) and not args.dataseries_file:
        print("Either start_date,stop_date or dataseries_file needs to be defined")
        sys.exit(1)

    if args.start_date and args.stop_date:
        args.start_date = datetime.datetime.strptime(args.start_date, '%Y-%m-%d')
        args.stop_date = datetime.datetime.strptime(args.stop_date, '%Y-%m-%d')

    return args, opts


def get_batch_size(img_size):
    if img_size[0] >= 384:
        batch_size = 3
    elif img_size[0] >= 256:
        batch_size = 8
    elif img_size[0] >= 128:
        batch_size = 32
    else:
        batch_size = 64

    return batch_size


def with_dataset(m, args, opts):
    img_size = get_img_size(args.preprocess)

    lds = LazyDataSeries(img_size=img_size, batch_size=get_batch_size(img_size), **vars(args))

    n = len(lds)

    train_ds = lds.dataset.take(math.floor(n * 0.9))
    val_ds = lds.dataset.skip(math.floor(n * 0.9))

    print("Number of train dataset elements: {}".format(math.floor(n * 0.9)))
    print("Number of validation dataset elements: {}".format(math.floor(n * 0.1)))

    hist = m.fit(train_ds, epochs = EPOCHS, validation_data = val_ds, callbacks=callbacks(args, opts))


def with_generator(m, args, opts):
    img_size = get_img_size(args.preprocess)

    batch_size = get_batch_size(img_size)

    train_gen, val_gen = create_generators(batch_size=batch_size, opts=opts, **vars(args))

    print("Number of train dataset elements: {}".format(len(train_gen.dataset)))
    print("Number of validation dataset elements: {}".format(len(val_gen.dataset)))

    hist = m.fit(train_gen, epochs = EPOCHS, validation_data = val_gen, callbacks=callbacks(args, opts))

    return hist

def callbacks(args, opts):
    cp_cb = tf.keras.callbacks.ModelCheckpoint(filepath='checkpoints/{}/cp.ckpt'.format(opts.get_label()),
                                                 save_weights_only=True)
    early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_loss", patience=7, min_delta=0.001, verbose=1)
    reduce_lr_cb = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)

    return [cp_cb, early_stopping_cb, reduce_lr_cb]


def save_model_info(args, opts, duration, hist, model_dir):
    with open('{}/info-{}.txt'.format(model_dir, datetime.datetime.now().strftime("%Y%m%dT%H%M%S")), 'w') as fp:
        fp.write(f'{args}\n')
        fp.write(f'{opts}\n')
        fp.write(f'duration: {duration}\n')
        fp.write(f'finished: {datetime.datetime.now()}\n')
        fp.write(f"hostname: {os.environ['HOSTNAME']}\n")

    with open('{}/hist-{}.txt'.format(model_dir, datetime.datetime.now().strftime("%Y%m%dT%H%M%S")), 'w') as fp:
        fp.write(f'{hist}')

def run_model(args, opts):
    model_dir = 'models/{}'.format(opts.get_label())

    pretrained_weights = 'checkpoints/{}/cp.ckpt'.format(opts.get_label()) if args.cont else None

    img_size = get_img_size(opts.preprocess)
    n_channels = int(opts.n_channels)

    if opts.include_datetime:
        n_channels += 2
    if opts.include_topography:
        n_channels += 1
    if opts.include_terrain_type:
        n_channels += 1
    if opts.leadtime_conditioning:
        if opts.onehot_encoding:
            n_channels += leadtime_conditioning
        else:
            n_channels += 1

    m = unet(pretrained_weights, input_size=img_size + (n_channels,), loss_function=args.loss_function, optimizer='adam')

    start = datetime.datetime.now()

    hist = with_dataset(m, args, opts)

    duration = datetime.datetime.now() - start

    save_model(m, model_dir)
    save_model_info(args, opts, duration, hist.history, model_dir)
    plot_hist(hist.history, model_dir)

    print(f"Model training finished in {duration}")


if __name__ == "__main__":
    args, opts = parse_command_line()
    run_model(args, opts)
