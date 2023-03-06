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
    parser.add_argument("--stop_date", action="store", type=str)
    parser.add_argument("--cont", action="store_true")
    parser.add_argument("--n_channels", action="store", type=int, default=4)
    parser.add_argument(
        "--loss_function", action="store", type=str, default="MeanSquaredError"
    )
    parser.add_argument(
        "--preprocess", action="store", type=str, default="img_size=128x128"
    )
    parser.add_argument("--label", action="store", type=str)
    parser.add_argument("--include_datetime", action="store_true", default=False)
    parser.add_argument("--include_topography", action="store_true", default=False)
    parser.add_argument("--include_terrain_type", action="store_true", default=False)
    parser.add_argument("--leadtime_conditioning", action="store", type=int, default=12)
    parser.add_argument("--reuse_y_as_x", action="store_true", default=False)
    parser.add_argument(
        "--include_sun_elevation_angle", action="store_true", default=False
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--start_date", action="store", type=str)
    group.add_argument("--dataseries_file", action="store", type=str, default=None)
    group.add_argument("--dataseries_directory", action="store", type=str, default=None)

    args = parser.parse_args()

    if args.label is not None:
        opts = CloudCastOptions(label=args.label)
    else:
        vars_ = vars(args)
        vars_["model"] = "unet"
        opts = CloudCastOptions(**vars_)

    if (
        (not args.start_date and not args.stop_date)
        and not args.dataseries_file
        and not args.dataseries_directory
    ):
        print(
            "Either start_date,stop_date or dataseries_file or dataseries_directory needs to be defined"
        )
        sys.exit(1)

    if args.start_date and args.stop_date:
        args.start_date = datetime.datetime.strptime(args.start_date, "%Y-%m-%d")
        args.stop_date = datetime.datetime.strptime(args.stop_date, "%Y-%m-%d")

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

    lds = LazyDataSeries(
        img_size=img_size,
        batch_size=get_batch_size(img_size),
        training_mode=True,
        **vars(args),
    )

    # number of samples
    n = len(lds)
    # train-val split ratio
    r = 0.85
    # training dataset
    train_ds = lds.get_dataset(take_ratio=r)
    # validation dataset
    val_ds = lds.get_dataset(skip_ratio=r)
    # number of train data set steps (step = one batch)
    train_ds_steps = int((n * r) / lds.batch_size)
    # number of val data set steps
    val_ds_steps = int((n * (1 - r)) / lds.batch_size)

    print(
        "Total number of train dataset samples: {:d} number of steps: {:d} (batch_size: {:d})".format(
            int(n * r), train_ds_steps, lds.batch_size
        )
    )

    print(
        "Total number of validation samples: {:d} number of steps: {:d}".format(
            int(n * (1 - r)), val_ds_steps
        )
    )

    hist = m.fit(
        train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=callbacks(args, opts)
    )

    return hist


def callbacks(args, opts):
    cp_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath="checkpoints/{}/cp.ckpt".format(opts.get_label()),
        save_weights_only=True,
        save_best_only=True,
    )
    early_stopping_cb = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=7, min_delta=0.001, verbose=1
    )
    reduce_lr_cb = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)
    term_nan = keras.callbacks.TerminateOnNaN()

    return [cp_cb, early_stopping_cb, reduce_lr_cb, term_nan]


def save_model_info(args, opts, duration, hist, model_dir):
    with open(
        "{}/info-{}.json".format(
            model_dir, datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        ),
        "w",
    ) as fp:
        data = {
            "args": args,
            "opts": opts,
            "duration": duration,
            "finished": datetime.datetime.now(),
            "hostname": os.environ["HOSTNAME"],
        }

        json.dump(data, fp)

    with open(
        "{}/hist-{}.txt".format(
            model_dir, datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        ),
        "w",
    ) as fp:
        json.dump(hist, fp)


def run_model(args, opts):
    model_dir = "models/{}".format(opts.get_label())

    pretrained_weights = None
    if args.cont:
        pretrained_weights = "checkpoints/{}/cp.ckpt".format(opts.get_label())
        print("Reading old weights from '{}'".format(pretrained_weights))

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
    if opts.include_sun_elevation_angle:
        n_channels += 1

    m = unet(
        pretrained_weights,
        input_size=img_size + (n_channels,),
        loss_function=args.loss_function,
        optimizer="adam",
    )

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
