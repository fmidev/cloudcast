import numpy as np
import sys
import argparse
import datetime
from tensorflow.keras.models import save_model
from model import *
from base.preprocess import *
from baselfileutils import *
from base.plotutils import *
from base.generators import *
from base.opts import CloudCastOptions


def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_date", action="store", type=str)
    parser.add_argument("--stop_date", action="store", type=str)
    parser.add_argument("--n_channels", action="store", type=int, default=12)
    parser.add_argument("--cont", action="store_true")
    parser.add_argument(
        "--loss_function", action="store", type=str, default="binary_crossentropy"
    )
    parser.add_argument(
        "--preprocess", action="store", type=str, default="img_size=128x128"
    )
    parser.add_argument("--label", action="store", type=str)
    parser.add_argument("--include_datetime", action="store_true", default=False)
    parser.add_argument("--include_topography", action="store_true", default=False)
    parser.add_argument("--include_terrain_type", action="store_true", default=False)
    parser.add_argument("--leadtime_conditioning", action="store_true", default=False)
    parser.add_argument("--dataseries_file", action="store", type=str, default="")

    args = parser.parse_args()

    if args.label is not None:
        opts = CloudCastOptions(label=args.label)
    else:
        vars_ = vars(args)
        vars_["model"] = "convlstm"
        opts = CloudCastOptions(**vars_)

    assert args.leadtime_conditioning == False
    assert args.include_datetime == False

    if (not args.start_date and not args.stop_date) and not args.dataseries_file:
        print("Either start_date,stop_date or dataseries_file needs to be defined")
        sys.exit(1)

    if args.start_date and args.stop_date:
        args.start_date = datetime.datetime.strptime(args.start_date, "%Y-%m-%d")
        args.stop_date = datetime.datetime.strptime(args.stop_date, "%Y-%m-%d")

    return args, opts


def fit(m, args, opts):
    batch_size = 1

    train_gen, val_gen = create_generators(
        batch_size=batch_size, opts=opts, output_is_timeseries=True, **vars(args)
    )

    print("Number of train dataset elements: {}".format(len(train_gen.dataset)))
    print("Number of validation dataset elements: {}".format(len(val_gen.dataset)))

    try:
        plot_timeseries([train_gen[0][0][0]], ["train_gen"], "Training data example")
    except Exception as e:
        print(f"Unable to show example data series: {e}")

    cp_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath="checkpoints/{}/cp.ckpt".format(opts.get_label()),
        save_weights_only=True,
        verbose=1,
    )

    early_stopping_cb = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=8, min_delta=0.001
    )
    reduce_lr_cb = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)

    hist = m.fit(
        train_gen,
        epochs=500,
        validation_data=val_gen,
        callbacks=[cp_cb, early_stopping_cb, reduce_lr_cb],
    )

    return hist


def save_model_info(args, opts, duration, hist, model_dir):
    with open(
        "{}/info-{}.txt".format(
            model_dir, datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        ),
        "w",
    ) as fp:
        fp.write(f"{args}\n")
        fp.write(f"{opts}\n")
        fp.write(f"duration: {duration}\n")
        fp.write(f"finished: {datetime.datetime.now()}\n")
        fp.write(f"hostname: {os.environ['HOSTNAME']}\n")

    with open(
        "{}/hist-{}.txt".format(
            model_dir, datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        ),
        "w",
    ) as fp:
        fp.write(f"{hist}")


def run_model(args, opts):
    model_dir = "models/{}".format(opts.get_label())

    pretrained_weights = (
        "checkpoints/{}/cp.ckpt".format(opts.get_label()) if args.cont else None
    )

    img_size = get_img_size(args.preprocess)
    n_channels = 1
    if args.include_topography_data:
        n_channels += 1
    if args.include_terrain_type_data:
        n_channels += 1

    m = convlstm(
        pretrained_weights=pretrained_weights,
        input_size=img_size + (n_channels,),
        loss_function=args.loss_function,
    )

    start = datetime.datetime.now()

    hist = fit(m, args, opts)

    duration = datetime.datetime.now() - start

    save_model(m, model_dir)
    save_model_info(args, opts, duration, hist.history, model_dir)
    plot_hist(hist.history, model_dir)

    print(f"Model training finished in {duration}")


if __name__ == "__main__":
    args, opts = parse_command_line()

    run_model(args, opts)
