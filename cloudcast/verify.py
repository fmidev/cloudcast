from tensorflow.keras.models import load_model
from model import *
import glob
import numpy as np
import matplotlib as mpl
# save plots as fiels when running inside a screen instance
mpl.use('Agg')
import matplotlib.pyplot as plt
import argparse
import copy
from dateutil import parser as dateparser
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error
from base.fileutils import *
from base.preprocess import *
from base.plotutils import *
from base.generators import *
from base.verifutils import *
from base.opts import CloudCastOptions

PRED_STEP = timedelta(minutes=15)
DSS = {}

def parse_time(timestr):
    masks = ['%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S']
    for m in masks:
        try:
            return datetime.datetime.strptime(timestr, m)
        except ValueError as e:
            pass

    return None


def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_date", action='store', type=str, required=False)
    parser.add_argument("--stop_date", action='store', type=str, required=False)
    parser.add_argument("--single_time", action='store', type=str, required=False)
    parser.add_argument("--label", action='store', nargs='+', type=str, required=True)
    parser.add_argument("--prediction_len", action='store', type=int, default=12)
    parser.add_argument("--include_additional", action='store', nargs='+', default=[])
    parser.add_argument("--top", action='store', type=int, default=-1, help='out of all models select the top n that perform best')
    parser.add_argument("--plot_dir", action='store', type=str, default=None, help='save plots to directory of choice')

    args = parser.parse_args()

    if (args.start_date is None and args.stop_date is None) and args.single_time is None:
        print("One of: (start_date, stop_date), (single_time) must be given")
        sys.exit(1)

    if args.start_date is not None and args.stop_date is not None:
        args.start_date = parse_time(args.start_date)
        args.stop_date = parse_time(args.stop_date)
    else:
        args.start_date = parse_time(args.single_time)
        args.stop_date = args.start_date

    return args


def normalize_label(label):
    normalized_labels = []

    for lbl in args.label:
        if lbl.find('*') != -1:
            lbls = [os.path.basename(x) for x in glob.glob(f"models/{lbl}")]
            normalized_labels.extend(lbls)

        else:
            normalized_labels.append(lbl)

    return normalized_labels

def infer_many(m, orig, num_predictions, **kwargs):
    datetime_weights = kwargs.get('datetime_weights', None)
    topography_weights = kwargs.get('topography_weights', None)
    terrain_type_weights = kwargs.get('terrain_type_weights', None)
    leadtime_conditioning = kwargs.get('leadtime_conditioning', None)
    onehot_econding = kwargs.get('onehot_encoding', False)

    predictions = []
    hist_len = len(orig)
    orig_sq = np.squeeze(np.moveaxis(orig, 0, 3), -2)

    def create_hist(predictions):
        if len(predictions) == 0:
            return np.squeeze(np.moveaxis(orig, 0, 3), -2)
        elif len(predictions) >= hist_len:
             return np.squeeze(np.moveaxis(np.asarray(predictions[-hist_len:]), 0, 3), -2)

        hist_a = orig[:hist_len-len(predictions)]
        hist_b = np.asarray(predictions[-len(predictions):])

        seed = np.squeeze(np.moveaxis(np.concatenate((hist_a, hist_b), axis=0), 0, 3), -2)

        return seed

    def append_auxiliary_weights(data, datetime_weights, topography_weights, terrain_type_weights, num_prediction):
        if leadtime_conditioning is not None:
            if False: # or onehot_encoding:
                lts = np.squeeze(np.swapaxes(leadtime_conditioning[num_prediction], 0, -1))
                data = np.concatenate((data, lts), axis=-1)
            else:
                data = np.concatenate((data, leadtime_conditioning[num_prediction]), axis=-1)

        if datetime_weights is not None:
            data = np.concatenate((data, datetime_weights[hist_len-1][0], datetime_weights[hist_len-1][1]), axis=-1)

        if topography_weights is not None:
            data = np.concatenate((data, topography_weights), axis=-1)

        if terrain_type_weights is not None:
            data = np.concatenate((data, terrain_type_weights), axis=-1)

        return data

    data = orig_sq

    for i in range(num_predictions):

        if leadtime_conditioning is None:
            # autoregression
            data = create_hist(predictions)

        alldata = append_auxiliary_weights(data, datetime_weights, topography_weights, terrain_type_weights, i)
        pred = infer(m, alldata)
        predictions.append(pred)

    return np.asarray(predictions)



def infer(m, img):
    img = np.expand_dims(img, axis=0)
    pred = m.predict(img)
    pred = np.squeeze(pred, axis=0)

    return pred


def predict_from_series(m, dataseries, num):
    pred = m.predict(np.expand_dims(dataseries, axis=0))
    pred = np.squeeze(pred, axis=0)

    if pred.shape[0] == num:
        return pred

    moar = m.predict(np.expand_dims(pred, axis=0))
    moar = np.squeeze(moar, axis=0)

    comb = np.concatenate((pred, moar), axis=0)
    return comb[:num]


def predict_many(args, opts_list):
    all_pred = {}

    for opts in opts_list:
#        elem = copy.deepcopy(args)
#        elem.label = lbl

        predictions = predict(args, opts)

        for i,k in enumerate(predictions.keys()):
            if not k in all_pred.keys():
                all_pred[k] = predictions[k]

    return all_pred

def predict(args, opts):
    global DSS

    model_file = 'models/{}'.format(opts.get_label())
    print(f"Loading {model_file}")
    m = load_model(model_file, compile=False)

    try:
        dss = DSS[opts.preprocess]
    except KeyError as e:
        print("NEW dss for {}".format(opts.preprocess))
        DSS[opts.preprocess] = {}
        dss = DSS[opts.preprocess]

    time_gen = TimeseriesGenerator(args.start_date, args.stop_date, opts.n_channels, args.prediction_len, step=PRED_STEP)
    topography_weights = None
    terrain_type_weights = None

    nwp = []
    climatology = False

    for k in args.include_additional:
        if k == 'climatology':
            climatology=True
        else:
            nwp.append(k)

    if not 'nwcsaf' in dss:
        dss['nwcsaf'] = DataSeries("nwcsaf", opts.preprocess)

    predictions = {
        opts.get_label() : { 'time' : [], 'data' : [] },
        'gt' : { 'time' : [], 'data' : [] },
    }

    if climatology:
        predictions['climatology'] = { 'time' : [], 'data' : [] }

    for k in nwp:
        if not k in dss:
            dss[k] = DataSeries(k, opts.preprocess)
        predictions[k] = { 'time' : [], 'data' : [] }

    def diff(a, b):
        b = set(b)
        return [i for i in a if i not in b]

    for times in time_gen:
        history = times[:opts.n_channels]
        leadtimes = times[opts.n_channels:]

        print("{}: using history {} to predict {}".format(
            history[-1].strftime("%Y-%m-%d"),
            list(map(lambda x: '{}'.format(x.strftime('%H:%M')), history)),
            list(map(lambda x: '{}'.format(x.strftime('%H:%M')), leadtimes))
        ))

        datas = {}

        for k in dss:
            analysis_time = None
            _leadtimes = leadtimes.copy()
            _leadtimes = [times[opts.n_channels-1]] + leadtimes
            if k in ['meps','mnwc']:
                analysis_time = times[opts.n_channels].replace(minute=0)
                datas[k] = dss[k].read_data(_leadtimes, analysis_time)
            elif k == 'nwcsaf':
                datas[k] = dss[k].read_data(times)

        if climatology:
            clim = generate_clim_values((len(leadtimes),) + get_img_size(opts.preprocess), int(leadtimes[0].strftime("%m")))

        if np.isnan(datas['nwcsaf']).any():
            print("Seed contains missing values, skipping")
            continue

        new_times = diff(times, predictions['gt']['time'])

        for t in new_times:
            i = times.index(t)
            predictions['gt']['time'].append(t)
            predictions['gt']['data'].append(datas['nwcsaf'][i])

        datetime_weights = None
        lt = None

        if opts.include_datetime:
            datetime_weights = list(map(lambda x: create_datetime(x, get_img_size(opts.preprocess)), history))
        if opts.include_topography and topography_weights is None:
            topography_weights = create_topography_data(opts.preprocess, opts.model == 'convlstm')
        if opts.include_terrain_type and terrain_type_weights is None:
            terrain_type_weights = create_terrain_type_data(opts.preprocess, opts.model == 'convlstm')

        if opts.leadtime_conditioning:
            assert(args.prediction_len <= opts.leadtime_conditioning)
            lt = []
            for i in range(args.prediction_len):
                if opts.onehot_encoding is False:
                    lt.append(create_squeezed_leadtime_conditioning(get_img_size(opts.preprocess), opts.leadtime_conditioning, i))
                else:
                    lt.append(create_onehot_leadtime_conditioning(get_img_size(opts.preprocess), opts.leadtime_conditioning, i))

            if opts.onehot_encoding is False:
                lt = np.squeeze(np.asarray(lt), axis=1)

        if opts.model == "unet":
            cc = infer_many(m, datas['nwcsaf'][:opts.n_channels], args.prediction_len, datetime_weights=datetime_weights, topography_weights=topography_weights, terrain_type_weights=terrain_type_weights, leadtime_conditioning=lt, onehot_econding=opts.onehot_encoding)
        else:
            hist = datas['nwcsaf'][:opts.n_channels]

            if opts.include_topography:
                topo = np.tile(topography_weights, 6)
                topo = np.swapaxes(np.expand_dims(topo, axis=0), 0, 3)
                hist = np.concatenate((hist, topo), axis=-1)
                assert(np.max(topo) <= 1)

            if opts.include_terrain_type:
                terr = np.tile(terrain_type_weights, 6)
                terr = np.swapaxes(np.expand_dims(terr, axis=0), 0, 3)
                hist = np.concatenate((hist, terr), axis=-1)
                assert(np.max(terr) <= 1)

            cc = predict_from_series(m, hist, args.prediction_len)

        initial = np.expand_dims(np.copy(datas['nwcsaf'][opts.n_channels-1]), axis=0)
        cc = np.concatenate((initial, cc), axis=0)
        leadtimes = [history[-1]] + leadtimes

        assert(cc.shape[0] == len(leadtimes))
        predictions[opts.get_label()]['time'].append(leadtimes)
        predictions[opts.get_label()]['data'].append(cc)

        for k in nwp:
            predictions[k]['time'].append(leadtimes)
            predictions[k]['data'].append(datas[k])

        if climatology:
            predictions['climatology']['time'].append(leadtimes)
            predictions['climatology']['data'].append(clim)

    if len(predictions[opts.get_label()]['data']) == 0:
        print("Zero valid predictions for {}".format(opts.get_label()))
        sys.exit(1)

    assert(len(predictions[opts.get_label()]['data']) == len(predictions[opts.get_label()]['time']))
    return predictions



def copy_range(gt, start, stop):
    a = gt['time'].index(start)
    b = gt['time'].index(stop)
    return np.asarray(gt['data'][a:b+1]) # need inclusive end



def sort_errors(errors, best_first=True):
    assert(best_first)

    labels = list(errors.keys())

    maes = {}

    for l in labels:
        maes[l] = np.mean(errors[l])

     # sort best first
    maes = dict(sorted(maes.items(), key=lambda item: item[1]))

    return list(maes.keys())


def filter_top_n(predictions, errors, n, keep=[]):

    if n == -1 or len(predictions) < n:
        return predictions, errors

    labels = sort_errors(errors)

    print("Filtering predictions from {} to {}".format(len(predictions), n))

    labels = labels[:n]

    for k in keep:
        labels.reverse()
        if k not in labels:
            for i,j in enumerate(labels):
                if j not in keep:
                    labels[i] = k
                    break
        labels.reverse()

    f_predictions = {}
    f_errors = {}

    for l in labels:
        predkey = l
        if l == 'persistence':
            predkey = 'gt'
        f_predictions[predkey] = predictions[predkey]
        f_errors[l] = errors[l]

    return f_predictions, f_errors


def plot_timeseries(args, predictions):

    while True:
        first = list(predictions.keys())[np.random.randint(len(predictions))]
        if first != 'gt':
            break

    if len(predictions) < 8:
        labels = list(predictions.keys())
        try:
            labels.remove('climatology') # not plotting this in stamp plot
        except ValueError as e:
            pass

        labels.sort()
        idx = np.random.randint(len(predictions[first]['data']))

        data = []
        times = predictions[first]['time'][idx]

        for l in labels:
            if l == 'gt':
                data.append(copy_range(predictions['gt'], times[0], times[-1]))
                continue
            data.append(predictions[l]['data'][idx])

        plot_stamps(data, labels, title='Prediction for t0={}'.format(times[0]), initial_data=None, start_from_zero=True, plot_dir=args.plot_dir)
    else:
        print("Too many predictions ({}) for timeseries plot".format(len(predictions)))


def intersection(opts_list, predictions):
    # take intersection of predicted data so that we get same amount of forecasts
    # for same times
    # sometimes this might differ if for example history is missing and one model
    # needs more history than another

    labels = list(map(lambda x: x.get_label(), opts_list))
    def _intersection(lst1, lst2):
        lst3 = [value for value in lst1 if value in lst2]
        return lst3

    utimes = None

    for i in range(len(labels)-1):
        label = labels[i]
        next_label = labels[i+1]
        if utimes == None:
            utimes = _intersection(predictions[label]['time'], predictions[next_label]['time'])
        else:
            utimes = _intersection(utimes, predictions[next_label]['time'])

    ret = {}

    for label in labels:
        ret[label] = { 'time' : utimes, 'data' : [] }

        for utime in utimes:
            idx = predictions[label]['time'].index(utime)
            ret[label]['data'].append(predictions[label]['data'][idx])

    # copy other
    for label in predictions:
        if label not in labels:
            ret[label] = predictions[label]

    return ret


if __name__ == "__main__":
    args = parse_command_line()

    labels = normalize_label(args.label)
    opts_list = []
    for l in labels:
        opts_list.append(CloudCastOptions(label=l))
        assert(opts_list[-1].onehot_encoding is False)

    predictions = predict_many(args, opts_list)
    predictions = intersection(opts_list, predictions)

    DSS = None

#    predictions, errors = filter_top_n(predictions, errors, args.top, keep=['persistence'] + args.include_additional)

    plot_timeseries(args, predictions)
    produce_scores(args, predictions)

    if args.plot_dir is None:
        plt.show()
