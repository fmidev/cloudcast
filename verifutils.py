import numpy as np
from sklearn.metrics import mean_absolute_error, confusion_matrix
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt
from plotutils import *

CATEGORIES =  ['cloudy', 'partly-cloudy', 'clear']

def produce_scores(args, predictions):
    mae(predictions)
    categorical_scores(predictions)
    histogram(predictions)
    ssim(args, predictions)


def histogram(predictions):
    datas=[]
    labels=[]

    for l in predictions:
        datas.append(predictions[l]['data'])
        labels.append(l)

    datas = np.asarray(datas)
    plot_histogram(datas, labels)


def mae(predictions):
    ae, times = ae2d(predictions)
    plot_mae_per_leadtime(ae)
    plot_mae2d(ae, times)
    plot_mae_timeseries(ae, times)

# absolute error 2d
def ae2d(predictions):
    ret = {}
    gtt = predictions['gt']['time']
    times = []
    first = True

    for l in predictions:
        if l == 'gt':
            continue

        ret[l] = []

        for pred_times in predictions[l]['time']:
            pred_times = pred_times[1:] # skip analysis time

            a = gtt.index(pred_times[0])
            b = gtt.index(pred_times[-1])

            gt_data = np.asarray(predictions['gt']['data'][a:b+1])
            pred_data = np.asarray(predictions[l]['data'][0])
            pred_data = pred_data[1:]

            assert(gt_data.shape == pred_data.shape)

            mae = np.abs(gt_data - pred_data)
            ret[l].append(mae)

            if first:
                times.append(pred_times)

        ret[l] = np.asarray(ret[l])

        first = False

    return ret, times


def plot_mae_per_leadtime(ae):
    for l in ae:
        newae = np.moveaxis(ae[l], 0, 1)
        maelt = []
        for i,lt in enumerate(newae):
            maelt.append(np.mean(lt))
        maelt = np.asarray(maelt)

        plot_linegraph([maelt], [l], title='MAE over {} predictions'.format(ae[l].shape[0]), ylabel='mae')


def plot_mae2d(ae, times):
    # dimensions of mae2d:
    # (9, 4, 128, 128, 1)
    # (forecasts, leadtimes, x, y, channels)
    # merge 0, 1 so that all mae2d fields are merged to one dimension

    title = 'MAE between {}..{}'.format(times[0][0].strftime('%Y%m%dT%H%M'), times[-1][-1].strftime('%Y%m%dT%H%M'))

    for l in ae:
        # calculate average 2d field
        img_size = ae[l][0].shape[1:3]
        mae = np.average(ae[l].reshape((-1, img_size[0], img_size[1], 1)), axis=0)
        plot_on_map(np.squeeze(mae), title=title)


def plot_mae_timeseries(ae, times):

    def process_data(ae_timeseries):
        counts=[]
        x=[]
        y=[]
        for t in ae_timeseries.keys():
            counts.append(len(ae_timeseries[t]))
            y.append(np.average(ae_timeseries[t]))
            x.append(t)
        return x, y, counts

    def aggregate_to_max_hour(ae_timeseries):
        x, y, counts = process_data(ae_timeseries)
        mcounts=[]
        mx=[]
        my=[]
        for i,t in enumerate(x):
            if t.strftime("%M") != "00":
                continue
            mx.append(t)
            my.append(np.max(y[i-3:i]))
            mcounts.append(np.sum(counts[i-3:i]))
        return mx, my, mcounts
 
    ret={}
    for l in ae:
        ret[l] = {}

        for i,aes in enumerate(ae[l]):
            _times = times[i]
            for j, ae in enumerate(aes):
                t = _times[j]
                try:
                    ret[l][t].append(ae)
                except KeyError as e:
                    ret[l][t] = [ ae ]

        mx, my, mcounts = aggregate_to_max_hour(ret[l])

        xlabels = list(map(lambda x: x.strftime('%H:%M'), mx))

        plot_normal(mx, [my], mcounts, [l],title='MAE between {}..{}'.format(times[0][0].strftime('%Y%m%dT%H%M'), times[-1][-1].strftime('%Y%m%dT%H%M')), xlabels=xlabels)



def calculate_categorical_score(category, cm, score):

    def calc_score(TN, FP, FN, TP, score):
        if score == 'POD':
            return TP / (TP + FN)
        elif score == 'FAR':
            return FP / (TP + FP)
        elif score == 'CSI':
            SR = 1 - calc_score(TN, FP, FN, TP, 'FAR')
            return 1 / (1 / SR + 1 / calc_score(TN, FP, FN, TP, 'POD') - 1)

    idx = CATEGORIES.index(category)
    TP = cm[idx,idx]
    FN = np.sum(cm[idx,]) - TP
    FP = np.sum(cm[:,idx]) - TP
    TN = np.sum(cm) - np.sum(cm[idx,]) - np.sum(cm[:,idx])

    return calc_score(TN, FP, FN, TP, score)

def categorize(arr):
    # cloudy = 2 when cloudiness > 85%
    # partly = 1 when 15% >= cloudiness >= 85%
    # clear = 0  when cloudiness < 15%
    return np.digitize(arr, [0.15, 0.85])


def categorical_scores(predictions):
    gtd = predictions['gt']['data']
    gtt = predictions['gt']['time']

    gtd = np.asarray(gtd).copy()
    gtd = categorize(gtd)

    keys = list(predictions.keys())
    keys.remove('gt')
    def calc_confusion_matrix(predictions):
        ret = {}
        for l in predictions:
            if l == 'gt':
                continue

            ret[l] = {}

            preds=[]
            gts=[]
            for pred_times in predictions[l]['time']:
#            pred_times = predictions[l]['time'][0]
                a = gtt.index(pred_times[0])
                b = gtt.index(pred_times[-1])

                gt_data = gtd[a:b+1]

                pred_data = np.asarray(predictions[l]['data'][0]).copy()
                pred_data = categorize(pred_data)

                assert(gt_data.shape == pred_data.shape)

                preds.append(pred_data.flatten())
                gts.append(gt_data.flatten())

            preds = np.asarray(preds)
            gts = np.asarray(gts)

            ret[l] = np.flip(confusion_matrix(gts.flatten(), preds.flatten(), normalize='all')) # wikipedia syntax
        return ret

    cm = calc_confusion_matrix(predictions)

    for l in keys:
        catscores = []
        for c in CATEGORIES:
            p = calculate_categorical_score(c, cm[l], 'POD')
            f = calculate_categorical_score(c, cm[l], 'FAR')
            catscores.append((p, f))

        plot_performance_diagram(catscores, CATEGORIES, title='Performance diagram over {} predictions'.format(len(predictions[keys[0]]['data'])))


def ssim(args, predictions):

    ssims = []
    for l in predictions.keys():

        ssims.append([])

        if l == 'gt':
            gtd = predictions['gt']['data']
            start = 0
            stop = args.prediction_len
            while stop < len(gtd):
                gtdata = gtd[start:stop+1]
                i = 0
                ssims[-1].append([])
                for cur in gtdata[1:]:
                    prev = gtdata[i]
                    i += 1
                    ssims[-1][-1].append(structural_similarity(np.squeeze(prev), np.squeeze(cur), data_range=1.0))

                start += args.prediction_len
                stop += args.prediction_len
                ssims[-1][-1] = np.asarray(ssims[-1][-1])
            ssims[-1] = np.average(np.asarray(ssims[-1]), axis=0)
            continue


        for pred_data in predictions[l]['data']:
            i = 0
            ssims[-1].append([])
            for cur in pred_data[1:]:
                prev = pred_data[i]
                i += 1
                ssims[-1][-1].append(structural_similarity(np.squeeze(prev), np.squeeze(cur), data_range=1.0))

            ssims[-1][-1] = np.asarray(ssims[-1][-1])

        ssims[-1] = np.average(np.asarray(ssims[-1]), axis=0)

    plot_linegraph(ssims, list(predictions.keys()), title='Mean SSIM over {} predictions'.format(len(predictions[list(predictions.keys())[0]]['data'])), ylabel='ssim')

