import numpy as np
from sklearn.metrics import mean_absolute_error, confusion_matrix


def recall(predictions):
    print(predictions.keys())
    gtd = predictions['gt']['data']
    gtt = predictions['gt']['time']

    def _cloudiness_category(arr, cloudiness):
        if cloudiness == 'cloudy':
            arr = np.where(arr < 0.8, 0, 1)
        elif cloudiness == 'partly-cloudy':
            arr = np.where((arr > 0.2) & (arr < 0.8), 1, 0)
        elif cloudiness == 'clear':
            arr = np.where(arr > 0.2, 0, 1)
        else:
            print("invalid category")
            sys.exit(1)

        return arr

    def cloudiness_category(arr):
        # cloudy = 2
        # partly = 1
        # clear = 0
        arr = np.digitize(arr, [0.2, 0.8])
        return arr

    gtd = np.asarray(gtd).copy()

    gtd = cloudiness_category(gtd)

    categories =  ['cloudy', 'partly-cloudy', 'clear']
    ret = {}
    for l in predictions:
        if l == 'gt':
            continue

        ret[l] = {}

        pred_times = predictions[l]['time'][0]
        a = gtt.index(pred_times[0])
        b = gtt.index(pred_times[-1])

        gt_data = gtd[a:b+1]

        pred_data = np.asarray(predictions[l]['data'][0]).copy()
        pred_data = cloudiness_category(pred_data)

        assert(gt_data.shape == pred_data.shape)
        ret[l] = np.flip(confusion_matrix(gt_data.flatten(), pred_data.flatten(), normalize='all')) # wikipedia syntax

    def calc_score(TN, FP, FN, TP, score):
        if score == 'POD':
            return TP / (TP + FN)
        elif score == 'FAR':
            return FP / (TP + FP)
        elif score == 'CSI':
            SR = 1 - calc_score(TN, FP, FN, TP, 'FAR')
            return 1 / (1 / SR + 1 / calc_score(TN, FP, FN, TP, 'POD') - 1)

    def calc_mc_score(category, cm, score):
        idx = categories.index(category)
        TP = cm[idx,idx]
        FN = np.sum(cm[idx,]) - TP
        FP = np.sum(cm[:,idx]) - TP
        TN = np.sum(cm) - np.sum(cm[idx,]) - np.sum(cm[:,idx])

        return calc_score(TN, FP, FN, TP, score)


    def print_score(categories, data, score):
        print('{:<45} {}'.format(score,'   '.join(categories)))
        for l in ret:
            outstr = '{:<45}'.format(l)
            for c in categories:
                val = calc_mc_score(c, ret[l], score)
                outstr += "    {:.3f}".format(val)

            print(outstr)

    print_score(categories, ret, 'POD')
    print_score(categories, ret, 'FAR')
    print_score(categories, ret, 'CSI')



