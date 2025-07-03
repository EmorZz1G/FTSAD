from sklearn import metrics
import numpy as np
import math
import copy
from itertools import groupby
from operator import itemgetter

try:
    from eTaPR_pkg.tapr import print_result, compute
    from eTaPR_pkg.etapr import evaluate_w_ranges
    from eTaPR_pkg.DataManage.File_IO import load_list
    from pate.PATE_metric import PATE
    from affiliation.generics import convert_vector_to_events
    from affiliation.metrics import pr_from_events
except Exception:
    import os
    import sys
    fil_pth = os.path.dirname(os.path.abspath(__file__))
    # proj_dir = os.path.dirname(fil_pth)  # 获取上上级路径
    sys.path.append(fil_pth)  # 将项目根目录添加到系统路径中
    from eTaPR_pkg.tapr import print_result, compute
    from eTaPR_pkg.etapr import evaluate_w_ranges
    from eTaPR_pkg.DataManage.File_IO import load_list
    from pate.PATE_metric import PATE
    from affiliation.generics import convert_vector_to_events
    from affiliation.metrics import pr_from_events





def generate_curve(label, score, slidingWindow, version='opt', thre=250):
    if version =='opt_mem':
        tpr_3d, fpr_3d, prec_3d, window_3d, avg_auc_3d, avg_ap_3d = basic_metricor().RangeAUC_volume_opt_mem(labels_original=label, score=score, windowSize=slidingWindow, thre=thre)
    else:
        tpr_3d, fpr_3d, prec_3d, window_3d, avg_auc_3d, avg_ap_3d = basic_metricor().RangeAUC_volume_opt(labels_original=label, score=score, windowSize=slidingWindow, thre=thre)

    X = np.array(tpr_3d).reshape(1,-1).ravel()
    X_ap = np.array(tpr_3d)[:,:-1].reshape(1,-1).ravel()
    Y = np.array(fpr_3d).reshape(1,-1).ravel()
    W = np.array(prec_3d).reshape(1,-1).ravel()
    Z = np.repeat(window_3d, len(tpr_3d[0]))
    Z_ap = np.repeat(window_3d, len(tpr_3d[0])-1)

    return Y, Z, X, X_ap, W, Z_ap,avg_auc_3d, avg_ap_3d

def generate_curve_ROC(label, score, slidingWindow, version='opt', thre=250):
    if version =='opt_mem':
        # tpr_3d, fpr_3d, prec_3d, window_3d, avg_auc_3d, avg_ap_3d = basic_metricor().RangeAUC_volume_opt_mem(labels_original=label, score=score, windowSize=slidingWindow, thre=thre)
        raise NotImplementedError("opt_mem version is not implemented for ROC curve generation.")
    else:
        tpr_3d, fpr_3d, prec_3d, window_3d, avg_auc_3d, avg_ap_3d = basic_metricor().RangeAUC_volume_opt_ROC_only(labels_original=label, score=score, windowSize=slidingWindow, thre=thre)

    X = np.array(tpr_3d).reshape(1,-1).ravel()
    X_ap = np.array(tpr_3d)[:,:-1].reshape(1,-1).ravel()
    Y = np.array(fpr_3d).reshape(1,-1).ravel()
    W = np.array(prec_3d).reshape(1,-1).ravel()
    Z = np.repeat(window_3d, len(tpr_3d[0]))
    Z_ap = np.repeat(window_3d, len(tpr_3d[0])-1)

    return Y, Z, X, X_ap, W, Z_ap,avg_auc_3d, None

def generate_curve_PR(label, score, slidingWindow, version='opt', thre=250):
    if version =='opt_mem':
        # tpr_3d, fpr_3d, prec_3d, window_3d, avg_auc_3d, avg_ap_3d = basic_metricor().RangeAUC_volume_opt_mem(labels_original=label, score=score, windowSize=slidingWindow, thre=thre)
        raise NotImplementedError("opt_mem version is not implemented for PR curve generation.")
    else:
        tpr_3d, fpr_3d, prec_3d, window_3d, avg_auc_3d, avg_ap_3d = basic_metricor().RangeAUC_volume_opt_PR_only(labels_original=label, score=score, windowSize=slidingWindow, thre=thre)

    X = np.array(tpr_3d).reshape(1,-1).ravel()
    X_ap = np.array(tpr_3d)[:,:-1].reshape(1,-1).ravel()
    Y = np.array(fpr_3d).reshape(1,-1).ravel()
    W = np.array(prec_3d).reshape(1,-1).ravel()
    Z = np.repeat(window_3d, len(tpr_3d[0]))
    Z_ap = np.repeat(window_3d, len(tpr_3d[0])-1)

    return Y, Z, X, X_ap, W, Z_ap,None, avg_ap_3d

import math,torch

class MyAffBiasCache:
    def __init__(self) -> None:
        self.cache = {}

    def get_label_hash(self, label):
        if isinstance(label, np.ndarray):
            label = label.reshape(-1).tolist()
        elif isinstance(label, torch.Tensor):
            label = label.reshape(-1).detach().cpu().numpy().tolist()

        label = tuple(label)  # Convert to tuple for caching
        label_hash = hash(label)
        return label_hash

    def cache_bias(self, label, bias):
        label_hash = self.get_label_hash(label)
        self.cache[label_hash] = bias

    def get_cached_bias(self, label):
        label_hash = self.get_label_hash(label)
        return self.cache.get(label_hash, None)
    


def convert_vector_to_events(vector = [0, 1, 1, 0, 0, 1, 0]):
    """
    Convert a binary vector (indicating 1 for the anomalous instances)
    to a list of events. The events are considered as durations,
    i.e. setting 1 at index i corresponds to an anomalous interval [i, i+1).
    
    :param vector: a list of elements belonging to {0, 1}
    :return: a list of couples, each couple representing the start and stop of
    each event
    """
    positive_indexes = [idx for idx, val in enumerate(vector) if val > 0]
    events = []
    for k, g in groupby(enumerate(positive_indexes), lambda ix : ix[0] - ix[1]):
        cur_cut = list(map(itemgetter(1), g))
        events.append((cur_cut[0], cur_cut[-1]))
    
    # Consistent conversion in case of range anomalies (for indexes):
    # A positive index i is considered as the interval [i, i+1),
    # so the last index should be moved by 1
    events = [(x, y+1) for (x,y) in events]
        
    return (events)


    
class basic_metricor():
    def __init__(self, a = 1, probability = True, bias = 'flat', ):
        self.a = a
        self.probability = probability
        self.bias = bias
        self.eps = 1e-15
        self.Unbiased_Aff_prec_bias_cache = MyAffBiasCache()

    def get_pred(self, score, quantile=0.95):
        """
        Get the predicted labels based on the score and quantile.
        """
        assert 0 < quantile < 1, "Quantile must be between 0 and 1."
        
        thres = np.percentile(score, quantile*100)
        preds = score > thres
        return preds

    def cal_unbiased_aff_prec_bias(self, label, sample_cnt=3, verbose=1):
        if self.Unbiased_Aff_prec_bias_cache.get_cached_bias(label) is not None:
            if verbose:
                print("[Info]: Unbiased_Aff_prec_bias is already cached. Use the cached value.")
            return self.Unbiased_Aff_prec_bias_cache.get_cached_bias(label)
        else:
            precs_bias = []
            for i in range(sample_cnt):
                rnd_score = np.random.rand(len(label))
                thres = np.percentile(rnd_score, 98)
                preds = rnd_score > thres
                Affiliation_F, Aff_pre, Aff_rec = self.metric_Affiliation(label, rnd_score, preds=preds)
                precs_bias.append(Aff_pre)
            avg_bias = np.mean(precs_bias)
            if verbose:
                print(f"[Info]: Unbiased_Aff_prec_bias is set to {avg_bias:.4f} by sampling {sample_cnt} times.")
                out_str = ", ".join([f"{v:.4}" for i, v in enumerate(precs_bias)])
                print(f"[Info]: Unbiased_Aff_prec_bias sampling results: [{out_str}]")          
                    
            self.Unbiased_Aff_prec_bias_cache.cache_bias(label, avg_bias)
        return avg_bias

    def _metric_PA_percentile_K_k(self, labels, score, preds=None, k = 0.1):
        if preds is None:
            thresholds = np.linspace(score.min(), score.max(), 100)
            PointF1PA_scores = []

            for threshold in thresholds:
                preds = (score > threshold).astype(int)

                adjust_preds = self._adjust_predicts_k(score, labels, pred=preds, k=k)
                PointF1PA = metrics.f1_score(labels, adjust_preds)

                PointF1PA_scores.append(PointF1PA)

            PointF1PA_Threshold = thresholds[np.argmax(PointF1PA_scores)]
            PointF1PA1 = max(PointF1PA_scores)
            preds = score > PointF1PA_Threshold
            adjust_preds = self._adjust_predicts_k(score, labels, pred=preds, k=k)
            adjust_preds = adjust_preds.astype(int)
            precision, recall, f_score, s = metrics.precision_recall_fscore_support(labels, adjust_preds, average='binary')
            assert f_score==PointF1PA1
            Pre = precision
            Rec = recall
        else:
            adjust_preds = self._adjust_predicts_k(score, labels, pred=preds, k=k)
            PointF1PA1 = metrics.f1_score(labels, adjust_preds)

            precision, recall, f_score, s = metrics.precision_recall_fscore_support(labels, adjust_preds, average='binary')
            assert f_score==PointF1PA1
            Pre = precision
            Rec = recall

        F1_Per_K = PointF1PA1
        return F1_Per_K, Pre, Rec

    def metric_PA_percentile_K(self, labels, score, preds=None, num_K=100):
        if preds is None:
            print("[Warning]: If the preds is None, the calculation will be time-consuming!!! We will calculate the threshold for each K percentile. Please set the preds manually if you want to speed up the calculation.")
        thresholds = np.linspace(0, 1, num_K)
        F1_Per_Ks = []
        Prec_Per_Ks = []
        Rec_Per_Ks = []
        for k in thresholds:
            F1_Per_K, Pre, Rec = self._metric_PA_percentile_K_k(labels, score, preds=preds, k=k)
            F1_Per_Ks.append(F1_Per_K)
            Prec_Per_Ks.append(Pre)
            Rec_Per_Ks.append(Rec)
        F1_Per_K = np.mean(F1_Per_Ks)
        Prec_Per_K = np.mean(Prec_Per_Ks)
        Rec_Per_K = np.mean(Rec_Per_Ks)
        return F1_Per_K, Prec_Per_K, Rec_Per_K
        

    def metric_PATE(self, labels, score, e_buffer=100, d_buffer=100, n_jobs=100, num_desired_thresholds=250):
        result = PATE(labels, score, e_buffer=e_buffer, d_buffer=d_buffer, n_jobs=n_jobs, num_desired_thresholds= num_desired_thresholds)
        return result

    def _metric_eTaPR_F1_pred(self, labels, preds, theta_p=0.5, theta_r=0.1, delta=0.0):
        anomalies = load_list(labels)
        predictions = load_list(preds)
        result = evaluate_w_ranges(anomalies, predictions, theta_p, theta_r, delta)
        eTaP = result['eTaP']
        eTaR = result['eTaR']
        eTaF1 = result['f1']
        return eTaP, eTaR, eTaF1

    def _metric_TaPR_F1_pred(self, labels, preds, alpha=0.6, theta=0.8, delta=600):
        anomalies = load_list(labels)
        predictions = load_list(preds)
        result = compute(anomalies, predictions, alpha, theta, delta)
        TaP = result['TaP']
        TaR = result['TaR']
        TaF1 = result['f1']
        return TaP, TaR, TaF1
    
    def metric_eTaPR_F1(self, labels, score=None, preds=None, theta_p=0.5, theta_r=0.1, delta=0.0):
        if score is None and preds is None:
            raise ValueError("Either score or preds must be provided.")
        if preds is None:
            thresholds = np.linspace(score.min(), score.max(), 100)
            eTaPR_F1_scores = []

            for threshold in thresholds:
                preds = (score > threshold).astype(int)
                eTaP, eTaR, eTaF1 = self._metric_eTaPR_F1_pred(labels, preds, theta_p=theta_p, theta_r=theta_r, delta=delta)
                eTaPR_F1_scores.append(eTaF1)

            eTaPR_F1_Threshold = thresholds[np.argmax(eTaPR_F1_scores)]
            eTaPR_F1 = max(eTaPR_F1_scores)
            preds = score > eTaPR_F1_Threshold
            eTaP, eTaR, eTaF1 = self._metric_eTaPR_F1_pred(labels, preds, theta_p=theta_p, theta_r=theta_r, delta=delta)

        else:
            eTaP, eTaR, eTaF1 = self._metric_eTaPR_F1_pred(labels, preds, theta_p=theta_p, theta_r=theta_r, delta=delta)

        return  eTaF1, eTaP, eTaR

    def metric_TaPR_F1(self, labels, score=None, preds=None, alpha=0.6, theta=0.8, delta=600):
        if score is None and preds is None:
            raise ValueError("Either score or preds must be provided.")
        if preds is None:
            thresholds = np.linspace(score.min(), score.max(), 100)
            TaPR_F1_scores = []

            for threshold in thresholds:
                preds = (score > threshold).astype(int)
                TaP, TaR, TaF1 = self._metric_TaPR_F1_pred(labels, preds, alpha=alpha, theta=theta, delta=delta)
                TaPR_F1_scores.append(TaF1)

            TaPR_F1_Threshold = thresholds[np.argmax(TaPR_F1_scores)]
            TaPR_F1 = max(TaPR_F1_scores)
            preds = score > TaPR_F1_Threshold
            TaP, TaR, TaF1 = self._metric_TaPR_F1_pred(labels, preds, alpha=alpha, theta=theta, delta=delta)
        else:
            TaP, TaR, TaF1 = self._metric_TaPR_F1_pred(labels, preds, alpha=alpha, theta=theta, delta=delta)
        return TaF1, TaP, TaR

    def _adjust_length(self, labels, preds, base=3):
        anom_ranges = self.range_convers_new(labels)
        new_label = []
        new_preds = []
        for start, end in anom_ranges:
            new_len = end - start
            new_len = math.floor(math.log(new_len+base, base))
            new_label.extend([1] * new_len)
            new_preds.extend([1] * new_len if preds[start] else [0] * new_len)
        norm_tmp = labels.copy()
        norm_tmp[norm_tmp == 0] = -1
        norm_tmp[norm_tmp == 1] = 0
        norm_tmp[norm_tmp == -1] = 1
        norm_ranges = self.range_convers_new(norm_tmp)
        norm_ranges = convert_vector_to_events(norm_tmp)
        for start, end in norm_ranges:
            lenz = end - start
            new_label.extend([0] * lenz)
            new_preds.extend(preds[start:end].tolist())

        return np.array(new_label), np.array(new_preds)

    def metric_Reduced_F1(self, labels, score, preds=None):
        if preds is None:
            thresholds = np.linspace(score.min(), score.max(), 100)
            PointF1PA_scores = []

            for threshold in thresholds:
                preds = (score > threshold).astype(int)
                adjust_preds = self._adjust_predicts(score, labels, pred=preds)
                PointF1PA = metrics.f1_score(labels, adjust_preds)
                PointF1PA_scores.append(PointF1PA)

            PointF1PA_Threshold = thresholds[np.argmax(PointF1PA_scores)]
            # PointF1PA1 = max(PointF1PA_scores)
            preds = score > PointF1PA_Threshold
            adjust_preds = self._adjust_predicts(score, labels, pred=preds)
            adjust_preds = adjust_preds.astype(int)
            score, adjust_preds = self._adjust_length(labels, adjust_preds)
            precision, recall, f_score, s = metrics.precision_recall_fscore_support(labels, adjust_preds, average='binary')
            Reduced_F1 = f_score
            Pre = precision
            Rec = recall

        else:
            adjust_preds = self._adjust_predicts(score, labels, pred=preds)
            labels, adjust_preds = self._adjust_length(labels, adjust_preds)
            precision, recall, f_score, s = metrics.precision_recall_fscore_support(labels, adjust_preds, average='binary')
            Reduced_F1 = f_score
            Pre = precision
            Rec = recall

        return Reduced_F1, Pre, Rec

    
        
    def metric_UN_Affiliation(self, labels, score, pred=None):
        Affiliation_F, Aff_pre, Aff_rec = self.metric_Affiliation(labels, score, preds=pred)
        bias_aff_pre = self.Unbiased_Aff_prec_bias_cache.get_cached_bias(labels)
        if bias_aff_pre is None:
            print("[Info]: Unbiased_Aff_prec_bias is not set. Calculate by sample bias. The time-cost will be higher than setting it manually. You can set it by calling `cal_unbiased_aff_prec_bias` function.")
            bias_aff_pre = self.cal_unbiased_aff_prec_bias(labels)
        UAff_Pre = (Aff_pre - bias_aff_pre)/(1-bias_aff_pre)
        UAff_F1 = 2*abs(UAff_Pre)*Aff_rec/(abs(UAff_Pre)+Aff_rec)
        if UAff_Pre < 0:
            UAff_F1 *= -1

        NAff_Pre = (Aff_pre - 0.5)/(1-0.5)
        NAff_F1 = 2*abs(NAff_Pre)*Aff_rec/(abs(NAff_Pre)+Aff_rec)
        if NAff_Pre < 0:
            NAff_F1 *= -1

        return UAff_F1, NAff_F1

    def metric_U_Affiliation_f1_pre_rec(self, labels, score, pred=None):
        Affiliation_F, Aff_pre, Aff_rec = self.metric_Affiliation(labels, score, preds=pred)
        bias_aff_pre = self.Unbiased_Aff_prec_bias_cache.get_cached_bias(labels)
        if bias_aff_pre is None:
            print("[Info]: Unbiased_Aff_prec_bias is not set. Calculate by sample bias. The time-cost will be higher than setting it manually. You can set it by calling `cal_unbiased_aff_prec_bias` function.")
            bias_aff_pre = self.cal_unbiased_aff_prec_bias(labels)
        UAff_Pre = (Aff_pre - bias_aff_pre)/(1-bias_aff_pre)
        UAff_F1 = 2*abs(UAff_Pre)*Aff_rec/(abs(UAff_Pre)+Aff_rec)
        if UAff_Pre < 0:
            UAff_F1 *= -1

        return UAff_F1, UAff_Pre, Aff_rec
    
    def metric_N_Affiliation_f1_pre_rec(self, labels, score, pred=None):
        Affiliation_F, Aff_pre, Aff_rec = self.metric_Affiliation(labels, score, preds=pred)
        NAff_Pre = (Aff_pre - 0.5)/(1-0.5)
        NAff_F1 = 2*abs(NAff_Pre)*Aff_rec/(abs(NAff_Pre)+Aff_rec)
        if NAff_Pre < 0:
            NAff_F1 *= -1

        return NAff_F1, NAff_Pre, Aff_rec

    def detect_model(self, model, label, contamination = 0.1, window = 100, is_A = False, is_threshold = True):
        if is_threshold:
            score = self.scale_threshold(model.decision_scores_, model._mu, model._sigma)
        else:
            score = self.scale_contamination(model.decision_scores_, contamination = contamination)
        if is_A is False:
            scoreX = np.zeros(len(score)+window)
            scoreX[math.ceil(window/2): len(score)+window - math.floor(window/2)] = score
        else:
            scoreX = score

        self.score_=scoreX
        L = self.metric(label, scoreX)
        return L
    
    def metric_VUS_ROC_PR(self, labels, scores, slidingWindow=100, version='opt', thre=250):
        _, _, _, _, _, _,VUS_ROC, VUS_PR = generate_curve(labels.astype(int), scores, 2*slidingWindow, version, thre)
        return VUS_ROC, VUS_PR
    
    def metric_VUS_ROC(self, labels, scores, slidingWindow=100, version='opt', thre=250):
        _, _, _, _, _, _,VUS_ROC, VUS_PR = generate_curve_ROC(labels.astype(int), scores, 2*slidingWindow, version, thre)
        return VUS_ROC
    
    def metric_VUS_PR(self, labels, scores, slidingWindow=100, version='opt', thre=250):
        _, _, _, _, _, _,VUS_ROC, VUS_PR = generate_curve_PR(labels.astype(int), scores, 2*slidingWindow, version, thre)
        return VUS_PR

    def w(self, AnomalyRange, p):
        MyValue = 0
        MaxValue = 0
        start = AnomalyRange[0]
        AnomalyLength = AnomalyRange[1] - AnomalyRange[0] + 1
        for i in range(start, start +AnomalyLength):
            bi = self.b(i, AnomalyLength)
            MaxValue +=  bi
            if i in p:
                MyValue += bi
        return MyValue/MaxValue

    def Cardinality_factor(self, Anomolyrange, Prange):
        score = 0
        start = Anomolyrange[0]
        end = Anomolyrange[1]
        for i in Prange:
            if i[0] >= start and i[0] <= end:
                score +=1
            elif start >= i[0] and start <= i[1]:
                score += 1
            elif end >= i[0] and end <= i[1]:
                score += 1
            elif start >= i[0] and end <= i[1]:
                score += 1
        if score == 0:
            return 0
        else:
            return 1/score

    def b(self, i, length):
        bias = self.bias
        if bias == 'flat':
            return 1
        elif bias == 'front-end bias':
            return length - i + 1
        elif bias == 'back-end bias':
            return i
        else:
            if i <= length/2:
                return i
            else:
                return length - i + 1

    def scale_threshold(self, score, score_mu, score_sigma):
        return (score >= (score_mu + 3*score_sigma)).astype(int)

    def _adjust_predicts_k(self, score, label, threshold=None, pred=None, k=0.):
        if len(score) != len(label):
            raise ValueError("score and label must have the same length")
        score = np.asarray(score)
        label = np.asarray(label)
        if pred is None:
            predict = score > threshold
        else:
            predict = copy.deepcopy(pred)
        anom_ranges = self.range_convers_new(label)
        for st, ed in anom_ranges:
            pred_corr = sum(predict[st:ed])
            proportion = pred_corr / (ed - st)
            if proportion >= k:
                predict[st:ed] = 1
        return predict
        
        

    def _adjust_predicts(self, score, label, threshold=None, pred=None, calc_latency=False):
        """
        Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.

        Args:
            score (np.ndarray): The anomaly score
            label (np.ndarray): The ground-truth label
            threshold (float): The threshold of anomaly score.
                A point is labeled as "anomaly" if its score is higher than the threshold.
            pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
            calc_latency (bool):

        Returns:
            np.ndarray: predict labels
        """
        if len(score) != len(label):
            raise ValueError("score and label must have the same length")
        score = np.asarray(score)
        label = np.asarray(label)
        latency = 0
        if pred is None:
            predict = score > threshold
        else:
            predict = copy.deepcopy(pred)
        actual = label > 0.1
        anomaly_state = False
        anomaly_count = 0
        for i in range(len(score)):
            if actual[i] and predict[i] and not anomaly_state:
                    anomaly_state = True
                    anomaly_count += 1
                    for j in range(i, 0, -1):
                        if not actual[j]:
                            break
                        else:
                            if not predict[j]:
                                predict[j] = True
                                latency += 1
            elif not actual[i]:
                anomaly_state = False
            if anomaly_state:
                predict[i] = True
        if calc_latency:
            return predict, latency / (anomaly_count + 1e-4)
        else:
            return predict

    def metric_new(self, label, score, preds, plot_ROC=False, alpha=0.2):
        '''input:
               Real labels and anomaly score in prediction

           output:
               AUC,
               Precision,
               Recall,
               F-score,
               Range-precision,
               Range-recall,
               Range-Fscore,
               Precison@k,

            k is chosen to be # of outliers in real labels
        '''
        if np.sum(label) == 0:
            print('All labels are 0. Label must have groud truth value for calculating AUC score.')
            return None

        if np.isnan(score).any() or score is None:
            print('Score must not be none.')
            return None

        #area under curve
        auc = metrics.roc_auc_score(label, score)
        # plor ROC curve
        if plot_ROC:
            fpr, tpr, thresholds  = metrics.roc_curve(label, score)
            # display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc)
            # display.plot()

        #precision, recall, F
        if preds is None:
            preds = score > (np.mean(score)+3*np.std(score))
        Precision, Recall, F, Support = metrics.precision_recall_fscore_support(label, preds, zero_division=0)
        precision = Precision[1]
        recall = Recall[1]
        f = F[1]

        #point-adjust
        adjust_preds = self._adjust_predicts(score, label, pred=preds)
        PointF1PA = metrics.f1_score(label, adjust_preds)

        #range anomaly
        Rrecall, ExistenceReward, OverlapReward = self.range_recall_new(label, preds, alpha)
        Rprecision = self.range_recall_new(preds, label, 0)[0]

        if Rprecision + Rrecall==0:
            Rf=0
        else:
            Rf = 2 * Rrecall * Rprecision / (Rprecision + Rrecall)

        # top-k
        k = int(np.sum(label))
        threshold = np.percentile(score, 100 * (1-k/len(label)))

        # precision_at_k = metrics.top_k_accuracy_score(label, score, k)
        p_at_k = np.where(preds > threshold)[0]
        TP_at_k = sum(label[p_at_k])
        precision_at_k = TP_at_k/k

        L = [auc, precision, recall, f, PointF1PA, Rrecall, ExistenceReward, OverlapReward, Rprecision, Rf, precision_at_k]
        if plot_ROC:
            return L, fpr, tpr
        return L

    def metric_ROC(self, label, score):
        return metrics.roc_auc_score(label, score)

    def metric_PR(self, label, score):
        return metrics.average_precision_score(label, score)
    
    def metric_PointF1(self, label, score, preds=None):
        if preds is None:
            precision, recall, thresholds = metrics.precision_recall_curve(label, score)
            f1_scores = 2 * (precision * recall) / (precision + recall + 0.00001)
            F1 = np.max(f1_scores)
            threshold = thresholds[np.argmax(f1_scores)]
        else:
            Precision, Recall, F, Support = metrics.precision_recall_fscore_support(label, preds, zero_division=0)
            F1 = F[1]
        return F1


    def metric_Affiliation(self, label, score, preds=None):
        

        if preds is None:
            thresholds = np.linspace(score.min(), score.max(), 100)
            Affiliation_scores = []
            Affiliation_pre = []
            Affiliation_rec = []
            print("[Warning]: If the preds is None, the calculation will be time-consuming!!! We will calculate the threshold for each percentile. Please set the preds manually if you want to speed up the calculation.")

            for threshold in thresholds:
                preds = (score > threshold).astype(int)

                events_pred = convert_vector_to_events(preds)
                events_gt = convert_vector_to_events(label)
                Trange = (0, len(preds))
                affiliation_metrics = pr_from_events(events_pred, events_gt, Trange)
                Affiliation_Precision = affiliation_metrics['Affiliation_Precision']
                Affiliation_Recall = affiliation_metrics['Affiliation_Recall']
                Affiliation_F = 2*Affiliation_Precision*Affiliation_Recall / (Affiliation_Precision+Affiliation_Recall+self.eps)

                if Affiliation_F == np.nan or Affiliation_F == np.inf or Affiliation_Precision == 0 or Affiliation_Recall == np.nan\
                    or Affiliation_F ==0 or math.isnan(Affiliation_F) or math.isinf(Affiliation_F) or math.isnan(Affiliation_Precision) or math.isnan(Affiliation_Recall):
                    pass
                else:
                    Affiliation_scores.append(Affiliation_F)
                    Affiliation_pre.append(Affiliation_Precision)
                    Affiliation_rec.append(Affiliation_Recall)

            Affiliation_F1_Threshold = thresholds[np.argmax(Affiliation_scores)]
            Affiliation_F1 = Affiliation_scores[np.argmax(Affiliation_scores)]
            Affiliation_Precision = Affiliation_pre[np.argmax(Affiliation_scores)]
            Affiliation_Recall = Affiliation_rec[np.argmax(Affiliation_scores)]


        else:
            events_pred = convert_vector_to_events(preds)
            events_gt = convert_vector_to_events(label)
            Trange = (0, len(preds))
            affiliation_metrics = pr_from_events(events_pred, events_gt, Trange)
            Affiliation_Precision = affiliation_metrics['Affiliation_Precision']
            Affiliation_Recall = affiliation_metrics['Affiliation_Recall']
            Affiliation_F1 = 2*Affiliation_Precision*Affiliation_Recall / (Affiliation_Precision+Affiliation_Recall+self.eps)

        return Affiliation_F1, Affiliation_Precision, Affiliation_Recall


    # range-based F1
    def metric_RF1(self, label, score, preds=None):

        if preds is None:
            thresholds = np.linspace(score.min(), score.max(), 100)
            Rf1_scores = []

            for threshold in thresholds:
                preds = (score > threshold).astype(int)

                Rrecall, ExistenceReward, OverlapReward = self.range_recall_new(label, preds, alpha=0.2)
                Rprecision = self.range_recall_new(preds, label, 0)[0]
                if Rprecision + Rrecall==0:
                    Rf=0
                else:
                    Rf = 2 * Rrecall * Rprecision / (Rprecision + Rrecall)

                Rf1_scores.append(Rf)

            RF1_Threshold = thresholds[np.argmax(Rf1_scores)]
            RF1 = max(Rf1_scores)
        else:
            Rrecall, ExistenceReward, OverlapReward = self.range_recall_new(label, preds, alpha=0.2)
            Rprecision = self.range_recall_new(preds, label, 0)[0]
            if Rprecision + Rrecall==0:
                RF1=0
            else:
                RF1 = 2 * Rrecall * Rprecision / (Rprecision + Rrecall)
        return RF1

    # F1-PA
    def metric_PointF1PA(self, label, score, preds=None):
        if preds is None:
            thresholds = np.linspace(score.min(), score.max(), 100)
            PointF1PA_scores = []

            for threshold in thresholds:
                preds = (score > threshold).astype(int)

                adjust_preds = self._adjust_predicts(score, label, pred=preds)
                PointF1PA = metrics.f1_score(label, adjust_preds)

                PointF1PA_scores.append(PointF1PA)

            PointF1PA_Threshold = thresholds[np.argmax(PointF1PA_scores)]
            PointF1PA1 = max(PointF1PA_scores)
            preds = score > PointF1PA_Threshold
            adjust_preds = self._adjust_predicts(score, label, pred=preds)
            adjust_preds = adjust_preds.astype(int)
            # Acc = metrics.accuracy_score(label, adjust_preds)
            precision, recall, f_score, s = metrics.precision_recall_fscore_support(label, adjust_preds, average='binary')
            assert f_score==PointF1PA1
            Pre = precision
            Rec = recall


        else:
            adjust_preds = self._adjust_predicts(score, label, pred=preds)
            PointF1PA1 = metrics.f1_score(label, adjust_preds)

            # Acc = metrics.accuracy_score(label, adjust_preds)
            precision, recall, f_score, s = metrics.precision_recall_fscore_support(label, adjust_preds, average='binary')
            assert f_score==PointF1PA1
            Pre = precision
            Rec = recall

        return PointF1PA1, Pre, Rec

    def _get_events(self, y_test, outlier=1, normal=0):
        events = dict()
        label_prev = normal
        event = 0  # corresponds to no event
        event_start = 0
        for tim, label in enumerate(y_test):
            if label == outlier:
                if label_prev == normal:
                    event += 1
                    event_start = tim
            else:
                if label_prev == outlier:
                    event_end = tim - 1
                    events[event] = (event_start, event_end)
            label_prev = label

        if label_prev == outlier:
            event_end = tim - 1
            events[event] = (event_start, event_end)
        return events

    def metric_EventF1PA(self, label, score, preds=None):
        from sklearn.metrics import precision_score
        true_events = self._get_events(label)

        if preds is None:
            thresholds = np.linspace(score.min(), score.max(), 100)
            EventF1PA_scores = []

            for threshold in thresholds:
                preds = (score > threshold).astype(int)

                tp = np.sum([preds[start:end + 1].any() for start, end in true_events.values()])
                fn = len(true_events) - tp
                rec_e = tp/(tp + fn)
                prec_t = precision_score(label, preds)
                EventF1PA = 2 * rec_e * prec_t / (rec_e + prec_t + self.eps)

                EventF1PA_scores.append(EventF1PA)

            EventF1PA_Threshold = thresholds[np.argmax(EventF1PA_scores)]
            EventF1PA1 = max(EventF1PA_scores)

        else:

            tp = np.sum([preds[start:end + 1].any() for start, end in true_events.values()])
            fn = len(true_events) - tp
            rec_e = tp/(tp + fn)
            prec_t = precision_score(label, preds)
            EventF1PA1 = 2 * rec_e * prec_t / (rec_e + prec_t + self.eps)

        return EventF1PA1

    def range_recall_new(self, labels, preds, alpha):
        p = np.where(preds == 1)[0]    # positions of predicted label==1
        range_pred = self.range_convers_new(preds)
        range_label = self.range_convers_new(labels)

        Nr = len(range_label)    # total # of real anomaly segments

        ExistenceReward = self.existence_reward(range_label, preds)

        OverlapReward = 0
        for i in range_label:
            OverlapReward += self.w(i, p) * self.Cardinality_factor(i, range_pred)


        score = alpha * ExistenceReward + (1-alpha) * OverlapReward
        if Nr != 0:
            return score/Nr, ExistenceReward/Nr, OverlapReward/Nr
        else:
            return 0,0,0

    def range_convers_new(self, label):
        '''
        input: arrays of binary values
        output: list of ordered pair [[a0,b0], [a1,b1]... ] of the inputs
        '''
        # anomaly_starts = np.where(np.diff(label) == 1)[0] + 1
        # anomaly_ends, = np.where(np.diff(label) == -1)
        # if len(anomaly_ends):
        #     if not len(anomaly_starts) or anomaly_ends[0] < anomaly_starts[0]:
        #         # we started with an anomaly, so the start of the first anomaly is the start of the labels
        #         anomaly_starts = np.concatenate([[0], anomaly_starts])
        # if len(anomaly_starts):
        #     if not len(anomaly_ends) or anomaly_ends[-1] < anomaly_starts[-1]:
        #         # we ended on an anomaly, so the end of the last anomaly is the end of the labels
        #         anomaly_ends = np.concatenate([anomaly_ends, [len(label) - 1]])
        # return list(zip(anomaly_starts, anomaly_ends))

        # EmorZz1G 2024-6-25 fix-bug
        return convert_vector_to_events(label)

    def existence_reward(self, labels, preds):
        '''
        labels: list of ordered pair
        preds predicted data
        '''

        score = 0
        for i in labels:
            if preds[i[0]:i[1]+1].any():
                score += 1
        return score

    def num_nonzero_segments(self, x):
        count=0
        if x[0]>0:
            count+=1
        for i in range(1, len(x)):
            if x[i]>0 and x[i-1]==0:
                count+=1
        return count

    def extend_postive_range(self, x, window=5):
        label = x.copy().astype(float)
        L = self.range_convers_new(label)   # index of non-zero segments
        length = len(label)
        for k in range(len(L)):
            s = L[k][0]
            e = L[k][1]


            x1 = np.arange(e,min(e+window//2,length))
            label[x1] += np.sqrt(1 - (x1-e)/(window))

            x2 = np.arange(max(s-window//2,0),s)
            label[x2] += np.sqrt(1 - (s-x2)/(window))

        label = np.minimum(np.ones(length), label)
        return label

    def extend_postive_range_individual(self, x, percentage=0.2):
        label = x.copy().astype(float)
        L = self.range_convers_new(label)   # index of non-zero segments
        length = len(label)
        for k in range(len(L)):
            s = L[k][0]
            e = L[k][1]

            l0 = int((e-s+1)*percentage)

            x1 = np.arange(e,min(e+l0,length))
            label[x1] += np.sqrt(1 - (x1-e)/(2*l0))

            x2 = np.arange(max(s-l0,0),s)
            label[x2] += np.sqrt(1 - (s-x2)/(2*l0))

        label = np.minimum(np.ones(length), label)
        return label

    def TPR_FPR_RangeAUC(self, labels, pred, P, L):
        indices = np.where(labels == 1)[0]
        product = labels * pred
        TP = np.sum(product)
        newlabels = product.copy()
        newlabels[indices] = 1

        # recall = min(TP/P,1)
        P_new = (P + np.sum(newlabels)) / 2  # so TPR is neither large nor small
        # P_new = np.sum(labels)
        recall = min(TP / P_new, 1)
        # recall = TP/np.sum(labels)
        # print('recall '+str(recall))

        existence = 0
        for seg in L:
            if np.sum(product[seg[0]:(seg[1] + 1)]) > 0:  # if newlabels>0, that segment must contained
                existence += 1

        existence_ratio = existence / len(L)
        # print(existence_ratio)

        # TPR_RangeAUC = np.sqrt(recall*existence_ratio)
        # print(existence_ratio)
        TPR_RangeAUC = recall * existence_ratio

        FP = np.sum(pred) - TP
        # TN = np.sum((1-pred) * (1-labels))

        # FPR_RangeAUC = FP/(FP+TN)
        N_new = len(labels) - P_new
        FPR_RangeAUC = FP / N_new

        Precision_RangeAUC = TP / np.sum(pred)

        return TPR_RangeAUC, FPR_RangeAUC, Precision_RangeAUC

    def RangeAUC(self, labels, score, window=0, percentage=0, plot_ROC=False, AUC_type='window'):
        # AUC_type='window'/'percentage'
        score_sorted = -np.sort(-score)

        P = np.sum(labels)
        # print(np.sum(labels))
        if AUC_type == 'window':
            labels = self.extend_postive_range(labels, window=window)
        else:
            labels = self.extend_postive_range_individual(labels, percentage=percentage)

        # print(np.sum(labels))
        L = self.range_convers_new(labels)
        TPR_list = [0]
        FPR_list = [0]
        Precision_list = [1]

        for i in np.linspace(0, len(score) - 1, 250).astype(int):
            threshold = score_sorted[i]
            # print('thre='+str(threshold))
            pred = score >= threshold
            TPR, FPR, Precision = self.TPR_FPR_RangeAUC(labels, pred, P, L)

            TPR_list.append(TPR)
            FPR_list.append(FPR)
            Precision_list.append(Precision)

        TPR_list.append(1)
        FPR_list.append(1)  # otherwise, range-AUC will stop earlier than (1,1)

        tpr = np.array(TPR_list)
        fpr = np.array(FPR_list)
        prec = np.array(Precision_list)

        width = fpr[1:] - fpr[:-1]
        height = (tpr[1:] + tpr[:-1]) / 2
        AUC_range = np.sum(width * height)

        width_PR = tpr[1:-1] - tpr[:-2]
        height_PR = prec[1:]
        AP_range = np.sum(width_PR * height_PR)

        if plot_ROC:
            return AUC_range, AP_range, fpr, tpr, prec

        return AUC_range


    def new_sequence(self, label, sequence_original, window):
        a = max(sequence_original[0][0] - window // 2, 0)
        sequence_new = []
        for i in range(len(sequence_original) - 1):
            if sequence_original[i][1] + window // 2 < sequence_original[i + 1][0] - window // 2:
                sequence_new.append((a, sequence_original[i][1] + window // 2))
                a = sequence_original[i + 1][0] - window // 2
        sequence_new.append((a, min(sequence_original[len(sequence_original) - 1][1] + window // 2, len(label) - 1)))
        return sequence_new

    def sequencing(self, x, L, window=5):
        label = x.copy().astype(float)
        length = len(label)

        for k in range(len(L)):
            s = L[k][0]
            e = L[k][1]

            x1 = np.arange(e + 1, min(e + window // 2 + 1, length))
            label[x1] += np.sqrt(1 - (x1 - e) / (window))

            x2 = np.arange(max(s - window // 2, 0), s)
            label[x2] += np.sqrt(1 - (s - x2) / (window))

        label = np.minimum(np.ones(length), label)
        return label

    # TPR_FPR_window
    def RangeAUC_volume_opt(self, labels_original, score, windowSize, thre=250):
        window_3d = np.arange(0, windowSize + 1, 1)
        P = np.sum(labels_original)
        seq = self.range_convers_new(labels_original)
        l = self.new_sequence(labels_original, seq, windowSize)

        score_sorted = -np.sort(-score)

        tpr_3d = np.zeros((windowSize + 1, thre + 2))
        fpr_3d = np.zeros((windowSize + 1, thre + 2))
        prec_3d = np.zeros((windowSize + 1, thre + 1))

        auc_3d = np.zeros(windowSize + 1)
        ap_3d = np.zeros(windowSize + 1)

        tp = np.zeros(thre)
        N_pred = np.zeros(thre)

        for k, i in enumerate(np.linspace(0, len(score) - 1, thre).astype(int)):
            threshold = score_sorted[i]
            pred = score >= threshold
            N_pred[k] = np.sum(pred)

        for window in window_3d:

            labels_extended = self.sequencing(labels_original, seq, window)
            L = self.new_sequence(labels_extended, seq, window)

            TF_list = np.zeros((thre + 2, 2))
            Precision_list = np.ones(thre + 1)
            j = 0

            for i in np.linspace(0, len(score) - 1, thre).astype(int):
                threshold = score_sorted[i]
                pred = score >= threshold
                labels = labels_extended.copy()
                existence = 0

                for seg in L:
                    labels[seg[0]:seg[1] + 1] = labels_extended[seg[0]:seg[1] + 1] * pred[seg[0]:seg[1] + 1]
                    if (pred[seg[0]:(seg[1] + 1)] > 0).any():
                        existence += 1
                for seg in seq:
                    labels[seg[0]:seg[1] + 1] = 1

                TP = 0
                N_labels = 0
                for seg in l:
                    TP += np.dot(labels[seg[0]:seg[1] + 1], pred[seg[0]:seg[1] + 1])
                    N_labels += np.sum(labels[seg[0]:seg[1] + 1])

                TP += tp[j]
                FP = N_pred[j] - TP

                existence_ratio = existence / len(L)

                P_new = (P + N_labels) / 2
                recall = min(TP / P_new, 1)

                TPR = recall * existence_ratio
                N_new = len(labels) - P_new
                FPR = FP / N_new

                Precision = TP / N_pred[j]

                j += 1
                TF_list[j] = [TPR, FPR]
                Precision_list[j] = Precision

            TF_list[j + 1] = [1, 1]  # otherwise, range-AUC will stop earlier than (1,1)

            tpr_3d[window] = TF_list[:, 0]
            fpr_3d[window] = TF_list[:, 1]
            prec_3d[window] = Precision_list

            width = TF_list[1:, 1] - TF_list[:-1, 1]
            height = (TF_list[1:, 0] + TF_list[:-1, 0]) / 2
            AUC_range = np.dot(width, height)
            auc_3d[window] = (AUC_range)

            width_PR = TF_list[1:-1, 0] - TF_list[:-2, 0]
            height_PR = Precision_list[1:]

            AP_range = np.dot(width_PR, height_PR)
            ap_3d[window] = AP_range

        return tpr_3d, fpr_3d, prec_3d, window_3d, sum(auc_3d) / len(window_3d), sum(ap_3d) / len(window_3d)
    
    # TPR_FPR_window
    def RangeAUC_volume_opt_PR_only(self, labels_original, score, windowSize, thre=250):
        window_3d = np.arange(0, windowSize + 1, 1)
        P = np.sum(labels_original)
        seq = self.range_convers_new(labels_original)
        l = self.new_sequence(labels_original, seq, windowSize)

        score_sorted = -np.sort(-score)

        tpr_3d = np.zeros((windowSize + 1, thre + 2))
        fpr_3d = np.zeros((windowSize + 1, thre + 2))
        prec_3d = np.zeros((windowSize + 1, thre + 1))

        # auc_3d = np.zeros(windowSize + 1)
        ap_3d = np.zeros(windowSize + 1)

        tp = np.zeros(thre)
        N_pred = np.zeros(thre)

        for k, i in enumerate(np.linspace(0, len(score) - 1, thre).astype(int)):
            threshold = score_sorted[i]
            pred = score >= threshold
            N_pred[k] = np.sum(pred)

        for window in window_3d:

            labels_extended = self.sequencing(labels_original, seq, window)
            L = self.new_sequence(labels_extended, seq, window)

            TF_list = np.zeros((thre + 2, 2))
            Precision_list = np.ones(thre + 1)
            j = 0

            for i in np.linspace(0, len(score) - 1, thre).astype(int):
                threshold = score_sorted[i]
                pred = score >= threshold
                labels = labels_extended.copy()
                existence = 0

                for seg in L:
                    labels[seg[0]:seg[1] + 1] = labels_extended[seg[0]:seg[1] + 1] * pred[seg[0]:seg[1] + 1]
                    if (pred[seg[0]:(seg[1] + 1)] > 0).any():
                        existence += 1
                for seg in seq:
                    labels[seg[0]:seg[1] + 1] = 1

                TP = 0
                N_labels = 0
                for seg in l:
                    TP += np.dot(labels[seg[0]:seg[1] + 1], pred[seg[0]:seg[1] + 1])
                    N_labels += np.sum(labels[seg[0]:seg[1] + 1])

                TP += tp[j]
                FP = N_pred[j] - TP

                existence_ratio = existence / len(L)

                P_new = (P + N_labels) / 2
                recall = min(TP / P_new, 1)

                TPR = recall * existence_ratio
                N_new = len(labels) - P_new
                FPR = FP / N_new

                Precision = TP / N_pred[j]

                j += 1
                TF_list[j] = [TPR, FPR]
                Precision_list[j] = Precision

            TF_list[j + 1] = [1, 1]  # otherwise, range-AUC will stop earlier than (1,1)

            tpr_3d[window] = TF_list[:, 0]
            fpr_3d[window] = TF_list[:, 1]
            prec_3d[window] = Precision_list

            # width = TF_list[1:, 1] - TF_list[:-1, 1]
            # height = (TF_list[1:, 0] + TF_list[:-1, 0]) / 2
            # AUC_range = np.dot(width, height)
            # auc_3d[window] = (AUC_range)

            width_PR = TF_list[1:-1, 0] - TF_list[:-2, 0]
            height_PR = Precision_list[1:]

            AP_range = np.dot(width_PR, height_PR)
            ap_3d[window] = AP_range

        return tpr_3d, fpr_3d, prec_3d, window_3d, None, sum(ap_3d) / len(window_3d)

    # TPR_FPR_window
    def RangeAUC_volume_opt_ROC_only(self, labels_original, score, windowSize, thre=250):
        window_3d = np.arange(0, windowSize + 1, 1)
        P = np.sum(labels_original)
        seq = self.range_convers_new(labels_original)
        l = self.new_sequence(labels_original, seq, windowSize)

        score_sorted = -np.sort(-score)

        tpr_3d = np.zeros((windowSize + 1, thre + 2))
        fpr_3d = np.zeros((windowSize + 1, thre + 2))
        prec_3d = np.zeros((windowSize + 1, thre + 1))

        auc_3d = np.zeros(windowSize + 1)
        # ap_3d = np.zeros(windowSize + 1)

        tp = np.zeros(thre)
        N_pred = np.zeros(thre)

        for k, i in enumerate(np.linspace(0, len(score) - 1, thre).astype(int)):
            threshold = score_sorted[i]
            pred = score >= threshold
            N_pred[k] = np.sum(pred)

        for window in window_3d:

            labels_extended = self.sequencing(labels_original, seq, window)
            L = self.new_sequence(labels_extended, seq, window)

            TF_list = np.zeros((thre + 2, 2))
            Precision_list = np.ones(thre + 1)
            j = 0

            for i in np.linspace(0, len(score) - 1, thre).astype(int):
                threshold = score_sorted[i]
                pred = score >= threshold
                labels = labels_extended.copy()
                existence = 0

                for seg in L:
                    labels[seg[0]:seg[1] + 1] = labels_extended[seg[0]:seg[1] + 1] * pred[seg[0]:seg[1] + 1]
                    if (pred[seg[0]:(seg[1] + 1)] > 0).any():
                        existence += 1
                for seg in seq:
                    labels[seg[0]:seg[1] + 1] = 1

                TP = 0
                N_labels = 0
                for seg in l:
                    TP += np.dot(labels[seg[0]:seg[1] + 1], pred[seg[0]:seg[1] + 1])
                    N_labels += np.sum(labels[seg[0]:seg[1] + 1])

                TP += tp[j]
                FP = N_pred[j] - TP

                existence_ratio = existence / len(L)

                P_new = (P + N_labels) / 2
                recall = min(TP / P_new, 1)

                TPR = recall * existence_ratio
                N_new = len(labels) - P_new
                FPR = FP / N_new

                Precision = TP / N_pred[j]

                j += 1
                TF_list[j] = [TPR, FPR]
                Precision_list[j] = Precision

            TF_list[j + 1] = [1, 1]  # otherwise, range-AUC will stop earlier than (1,1)

            tpr_3d[window] = TF_list[:, 0]
            fpr_3d[window] = TF_list[:, 1]
            prec_3d[window] = Precision_list

            width = TF_list[1:, 1] - TF_list[:-1, 1]
            height = (TF_list[1:, 0] + TF_list[:-1, 0]) / 2
            AUC_range = np.dot(width, height)
            auc_3d[window] = (AUC_range)

            # width_PR = TF_list[1:-1, 0] - TF_list[:-2, 0]
            # height_PR = Precision_list[1:]

            # AP_range = np.dot(width_PR, height_PR)
            # ap_3d[window] = AP_range

        return tpr_3d, fpr_3d, prec_3d, window_3d, sum(auc_3d) / len(window_3d), None

    def RangeAUC_volume_opt_mem(self, labels_original, score, windowSize, thre=250):
        window_3d = np.arange(0, windowSize + 1, 1)
        P = np.sum(labels_original)
        seq = self.range_convers_new(labels_original)
        l = self.new_sequence(labels_original, seq, windowSize)

        score_sorted = -np.sort(-score)

        tpr_3d = np.zeros((windowSize + 1, thre + 2))
        fpr_3d = np.zeros((windowSize + 1, thre + 2))
        prec_3d = np.zeros((windowSize + 1, thre + 1))

        auc_3d = np.zeros(windowSize + 1)
        ap_3d = np.zeros(windowSize + 1)

        tp = np.zeros(thre)
        N_pred = np.zeros(thre)
        p = np.zeros((thre, len(score)))

        for k, i in enumerate(np.linspace(0, len(score) - 1, thre).astype(int)):
            threshold = score_sorted[i]
            pred = score >= threshold
            p[k] = pred
            N_pred[k] = np.sum(pred)

        for window in window_3d:
            labels_extended = self.sequencing(labels_original, seq, window)
            L = self.new_sequence(labels_extended, seq, window)

            TF_list = np.zeros((thre + 2, 2))
            Precision_list = np.ones(thre + 1)
            j = 0

            for i in np.linspace(0, len(score) - 1, thre).astype(int):
                labels = labels_extended.copy()
                existence = 0

                for seg in L:
                    labels[seg[0]:seg[1] + 1] = labels_extended[seg[0]:seg[1] + 1] * p[j][seg[0]:seg[1] + 1]
                    if (p[j][seg[0]:(seg[1] + 1)] > 0).any():
                        existence += 1
                for seg in seq:
                    labels[seg[0]:seg[1] + 1] = 1

                N_labels = 0
                TP = 0
                for seg in l:
                    TP += np.dot(labels[seg[0]:seg[1] + 1], p[j][seg[0]:seg[1] + 1])
                    N_labels += np.sum(labels[seg[0]:seg[1] + 1])

                TP += tp[j]
                FP = N_pred[j] - TP

                existence_ratio = existence / len(L)

                P_new = (P + N_labels) / 2
                recall = min(TP / P_new, 1)

                TPR = recall * existence_ratio

                N_new = len(labels) - P_new
                FPR = FP / N_new
                Precision = TP / N_pred[j]
                j += 1

                TF_list[j] = [TPR, FPR]
                Precision_list[j] = Precision

            TF_list[j + 1] = [1, 1]
            tpr_3d[window] = TF_list[:, 0]
            fpr_3d[window] = TF_list[:, 1]
            prec_3d[window] = Precision_list

            width = TF_list[1:, 1] - TF_list[:-1, 1]
            height = (TF_list[1:, 0] + TF_list[:-1, 0]) / 2
            AUC_range = np.dot(width, height)
            auc_3d[window] = (AUC_range)

            width_PR = TF_list[1:-1, 0] - TF_list[:-2, 0]
            height_PR = Precision_list[1:]
            AP_range = np.dot(width_PR, height_PR)
            ap_3d[window] = (AP_range)
        return tpr_3d, fpr_3d, prec_3d, window_3d, sum(auc_3d) / len(window_3d), sum(ap_3d) / len(window_3d)


    def metric_VUS_pred(self, labels, preds, windowSize):
        window_3d = np.arange(0, windowSize + 1, 1)
        P = np.sum(labels)
        seq = self.range_convers_new(labels)
        l = self.new_sequence(labels, seq, windowSize)

        recall_3d = np.zeros((windowSize + 1))
        prec_3d = np.zeros((windowSize + 1))
        f_3d = np.zeros((windowSize + 1))

        N_pred = np.sum(preds)

        for window in window_3d:

            labels_extended = self.sequencing(labels, seq, window)
            L = self.new_sequence(labels_extended, seq, window)
                
            labels = labels_extended.copy()
            existence = 0

            for seg in L:
                labels[seg[0]:seg[1] + 1] = labels_extended[seg[0]:seg[1] + 1] * preds[seg[0]:seg[1] + 1]
                if (preds[seg[0]:(seg[1] + 1)] > 0).any():
                    existence += 1
            for seg in seq:
                labels[seg[0]:seg[1] + 1] = 1

            TP = 0
            N_labels = 0
            for seg in l:
                TP += np.dot(labels[seg[0]:seg[1] + 1], preds[seg[0]:seg[1] + 1])
                N_labels += np.sum(labels[seg[0]:seg[1] + 1])

            P_new = (P + N_labels) / 2
            recall = min(TP / P_new, 1)
            Precision = TP / N_pred            

            recall_3d[window] = recall
            prec_3d[window] = Precision
            f_3d[window] = 2 * Precision * recall / (Precision + recall) if (Precision + recall) > 0 else 0

        return sum(recall_3d) / len(window_3d), sum(prec_3d) / len(window_3d), sum(f_3d) / len(window_3d)
    




if __name__ == "__main__":
    # labels = "0010001111111000011000"
    # labels = np.array([int(i) for i in labels])
    # preds =  "0010000011000001000001"
    # preds = np.array([int(i) for i in preds])

    metricor = basic_metricor()
    # f1,pre,rec = metricor.metric_Reduced_F1(labels, preds, preds)
    # print("Reduced F1:", f1, "Precision:", pre, "Recall:", rec)


    
    theta, alpha, delta, graph = 0.5, 0.8, 600, 'none'  #default values
    assert(0.0 <= theta <= 1.0)
    assert(0.0 <= alpha <= 1.0)
    assert(isinstance(delta, int))
    assert(graph == 'screen' or graph == 'file' or graph == 'none' or graph == 'all')
    import numpy as np
    labels = "0010001111111000011000"
    labels = np.array([int(i) for i in labels])
    preds =  "0010000011000001000001"
    preds = np.array([int(i) for i in preds])

    result = metricor.metric_PA_percentile_K(labels, score=preds)
    print(result)

    # f1, pre, rec = metricor.metric_eTaPR_F1(labels, preds=preds)
    # print(f1, pre, rec)
    # for key, value in result.items():
    #     print(f"{key}: {value}")