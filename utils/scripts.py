import torch
import os
from sklearn import metrics
from matplotlib import tri as mtri
import numpy as np


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(gt, pred):
    return metrics.accuracy_score(gt, pred)


def calculate_fpr(y_true, y_pred):
    true_spoof = 0  # Spoof being 1
    false_real = 0  # real being 0
    for i in range(len(y_true)):
        target = y_true[i]
        pred = y_pred[i]
        if target:
            true_spoof += 1
            if not pred:
                false_real += 1
    return false_real / true_spoof if true_spoof else 0


def calculate_fnr(y_true, y_pred):
    true_real = 0  # Spoof being 1
    false_spoof = 0  # real being 0
    for i in range(len(y_true)):
        target = y_true[i]
        pred = y_pred[i]
        if not target:
            true_real += 1
            if pred:
                false_spoof += 1
    return false_spoof / true_real if true_real else 0


def calculate_eer(APCER, BPCER):
    return abs(APCER - BPCER)


def get_labels(probability, taw):
    pred = 0 if float(probability) < taw else 1
    return pred


def create_data(scores, taw):
    data = []
    for i, prob in enumerate(scores):
        pred = get_labels(prob, taw)
        data.append(pred)

    return data


def find_optimal_taw(scores, taw_range, taw_record, y_true):
    for t in taw_range:
        y_hat = create_data(scores, t)
        APCER = calculate_fpr(y_true, y_hat)
        BPCER = calculate_fnr(y_true, y_hat)
        if t not in taw_record.keys():
            taw_record[t] = calculate_eer(APCER, BPCER)
    return min(taw_record, key=taw_record.get)


def load_checkpoint(model, flag):
    cp = torch.load(flag.load_path)
    model.load_state_dict(cp['state_dict'])
    start_epoch = cp['epoch']
    losses = cp['loss']

    return {"model": model,
            "losses": losses,
            "start_epoch": start_epoch}


def save_checkpoint(save_dir, info_dict):
    save_filename = f"{save_dir}/model_{info_dict['epoch']}_{info_dict['acc']}_{info_dict['apcer']}_{info_dict['bpcer']}.pth"
    os.makedirs(save_dir, exist_ok=True)
    cp = {
        'epoch': info_dict['epoch'],
        'state_dict': info_dict['model'].cpu().state_dict(),
        'loss': info_dict['loss']
    }

    torch.save(cp, save_filename)
    print(f"INFO: MODEL SAVED AT {save_filename}")
