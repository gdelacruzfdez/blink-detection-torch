
import os
import sys
import argparse

import numpy as np
from PIL import Image
from sklearn import metrics
from tqdm import tqdm

import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.nn import BCELoss, CrossEntropyLoss

from dataloader import SiameseDataset, LSTMDataset, BalancedBatchSampler
from network import EmbeddingNet, SiameseNet, SiameseNetV2, BiRNN, LSTM
from loss import OnlineTripletLoss, ContrastiveLoss
from augmentator import ImgAugTransform, LSTMImgAugTransform
from evaluation import evaluate, convertAnnotationToBlinks, deleteNonVisibleBlinks

DATA_BASE_PATH = '/mnt/hdd/gcruz/eyesOriginalSize'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dirs', type=str)
    parser.add_argument('--input_size', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--dims', type=int, default=256)
    parser.add_argument('--lstm_layers', type=int, default=2)
    parser.add_argument('--lstm_hidden_units', type=int, default=512)
    parser.add_argument('--cnn_model', type=str)
    parser.add_argument('--lstm_model', type=str)
    parser.add_argument('--sequence_len', type=int, default=100)
    return parser.parse_args()

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN)


def test(test_loader, model, cnn_model, cuda, args):
    accuracies = []
    TN = 0
    FN = 0
    TP = 0
    FP = 0
    progress = tqdm(enumerate(test_loader), total=len(test_loader), desc='Testing', file=sys.stdout)
    predictions = np.array([])

    for batch_idx, data in progress:
        samples, targets = data
        if cuda:
            samples = samples.cuda()
            targets = targets.cuda()
        
        with torch.no_grad():
            features = cnn_model.get_embedding(samples)
            if features.numel() <  args.dims * args.batch_size:
                continue
            features = features.reshape(-1, args.sequence_len, args.dims)
            outputs = model(features)
            targets = targets.data.cpu()
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.data.cpu()
            predictions = np.concatenate((predictions, predicted))

            perf = perf_measure(targets, predicted)

            TN += perf[2]
            FN += perf[3]
            TP += perf[0]
            FP += perf[1] 

            acc = (TP + TN) / (FP + FN + TP + TN)
            accuracies.append(acc)
            precision = TP / (TP + FP) if TP + FP > 0 else 0
            recall = TP / (TP + FN) if TP + FN > 0 else 0
            f1 = 2 * ( precision * recall ) / ( precision + recall ) if precision + recall > 0 else 0
            progress.set_description('Test Accuracy: {:.4f} | F1: {:.4f} | Precision: {:.4f} | Recall: {:.4f} | TP: {} | TN: {} | FP: {} | FN: {}'.format(acc,f1, precision, recall, TP, TN, FP, FN))
    dataframe = test_loader.dataset.getDataframe().copy()
    dataframe = dataframe[:len(predictions)]
    dataframe['blink_id_pred'] = predictions
    return evaluate(dataframe)


def main():
    args = parse_args()
    cuda = torch.cuda.is_available()
    if cuda:
        print('Device: {}'.format(torch.cuda.get_device_name(0)))


    cnn_model = SiameseNetV2(args.dims)
    cnn_model.load_state_dict(torch.load(args.cnn_model))
    cnn_model.eval()
    lstm_model = BiRNN(args.dims, args.lstm_hidden_units, args.lstm_layers,2)
    lstm_model.load_state_dict(torch.load(args.lstm_model))
    lstm_model.eval()
    if cuda:
        cnn_model = cnn_model.cuda()
        lstm_model = lstm_model.cuda()
    

    test_transform = transforms.Compose([
        transforms.Resize((100,100), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    ])

    dataset_dirs = args.dataset_dirs.split(',')
    dataset_dirs = list(map(lambda x: "{}/{}".format(DATA_BASE_PATH,x), dataset_dirs))

    dataset = LSTMDataset(dataset_dirs, test_transform)    
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    dataframe = loader.dataset.getDataframe().copy()
    true_blinks = deleteNonVisibleBlinks(convertAnnotationToBlinks(dataframe, 'blink_id'))
    print('BLINKS', len(true_blinks))

    f1, precision, recall, fp, fn, tp= test(loader, lstm_model, cnn_model, cuda, args)
    print('F1: {:.4f} | Precision: {:.4f} | Recall: {:.4f} | TP: {} | FP: {} | FN: {}'.format(f1, precision, recall, tp, fp, fn))
if __name__ == '__main__':
    main()
