
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
from evaluation import  extractPartialCompleteBlinks,evaluatePartialBlinks

DATA_BASE_PATH = '/mnt/hdd/gcruz/eyesOriginalSize'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dirs', type=str)
    parser.add_argument('--test_dataset_dirs', type=str)
    parser.add_argument('--epochs', type=int, default = 20)
    parser.add_argument('--input_size', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--dims', type=int, default=256)
    parser.add_argument('--lstm_layers', type=int, default=2)
    parser.add_argument('--lstm_hidden_units', type=int, default=512)
    parser.add_argument('--cnn_model', type=str)
    parser.add_argument('--sequence_len', type=int, default=100)
    return parser.parse_args()

def fit(train_loader, test_loader, model, cnn_model, criterion, optimizer, scheduler, n_epochs, cuda, args):
    bestf1 = 0
    for epoch in range(1, n_epochs + 1):
        scheduler.step()

        train_loss, train_accuracy= train_epoch(train_loader, model, cnn_model, criterion, optimizer, cuda, args)
        print('Epoch: {}/{}, Average train loss: {:.4f}, Average train accuracy: {:.4f}'.format(epoch, n_epochs, train_loss, train_accuracy))

        if test_loader is not None:
            partial, complete= test_epoch(test_loader, model, cnn_model, criterion, cuda, args)
            f1_partial, precision_partial, recall_partial, fp_partial, fn_partial, tp_partial, db_partial = partial
            f1_complete, precision_complete, recall_complete, fp_complete, fn_complete, tp_complete, db_complete= complete
            print('PARTIAL Epoch: {}/{}, F1: {:.4f} | Precision: {:.4f} | Recall: {:.4f} | TP: {} | FP: {} | FN: {} | Predicted Blinks: {}'.format(epoch, n_epochs, f1_partial, precision_partial, recall_partial, tp_partial, fp_partial, fn_partial, db_partial))
            print('COMPLETE Epoch: {}/{}, F1: {:.4f} | Precision: {:.4f} | Recall: {:.4f} | TP: {} | FP: {} | FN: {} | Predicted Blinks: {}'.format(epoch, n_epochs, f1_complete, precision_complete, recall_complete, tp_complete, fp_complete, fn_complete, db_complete))
            if f1_partial + f1_complete > bestf1:
                print('Best model! New F1 Partial:{:.4f} | Complete: {:.4f}'.format(f1_partial,f1_complete))
                print('')
                bestf1 = f1_partial + f1_complete
                torch.save(model.state_dict(),"lstm_partial_best.pt")


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

def train_epoch(train_loader, model, cnn_model, criterion, optimizer, cuda, args):
    model.train()

    losses = []
    accuracies = []
    TN = 0
    FN = 0
    TP = 0
    FP = 0
    previousFeatures = [np.zeros(()) for i in range(5)]
    progress = tqdm(enumerate(train_loader), total=len(train_loader), desc='Training', file=sys.stdout)
    for batch_idx, data in progress:
        samples, targets = data
        if cuda:
            samples = samples.cuda()
            targets = targets.cuda()
        
        features = cnn_model.get_embedding(samples)
        if features.numel() <  args.dims * args.batch_size:
            zeros_features = torch.zeros(args.batch_size - features.shape[0], args.dims)
            zeros_features = zeros_features.cuda()
            features = torch.cat((features, zeros_features))
            zeros_targets = torch.zeros(args.batch_size - targets.shape[0], dtype=torch.long)
            zeros_targets = zeros_targets.cuda()
            targets = torch.cat((targets, zeros_targets))
        features = features.reshape(-1, args.sequence_len, args.dims)
        

        outputs = model(features)


        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        targets = targets.data.cpu()
        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.data.cpu()
        perf = perf_measure(targets, predicted)

        TN += perf[2]
        FN += perf[3]
        TP += perf[0]
        FP += perf[1] 

        acc = (TP + TN) / (FP + FN + TP + TN)
        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        f1 = 2 * ( precision * recall ) / ( precision + recall ) if precision + recall > 0 else 0
        accuracies.append(acc)
        #progress.set_description('Training Loss: {} | Accuracy: {} | F1: {}'.format(loss.item(), accuracy, f1))
        progress.set_description('Training Loss: {:.4f} | Accuracy: {:.4f} | F1: {:.4f} | Precision: {:.4f} | Recall: {:.4f} | TP: {} | TN: {} | FP: {} | FN: {}'.format(loss.item(), acc,f1, precision, recall, TP, TN, FP, FN))

    return np.mean(losses), np.mean(accuracies)


def test_epoch(test_loader, model, cnn_model, criterion, cuda, args):
    losses = []
    accuracies = []
    TN = 0
    FN = 0
    TP = 0
    FP = 0
    progress = tqdm(enumerate(test_loader), total=len(test_loader), desc='Testing', file=sys.stdout)
    predictions = np.array([])
    alltargets = np.array([])

    for batch_idx, data in progress:
        samples, targets = data
        if cuda:
            samples = samples.cuda()
            targets = targets.cuda()
        
        with torch.no_grad():
            features = cnn_model.get_embedding(samples)
            if features.numel() <  args.dims * args.batch_size:
                zeros_features = torch.zeros(args.batch_size - features.shape[0], args.dims)
                zeros_features = zeros_features.cuda()
                features = torch.cat((features, zeros_features))
                zeros_targets = torch.zeros(args.batch_size - targets.shape[0], dtype=torch.long)
                zeros_targets = zeros_targets.cuda()
                targets = torch.cat((targets, zeros_targets))
            features = features.reshape(-1, args.sequence_len, args.dims)
            outputs = model(features)

            loss = criterion(outputs, targets)

            losses.append(loss.item())
            targets = targets.data.cpu()
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.data.cpu()
            predictions = np.concatenate((predictions, predicted))
            alltargets = np.concatenate((alltargets, targets))

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
            progress.set_description('Test Loss: {:.4f} | Accuracy: {:.4f} | F1: {:.4f} | Precision: {:.4f} | Recall: {:.4f} | TP: {} | TN: {} | FP: {} | FN: {}'.format(loss.item(), acc,f1, precision, recall, TP, TN, FP, FN))
            #progress.set_description('Test: {} | Accuracy: {} | F1: {}'.format(loss.item(), accuracy, f1))
    dataframe = test_loader.dataset.getDataframe().copy()
    predictions = predictions[:len(dataframe)]
    dataframe['blink_id_pred'] = predictions
    return evaluatePartialBlinks(dataframe)

def main():
    args = parse_args()
    print(args)
    cuda = torch.cuda.is_available()
    if cuda:
        print('Device: {}'.format(torch.cuda.get_device_name(0)))

    cnn_model = SiameseNetV2(args.dims)
    cnn_model.load_state_dict(torch.load(args.cnn_model))
    cnn_model.eval()
    lstm_model = BiRNN(args.dims, args.lstm_hidden_units, args.lstm_layers,3)
    if cuda:
        lstm_model = lstm_model.cuda()
        cnn_model = cnn_model.cuda()
    

    train_transform = transforms.Compose([
        transforms.Resize((100,100), interpolation=Image.BICUBIC),
        #LSTMImgAugTransform(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((100,100), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    ])

    dataset_dirs = args.dataset_dirs.split(',')
    dataset_dirs = list(map(lambda x: "{}/{}".format(DATA_BASE_PATH,x), dataset_dirs))

    train_set = LSTMDataset(dataset_dirs, train_transform, partial_blinks=True)    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=8)
    print('DATASET LENGTH', len(train_loader.dataset.getDataframe()))

    test_dataset_dirs = args.test_dataset_dirs.split(',')
    test_dataset_dirs = list(map(lambda x: "{}/{}".format(DATA_BASE_PATH,x), test_dataset_dirs))
    test_set = LSTMDataset(test_dataset_dirs, test_transform, partial_blinks=True)    
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=8)
    print('TEST LENGTH', len(test_loader.dataset.getDataframe()))

    criterion = CrossEntropyLoss()

    optimizer = Adam(lstm_model.parameters(), lr=1e-4)
    scheduler = StepLR(optimizer, 20, gamma=0.1, last_epoch=-1)

    fit(train_loader, test_loader, lstm_model, cnn_model, criterion, optimizer, scheduler, args.epochs, cuda, args)

if __name__ == '__main__':
    main()