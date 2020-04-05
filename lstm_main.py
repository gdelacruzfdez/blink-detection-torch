
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
from torch.nn import CrossEntropyLoss

from dataloader import SiameseDataset, LSTMDataset, BalancedBatchSampler
from network import EmbeddingNet, SiameseNet, BiRNN, LSTM
from loss import OnlineTripletLoss, ContrastiveLoss
from augmentator import ImgAugTransform

DATA_BASE_PATH = '/mnt/hdd/gcruz/eyesOriginalSize'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dirs', type=str)
    parser.add_argument('--test_dataset_dirs', type=str)
    parser.add_argument('--epochs', type=int, default = 20)
    parser.add_argument('--input_size', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=150)
    parser.add_argument('--dims', type=int, default=256)
    parser.add_argument('--lstm_layers', type=int, default=2)
    parser.add_argument('--lstm_hidden_units', type=int, default=512)
    parser.add_argument('--cnn_model', type=str)
    parser.add_argument('--sequence_len', type=int, default=30)
    return parser.parse_args()

def fit(train_loader, test_loader, model, cnn_model, criterion, optimizer, scheduler, n_epochs, cuda, args):
    for epoch in range(1, n_epochs + 1):
        scheduler.step()

        train_loss, train_accuracy= train_epoch(train_loader, model, cnn_model, criterion, optimizer, cuda, args)
        print('Epoch: {}/{}, Average train loss: {:.4f}, Average train accuracy: {:.4f}'.format(epoch, n_epochs, train_loss, train_accuracy))

        if test_loader is not None:
            test_loss, test_accuracy= test_epoch(test_loader, model, cnn_model, criterion, cuda, args)
            print('Epoch: {}/{}, Average test loss: {:.4f}, Average test accuracy: {:.4f}'.format(epoch, n_epochs, test_loss, test_accuracy))

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
    progress = tqdm(enumerate(train_loader), total=len(train_loader), desc='Training', file=sys.stdout)
    for batch_idx, data in progress:
        samples, targets = data
        if cuda:
            samples = samples.cuda()
            targets = targets.cuda()
        
        features = cnn_model.get_embedding(samples)

        if features.numel() <  args.dims * args.batch_size:
            continue
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
        accuracies.append(acc)
        #progress.set_description('Training Loss: {} | Accuracy: {} | F1: {}'.format(loss.item(), accuracy, f1))
        progress.set_description('Training Loss: {} | Accuracy: {} | TP: {} | TN: {} | FP: {} | FN: {}'.format(loss.item(), acc, TP, TN, FP, FN))

    return np.mean(losses), np.mean(accuracies)


def test_epoch(test_loader, model, cnn_model, criterion, cuda, args):
    losses = []
    accuracies = []
    TN = 0
    FN = 0
    TP = 0
    FP = 0
    progress = tqdm(enumerate(test_loader), total=len(test_loader), desc='Testing', file=sys.stdout)
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

            loss = criterion(outputs, targets)

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
            accuracies.append(acc)
            progress.set_description('Test Loss: {} | Accuracy: {} | TP: {} | TN: {} | FP: {} | FN: {}'.format(loss.item(), acc, TP, TN, FP, FN))
            #progress.set_description('Test: {} | Accuracy: {} | F1: {}'.format(loss.item(), accuracy, f1))
    return np.mean(losses), np.mean(accuracies)


def main():
    args = parse_args()
    cuda = torch.cuda.is_available()
    if cuda:
        print('Device: {}'.format(torch.cuda.get_device_name(0)))

    cnn_model = SiameseNet(args.dims)
    cnn_model.load_state_dict(torch.load(args.cnn_model))
    cnn_model.eval()
    lstm_model = BiRNN(args.dims, args.lstm_hidden_units, args.lstm_layers,2)
    if cuda:
        lstm_model = lstm_model.cuda()
        cnn_model = cnn_model.cuda()
    

    train_transform = transforms.Compose([
        transforms.Resize((100,100), interpolation=Image.BICUBIC),
    #    ImgAugTransform(),
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

    train_set = LSTMDataset(dataset_dirs, train_transform)    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False)


    test_dataset_dirs = args.test_dataset_dirs.split(',')
    test_dataset_dirs = list(map(lambda x: "{}/{}".format(DATA_BASE_PATH,x), test_dataset_dirs))
    test_set = LSTMDataset(test_dataset_dirs, test_transform)    
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    criterion = CrossEntropyLoss()
    optimizer = Adam(lstm_model.parameters(), lr=1e-4)
    scheduler = StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)

    fit(train_loader, test_loader, lstm_model, cnn_model, criterion, optimizer, scheduler, args.epochs, cuda, args)

if __name__ == '__main__':
    main()
