
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
from network import EmbeddingNet, SiameseNet, BiRNN
from loss import OnlineTripletLoss, ContrastiveLoss
from augmentator import ImgAugTransform

DATA_BASE_PATH = '/mnt/hdd/gcruz/eyesOriginalSize'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dirs', type=str)
    parser.add_argument('--test_dataset_dirs', type=str)
    parser.add_argument('--epochs', type=int, default = 20)
    parser.add_argument('--input_size', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--dims', type=int, default=256)
    parser.add_argument('--lstm_layers', type=int, default=2)
    parser.add_argument('--lstm_hidden_units', type=int, default=1024)
    parser.add_argument('--cnn_model', type=str)
    return parser.parse_args()

def fit(train_loader, test_loader, model, criterion, optimizer, scheduler, n_epochs, cuda):
    for epoch in range(1, n_epochs + 1):
        scheduler.step()

        train_loss = train_epoch(train_loader, model, criterion, optimizer, cuda)
        print('Epoch: {}/{}, Average train loss: {:.4f}'.format(epoch, n_epochs, train_loss))

        if test_loader is not None:
            test_loss = test_epoch(test_loader, model, criterion, cuda)
            print('Epoch: {}/{}, Test Loss: {:.4f}'.format(epoch, n_epochs, test_loss))


def main():
    args = parse_args()
    cuda = torch.cuda.is_available()
    if cuda:
        print('Device: {}'.format(torch.cuda.get_device_name(0)))

    cnn_model = SiameseNet(args.dims)
    cnn_model.load_state_dict(torch.load(args.cnn_model))
    model.eval()
    lstm_model = BiRNN(args.dims, args.lstm_hidden_units, args.lstm_layers,2)
    if cuda:
        lstm_model = lstm_model.cuda()
    

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
    dataset_dirs = map(lambda x: "{}/{}".format(DATA_BASE_PATH,x), dataset_dirs)

    train_set = LSTMDataset(dataset_dirs, train_transform, cnn_model, cuda)    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False)

    test_set = LSTMDataset(test_dataset_dirs, test_transform, cnn_model, cuda)    
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    criterion = CrossEntropyLoss()
    optimizer = Adam(lstm_model.parameters(), lr=1e-4)
    scheduler = StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)

    fit(train_loader, test_loader, model, criterion, optimizer, scheduler)

if __name__ == '__main__':
    main()
