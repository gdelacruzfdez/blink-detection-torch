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

from dataloader import SiameseDataset, BalancedBatchSampler
from network import EmbeddingNet, SiameseNet
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
    return parser.parse_args()

def fit(train_loader, test_loader, model, criterion, optimizer, scheduler, n_epochs, cuda):
    for epoch in range(1, n_epochs+1):
        scheduler.step()

        train_loss = train_epoch(train_loader, model, criterion, optimizer, cuda)
        print('Epoch: {}/{}, Average train loss: {:.4f}'.format(epoch, n_epochs, train_loss))

        if test_loader is not None:
            test_loss = test_epoch(test_loader, model, criterion, cuda)
            print('Epoch: {}/{}, Test Loss: {:.4f}'.format(epoch, n_epochs, test_loss))


def train_epoch(train_loader, model, criterion, optimizer, cuda):
    model.train()

    losses = []
    progress = tqdm(enumerate(train_loader), total=len(train_loader), desc='Training', file=sys.stdout)
    for batch_idx, data in progress:
        samples1, samples2, targets = data
        if cuda:
            samples1 = samples1.cuda()
            samples2 = samples2.cuda()
            targets = targets.cuda()
        
        optimizer.zero_grad()
        outputs1, outputs2 = model(samples1, samples2)

        loss = criterion(outputs1, outputs2, targets)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        progress.set_description('Training Loss: ' + str(loss.item()))
    
    return np.mean(losses)


def test_epoch(test_loader, model, criterion, cuda):
    
    losses = []
    progress = tqdm(enumerate(test_loader), total=len(test_loader), desc='Testing', file=sys.stdout)
    for batch_idx, data in progress:
        samples1, samples2, targets = data
        if cuda:
            samples1 = samples1.cuda()
            samples2 = samples2.cuda()
            targets = targets.cuda()
        
        with torch.no_grad():
            outputs1 = model.get_embedding(samples1)
            outputs2 = model.get_embedding(samples2)
            loss = criterion(outputs1, outputs2, targets)
            losses.append(loss.item())
            progress.set_description('Test Loss: ' + str(loss.item()))
    return np.mean(losses)

def extract_embeddings(loader, model, cuda):
    model.eval()

    embeddings = []
    targets = []

    with torch.no_grad():
        for sample, target in tqdm(loader, total=len(loader), desc='Testing', file=sys.stdout):
            if cuda:
                sample = sample.cuda()
            
            output = model.get_embedding(sample)

            embeddings.append(output.cpu().numpy())
            targets.append(target)
    
    embeddings = np.vstack(embeddings)
    targets = np.concatenate(targets)

    return embeddings, targets



def main():
    args = parse_args()
    print(vars(args))

    cuda = torch.cuda.is_available()
    if cuda:
        print('Device: {}'.format(torch.cuda.get_device_name(0)))

    train_transform = transforms.Compose([
        transforms.Resize((100,100), interpolation=Image.BICUBIC),
        ImgAugTransform(),
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

    train_set = SiameseDataset(dataset_dirs, train_transform)    
    train_batch_sampler = BalancedBatchSampler(train_set.targets, n_classes=2, n_samples=args.batch_size)
    train_loader = DataLoader(train_set, batch_sampler=train_batch_sampler, num_workers=8)

    print(train_set)

    test_loader = None

    if args.test_dataset_dirs != None:
        test_dataset_dirs = args.test_dataset_dirs.split(',')
        test_dataset_dirs = map(lambda x: "{}/{}".format(DATA_BASE_PATH,x), test_dataset_dirs)
        test_set = SiameseDataset(test_dataset_dirs, test_transform)
        test_batch_sampler = BalancedBatchSampler(test_set.targets, n_classes=2, n_samples=args.batch_size)
        test_loader = DataLoader(test_set, batch_sampler=test_batch_sampler, num_workers=8)
        print(test_set)

    model = SiameseNet(args.dims)
    if cuda:
        model = model.cuda()

    criterion = ContrastiveLoss(margin=1)
    optimizer = Adam(model.parameters(), lr=1e-4)
    scheduler = StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)

    fit(train_loader, test_loader, model, criterion, optimizer, scheduler, args.epochs, cuda)
    
    torch.save(model.state_dict(),'siamese_model_resnet50_50ep.pt')



if __name__ == '__main__':
    main()


