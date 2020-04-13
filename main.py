import os
import sys
import argparse

import numpy as np
from PIL import Image
from sklearn import metrics
from sklearn.neighbors import NearestCentroid
from tqdm import tqdm

import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.nn import CrossEntropyLoss, BCELoss  

from dataloader import SiameseDataset, BalancedBatchSampler, LSTMDataset
from network import EmbeddingNet, SiameseNet, SiameseNetV2
from loss import OnlineTripletLoss, ContrastiveLoss
from augmentator import ImgAugTransform

DATA_BASE_PATH = '/mnt/hdd/gcruz/eyesOriginalSize'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dirs', type=str)
    parser.add_argument('--test_dataset_dirs', type=str)
    parser.add_argument('--epochs', type=int, default = 10)
    parser.add_argument('--input_size', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--dims', type=int, default=256)
    return parser.parse_args()

def fit(train_loader, test_loader,eval_train_loader, eval_test_loader, model, criterion, optimizer, scheduler, n_epochs, cuda):
    bestf1 = 0
    for epoch in range(1, n_epochs+1):
        scheduler.step()

        train_loss = train_epoch(train_loader, model, criterion, optimizer, cuda)
        print('Epoch: {}/{}, Average train loss: {:.4f}'.format(epoch, n_epochs, train_loss))

        if test_loader is not None:
            classification_report, classification_metrics = test_epoch(eval_train_loader, eval_test_loader, model, criterion, cuda)
            print('Test Epoch: {}/{}'.format(epoch, n_epochs))
            print(classification_report)
            print(classification_metrics)
            if classification_metrics[2] > bestf1:
                print('Best model! New F1:{:.4f} | Previous F1 {:.4f}'.format(classification_metrics[2],bestf1))
                bestf1 = classification_metrics[2]
                torch.save(model.state_dict(),"siamese_model_resnet18_best_ep_2.pt")


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
        outputs = model(samples1, samples2)
        #outputs1, outputs2 = model(samples1, samples2)
        outputs = outputs.squeeze(1)
        #loss = criterion(outputs1, outputs2, targets)
        loss = criterion(outputs, targets.float())
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        progress.set_description('Mean Training Loss: {:.4f}'.format(np.mean(losses)))
    
    return np.mean(losses)


def test_epoch(train_loader, test_loader, model, criterion, cuda):
    train_embeddings, train_targets = extract_embeddings(train_loader, model, cuda)
    test_embeddings, test_targets = extract_embeddings(test_loader, model, cuda)

    nc = NearestCentroid()
    nc.fit(train_embeddings, train_targets)
    predictions = nc.predict(test_embeddings)
    classification_report = metrics.classification_report(test_targets, predictions, target_names=['Open', 'Closed'])
    classification_metrics = metrics.precision_recall_fscore_support(test_targets, predictions, average='macro')
    return classification_report, classification_metrics


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
    dataset_dirs = list(map(lambda x: "{}/{}".format(DATA_BASE_PATH,x), dataset_dirs))
    train_set = SiameseDataset(dataset_dirs, train_transform)    
    train_batch_sampler = BalancedBatchSampler(train_set.targets, n_classes=2, n_samples=args.batch_size)
    train_loader = DataLoader(train_set, batch_sampler=train_batch_sampler, num_workers=8)


    eval_train_set = LSTMDataset(dataset_dirs, test_transform)    
    eval_train_loader = DataLoader(eval_train_set, batch_size=args.batch_size, shuffle=False)


    test_loader = None
    eval_test_loader = None

    if args.test_dataset_dirs != None:
        test_dataset_dirs = args.test_dataset_dirs.split(',')
        test_dataset_dirs = list(map(lambda x: "{}/{}".format(DATA_BASE_PATH,x), test_dataset_dirs))
        test_set = SiameseDataset(test_dataset_dirs, test_transform)
        test_batch_sampler = BalancedBatchSampler(test_set.targets, n_classes=2, n_samples=args.batch_size)
        #test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=8)
        test_loader = DataLoader(test_set, batch_sampler=test_batch_sampler, num_workers=8)
    
        eval_test_set = LSTMDataset(test_dataset_dirs, test_transform)    
        eval_test_loader = DataLoader(eval_test_set, batch_size=args.batch_size, shuffle=False)
        print(test_set)

    model = SiameseNetV2(args.dims)
    if cuda:
        model = model.cuda()

    criterion = BCELoss()
    optimizer = Adam(model.parameters(), lr=1e-4)
    #optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
    scheduler = StepLR(optimizer, 5, gamma=0.1, last_epoch=-1)
    #scheduler = ReduceLROnPlateau(optimizer, 'min', patience=8)

    fit(train_loader, test_loader,eval_train_loader, eval_test_loader, model, criterion, optimizer, scheduler, args.epochs, cuda)
    



if __name__ == '__main__':
    main()


