import sys
import dataloader
import network
import numpy as np
import torch
from PIL import Image
from skorch import NeuralClassifier
from functional import seq
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from augmentator import ImgAugTransform
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.nn import BCELoss  
from sklearn.neighbors import NearestCentroid
from tqdm import tqdm
import sklearn.metrics
from sklearn.model_selection import GridSearchCV

class SiameseModel:

    TRAIN_TRANSFORM =  transforms.Compose([
        transforms.Resize((100,100), interpolation=Image.BICUBIC),
        ImgAugTransform(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    ])

    TEST_TRANSFORM =  transforms.Compose([
        transforms.Resize((100,100), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    ])


    def __init__(self, params, cuda):
        self.params = params
        self.cuda = cuda
        self.train_videos = None if not 'train_videos' in params else params.get('train_videos').split(',')
        self.test_videos = None if not 'test_videos' in params else params.get('test_videos').split(',')
        self.train_dataset_dirs = seq(params.get('train_dataset_dirs'))\
            .map(lambda x: "{}/{}".format(self.params.get('datasets_base_path'), x))
        self.test_dataset_dirs = seq(params.get('test_dataset_dirs'))\
            .map(lambda x: "{}/{}".format(self.params.get('datasets_base_path'), x))

        self.__initialize_train_loader()
        self.__initialize_evaluation_loader()
        self.__initialize_model()
        self.__initialize_training_parameters()

    
    def __initialize_train_loader(self):
        self.train_set = dataloader.SiameseDataset(self.train_dataset_dirs, self.TRAIN_TRANSFORM, videos = self.train_videos)
        self.train_batch_sampler = dataloader.BalancedBatchSampler(self.train_set.targets, n_classes=2, n_samples=self.params.get('batch_size'))
        self.train_loader = DataLoader(self.train_set, batch_sampler = self.train_batch_sampler, num_workers=32)

    def __initialize_evaluation_loader(self):
       self.eval_train_set = dataloader.LSTMDataset(self.train_dataset_dirs, self.TEST_TRANSFORM, mode = dataloader.EYE_STATE_DETECTION_MODE, videos=self.train_videos)
       self.eval_train_loader = DataLoader(self.eval_train_set, batch_size=self.params.get('batch_size'), shuffle=False, num_workers=16)
       self.eval_test_set = dataloader.LSTMDataset(self.test_dataset_dirs, self.TEST_TRANSFORM, mode = dataloader.EYE_STATE_DETECTION_MODE, videos=self.test_videos)
       self.eval_test_loader = DataLoader(self.eval_test_set, batch_size=self.params.get('batch_size'), shuffle=False, num_workers=16)

    def __initialize_model(self):
        self.model = network.SiameseNetV2(self.params.get('dims'))
        if self.cuda:
            self.model = self.model.cuda()

    def __initialize_training_parameters(self):
        self.criterion = BCELoss()
        self.optimizer = Adam(self.model.parameters(), lr=self.params.get('lr'))
        self.scheduler = StepLR(self.optimizer, 8, gamma=0.1, last_epoch=-1)

    def __train_epoch(self):
        self.model.train()
        losses = []
        progress = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc='Training', file=sys.stdout)
        for batch_idx, data in progress:
            samples1, samples2, targets = data
            if self.cuda:
                samples1 = samples1.cuda()
                samples2 = samples2.cuda()
                targets = targets.cuda()
            
            self.optimizer.zero_grad()
            outputs = self.model(samples1, samples2)
            outputs = outputs.squeeze(1)
            loss = self.criterion(outputs, targets.float())
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())
            progress.set_description('Mean Training Loss: {:.4f}'.format(np.mean(losses)))

        return np.mean(losses)

    def __test_epoch(self):
        train_embeddings, train_targets = self.__extract_embeddings(self.eval_train_loader)
        test_embeddings, test_targets = self.__extract_embeddings(self.eval_test_loader)

        nc = NearestCentroid()
        nc.fit(train_embeddings, train_targets)
        predictions = nc.predict(test_embeddings)
        classification_report = sklearn.metrics.classification_report(test_targets, predictions, target_names=['Open','Closed'])
        classification_metrics = sklearn.metrics.precision_recall_fscore_support(test_targets, predictions, average='macro')
        confussion_matrix = sklearn.metrics.confusion_matrix(test_targets, predictions)

        return classification_report, classification_metrics, confussion_matrix



    def __extract_embeddings(self,loader):
        self.model.eval()

        embeddings = []
        targets = []

        with torch.no_grad():
            for sample, target in tqdm(loader, total=len(loader), desc='Testing', file=sys.stdout):
                if self.cuda:
                    sample = sample.cuda()

                output = self.model.get_embedding(sample)
                embeddings.append(output.cpu().numpy())
                targets.append(target)
        
        embeddings = np.vstack(embeddings)
        targets = np.concatenate(targets)

        return embeddings, targets


    def fit(self):
        bestf1 = 0
        currentf1 = 0
        epochs = self.params.get('epochs')
        for epoch in range(1,epochs + 1):
            self.scheduler.step()
            train_loss = self.__train_epoch()
            print('Epoch: {}/{}, Average train loss: {:.4f}'.format(epoch, epochs, train_loss))

            classification_report, classification_metrics , confussion_matrix = self.__test_epoch()
            print('Test Epoch: {}/{}'.format(epoch, epochs))
            print(classification_report)
            print(classification_metrics)
            print(confussion_matrix)

            currentf1 = classification_metrics[2]
            if classification_metrics[2] > bestf1:
                print('Best model! New F1:{:.4f} | Previous F1 {:.4f}'.format(classification_metrics[2],bestf1))
                bestf1 = classification_metrics[2]
                torch.save(self.model.state_dict(), self.params.get('model_file'))
    
    def hyperparameter_tunning(self):
        net = NeuralClassifier(
            SiameseModel,
            max_epochs=2,
            criterion= BCELoss,
            optimizer= Adam,
            iterator_train__shuffle=True)
        
        net.set_params(train_split=False, verbose=0)
        params = {
            'lr': [0.01, 0.001],
            'module__dims': [128, 256]
        }
        gs = GridSearchCV(net, params, refit=False, cv=3, scoring='f1', verbose=2)
        gs.fit(self.train_set)
