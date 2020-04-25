import os

import numpy as np
import numpy.random as rand
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import BatchSampler
import pandas as pd
from PIL import Image
from network import SiameseNet

POSITIVE_NEGATIVE_RATIO = 0.5
SAME_CLASS = 1


class LSTMDataset(Dataset):

    def __init__(self, paths, transform, partial_blinks=False):
        self.x_col = 'complete_path'
        self.y_col = 'blink_id'
        self.transform = transform
        self.partial_blinks = partial_blinks
        dataframes = []
        maxNumVideo = 0
        for root in paths:
            csvFilePath = root + '.csv'
            dataframe = pd.read_csv(csvFilePath)
            dataframe['base_path'] = root

            completePaths = []
            for idx, row in dataframe.iterrows():
                completePaths.append(os.path.join(row['base_path'], row['frame']))
            dataframe['complete_path'] = completePaths
            dataframeMaxVideo = dataframe['video'].max()
            dataframe['video'] = dataframe['video'] +  maxNumVideo
            maxNumVideo = dataframeMaxVideo
            dataframes.append(dataframe)
        
        self.dataframe = pd.concat(dataframes, ignore_index=True, sort=False)
        self.dataframe['blink_type'] = (self.dataframe['blink_id'].astype(int) > 0) + self.dataframe['blink'].astype(int)
        self.dataframe['blink_id_pred'] = 0
        self.dataframe = self.dataframe.rename_axis('idx').sort_values(by=['eye', 'idx'], ascending=[True, True]).reset_index()
        self.targets = self.dataframe[self.y_col]
        self.classes = np.unique(self.dataframe[self.y_col])

    def __len__(self):
        return len(self.dataframe) 

    def getDataframeRow(self, idx):
        return self.dataframe.iloc[idx]

    def getDataframe(self):
        return self.dataframe
    
    def __getitem__(self, idx):
        selectedRow = self.dataframe.iloc[idx]
        if 'NOT_VISIBLE' in selectedRow['complete_path']:
            sample = Image.new('RGB', (100,100))
        else:
            sample = Image.open(selectedRow['complete_path'])
        if self.partial_blinks:
            target = selectedRow['blink_type']
        else:
            target = (selectedRow[self.y_col].astype(int) >= 0).astype(int)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target



class LSTMDatasetOld(Dataset):

    def __init__(self, paths, transform, partial_blinks=False):
        self.x_col = 'complete_path'
        self.y_col = 'blink_id'
        self.transform = transform
        self.partial_blinks = partial_blinks
        dataframes = []
        for root in paths:
            csvFilePath = root + '.csv'
            dataframe = pd.read_csv(csvFilePath)
            dataframe['base_path'] = root

            completePaths = []
            for idx, row in dataframe.iterrows():
                completePaths.append(os.path.join(row['base_path'], row['frame']))
            dataframe['complete_path'] = completePaths
            dataframes.append(dataframe)
        
        self.dataframe = pd.concat(dataframes, ignore_index=True, sort=False)
        self.dataframe['blink_type'] = (self.dataframe['blink_id'].astype(int) > 0) + self.dataframe['blink'].astype(int)
        self.dataframe = self.dataframe.rename_axis('idx').sort_values(by=['eye', 'idx'], ascending=[True, True]).reset_index()
        blinks = deleteNonVisibleBlinks(realBlinks(self.dataframe))
        true_blinks = deleteNonVisibleBlinks(convertAnnotationToBlinks(self.dataframe, 'blink_id'))
        print('BLINKS:', len(blinks), len(true_blinks))
        true_blinks =  convertToIntervalsPartialComplete(self.dataframe,'blink_type')
        partial_true_blinks, complete_true_blinks =  extractPartialAndFullBlinks(true_blinks)
        print('LEN TRUE', len(partial_true_blinks), len(complete_true_blinks))
        self.targets = self.dataframe[self.y_col]
        self.classes = np.unique(self.dataframe[self.y_col])

    def __len__(self):
        return len(self.dataframe) 

    def getDataframeRow(self, idx):
        return self.dataframe.iloc[idx]

    def getDataframe(self):
        return self.dataframe
    
    def __getitem__(self, idx):
        selectedRow = self.dataframe.iloc[idx]
        sample = Image.open(selectedRow['complete_path'])
        if self.partial_blinks:
            target = selectedRow['blink_type']
        else:
            target = (selectedRow[self.y_col].astype(int) >= 0).astype(int)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target



class SiameseDataset(Dataset):

    def __init__(self, paths, transform):
        self.x_col = 'complete_path'
        self.y_col = 'blink'
        self.transform = transform
        dataframes = []
        for root in paths:
            csvFilePath = root + '.csv'
            dataframe = pd.read_csv(csvFilePath)
            dataframe['base_path'] = root

            completePaths = []
            for idx, row in dataframe.iterrows():
                completePaths.append(os.path.join(row['base_path'], row['frame']))
            dataframe['complete_path'] = completePaths
            dataframes.append(dataframe)
        
        self.dataframe = pd.concat(dataframes, ignore_index=True, sort=False)
        self.targets = self.dataframe[self.y_col]
        self.classes = np.unique(self.dataframe[self.y_col])

    def __len__(self):
        return len(self.dataframe)

    def getDataframeRow(self, idx):
        return self.dataframe.iloc[idx]
    
    def __getitem__(self, idx):
        target = int(rand.random_sample() > POSITIVE_NEGATIVE_RATIO)

        y = self.dataframe[self.y_col].to_numpy().astype(int)

        class1 = rand.choice(self.classes)
        if target == SAME_CLASS:
            class2 = class1
        else:
            class2 = rand.choice(list((set(self.classes) - {class1})))

        idx1 = rand.choice(np.argwhere(y == class1).flatten())
        selectedRow1 = self.dataframe.iloc[idx1]

        idx2 = rand.choice(np.argwhere(y == class2).flatten())
        selectedRow2 = self.dataframe.iloc[idx2]

        sample1 = Image.open(selectedRow1['complete_path'])
        sample2 = Image.open(selectedRow2['complete_path'])


        if self.transform is not None:
            sample1 = self.transform(sample1)
            sample2 = self.transform(sample2)

        return sample1, sample2, target


class BalancedBatchSampler(BatchSampler):

    def __init__(self, targets, n_classes, n_samples):
        self.targets = targets
        self.classes = list(set(self.targets))
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.targets)
        self.batch_size = self.n_classes * self.n_samples

        self.target_to_idxs = {target: np.where(np.array(self.targets) == target)[0] for target in self.classes}
        #np.random.seed(42)
    
    def __iter__(self):
        count = 0
        while count + self.batch_size < self.n_dataset:
            indices = []
            for target in np.random.choice(self.classes, self.n_classes, replace = False):
                indices.extend(np.random.choice(self.target_to_idxs[target], self.n_samples, replace=False))
            yield indices
            count += self.batch_size

    def __len__(self):
        return self.n_dataset // self.batch_size

        

