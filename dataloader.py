import os

import numpy as np
import numpy.random as rand
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import BatchSampler
import pandas as pd
from PIL import Image

POSITIVE_NEGATIVE_RATIO = 0.5
SAME_CLASS = 1

class SiameseDataset(Dataset):

    def __init__(self, paths, transform):
        self.x_col = 'frame'
        self.y_col = 'blink'
        self.transform = transform
        dataframes = []
        for root in paths:
            csvFilePath = root + '.csv'
            dataframe = pd.read_csv(self.csvFilePath)
            dataframe['base_path'] = root

            completePaths = []
            for idx, row in dataframe.iterrows():
                completePaths.append(os.path.join(row['base_path'], row['frame']))
            dataframe['complete_path'] = completePaths
            dataframes.append(dataframe)
        
        self.dataframe = pd.concat(dataframes, ignore_index=True, sort=False)
        self.classes = np.unique(self.dataframe[self.y_col])

    def __len__(self):
        return len(self.dataframe)

    def getDataframeRow(self, idx):
        return self.dataframe.iloc[idx]
    
    def __getitem__(self, idx):
        target = int(rand.random_sample() > POSITIVE_NEGATIVE_RATIO)

        y = self.dataframe[y_col].to_numpy().astype(int)

        class1 = rand.choice(self.classes)
        if target == SAME_CLASS:
            class2 = class1
        else:
            class2 = rand.choice(list((set(self.classes) - {class1})))

        idx1 = rand.choice(np.where(y == class1))
        selectedRow1 = x.iloc[idx1]

        idx2 = rand.choice(np.where(y == class2))
        selectedRow2 = x.iloc[idx2]

        sample1 = Image.open(selectedRow1['complete_path'])
        sample2 = Image.open(selectedRow2['complete_path'])

        sample1 = np.array(sample1)
        sample2 = np.array(sample2)

        if self.transform is not None:
            sample1 = self.transform(sample1)
            sample2 = self.transform(sample2)
        
        return (sample1, sample2), target




        

