import sys
import json
import torch
import argparse
from siamese import SiameseModel
from sklearn.model_selection import ParameterGrid

HYPERPARAM_MODE = 'HYPERPARAM_MODE'
TRAINING_MODE = 'TRAINING_MODE'


def main():
    cuda = torch.cuda.is_available()
    if cuda:
        print('Device: {}'.format(torch.cuda.get_device_name(0)))

    json_params_file = sys.argv[1]
    with open(json_params_file, "r") as json_file:
        params = json.load(json_file)
        print(params)

        if HYPERPARAM_MODE == params['mode']:
            param_grid = ParameterGrid(params)
            for search_params in param_grid:
                print('Fitting model with params:',search_params)
                siamese_model = SiameseModel(search_params, cuda)
                siamese_model.fit()
        else:
            siamese_model = SiameseModel(params, cuda)
            siamese_model.fit()




if __name__ == '__main__':
    main()
