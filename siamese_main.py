import sys
import json
import torch
from siamese import SiameseModel

def main():
    cuda = torch.cuda.is_available()
    if cuda:
        print('Device: {}'.format(torch.cuda.get_device_name(0)))

    json_params_file = sys.argv[1]
    with open(json_params_file, "r") as json_file:
        params = json.load(json_file)
        print(params)

        siamese_model = SiameseModel(params, cuda)
        siamese_model.hyperparameter_tunning()
        #siamese_model.fit()




if __name__ == '__main__':
    main()
