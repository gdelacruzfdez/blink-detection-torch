from torch import nn
from torchvision import models
import multiprocessing.dummy as mp
import torch
from pytorchcv.model_provider import get_model as ptcv_get_model
from torch.autograd import Variable


class EmbeddingNet(nn.Module):

    def __init__(self, num_dims):
        super().__init__()

        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_dims)

    def forward(self, x):
        return self.model(x)        

    def get_embedding(self, x):
        return self.forward(x)

class EmbeddingNetV2(nn.Module):

    def __init__(self, num_dims):
        super().__init__()

        self.model = ptcv_get_model("resnet18", pretrained=True)
        print('SENET-28')

    def forward(self, x):
        return self.model(x)        

    def get_embedding(self, x):
        return self.forward(x)
class EmbeddingNetDenseNet(nn.Module):

    def __init__(self, num_dims):
        super().__init__()

        self.model = models.densenet121(pretrained=True)
        num_ftrs = self.model.classifier.in_features
        self.model.fc = nn.Linear(num_ftrs, num_dims)

    def forward(self, x):
        return self.model(x)        

    def get_embedding(self, x):
        return self.forward(x)


class SiameseNet(nn.Module):

    def __init__(self, num_dims: int = 256):
        super().__init__()

        self.embedding_net = EmbeddingNet(num_dims)
        self.pool = mp.Pool(processes = 2)

    def forward(self, x1, x2):
        output1, output2 = self.pool.map(self.get_embedding, [x1, x2])
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)

class SiameseNetV2(nn.Module):

    def __init__(self, num_dims: int = 256):
        super().__init__()

        self.embedding_net = EmbeddingNet(num_dims)
        self.pool = mp.Pool(processes = 2)
        self.fc = nn.Linear(num_dims, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        output1, output2 = self.pool.map(self.get_embedding, [x1, x2])
        l1_distance = torch.abs(output1 - output2)
        out = self.fc(l1_distance)
        y_prob = self.sigmoid(out)
        return y_prob

    def get_embedding(self, x):
        return self.embedding_net(x)

class TripletNet(nn.Module):

    def __init__(self, num_dims: int = 16):
        super().__init__()

        self.embedding_net = EmbeddingNet(num_dims)
        self.pool = mp.Pool(processes=3)

    def forward(self, x1, x2, x3):
        output1, output2, output3 = self.pool.map(self.get_embedding, [x1, x2, x3])
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)


class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size*2, num_classes)  # 2 for bidirection

    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).cuda() # 2 for bidirection 
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).cuda()
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        
        # Decode the hidden state of the last time step
        out = self.fc(out)

        out = out.reshape(-1, self.num_classes)
        predictions =  nn.functional.softmax(out)
        return predictions


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_size, num_layers, output_size):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size).cuda(),
                            torch.zeros(1,1,self.hidden_layer_size).cuda())

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = nn.functional.softmax(self.linear(lstm_out.view(len(input_seq), -1)))
        return predictions