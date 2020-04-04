from torch import nn
from torchvision import models
import multiprocessing.dummy as mp


class EmbeddingNet(nn.Module):

    def __init__(self, num_dims):
        super().__init__()

        self.model = models.resnet50(pretrained=True)
        num_ftrs = self.model.fc.in_features
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
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)  # 2 for bidirection
    
    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device) # 2 for bidirection 
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out