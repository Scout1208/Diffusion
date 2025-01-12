import torch.nn as nn

class BulkRNAencoder(nn.Module):
    def __init__(self, inputDim, encodeDim):
        super(BulkRNAencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(inputDim, 512),
            nn.ReLU(),
            nn.Linear(512, encodeDim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encodeDim, 512),
            nn.ReLU(),
            nn.Linear(512, inputDim),
            nn.ReLU()
        )
    def forward(self, input):
        encoded = self.encoder(input)
        decoded = self.decoder(encoded)
        return encoded, decoded