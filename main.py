import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from dataset import DiffusionDataset
from model import Autoencoder
from metrics import Metrics
data = DiffusionDataset()
adata = data.adata
bulkRNADict = data.bulkRNA
print(bulkRNADict)
scaler = MinMaxScaler()
bulkRNAMatrix = np.array(list(bulkRNADict.values()))
bulkRNAMatrixNormalized = scaler.fit_transform(bulkRNAMatrix)
bulkRNATensor = torch.FloatTensor(bulkRNAMatrixNormalized)
print(bulkRNATensor.shape)
print(bulkRNATensor.mean(dim=1))  # 打印每個 RNA bulk 的均值
print(bulkRNATensor.std(dim=1))   # 打印每個 RNA bulk 的標準差

inputDim = bulkRNATensor.shape[1]
encodeDim = 128
model = Autoencoder(inputDim, encodeDim)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 100
for epoch in range(epochs):
    

    encoded, decoded = model(bulkRNATensor)
    loss = criterion(decoded, bulkRNATensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

encodedTensor = model.encoder(bulkRNATensor).detach().numpy()
print(f"Latent representation shape: {encodedTensor.shape}")
decodedTensor = model.decoder(torch.FloatTensor(encodedTensor)).detach().numpy()
decodedOutput = scaler.inverse_transform(decodedTensor)
print(f"Generated single-cell data shape: {decodedOutput.shape}")

mse = Metrics.mse(bulkRNATensor.flatten(), decodedTensor.flatten())
scc = Metrics.scc(bulkRNATensor.flatten(), decodedTensor.flatten())
print(f"MSE: {mse:.4f}")
print(f"Spearman Correlation Coefficient: {scc:.4f}")

