from dataclasses import dataclass
from typing import Any, Dict

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


# Supervised models

def build_rf(**kwargs):
    return RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, **kwargs)


def build_svm(**kwargs):
    return SVC(kernel="rbf", probability=True, **kwargs)


def build_knn(**kwargs):
    return KNeighborsClassifier(n_neighbors=7, **kwargs)


# Unsupervised models

def build_isolation_forest(**kwargs):
    return IsolationForest(n_estimators=200, random_state=42, **kwargs)


def build_kmeans(**kwargs):
    return KMeans(n_clusters=2, n_init=10, random_state=42, **kwargs)


# PyTorch Autoencoder for anomaly detection

class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


@dataclass
class AEConfig:
    epochs: int = 10
    batch_size: int = 256
    lr: float = 1e-3


class AutoencoderWrapper:
    def __init__(self, input_dim: int, config: AEConfig | None = None):
        self.config = config or AEConfig()
        self.model = Autoencoder(input_dim)
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)

    def fit(self, X_train: np.ndarray):
        dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        self.model.train()
        for epoch in range(self.config.epochs):
            total_loss = 0.0
            for (batch,) in loader:
                self.optimizer.zero_grad()
                recon = self.model(batch)
                loss = self.loss_fn(recon, batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * len(batch)

    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            recon = self.model(X_t)
            loss = ((recon - X_t) ** 2).mean(dim=1)
        return loss.numpy()

    def predict(self, X: np.ndarray, threshold: float | None = None) -> np.ndarray:
        scores = self.anomaly_scores(X)
        if threshold is None:
            threshold = np.percentile(scores, 95)
        return np.where(scores > threshold, "attack", "normal")

