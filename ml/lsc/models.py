"""Modelos comparados: 3 clásicos (scikit-learn) y 4 redes (PyTorch)."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

# ---------------------------------------------------------------- clásicos

def modelos_clasicos(semilla: int = 0) -> dict:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    return {
        "kNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, weights="distance")),
        "SVM-RBF": make_pipeline(StandardScaler(), SVC(C=10, gamma="scale")),
        "RandomForest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=semilla),
    }


# ---------------------------------------------------------------- redes

class Normalizador(nn.Module):
    """Estandarización con media/desviación del entrenamiento, dentro del
    modelo para que el ONNX exportado reciba las características tal cual."""

    def __init__(self, n_feat: int):
        super().__init__()
        self.register_buffer("media", torch.zeros(n_feat))
        self.register_buffer("desv", torch.ones(n_feat))

    def ajustar(self, X: np.ndarray):
        flat = X.reshape(-1, X.shape[-1])
        self.media.copy_(torch.from_numpy(flat.mean(0)))
        self.desv.copy_(torch.from_numpy(np.maximum(flat.std(0), 1e-3)))

    def forward(self, x):
        return (x - self.media) / self.desv


class MLP(nn.Module):
    def __init__(self, n_feat, n_cls, T, h=256, p=0.3):
        super().__init__()
        self.norm = Normalizador(n_feat)
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(T * n_feat, h), nn.ReLU(), nn.Dropout(p),
            nn.Linear(h, h // 2), nn.ReLU(), nn.Dropout(p), nn.Linear(h // 2, n_cls))

    def forward(self, x):
        return self.net(self.norm(x))


class CNN1D(nn.Module):
    """Convoluciones temporales sobre la secuencia de cuadros."""

    def __init__(self, n_feat, n_cls, T, h=128, p=0.3):
        super().__init__()
        self.norm = Normalizador(n_feat)
        self.conv = nn.Sequential(
            nn.Conv1d(n_feat, h, 3, padding=1), nn.BatchNorm1d(h), nn.ReLU(),
            nn.Conv1d(h, h, 3, padding=1), nn.BatchNorm1d(h), nn.ReLU())
        self.cabeza = nn.Sequential(nn.Dropout(p), nn.Linear(2 * h, n_cls))

    def forward(self, x):
        z = self.conv(self.norm(x).transpose(1, 2))
        z = torch.cat([z.mean(2), z.amax(2)], 1)
        return self.cabeza(z)


class Recurrente(nn.Module):
    """GRU o LSTM bidireccional: modela la dinámica de la seña."""

    def __init__(self, n_feat, n_cls, T, tipo="gru", h=128, p=0.3):
        super().__init__()
        self.norm = Normalizador(n_feat)
        self.proy = nn.Sequential(nn.Linear(n_feat, h), nn.ReLU(), nn.Dropout(0.1))
        rnn = nn.GRU if tipo == "gru" else nn.LSTM
        self.rnn = rnn(h, h, batch_first=True, bidirectional=True)
        self.cabeza = nn.Sequential(nn.Dropout(p), nn.Linear(4 * h, n_cls))

    def forward(self, x):
        z, _ = self.rnn(self.proy(self.norm(x)))
        z = torch.cat([z.mean(1), z[:, -1]], 1)
        return self.cabeza(z)


REDES = {
    "MLP": lambda f, c, T: MLP(f, c, T),
    "CNN-1D": lambda f, c, T: CNN1D(f, c, T),
    "LSTM": lambda f, c, T: Recurrente(f, c, T, "lstm"),
    "GRU": lambda f, c, T: Recurrente(f, c, T, "gru"),
}


class ConSoftmax(nn.Module):
    """Envoltura para exportar: devuelve probabilidades."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return torch.softmax(self.m(x), dim=-1)


def n_parametros(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
