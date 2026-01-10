# u_osls.py
# Unified Open-Set & Label-Shift Self-Training (U-OSLS)

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import f1_score, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# -------------------------
# Model
# -------------------------
class UOSLS(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Linear(d, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.cls = nn.Linear(128, 2)      # known classes
        self.ood = nn.Linear(128, 1)      # unknown detector

    def forward(self, x):
        h = self.feat(x)
        return self.cls(h), self.ood(h)

# -------------------------
# Utils
# -------------------------
def energy(logits):
    return -torch.logsumexp(logits, dim=1)

def fit_gmm(scores):
    gmm = GaussianMixture(2, covariance_type="full")
    gmm.fit(scores.reshape(-1,1))
    return gmm

def saerens_em(P, pi0, it=50):
    pi = pi0
    for _ in range(it):
        w = (pi * P[:,1]) / (pi * P[:,1] + (1-pi) * P[:,0] + 1e-8)
        pi = w.mean()
    return pi

# -------------------------
# Training
# -------------------------
def train_uosls(Xs, ys, Xt, Xt_test, yt_test, epochs=20):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UOSLS(Xs.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # source loader
    src_loader = DataLoader(
        TensorDataset(torch.tensor(Xs).float(),
                      torch.tensor(ys).long()),
        batch_size=128, shuffle=True)

    # --- Stage 0 ---
    for _ in range(5):
        for x,y in src_loader:
            x,y = x.to(device), y.to(device)
            logit,_ = model(x)
            loss = F.cross_entropy(logit, y)
            opt.zero_grad(); loss.backward(); opt.step()

    # --- Adaptation ---
    for _ in range(epochs):
        with torch.no_grad():
            logits,_ = model(torch.tensor(Xt).float().to(device))
            P = F.softmax(logits,1).cpu().numpy()
            E = energy(logits).cpu().numpy()

            gmm = fit_gmm(E)
            unk = gmm.predict(E) == gmm.means_.argmax()

        # pseudo known
        idx_k = np.where(~unk)[0]
        Xk = torch.tensor(Xt[idx_k]).float()
        yk = torch.tensor(P[idx_k].argmax(1)).long()

        loader_k = DataLoader(
            TensorDataset(Xk, yk),
            batch_size=128, shuffle=True)

        for x,y in loader_k:
            x,y = x.to(device), y.to(device)
            logit,ood = model(x)
            loss_cls = F.cross_entropy(logit, y)
            loss_ood = F.binary_cross_entropy_with_logits(
                ood.squeeze(), torch.zeros(len(x),device=device))
            loss = loss_cls + loss_ood
            opt.zero_grad(); loss.backward(); opt.step()

        # OE on unknown
        idx_u = np.where(unk)[0]
        if len(idx_u)>0:
            Xu = torch.tensor(Xt[idx_u]).float().to(device)
            logit,_ = model(Xu)
            loss_oe = (F.softmax(logit,1)
                       * torch.log(F.softmax(logit,1)+1e-8)).sum(1).mean()
            opt.zero_grad(); loss_oe.backward(); opt.step()

    # --- Eval ---
    with torch.no_grad():
        x = torch.tensor(Xt_test).float().to(device)
        logit,ood = model(x)
        P = F.softmax(logit,1).cpu().numpy()
        ood_score = torch.sigmoid(ood).cpu().numpy()

    mask = yt_test != -1
    f1 = f1_score(yt_test[mask], P[mask].argmax(1), average="macro")
    auroc = roc_auc_score((yt_test==-1).astype(int), ood_score)

    return f1, auroc
