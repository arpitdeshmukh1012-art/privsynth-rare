"""Core pipeline: same logic as the Colab notebook / main.py, plus loss logging and ROC/PR outputs."""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import ks_2samp
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, recall_score, f1_score, precision_score,
                             pairwise_distances, roc_curve, precision_recall_curve,
                             average_precision_score, confusion_matrix)
from torch.utils.data import DataLoader, TensorDataset

BETA = 0.5
DELTA = 1e-5


def load_data(seed=42, n_rare=60):
    raw = load_breast_cancer(as_frame=True).frame.copy()
    raw["target"] = 1 - raw["target"]                      # 1 = malignant (rare)
    benign = raw[raw.target == 0]
    rare = raw[raw.target == 1].sample(n_rare, random_state=seed)
    data = pd.concat([benign, rare]).sample(frac=1, random_state=seed).reset_index(drop=True)
    cols = [c for c in data.columns if c != "target"]
    X = data[cols].values.astype(np.float32)
    y = data["target"].values.astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    sc = StandardScaler().fit(Xtr)
    Ztr = sc.transform(Xtr).astype(np.float32)
    Zte = sc.transform(Xte).astype(np.float32)
    return dict(data=data, cols=cols, Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte, sc=sc,
                Ztr=Ztr, Zte=Zte, lo=Ztr.min(0), hi=Ztr.max(0), D=Ztr.shape[1], seed=seed)


class CVAE(nn.Module):
    def __init__(self, d, k=2, h=64, z=8):
        super().__init__()
        self.d, self.k, self.z = d, k, z
        self.e1 = nn.Linear(d + k, h)
        self.e2 = nn.Linear(h, h)
        self.mu = nn.Linear(h, z)
        self.lv = nn.Linear(h, z)
        self.d1 = nn.Linear(z + k, h)
        self.d2 = nn.Linear(h, h)
        self.out = nn.Linear(h, d)

    def decode(self, z, yo):
        h = F.relu(self.d1(torch.cat([z, yo], 1)))
        h = F.relu(self.d2(h))
        return self.out(h)

    def forward(self, xy):
        yo = xy[:, self.d:]
        h = F.relu(self.e1(xy))
        h = F.relu(self.e2(h))
        mu = self.mu(h)
        lv = self.lv(h).clamp(-8, 8)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        return self.decode(z, yo), mu, lv


def loss_fn(out, xy, D):
    rec, mu, lv = out
    x = xy[:, :D]
    r = ((rec - x) ** 2).sum(1)
    kl = -0.5 * (1 + lv - mu ** 2 - lv.exp()).sum(1)
    return (r + BETA * kl).mean()


def train(d, eps=None, epochs=200, bs=64, lr=2e-3, clip=1.0, progress=None):
    """Train the CVAE. eps=None -> non-private. Returns (model, eps_spent, per-epoch loss list)."""
    torch.manual_seed(d["seed"])
    D = d["D"]
    net = CVAE(D)
    xy = torch.tensor(np.hstack([d["Ztr"], np.eye(2, dtype=np.float32)[d["ytr"]]]))
    dl = DataLoader(TensorDataset(xy), batch_size=bs, shuffle=True)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    model, pe = net, None
    if eps is not None:
        from opacus import PrivacyEngine
        pe = PrivacyEngine()
        model, opt, dl = pe.make_private_with_epsilon(
            module=net, optimizer=opt, data_loader=dl, epochs=epochs,
            target_epsilon=eps, target_delta=DELTA, max_grad_norm=clip)
    losses = []
    for ep in range(epochs):
        tot, n = 0.0, 0
        for (b,) in dl:
            if b.shape[0] == 0:
                continue
            opt.zero_grad()
            loss = loss_fn(model(b), b, D)
            loss.backward()
            opt.step()
            tot += loss.item() * b.shape[0]
            n += b.shape[0]
        losses.append(tot / max(n, 1))
        if progress:
            progress((ep + 1) / epochs)
    spent = pe.get_epsilon(DELTA) if pe else float("inf")
    return net, spent, losses


def synth(d, net, n, p_rare, seed=0):
    net.eval()
    g = torch.Generator().manual_seed(d["seed"] + seed)
    with torch.no_grad():
        yy = (torch.rand(n, generator=g) < p_rare).long()
        z = torch.randn(n, net.z, generator=g)
        x = net.decode(z, F.one_hot(yy, 2).float()).numpy()
    return np.clip(x, d["lo"], d["hi"]), yy.numpy()


def clf_eval(d, Xa, ya):
    m = RandomForestClassifier(300, random_state=d["seed"], n_jobs=-1).fit(Xa, ya)
    p = m.predict_proba(d["Zte"])[:, 1]
    pr = (p >= 0.5).astype(int)
    yte = d["yte"]
    return dict(auroc=roc_auc_score(yte, p), recall=recall_score(yte, pr), f1=f1_score(yte, pr),
                precision=precision_score(yte, pr, zero_division=0), ap=average_precision_score(yte, p),
                roc=roc_curve(yte, p), pr_curve=precision_recall_curve(yte, p),
                cm=confusion_matrix(yte, pr, labels=[0, 1]))


def fidelity(d, syn):
    Ztr = d["Ztr"]
    ks_each = np.array([ks_2samp(Ztr[:, j], syn[:, j]).statistic for j in range(d["D"])])
    cd = np.abs(np.corrcoef(Ztr.T) - np.corrcoef(syn.T)).mean()
    return ks_each.mean(), cd, ks_each


def privacy(d, syn):
    Ztr, Zte = d["Ztr"], d["Zte"]
    m = min(len(Ztr), len(Zte))
    idx = np.random.RandomState(d["seed"]).choice(len(Ztr), m, replace=False)
    dm = pairwise_distances(Ztr[idx], syn).min(1)
    dn = pairwise_distances(Zte[:m], syn).min(1)
    auc = roc_auc_score(np.r_[np.ones(m), np.zeros(m)], np.r_[-dm, -dn])
    return auc, float(np.median(dm)), float((dm < 0.05).mean())


def real_baseline(d):
    return clf_eval(d, d["Ztr"], d["ytr"])


def run_config(d, eps=None, epochs=200, n_syn=2000, progress=None):
    """Full experiment for one privacy setting (identical to one loop iteration of main.py)."""
    ytr = d["ytr"]
    net, spent, losses = train(d, eps, epochs=epochs, progress=progress)
    sn, yn = synth(d, net, n_syn, ytr.mean(), 1)
    sb, yb = synth(d, net, n_syn, 0.5, 2)
    ks, cd, ks_each = fidelity(d, sn)
    mia, dcr, cp = privacy(d, sn)
    tstr = clf_eval(d, sb, yb)
    need = int((ytr == 0).sum() - (ytr == 1).sum())
    ax = np.vstack([d["Ztr"], sb[yb == 1][:need]])
    ay = np.r_[ytr, np.ones(min(need, int((yb == 1).sum())), dtype=int)]
    aug = clf_eval(d, ax, ay)
    syn_df = pd.DataFrame(d["sc"].inverse_transform(sb), columns=d["cols"])
    syn_df["malignant"] = yb
    return dict(eps_target=eps, eps_spent=spent, losses=losses, natural=(sn, yn), balanced=(sb, yb),
                KS=ks, corr_diff=cd, ks_each=ks_each, MIA_AUC=mia, median_DCR=dcr, copy_rate=cp,
                tstr=tstr, aug=aug, syn_df=syn_df)
