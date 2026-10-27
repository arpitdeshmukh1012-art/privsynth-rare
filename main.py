import os, shutil, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, recall_score, f1_score, pairwise_distances
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine

warnings.filterwarnings("ignore")
SEED = int(os.environ.get("SEED", 42))
EPOCHS = int(os.environ.get("EPOCHS", 200))
N_SYN = int(os.environ.get("N_SYN", 2000))
BETA = 0.5
DELTA = 1e-5
torch.manual_seed(SEED)
np.random.seed(SEED)

raw = load_breast_cancer(as_frame=True).frame.copy()
raw["target"] = 1 - raw["target"]
benign = raw[raw.target == 0]
rare = raw[raw.target == 1].sample(60, random_state=SEED)
data = pd.concat([benign, rare]).sample(frac=1, random_state=SEED).reset_index(drop=True)
cols = [c for c in data.columns if c != "target"]
X = data[cols].values.astype(np.float32)
y = data["target"].values.astype(int)

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
sc = StandardScaler().fit(Xtr)
Ztr = sc.transform(Xtr).astype(np.float32)
Zte = sc.transform(Xte).astype(np.float32)
lo, hi = Ztr.min(0), Ztr.max(0)
D = Ztr.shape[1]
print(f"train={len(Ztr)} test={len(Zte)} rare_train={ytr.sum()} ({ytr.mean():.1%}) rare_test={yte.sum()}")


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


def loss_fn(out, xy):
    rec, mu, lv = out
    x = xy[:, :D]
    r = ((rec - x) ** 2).sum(1)
    kl = -0.5 * (1 + lv - mu ** 2 - lv.exp()).sum(1)
    return (r + BETA * kl).mean()


def train(eps=None, bs=64, lr=2e-3, clip=1.0):
    torch.manual_seed(SEED)
    net = CVAE(D)
    xy = torch.tensor(np.hstack([Ztr, np.eye(2, dtype=np.float32)[ytr]]))
    dl = DataLoader(TensorDataset(xy), batch_size=bs, shuffle=True)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    model, pe = net, None
    if eps is not None:
        pe = PrivacyEngine()
        model, opt, dl = pe.make_private_with_epsilon(
            module=net, optimizer=opt, data_loader=dl, epochs=EPOCHS,
            target_epsilon=eps, target_delta=DELTA, max_grad_norm=clip)
    for _ in range(EPOCHS):
        for (b,) in dl:
            if b.shape[0] == 0:
                continue
            opt.zero_grad()
            loss_fn(model(b), b).backward()
            opt.step()
    spent = pe.get_epsilon(DELTA) if pe else float("inf")
    return net, spent


def synth(net, n, p_rare, seed=0):
    net.eval()
    g = torch.Generator().manual_seed(SEED + seed)
    with torch.no_grad():
        yy = (torch.rand(n, generator=g) < p_rare).long()
        z = torch.randn(n, net.z, generator=g)
        x = net.decode(z, F.one_hot(yy, 2).float()).numpy()
    return np.clip(x, lo, hi), yy.numpy()


def clf_eval(Xa, ya):
    m = RandomForestClassifier(300, random_state=SEED, n_jobs=-1).fit(Xa, ya)
    p = m.predict_proba(Zte)[:, 1]
    pr = (p >= 0.5).astype(int)
    return roc_auc_score(yte, p), recall_score(yte, pr), f1_score(yte, pr)


def fidelity(syn):
    ks = np.mean([ks_2samp(Ztr[:, j], syn[:, j]).statistic for j in range(D)])
    cd = np.abs(np.corrcoef(Ztr.T) - np.corrcoef(syn.T)).mean()
    return ks, cd


def privacy(syn):
    m = min(len(Ztr), len(Zte))
    idx = np.random.RandomState(SEED).choice(len(Ztr), m, replace=False)
    dm = pairwise_distances(Ztr[idx], syn).min(1)
    dn = pairwise_distances(Zte[:m], syn).min(1)
    auc = roc_auc_score(np.r_[np.ones(m), np.zeros(m)], np.r_[-dm, -dn])
    return auc, float(np.median(dm)), float((dm < 0.05).mean())


rows = []
b = clf_eval(Ztr, ytr)
rows.append({"model": "Real_only(TRTR)", "epsilon": np.nan, "KS": np.nan, "corr_diff": np.nan,
             "MIA_AUC": np.nan, "median_DCR": np.nan, "copy_rate": np.nan,
             "TSTR_AUROC": b[0], "TSTR_rare_recall": b[1], "TSTR_rare_F1": b[2],
             "AUG_AUROC": np.nan, "AUG_rare_recall": np.nan, "AUG_rare_F1": np.nan})

configs = [("NonDP", None), ("DP_eps10", 10), ("DP_eps5", 5), ("DP_eps1", 1)]
nat_store, bal_store = {}, {}
need = int((ytr == 0).sum() - (ytr == 1).sum())

for name, eps in configs:
    net, spent = train(eps)
    sn, yn = synth(net, N_SYN, ytr.mean(), 1)
    sb, yb = synth(net, N_SYN, 0.5, 2)
    ks, cd = fidelity(sn)
    mia, dcr, cp = privacy(sn)
    t = clf_eval(sb, yb)
    ax = np.vstack([Ztr, sb[yb == 1][:need]])
    ay = np.r_[ytr, np.ones(min(need, int((yb == 1).sum())), dtype=int)]
    a = clf_eval(ax, ay)
    nat_store[name], bal_store[name] = (sn, yn), (sb, yb)
    rows.append({"model": name, "epsilon": spent, "KS": ks, "corr_diff": cd,
                 "MIA_AUC": mia, "median_DCR": dcr, "copy_rate": cp,
                 "TSTR_AUROC": t[0], "TSTR_rare_recall": t[1], "TSTR_rare_F1": t[2],
                 "AUG_AUROC": a[0], "AUG_rare_recall": a[1], "AUG_rare_F1": a[2]})
    print(f"done {name} eps={spent:.2f}")

res = pd.DataFrame(rows).round(3)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
print(res.to_string(index=False))

os.makedirs("out", exist_ok=True)
res.to_csv("out/results.csv", index=False)
for name, _ in configs:
    sb, yb = bal_store[name]
    df = pd.DataFrame(sc.inverse_transform(sb), columns=cols)
    df["malignant"] = yb
    df.to_csv(f"out/synthetic_{name}.csv", index=False)
pd.DataFrame(Xtr, columns=cols).assign(malignant=ytr).to_csv("out/real_train.csv", index=False)
pd.DataFrame(Xte, columns=cols).assign(malignant=yte).to_csv("out/real_test.csv", index=False)

pca = PCA(2).fit(Ztr)
fig, axs = plt.subplots(1, 5, figsize=(22, 4), sharex=True, sharey=True)
panels = [("Real train", Ztr, ytr)] + [(n, *nat_store[n]) for n, _ in configs]
for ax, (title, Z, yy) in zip(axs, panels):
    P = pca.transform(Z)
    ax.scatter(P[yy == 0, 0], P[yy == 0, 1], s=6, alpha=0.4, label="benign")
    ax.scatter(P[yy == 1, 0], P[yy == 1, 1], s=10, alpha=0.8, label="rare")
    ax.set_title(title)
axs[0].legend()
plt.tight_layout()
plt.savefig("out/pca_comparison.png", dpi=130)
plt.close()

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
d = res[res.model.str.startswith("DP_")].copy()
d["epsilon"] = d["epsilon"].astype(float)
d = d.sort_values("epsilon")
ax[0].plot(d.epsilon, d.TSTR_AUROC, "o-", label="TSTR AUROC")
ax[0].plot(d.epsilon, d.AUG_AUROC, "s-", label="Real+synthetic AUROC")
ax[0].axhline(res.iloc[0].TSTR_AUROC, ls="--", c="gray", label="Real only")
ax[0].set_xlabel("epsilon")
ax[0].set_title("Utility vs privacy budget")
ax[0].legend()
ax[1].plot(d.epsilon, d.MIA_AUC, "o-", c="red")
ax[1].axhline(0.5, ls="--", c="gray")
ax[1].set_xlabel("epsilon")
ax[1].set_title("Membership inference AUC (0.5 = safe)")
plt.tight_layout()
plt.savefig("out/privacy_utility_tradeoff.png", dpi=130)
plt.close()

shutil.make_archive("synthetic_medical_project", "zip", "out")
print("saved: out/ and synthetic_medical_project.zip")
