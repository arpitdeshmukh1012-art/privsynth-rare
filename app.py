import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.decomposition import PCA

import core

st.set_page_config(page_title="PrivSynth-Rare", page_icon="🩺", layout="wide")

CONFIGS = {"Non-DP (no privacy)": None, "DP ε = 10": 10.0, "DP ε = 5": 5.0, "DP ε = 1": 1.0}
BLUE, ORANGE, GREEN, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8985"


@st.cache_resource(show_spinner=False)
def get_data(seed):
    return core.load_data(seed)


@st.cache_resource(show_spinner=False)
def get_baseline(seed):
    return core.real_baseline(get_data(seed))


@st.cache_resource(show_spinner=False)
def get_run(seed, eps, epochs, n_syn):
    return core.run_config(get_data(seed), eps, epochs, n_syn)


def fig_ax(w=5, h=3.4):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.3)
    return fig, ax


# ---------------- Sidebar ----------------
st.sidebar.title("Settings")
choice = st.sidebar.selectbox("Privacy setting", list(CONFIGS) + ["Custom ε"])
if choice == "Custom ε":
    eps = float(st.sidebar.slider("Target ε (lower = more private)", 0.5, 20.0, 3.0, 0.5))
else:
    eps = CONFIGS[choice]
epochs = st.sidebar.select_slider("Training epochs", [50, 100, 200], value=200,
                                  help="200 matches the report. Lower = faster on the free cloud tier.")
n_syn = st.sidebar.select_slider("Synthetic records per set", [500, 1000, 2000], value=2000)
seed = int(st.sidebar.number_input("Random seed", 0, 9999, 42))
st.sidebar.caption("δ = 1e-5 · clipping norm C = 1.0 · batch 64 · lr 2e-3")

# ---------------- Header ----------------
st.title("PrivSynth-Rare")
st.markdown("**Differentially private synthetic medical data for a rare disease class.** "
            "A conditional VAE is trained with DP-SGD (Opacus) on a rare-class version of the Wisconsin "
            "breast-cancer dataset, then scored on **fidelity**, **privacy** and **usefulness** against real, held-out patients.")

d = get_data(seed)
tab_data, tab_run, tab_cmp = st.tabs(["📊 Dataset", "🔬 Train & evaluate", "⚖️ Compare all privacy levels"])

# ---------------- Dataset tab ----------------
with tab_data:
    c = st.columns(4)
    c[0].metric("Records", len(d["data"]))
    c[1].metric("Rare (malignant)", int(d["data"].target.sum()), f"{d['data'].target.mean():.1%}", delta_color="off")
    c[2].metric("Train / test", f"{len(d['Ztr'])} / {len(d['Zte'])}")
    c[3].metric("Features", d["D"])
    st.markdown("All 357 benign records are kept and only 60 malignant records are sampled, so malignant "
                "becomes a rare class (14.4%). The generator never sees the test set.")
    st.dataframe(d["data"].groupby("target")[d["cols"][:8]].mean().rename(index={0: "Benign", 1: "Malignant"}).round(2),
                 use_container_width=True)

# ---------------- Single run tab ----------------
with tab_run:
    label = "no privacy" if eps is None else f"target ε = {eps:g}"
    st.subheader(f"Run: {label}, {epochs} epochs")
    if st.button("▶ Train & evaluate", type="primary"):
        st.session_state["go"] = (seed, eps, epochs, n_syn)
    if st.session_state.get("go") == (seed, eps, epochs, n_syn):
        with st.spinner("Training the CVAE and evaluating (≈20–60 s on the free tier)…"):
            r = get_run(seed, eps, epochs, n_syn)
            base = get_baseline(seed)

        st.markdown("#### Privacy")
        c = st.columns(4)
        c[0].metric("ε spent", "∞" if r["eps_spent"] == float("inf") else f"{r['eps_spent']:.2f}")
        c[1].metric("Membership-attack AUC", f"{r['MIA_AUC']:.3f}", help="0.5 = attacker is guessing")
        c[2].metric("Median DCR", f"{r['median_DCR']:.2f}", help="Distance from real records to the nearest synthetic one")
        c[3].metric("Copy rate", f"{r['copy_rate']:.1%}")
        st.markdown("#### Fidelity & usefulness")
        c = st.columns(4)
        c[0].metric("Mean KS ↓", f"{r['KS']:.3f}")
        c[1].metric("Correlation Δ ↓", f"{r['corr_diff']:.3f}")
        c[2].metric("TSTR AUROC", f"{r['tstr']['auroc']:.3f}", f"{r['tstr']['auroc'] - base['auroc']:+.3f} vs real")
        c[3].metric("Real+synthetic AUROC", f"{r['aug']['auroc']:.3f}", f"{r['aug']['auroc'] - base['auroc']:+.3f} vs real")

        tbl = pd.DataFrame({k: {"AUROC": v["auroc"], "Avg precision": v["ap"], "Recall": v["recall"],
                                "Precision": v["precision"], "F1": v["f1"]}
                            for k, v in [("Real only", base), ("Train on synthetic (TSTR)", r["tstr"]),
                                         ("Real + synthetic (AUG)", r["aug"])]}).T.round(3)
        st.dataframe(tbl, use_container_width=True)

        g1, g2 = st.columns(2)
        with g1:
            fig, ax = fig_ax()
            ax.plot(range(1, len(r["losses"]) + 1), r["losses"], color=BLUE, lw=2)
            ax.set_xlabel("Epoch"); ax.set_ylabel("Training loss (recon + β·KL)"); ax.set_title("Training loss curve")
            st.pyplot(fig)
        with g2:
            pca = PCA(2).fit(d["Ztr"])
            fig, axs = plt.subplots(1, 2, figsize=(6, 3.4), sharex=True, sharey=True)
            for a, (t, Z, yy) in zip(axs, [("Real train", d["Ztr"], d["ytr"]), ("Synthetic", *r["natural"])]):
                P = pca.transform(Z)
                a.scatter(P[yy == 0, 0], P[yy == 0, 1], s=6, alpha=0.4, c=BLUE, label="benign")
                a.scatter(P[yy == 1, 0], P[yy == 1, 1], s=10, alpha=0.8, c=ORANGE, label="malignant")
                a.set_title(t)
            axs[0].legend(fontsize=8)
            plt.tight_layout(); st.pyplot(fig)

        g1, g2, g3 = st.columns(3)
        with g1:
            fig, ax = fig_ax(4, 3.4)
            for nm, v, col in [("Real only", base, GRAY), ("TSTR", r["tstr"], BLUE), ("Real+syn", r["aug"], ORANGE)]:
                fpr, tpr, _ = v["roc"]
                ax.plot(fpr, tpr, color=col, lw=2, label=f"{nm} ({v['auroc']:.3f})")
            ax.plot([0, 1], [0, 1], "--", color=GRAY, lw=1)
            ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.set_title("ROC curve")
            ax.legend(fontsize=8); st.pyplot(fig)
        with g2:
            fig, ax = fig_ax(4, 3.4)
            for nm, v, col in [("Real only", base, GRAY), ("TSTR", r["tstr"], BLUE), ("Real+syn", r["aug"], ORANGE)]:
                p, rc, _ = v["pr_curve"]
                ax.plot(rc, p, color=col, lw=2, label=f"{nm} (AP {v['ap']:.3f})")
            ax.axhline(d["yte"].mean(), ls="--", color=GRAY, lw=1)
            ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision–recall curve")
            ax.legend(fontsize=8); st.pyplot(fig)
        with g3:
            fig, axs = plt.subplots(1, 2, figsize=(4.4, 2.6))
            for a, (nm, v) in zip(axs, [("TSTR", r["tstr"]), ("Real+syn", r["aug"])]):
                cm = v["cm"]
                a.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
                for (i, j), val in np.ndenumerate(cm):
                    a.text(j, i, val, ha="center", va="center", color="white" if cm[i, j] / cm[i].sum() > 0.55 else "black")
                a.set_xticks([0, 1], ["Pred 0", "Pred 1"], fontsize=7); a.set_yticks([0, 1], ["True 0", "True 1"], fontsize=7)
                a.set_title(nm, fontsize=9)
            plt.tight_layout(); st.pyplot(fig)

        st.download_button("⬇ Download synthetic data (balanced, original units)",
                           r["syn_df"].to_csv(index=False).encode(), f"synthetic_eps_{label.replace(' ', '')}.csv", "text/csv")
    else:
        st.info("Pick a privacy setting in the sidebar, then press **Train & evaluate**.")

# ---------------- Comparison tab ----------------
with tab_cmp:
    st.subheader("Privacy–utility trade-off across all four settings")
    st.caption("Trains four models (≈1–3 min on the free tier). Results are cached afterwards.")
    if st.button("▶ Run full comparison"):
        st.session_state["cmp"] = (seed, epochs, n_syn)
    if st.session_state.get("cmp") == (seed, epochs, n_syn):
        base = get_baseline(seed)
        rows, runs = [], {}
        bar = st.progress(0.0, "Training…")
        for i, (nm, e) in enumerate(CONFIGS.items()):
            runs[nm] = get_run(seed, e, epochs, n_syn)
            bar.progress((i + 1) / len(CONFIGS), f"Done: {nm}")
        bar.empty()
        rows.append({"Model": "Real only", "ε spent": None, "KS": None, "Corr Δ": None, "MIA AUC": None,
                     "TSTR AUROC": base["auroc"], "TSTR F1": base["f1"], "AUG AUROC": None, "AUG F1": None})
        for nm, r in runs.items():
            rows.append({"Model": nm, "ε spent": r["eps_spent"], "KS": r["KS"], "Corr Δ": r["corr_diff"],
                         "MIA AUC": r["MIA_AUC"], "TSTR AUROC": r["tstr"]["auroc"], "TSTR F1": r["tstr"]["f1"],
                         "AUG AUROC": r["aug"]["auroc"], "AUG F1": r["aug"]["f1"]})
        res = pd.DataFrame(rows)
        st.dataframe(res.round(3), use_container_width=True, hide_index=True)
        st.download_button("⬇ Download results.csv", res.to_csv(index=False).encode(), "results.csv", "text/csv")

        dp = [(r["eps_spent"], r) for nm, r in runs.items() if r["eps_target"] is not None]
        dp.sort(key=lambda t: t[0])
        xs = [e for e, _ in dp]
        g1, g2, g3 = st.columns(3)
        with g1:
            fig, ax = fig_ax(4, 3.4)
            ax.plot(xs, [r["tstr"]["auroc"] for _, r in dp], "o-", color=BLUE, lw=2, label="TSTR")
            ax.plot(xs, [r["aug"]["auroc"] for _, r in dp], "s-", color=ORANGE, lw=2, label="Real + synthetic")
            ax.axhline(base["auroc"], ls="--", color=GRAY, label="Real only")
            ax.set_xlabel("ε spent"); ax.set_ylabel("AUROC"); ax.set_title("Utility vs privacy budget"); ax.legend(fontsize=8)
            st.pyplot(fig)
        with g2:
            fig, ax = fig_ax(4, 3.4)
            ax.plot(xs, [r["MIA_AUC"] for _, r in dp], "o-", color=ORANGE, lw=2, label="DP-CVAE")
            ax.axhline(runs["Non-DP (no privacy)"]["MIA_AUC"], ls=":", color=GREEN, label="Non-DP")
            ax.axhline(0.5, ls="--", color=GRAY, label="Random guess")
            ax.set_xlabel("ε spent"); ax.set_ylabel("Attack AUC"); ax.set_title("Membership-inference risk"); ax.legend(fontsize=8)
            st.pyplot(fig)
        with g3:
            fig, ax = fig_ax(4, 3.4)
            for (nm, r), col in zip(runs.items(), [GREEN, BLUE, ORANGE, "#6250d6"]):
                ax.plot(r["losses"], color=col, lw=1.6, label=nm.replace(" (no privacy)", ""))
            ax.set_xlabel("Epoch"); ax.set_ylabel("Training loss"); ax.set_title("Loss curves"); ax.legend(fontsize=8)
            st.pyplot(fig)

        fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
        for a, key, ttl in [(axs[0], "tstr", "ROC — train on synthetic"), (axs[1], "aug", "ROC — real + synthetic")]:
            a.plot(*base["roc"][:2], color=GRAY, lw=2, label=f"Real only ({base['auroc']:.3f})")
            for (nm, r), col in zip(runs.items(), [GREEN, BLUE, ORANGE, "#6250d6"]):
                a.plot(*r[key]["roc"][:2], color=col, lw=1.6, label=f"{nm.replace(' (no privacy)', '')} ({r[key]['auroc']:.3f})")
            a.plot([0, 1], [0, 1], "--", color=GRAY, lw=1); a.set_title(ttl); a.set_xlabel("FPR"); a.set_ylabel("TPR")
            a.legend(fontsize=7); a.spines[["top", "right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig)

st.divider()
st.caption("Note: the DP guarantee covers CVAE training only; scaler statistics and clipping bounds are computed from "
           "raw training data. Educational project, not for clinical use.")
