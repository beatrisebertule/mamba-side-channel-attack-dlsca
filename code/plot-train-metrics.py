import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_val_ge_epoch_mlp_mamba(
    mlp_csv,
    mamba_csv,
    D_MODEL=64,
    N_LAYERS=2
):
    os.makedirs("plots-final", exist_ok=True)

    df_mlp = pd.read_csv(mlp_csv)
    df_mamba = pd.read_csv(mamba_csv)

    df_mlp = df_mlp[df_mlp["val_ge_mean"].notna()]
    df_mamba = df_mamba[df_mamba["val_ge_mean"].notna()]

    epoch_mlp = df_mlp["epoch"].values
    ge_mlp = df_mlp["val_ge_mean"].values

    epoch_mamba = df_mamba["epoch"].values
    ge_mamba = df_mamba["val_ge_mean"].values

    plt.figure(figsize=(10, 4), dpi=300)

    # MLP – red
    plt.plot(
        epoch_mlp,
        ge_mlp,
        linewidth=1.4,
        # marker="o",
        color="#c1121f",
        label=f"MLP (min GE = {ge_mlp.min():.2f} at epoch {epoch_mlp[np.argmin(ge_mlp)]})"
    )

    # Mamba – purple
    plt.plot(
        epoch_mamba,
        ge_mamba,
        alpha=0.7,
        linewidth=1.4,
        # marker="o",
        color="#5e548e",
        label=f"Mamba (min GE = {ge_mamba.min():.2f} at epoch {epoch_mamba[np.argmin(ge_mamba)]})"
    )

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Validation Guessing Entropy (GE)", fontsize=12)
    plt.title("Validation GE during Training (CW-Target)", fontsize=14)
    plt.ylim(-0.2, 5)
    plt.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
    plt.tick_params(axis="both", labelsize=10)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    plt.legend(fontsize=10)
    plt.tight_layout()

    plt.savefig(
        "plots-final/FIN-cw-target-val-ge-epoch-mlp-vs-mamba.pdf"
    )
    plt.show()

if __name__=="__main__":
    plot_val_ge_epoch_mlp_mamba(
        # mlp_csv="./experiments-final/experiments-ascadv1-fixed-key/mlp-128/logs/version_0/metrics.csv",
        # mamba_csv="./experiments-final/experiments-ascadv1-fixed-key/mamba-d64-l2/logs/version_0/metrics.csv"
        mlp_csv="./experiments-final/experiments-chipwh/mlp-128/logs/version_0/metrics.csv",
        mamba_csv="./experiments-final/experiments-chipwh/mamba-d64-l2/logs/version_0/metrics.csv"
    )
