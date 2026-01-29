import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_ge_convergence_mlp_mamba(
    mlp_csv,
    mamba_csv,
    D_MODEL=64,
    N_LAYERS=2
):
    os.makedirs("plots-final", exist_ok=True)

    # --- Load CSVs ---
    df_mlp = pd.read_csv(mlp_csv)
    df_mamba = pd.read_csv(mamba_csv)

    # --- Extract GE vectors ---
    # If num_traces exists, use it; otherwise use index
    if "num_traces" in df_mlp.columns:
        x_mlp = df_mlp["num_traces"]
    else:
        x_mlp = np.arange(0, len(df_mlp))

    if "num_traces" in df_mamba.columns:
        x_mamba = df_mamba["num_traces"]
    else:
        x_mamba = np.arange(0, len(df_mamba))

    ge_mlp = df_mlp["ge_mean"].values
    ge_mamba = df_mamba["ge_mean"].values
    
    # # --- Only first 5 traces ---
    # x_mlp = x_mlp[:5]
    # ge_mlp = ge_mlp[:5]
    # x_mamba = x_mamba[:5]
    # ge_mamba = ge_mamba[:5]

    # --- Plot ---
    plt.figure(figsize=(10, 4), dpi=300)

    plt.plot(
        x_mlp,
        ge_mlp,
        # alpha=0.5,
        linewidth=1.4,
        color="#c1121f",
        label=f"MLP (min GE = {ge_mlp.min():.2f} with {x_mlp[np.argmin(ge_mlp)]} traces)"
    )

    plt.plot(
        x_mamba,
        ge_mamba,
        alpha=0.7,
        linewidth=1.4,
        color="#5e548e",
        label=f"Mamba (min GE = {ge_mamba.min():.2f} with {x_mamba[np.argmin(ge_mamba)]} traces)"
    )

    plt.xlabel("Number of Traces", fontsize=12)
    plt.ylabel("Guessing Entropy (GE)", fontsize=12)
    plt.title("GE Convergence: MLP vs Mamba (CW-Target)", fontsize=14)

    plt.ylim(-0.2, 3)
    plt.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
    plt.tick_params(axis="both", labelsize=10)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    plt.legend(fontsize=10)
    plt.tight_layout()

    plt.savefig(
        # f"plots-final/ascadv1f-ge-convergence-mlp-vs-mamba-d{D_MODEL}-l{N_LAYERS}.pdf"
        "plots-final/FIN-cw-ge-convergence-mlp-vs-mamba.pdf"
    )
    plt.show()

if __name__=="__main__":
    plot_ge_convergence_mlp_mamba(
    # mlp_csv="./experiments-final/experiments-ascadv1-fixed-key/mlp-128/avvge-ascadv1-mlp.csv",
    # mamba_csv="./experiments-final/experiments-ascadv1-fixed-key/mamba-d64-l2/avvge-ascadv1-mamba.csv"
    mlp_csv="./experiments-final/experiments-chipwh/avvge-chipwh-mamba.csv",
    mamba_csv="./experiments-final/experiments-chipwh/mlp-128avvge-chipwh-mlp.csv"
)
