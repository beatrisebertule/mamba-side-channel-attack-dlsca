
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os
import sys
import h5py

# my package - utilities
from utilities.profile import create_labels
from utilities.attack import AES_Sbox, score_keys, guessing_entropy_convergence, guessing_entropy
from network import MAMBA_model

# torch
import torch
from torch.utils.data import DataLoader, TensorDataset

# pytorch lightning
import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical


FILE_PATH_TRAIN = "../data/chip_whisperer/seeded-train/device_A/dataset_device_A.npy"
FILE_PATH_TEST  = "../data/chip_whisperer/seeded-test/device_A/dataset_device_A.npy"

PATH_MAMBA  = "./experiments-final/experiments-chipwh/"

TARGET_BYTE = 0
WINDOW_SIZE = 50
LEAKAGE_MODEL = "ID"

NB_PROF_TRACES = 40000
NB_VAL_TRACES = 10000

BATCH_SIZE = 512
NR_EPOCHS = 100
LR = 0.0005

CLASSES = 256
D_MODEL = 64
N_LAYERS = 2

def load_data(data):
    trace_idx = data.files[0]
    traces = data[trace_idx]
    plaintext_idx = data.files[1]
    plaintexts = data[plaintext_idx]
    key_idx = data.files[2]
    keys = data[key_idx]
    return traces, plaintexts, keys


def plot_ge_convergance(avgge):
    plt.figure(figsize=(10, 4), dpi=300)

    plt.plot(avgge, linewidth=1, color="#5e548e",
            label=f"Min GE = {np.min(avgge):.2f} at trace {np.argmin(avgge)}")
    plt.xlabel("Number of Traces", fontsize=12)
    plt.ylabel("Guessing Entropy (GE)", fontsize=12)
    plt.title("GE Convergence Mamba Model", fontsize=14)
    plt.tick_params(axis='both', labelsize=10)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.grid(True, linestyle="--", linewidth=0.4, alpha=0.6) 
    plt.ylim(0, 3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(f"plots-final/chipwh-ge-conv-mamba-d{D_MODEL}-l{N_LAYERS}.pdf")

def save_ge_convergence(avgge, medge, filename):
    df = pd.DataFrame({
        "num_traces": np.arange(1, len(avgge) + 1),
        "ge_mean": avgge,
        "ge_median": medge
    })
    df.to_csv(filename, index=False)
    

if __name__=="__main__":
	
 # load data
    data_train = np.load(FILE_PATH_TRAIN)
    data_test = np.load(FILE_PATH_TEST)

    traces_train, plaintexts_train, keys_train = load_data(data_train) 
    traces_test, plaintexts_test, keys_test = load_data(data_test)

    # visualize one trace
    # plot_raw_trace(traces_train)

    # create labels, return target byte labels only
    y_train, y_test = create_labels(plaintexts_train, keys_train, plaintexts_test, keys_test, TARGET_BYTE)

    # select features
    # leakage_model = 'HW'
    # if leakage_model == 'HW':
    #     y_train_new = calculate_HW(y_train) 
    #     y_test_new = calculate_HW(y_test)

    # snr_values_train = snr_trace(traces_train, y_train_new)
    # pois_train = points_of_interets(snr_values_train)
    # selected_features_train = traces_train[:, pois_train]

    loc = 1320 # hardiced location of leakage taken from the function above , 
    # the value of the location stays the same and I am not doing this again bc it is very expensive to compute this.
    traces_train = traces_train[:, loc-WINDOW_SIZE:loc+WINDOW_SIZE]
    traces_test = traces_test[:, loc-WINDOW_SIZE:loc+WINDOW_SIZE]

    # plot_selected_features(X_train)
    traces_val 	 = traces_train[NB_PROF_TRACES:NB_PROF_TRACES+NB_VAL_TRACES]
    traces_train = traces_train[:NB_PROF_TRACES]
	
    # convert labels to one-hot encoded arrays
    y_train = to_categorical(y_train, num_classes=CLASSES)
    y_test = to_categorical(y_test, num_classes=CLASSES)
        
    y_val = y_train[NB_PROF_TRACES:NB_PROF_TRACES+NB_VAL_TRACES]
    y_train = y_train[:NB_PROF_TRACES]
        
    plaintexts_val = plaintexts_train[NB_PROF_TRACES:NB_PROF_TRACES+NB_VAL_TRACES]
    plaintexts_train = plaintexts_train[NB_PROF_TRACES:NB_PROF_TRACES+NB_VAL_TRACES]

    correct_keys_val = keys_train[NB_PROF_TRACES:NB_PROF_TRACES+NB_VAL_TRACES]
    correct_keys_train = keys_train[NB_PROF_TRACES:NB_PROF_TRACES+NB_VAL_TRACES]
    
    scaler = StandardScaler() 
    traces_train = scaler.fit_transform(traces_train)
    traces_val = scaler.transform(traces_val)
    traces_test = scaler.transform(traces_test)

    X_test_tensor = torch.tensor(traces_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    seed_everything(83545, workers=True)
    trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
    )
    
    input_size = len(traces_train[0])
    mamba = MAMBA_model.load_from_checkpoint("./experiments-final/experiments-chipwh/mamba-d64-l2/mamba-epoch=06-val_ge_mean=0.00.ckpt", channels_in=1, d_model=D_MODEL, num_classes=CLASSES, n_layers=N_LAYERS, learning_rate=LR)
    trainer.test(mamba, dataloaders=test_loader)

    # make predictions
    probs = []
    with torch.no_grad():
        mamba_predictions = trainer.predict(mamba, test_loader)

    for prediction in mamba_predictions:
        prob = torch.nn.functional.softmax(prediction, dim=1)
        probs.append(prob)

    mamba_predictions = torch.cat(probs, dim=0)
    mamba_predictions = mamba_predictions.cpu().float().numpy()

    # recover the key 
    nb_traces = mamba_predictions.shape[0]
    correct_key = keys_test[0, TARGET_BYTE] # 0 because we have a fixed key, we get the 1st element from the array

    print("\nCalculating Key Rank...")
    key_scores = score_keys(mamba_predictions, plaintexts_test[:, TARGET_BYTE], LEAKAGE_MODEL)
    order_keys = np.argsort(key_scores)[::-1]
    key_rank , = np.where(order_keys == correct_key)
    print(f"Key rank: {key_rank}.")
    
    print("\nCalculating Guessing Entropy...")
    num_attack_traces = 1000
    ge_median, ge_mean = guessing_entropy(mamba_predictions, plaintexts_test[:, TARGET_BYTE], correct_key, num_attack_traces, LEAKAGE_MODEL)
    print(f"The median of GE: {ge_median}.")
    print(f"The mean of GE: {ge_mean}.")

    print("\nCalculating Guessing Entropy Convergance...")
    medge, avgge = guessing_entropy_convergence(mamba_predictions, plaintexts_test[:, TARGET_BYTE], correct_key, LEAKAGE_MODEL, num_attack_traces)
    print(f"GE of {np.min(avgge)} with {np.argmin(avgge)} traces.")
    plot_ge_convergance(avgge)
    
    save_ge_convergence(avgge,medge, f"{PATH_MAMBA}avvge-chipwh-mamba.csv")


