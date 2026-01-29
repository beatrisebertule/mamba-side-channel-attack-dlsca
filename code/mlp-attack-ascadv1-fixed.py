import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os
import sys
import h5py

# my package - utilities
from utilities.attack import AES_Sbox, score_keys, guessing_entropy_convergence, guessing_entropy
from network import MLP_model

# torch
import torch
from torch.utils.data import DataLoader, TensorDataset

# pytorch lightning
import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical



TARGET_BYTE = 2
LEAKAGE_MODEL = "ID"

BATCH_SIZE = 512
NR_EPOCHS = 100
LR = 0.0005

CLASSES = 256
NB_PROF_TRACES = 40000
NB_VAL_TRACES = 10000

MLP_DIM = 128

FILE_PATH = "./data/ascadv1-fixed.h5"
PATH_MLP  = f"./experiments-final/experiments-ascadv1-fixed-key/mlp-{MLP_DIM}/"


def check_file_exists(file_path):
    if os.path.exists(file_path) == False:
        print("Error: provided file path '%s' does not exist!" % file_path)
        sys.exit(-1)
    return

def load_ascad(ascad_database_file, load_metadata=False, target_byte=2):
	check_file_exists(ascad_database_file)
	try:
		in_file	 = h5py.File(ascad_database_file, "r")
	except:
		print("Error: can't open HDF5 file '%s' for reading (it might be malformed) ..." % ascad_database_file)
		sys.exit(-1)
	X_profiling = np.array(in_file['Profiling_traces/traces'], dtype=np.int8)
	Y_profiling = np.array(in_file['Profiling_traces/labels'])
	X_attack = np.array(in_file['Attack_traces/traces'], dtype=np.int8)
	Y_attack = np.array(in_file['Attack_traces/labels'])
	if load_metadata == False:
		return (X_profiling, Y_profiling), (X_attack, Y_attack)
	else:
		return (X_profiling, Y_profiling), (X_attack, Y_attack), (in_file['Profiling_traces/metadata'], in_file['Attack_traces/metadata'])

def plot_ge_convergance(avgge):
    plt.figure(figsize=(10, 4), dpi=300)

    plt.plot(avgge, linewidth=1.4, color="#5e548e",
            label=f"Min GE = {np.min(avgge):.2f} at trace {np.argmin(avgge)}")
    plt.xlabel("Number of Traces", fontsize=12)
    plt.ylabel("Guessing Entropy (GE)", fontsize=12)
    plt.title("GE Convergence MLP Model", fontsize=14)
    plt.tick_params(axis='both', labelsize=10)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.grid(True, linestyle="--", linewidth=0.4, alpha=0.6) 
    plt.ylim(0, 256)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(f"plots-final/ascadv1r-mlp-{MLP_DIM}-ge-convergance.pdf")


def save_ge_convergence(avgge, medge, filename):
    df = pd.DataFrame({
        "num_traces": np.arange(1, len(avgge) + 1),
        "ge_mean": avgge,
        "ge_median": medge
    })
    df.to_csv(filename, index=False)
    

if __name__=="__main__":
	
    # load traces
    (X_profiling, Y_profiling), (X_attack, Y_attack), (profiling_metadata, attack_metadata) = load_ascad(FILE_PATH, load_metadata=True, target_byte=TARGET_BYTE)
	
    profiling_plaintexts = profiling_metadata["plaintext"][:, TARGET_BYTE].astype(np.uint8)
    profiling_keys       = profiling_metadata["key"][:, TARGET_BYTE].astype(np.uint8)
	
    attack_plaintexts = attack_metadata["plaintext"][:, TARGET_BYTE].astype(np.uint8)
    attack_keys       = attack_metadata["key"][:, TARGET_BYTE].astype(np.uint8)

    # reshape traces
    traces_val 	 = X_profiling[NB_PROF_TRACES:NB_PROF_TRACES+NB_VAL_TRACES]
    traces_train = X_profiling[:NB_PROF_TRACES]
        
    val_metadata   = profiling_metadata[NB_PROF_TRACES:NB_PROF_TRACES+NB_VAL_TRACES]
    train_metadata = profiling_metadata[:NB_PROF_TRACES]

    # standardize traces
    y_train = to_categorical(Y_profiling, num_classes=CLASSES)
    y_test = to_categorical(Y_attack, num_classes=CLASSES)
        
    y_val = y_train[NB_PROF_TRACES:NB_PROF_TRACES+NB_VAL_TRACES]
    y_train = y_train[:NB_PROF_TRACES]
        
    val_plaintext = val_metadata["plaintext"][:, TARGET_BYTE]
    val_correct_key = val_metadata["key"][:, TARGET_BYTE]
        
    train_plaintext = train_metadata["plaintext"][:, TARGET_BYTE]
    train_correct_key = train_metadata["key"][:, TARGET_BYTE]
    scaler = StandardScaler()
    traces_train = scaler.fit_transform(traces_train)
    traces_val = scaler.transform(traces_val)
    traces_test = scaler.transform(X_attack)

    # create test (attack) tensor and data loader
    X_test_tensor = torch.tensor(traces_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # laod model
    seed_everything(83545, workers=True)
    trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
    )

    input_size = len(X_profiling[0])
    mlp = MLP_model.load_from_checkpoint(f"./experiments-final/experiments-ascadv1-fixed-key/mlp-128/mlp-epoch=15-val_ge_mean=0.00.ckpt", input_dim=input_size, mlp_dim=MLP_DIM, num_classes=CLASSES)
    trainer.test(mlp, dataloaders=test_loader)

    # make predictions
    probs = []
    with torch.no_grad():
        mlp_predictions = trainer.predict(mlp, test_loader)

    for prediction in mlp_predictions:
        prob = torch.nn.functional.softmax(prediction, dim=1)
        probs.append(prob)

    mlp_predictions = torch.cat(probs, dim=0)
    mlp_predictions = mlp_predictions.cpu().float().numpy()

    # recover the key 
    correct_key = attack_metadata["key"][0, TARGET_BYTE] 

    print("\nCalculating Key Rank...")

    key_scores = score_keys(mlp_predictions, attack_metadata["plaintext"][:, TARGET_BYTE], LEAKAGE_MODEL)
    order_keys = np.argsort(key_scores)[::-1]
    key_rank , = np.where(order_keys == correct_key)
    print(f"Key rank: {key_rank}.")

    print("\nCalculating GE...")

    # num_traces = mlp_predictions.shape[0]
    num_attack_traces = 1000
    ge_median, ge_mean = guessing_entropy(mlp_predictions, attack_metadata["plaintext"][:, TARGET_BYTE], correct_key, num_attack_traces, LEAKAGE_MODEL)
    print(f"The median of GE: {ge_median}.")
    print(f"The mean of GE: {ge_mean}.")

    print("\nCalculating GE Convergance...")

    correct_key = attack_metadata["key"][0, TARGET_BYTE] 
    num_attack_traces = 1000
    medge, avgge = guessing_entropy_convergence(mlp_predictions, attack_metadata["plaintext"][:, TARGET_BYTE], correct_key, LEAKAGE_MODEL, num_attack_traces)
    print(f"GE of {np.min(avgge)} with {np.argmin(avgge)} traces.")

    plot_ge_convergance(avgge)
    
    save_ge_convergence(avgge,medge, f"{PATH_MLP}avvge-ascadv1-mlp.csv")

