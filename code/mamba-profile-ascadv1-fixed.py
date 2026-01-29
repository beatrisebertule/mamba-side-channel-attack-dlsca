import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import sys
import h5py

# my package - utilities
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

from lightning.pytorch.loggers import CSVLogger

#Change bi-directionality
#Change convolution

TARGET_BYTE = 2
LEAKAGE_MODEL = "ID"

BATCH_SIZE = 768
NR_EPOCHS = 100
LR = 0.0005 # rerun with lower

CLASSES = 256
D_MODEL = 64
N_LAYERS = 2

FILE_PATH = "./data/ascadv1-fixed.h5"
PATH_MAMBA  = f"./experiments-final/experiments-ascadv1-fixed-key/mamba-d{D_MODEL}-l{N_LAYERS}-four-forw-encoder/"

NB_PROF_TRACES = 40000
NB_VAL_TRACES = 10000


class KeyRankCallback(L.Callback):
    def __init__(
        self,
        val_metadata,
        correct_key,
        leakage_model,
        target_byte,
    ):
        super().__init__()

        self.val_metadata = val_metadata
        self.correct_key = correct_key
        self.leakage_model = leakage_model
        self.target_byte = target_byte

    def on_validation_epoch_end(self, trainer, pl_module): 
        print("\nRunning Key Rank...", flush = True) 
        preds = torch.cat(pl_module.val_predictions, dim = 0)
        preds = torch.nn.functional.softmax(preds, dim=1)
        preds = preds.cpu().float().numpy()

        key_scores = score_keys(preds, val_metadata["plaintext"][:, TARGET_BYTE], LEAKAGE_MODEL)
        order_keys = np.argsort(key_scores)[::-1]
        key_rank , = np.where(order_keys == self.correct_key)
        pl_module.log("val_key_rank", key_rank.item(), prog_bar=True, on_step=False, on_epoch=True)


class GuessingEntropyCallback(L.Callback):
    def __init__(
        self,
        val_metadata,
        correct_key,
        num_traces,
        leakage_model,
        target_byte,
    ):
        super().__init__()

        self.val_metadata = val_metadata
        self.correct_key = correct_key
        self.num_traces = num_traces
        self.leakage_model = leakage_model
        self.target_byte = target_byte

    def on_validation_epoch_end(self, trainer, pl_module): 
        print("\nRunning Guessing Entropy...", flush = True) 
        preds = torch.cat(pl_module.val_predictions, dim = 0)
        preds = torch.nn.functional.softmax(preds, dim=1)
        preds = preds.cpu().float().numpy()
        plaintext = self.val_metadata["plaintext"][:, self.target_byte]
        
        _, ge_mean = guessing_entropy(
            preds,
            plaintext,
            self.correct_key,
            # preds.shape[0],
            1000,
            self.leakage_model,
        )
		
        pl_module.log(f"val_ge_mean", ge_mean, prog_bar=True, on_step=False, on_epoch=True)


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

def plot_raw_traces(traces):
	idx = 0
	raw_trace = traces[idx]
	plt.figure(figsize=(10,4), dpi=300)
	plt.plot(raw_trace, linewidth=0.8, color="#5e548e")
	plt.title("Raw Power Trace ASCADv1 Fixed Key", fontsize=14)
	plt.xlabel("Time Samples", fontsize=12)
	plt.ylabel("Power Consumption (a.u.)", fontsize=12)
	plt.tick_params(axis='both', labelsize=10)
	plt.gca().spines['top'].set_visible(False)
	plt.gca().spines['right'].set_visible(False)
	plt.tight_layout()
	plt.savefig("plots/mamba-raw-ascadv1-variable-trace.pdf")
	
def plot_val_ge():
	log_path = f"{PATH_MAMBA}/logs/version_0/metrics.csv"
	df = pd.read_csv(log_path)
	
	df_ge = df[df["val_ge_mean"].notna()]
	plt.figure(figsize=(6,4))
	plt.plot(df_ge["epoch"], df_ge["val_ge_mean"], marker="o")
	plt.xlabel("Epoch")
	plt.ylabel("Validation Guessing Entropy")
	plt.title("GE Convergence")
	plt.grid(True)
	plt.tight_layout()
	plt.savefig("plots/mamba-d{D_MODEL}-l{N_LAYERS}-ascadv1-fixed-ge-conv-ge-val.pdf")

if __name__ == "__main__":

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
	
	# plot traces
	plot_raw_traces(X_profiling)
	
	# convert labels to one-hot encoded arrays
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

	# create tensors and data loaders
	X_train_tensor = torch.tensor(traces_train, dtype=torch.float32)
	y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

	X_val_tensor = torch.tensor(traces_val, dtype=torch.float32)
	y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

	X_test_tensor = torch.tensor(traces_test, dtype=torch.float32)
	y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

	train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
	val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
	test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

	train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=4, shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)
	test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

	# get model
	input_size = len(traces_train[0])
	mamba = MAMBA_model(channels_in=1, d_model=D_MODEL, num_classes=CLASSES, n_layers=N_LAYERS, learning_rate=LR)
      	
	save = ModelCheckpoint(
					monitor="val_ge_mean",
					mode="min",
					dirpath=PATH_MAMBA,
					filename="mamba-{epoch:02d}-{val_ge_mean:.2f}",
					save_top_k=-1
	)
	
	ge_callback = GuessingEntropyCallback(
		val_metadata = val_metadata,
		correct_key = val_correct_key[0], # 0 because fixed key
        num_traces = NB_VAL_TRACES,
		leakage_model = LEAKAGE_MODEL,
		target_byte = TARGET_BYTE,
	)
      
	kr_callback = KeyRankCallback(
		val_metadata=val_metadata,
		correct_key=val_correct_key[0], # 0 because fixed key
		leakage_model=LEAKAGE_MODEL,
		target_byte=TARGET_BYTE,
	)
      
	csv_logger = CSVLogger(
		save_dir=PATH_MAMBA,
		name="logs"
	)

	seed_everything(83545, workers=True)
	trainer = L.Trainer(
		max_epochs=100,
		callbacks=[save, ge_callback],
            logger=csv_logger,
		accelerator="gpu" if torch.cuda.is_available() else "cpu",
		devices=1
	)

	# train model
	trainer.fit(mamba, train_loader, val_loader)

	# resume train
	# trainer.fit(mamba, train_loader, val_loader, ckpt_path="./experiments/experiments-ascadv1-fixed-key/mamba-d64-l2/mamba-epoch=04-val_ge_mean=63.00.ckpt")
      
	# plot 
	# plot_val_ge()

