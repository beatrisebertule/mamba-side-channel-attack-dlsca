import numpy as np
import matplotlib.pyplot as plt

import os
import sys
import h5py

# my package - utilities
from utilities.profile import create_labels, calculate_HW, snr_trace, points_of_interets
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


FILE_PATH_TRAIN = "../data/chip_whisperer/seeded-train/device_A/dataset_device_A.npy"
FILE_PATH_TEST  = "../data/chip_whisperer/seeded-test/device_A/dataset_device_A.npy"

TARGET_BYTE = 0
WINDOW_SIZE = 50
LEAKAGE_MODEL = "ID"

NB_PROF_TRACES = 40000
NB_VAL_TRACES = 10000

BATCH_SIZE = 512
NR_EPOCHS = 100
LR = 0.0005

CLASSES = 256
MLP_DIM = 128

PATH_MLP  = f"./experiments-final/experiments-chipwh/mlp-{MLP_DIM}"

class KeyRankCallback(L.Callback):
    def __init__(
        self,
        plaintexts,
        correct_key,
        leakage_model,
        target_byte,
    ):
        super().__init__()

        self.plaintexts = plaintexts
        self.correct_key = correct_key
        self.leakage_model = leakage_model
        self.target_byte = target_byte

    def on_validation_epoch_end(self, trainer, pl_module): 
        print("Running Key Rank...", flush = True) 
        preds = torch.cat(pl_module.val_predictions, dim = 0)
        preds = torch.nn.functional.softmax(preds, dim=1)
        preds = preds.cpu().float().numpy()
        plaintexts = self.plaintexts

        key_scores = score_keys(preds, plaintexts, LEAKAGE_MODEL)
        order_keys = np.argsort(key_scores)[::-1]
        key_rank , = np.where(order_keys == self.correct_key)
        pl_module.log("val_key_rank", key_rank.item(), prog_bar=True, on_step=False, on_epoch=True)


class GuessingEntropyCallback(L.Callback):
    def __init__(
        self,
        plaintexts,
        correct_key,
        num_traces,
        leakage_model,
        target_byte,
    ):
        super().__init__()

        self.plaintexts = plaintexts
        self.correct_key = correct_key
        self.num_traces = num_traces
        self.leakage_model = leakage_model
        self.target_byte = target_byte

    def on_validation_epoch_end(self, trainer, pl_module): 
        print("\nRunning Guessing Entropy...", flush = True) 
        preds = torch.cat(pl_module.val_predictions, dim = 0)
        preds = torch.nn.functional.softmax(preds, dim=1)
        preds = preds.cpu().float().numpy()
        
        _, ge_mean = guessing_entropy(
            preds,
            self.plaintexts,
            self.correct_key,
            1000,
            self.leakage_model,
        )
		
        pl_module.log(f"val_ge_mean", ge_mean, prog_bar=True, on_step=False, on_epoch=True)


def load_data(data):
    trace_idx = data.files[0]
    traces = data[trace_idx]
    plaintext_idx = data.files[1]
    plaintexts = data[plaintext_idx]
    key_idx = data.files[2]
    keys = data[key_idx]
    return traces, plaintexts, keys

def plot_raw_trace(traces):
    idx = 0
    trace = traces[idx]

    plt.figure(figsize=(10, 4), dpi=300)
    plt.plot(trace, linewidth=0.8, color="#5e548e") # dark purple 
    plt.title("Raw Power Trace ChipWhisperer", fontsize=14) 
    plt.xlabel("Time Samples", fontsize=12) 
    plt.ylabel("Power Consumption (a.u.)", fontsize=12) 
    plt.tick_params(axis='both', labelsize=10) 
    plt.gca().spines['top'].set_visible(False) 
    plt.gca().spines['right'].set_visible(False) 
    plt.tight_layout() 
    # plt.show()
    plt.savefig("plots/raw-chipwh-trace.pdf")

def plot_selected_features(traces):
    idx = 0
    trace = traces[idx]

    plt.figure(figsize=(10, 4), dpi=300)
    plt.plot(trace, linewidth=3, color="#5e548e") # dark purple 
    # plt.title("Raw Power Trace", fontsize=14) 
    # plt.xlabel("Time Samples", fontsize=12) 
    # plt.ylabel("Power Consumption (a.u.)", fontsize=12) 
    plt.tick_params(axis='both', labelsize=10) 
    plt.gca().spines['top'].set_visible(False) 
    plt.gca().spines['right'].set_visible(False) 
    plt.tight_layout() 
    plt.savefig("plots/chipwh-selected-features.png")


if __name__ == "__main__":

    # load data
    data_train = np.load(FILE_PATH_TRAIN)
    data_test = np.load(FILE_PATH_TEST)

    traces_train, plaintexts_train, keys_train = load_data(data_train) 
    traces_test, plaintexts_test, keys_test = load_data(data_test)

    # visualize one trace
    plot_raw_trace(traces_train)

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

    # create data loaders
    X_train_tensor = torch.tensor(traces_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                            num_workers=16, 
                            shuffle=True)

    X_val_tensor = torch.tensor(traces_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, 
                            batch_size=BATCH_SIZE, 
                            num_workers=16,
                            shuffle=False)

    X_test_tensor = torch.tensor(traces_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    input_size = len(traces_train[0])
    mlp = MLP_model(input_dim=input_size, num_classes=CLASSES, mlp_dim=MLP_DIM, learning_rate=LR)
        	

    save = ModelCheckpoint(
                    monitor="val_ge_mean",
                    mode="min",
                    dirpath=PATH_MLP,
                    filename="mlp-{epoch:02d}-{val_ge_mean:.2f}",
                    save_top_k=-1
    )
    
    ge_callback = GuessingEntropyCallback(
        plaintexts=plaintexts_val[:, TARGET_BYTE],
        correct_key = correct_keys_val[0, TARGET_BYTE], # 0 because fixed key
        num_traces = NB_VAL_TRACES,
        leakage_model = LEAKAGE_MODEL,
        target_byte = TARGET_BYTE,
    )
        

    kr_callback = KeyRankCallback(
        plaintexts=plaintexts_val[:, TARGET_BYTE],
        correct_key=correct_keys_val[0, TARGET_BYTE], # 0 because fixed key
        leakage_model=LEAKAGE_MODEL,
        target_byte=TARGET_BYTE,
    )
        

    csv_logger = CSVLogger(
        save_dir=PATH_MLP,
        name="logs"
    )

    seed_everything(83545, workers=True)
    trainer = L.Trainer(
        max_epochs=100,
        callbacks=[save, ge_callback],
        logger=csv_logger,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        num_sanity_val_steps=0
    )

    # train model
    trainer.fit(mlp, train_loader, val_loader)
