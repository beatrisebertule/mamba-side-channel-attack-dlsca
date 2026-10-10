# Python >= 3.10, Linux with an NVIDIA GPU + CUDA (required by mamba-ssm)
#
# Install PyTorch first (pick the build matching your CUDA version from
# https://pytorch.org/get-started/locally/), then the rest:
#   pip install -r requirements.txt
# If mamba-ssm / causal-conv1d fail to build, install them separately with:
#   pip install causal-conv1d mamba-ssm --no-build-isolation

# Deep learning
torch>=2.1
lightning>=2.1
torchmetrics>=1.0
mamba-ssm>=2.0          # provides mamba_ssm.Mamba and mamba_ssm.modules.block.Block
causal-conv1d>=1.2      # optional, fast CUDA kernel used by mamba-ssm

# Data handling and preprocessing
numpy
pandas
h5py
scikit-learn

# Plotting
matplotlib

# Only used for tensorflow.keras.utils.to_categorical (one-hot labels)
tensorflow
