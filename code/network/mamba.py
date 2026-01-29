import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy

import lightning as L
from mamba_ssm import Mamba
from mamba_ssm.modules.block import Block 
from functools import partial

import math


def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)
                    

class MAMBA(nn.Module):
    def __init__(self, channels_in=1, d_model=64, num_classes=256, n_layers=4, learning_rate=0.0005):
        super(MAMBA, self).__init__()
        self.channels_in = channels_in
        self.d_model = d_model
        self.lr = learning_rate

        self.norm_f = (nn.LayerNorm)(
            d_model, eps=1e-5
        )

        self.conv = nn.Conv1d(
            channels_in, d_model, kernel_size=3, stride=3, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)

        # self.feature_extractor = nn.Sequential(
        #     nn.Conv1d(channels_in, 16, kernel_size=3, stride=3, padding=1),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(16),
        #     nn.Conv1d(16, 32, kernel_size=3, stride=3, padding=1),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(32),
        #     nn.Conv1d(32, d_model, kernel_size=3, stride=3, padding=1),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(d_model)
        # )

        def create_block(layer_idx):
            mixer_cls = partial(
                Mamba,
                layer_idx=layer_idx,
            )

            norm_cls = partial(
                nn.LayerNorm, eps=1e-5
            )
            block = Block(
                d_model,
                mixer_cls = mixer_cls,
                mlp_cls = nn.Identity,
                norm_cls = norm_cls,
                fused_add_norm = False,
                residual_in_fp32 = False,
            )
            block.layer_idx = layer_idx
            return block


        self.layers_forw = nn.ModuleList(
            [
                create_block(i)
                for i in range(n_layers)
            ]
        )

        self.layers_backw = nn.ModuleList(
            [
                create_block(i)
                for i in range(n_layers)
            ]
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layers,
                **({}),
                n_residuals_per_layer=1
            )
        )

        self.net = nn.Linear(d_model, num_classes)
    
        self.loss = nn.CrossEntropyLoss()
        self.metric = Accuracy(task="multiclass", num_classes=num_classes)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight, -0.05, 0.05)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def mamba_forw(self, hidden_states):
        residual = None
        for layer in self.layers_forw:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=None
            )
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        return hidden_states
    
    def mamba_backw(self, hidden_states):
        residual = None
        for layer in self.layers_backw:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=None
            )
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        return hidden_states
        
    
    def forward(self, x):
        output = torch.reshape(x, (x.shape[0], self.channels_in, x.shape[1] ))
        output = self.conv(output)
        # output = self.feature_extractor(output)
        output = torch.reshape(output, (output.shape[0], output.shape[2], output.shape[1]))
        forw_output = self.mamba_forw(output)
        flipped = torch.flip(output, dims=[1])
        backw_output = self.mamba_backw(flipped)
        combined = torch.cat([forw_output[:, -1, :], backw_output[:, -1, :]], dim=-1)
        # combined = forw_output[:, -1, :]
        output = self.net(combined)
        return output


class MAMBA_model(L.LightningModule):
    def __init__(self, channels_in :int, d_model : int, num_classes : int, n_layers : int, learning_rate):
        super().__init__()

        self.net = MAMBA(channels_in=channels_in, d_model=d_model, num_classes=num_classes, n_layers=n_layers).to(self.dtype)
        
        self.loss = nn.CrossEntropyLoss()
        self.lr = learning_rate
        self.metric = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.net(x)
    
    def training_step(self, train_batch, batch_idx):
        X, y = train_batch
        X, y = X.to(self.dtype), y.to(self.dtype)
        y = torch.argmax(y, dim=1)
        outputs = self.forward(X)
        loss = self.loss(outputs, y)
        acc = self.metric(outputs, y)
        self.log('loss_train', loss, on_epoch=True, prog_bar=True)
        self.log('acc_train', acc, on_epoch=True, prog_bar=True)
        return loss
    
    def on_validation_epoch_start(self, *args):
        self.val_predictions = []

    def validation_step(self, val_batch, batch_idx):
        X, y = val_batch
        X, y = X.to(self.dtype), y.to(self.dtype)
        y = torch.argmax(y, dim=1)
        outputs = self.forward(X)
        self.val_predictions.append(outputs.detach().cpu())
        loss = self.loss(outputs, y)
        acc = self.metric(outputs, y)

        self.log("loss_val", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("acc_val", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    

    def test_step(self, test_batch, batch_idx):
        X, y = test_batch
        X, y = X.to(self.dtype), y.to(self.dtype)
        y = torch.argmax(y, dim=1)
        outputs = self.forward(X)
        # print(y.dtype, outputs.dtype)
        loss = self.loss(outputs, y)
        acc = self.metric(outputs, y)

        self.log('loss_test', loss, on_epoch=True, prog_bar=True)
        self.log('acc_test', acc, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        return optimizer
    
    def predict_step(self, test_batch, batch_idx, dataloader_idx=0):
        X, y = test_batch
        X, y = X.to(self.dtype), y.to(self.dtype)
        # print(X.shape)
        outputs = self.forward(X)
        return outputs
