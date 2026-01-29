import lightning as L
from torchmetrics import Accuracy

import torch
import torch.optim as optim
import torch.nn as nn

# class MLP_model(L.LightningModule):
#     def __init__(self, input_dim : int, num_classes : int, learning_rate = 0.0001):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, 500),
#             nn.ReLU(),
#             nn.Linear(500, 500),
#             nn.ReLU(),
#             nn.Linear(500, num_classes),
#         ).to(torch.float64)
#         self.loss = nn.CrossEntropyLoss()
#         self.lr = learning_rate
#         self.metric = Accuracy(task="multiclass", num_classes=num_classes)

#     def forward(self, x):
#         return self.net(x)
    
#     def training_step(self, train_batch, batch_idx):
#         X, y = train_batch
#         y = torch.argmax(y, dim=1)
#         outputs = self.forward(X)
#         loss = self.loss(outputs, y)
#         acc = self.metric(outputs, y)
#         self.log('loss_train', loss, on_epoch=True, prog_bar=True)
#         self.log('acc_train', acc, on_epoch=True, prog_bar=True)
#         return loss
        
#     def validation_step(self, val_batch, batch_idx):
#         X, y = val_batch
#         y = torch.argmax(y, dim=1)
#         outputs = self.forward(X)
#         loss = self.loss(outputs, y)
#         acc = self.metric(outputs, y)
#         self.log('loss_val', loss, on_epoch=True, prog_bar=True)
#         self.log('acc_val', acc, on_epoch=True, prog_bar=True)
#         return loss
    
#     def test_step(self, test_batch, batch_idx):
#         X, y = test_batch
#         y = torch.argmax(y, dim=1)
#         outputs = self.forward(X)
#         loss = self.loss(outputs, y)
#         acc = self.metric(outputs, y)

#         self.log('loss_test', loss, on_epoch=True, prog_bar=True)
#         self.log('acc_test', acc, on_epoch=True, prog_bar=True)
#         return loss
    
#     def predict_step(self, test_batch, batch_idx, dataloader_idx=0):
#         X, y = test_batch
#         # X, y = X.to(self.dtype), y.to(self.dtype)
#         X = X.to(torch.float64)
#         outputs = self.forward(X)
#         return outputs
#         # return self(X)
    
#     def configure_optimizers(self):
#         optimizer = optim.Adam(self.parameters(), lr=self.lr)
#         return optimizer

class MLP_model(L.LightningModule):
    def __init__(self, input_dim: int, num_classes: int, mlp_dim : int, learning_rate):
        super().__init__()
        
        self.save_hyperparameters()
        self.lr = learning_rate
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, mlp_dim),
            nn.SELU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.SELU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.SELU(),
            nn.Linear(mlp_dim, num_classes)
        )
        
        self.net.apply(self._init_weights)

        self.loss = nn.CrossEntropyLoss()
        self.metric = Accuracy(task="multiclass", num_classes=num_classes)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight, -0.05, 0.05)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        X, y = batch
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
        return loss
        
    # def validation_step(self, batch, batch_idx):
    #     X, y = batch
    #     y = torch.argmax(y, dim=1)
    #     outputs = self.forward(X)
    #     loss = self.loss(outputs, y)
    #     acc = self.metric(outputs, y)
    #     self.log('loss_val', loss, on_epoch=True, prog_bar=True)
    #     self.log('acc_val', acc, on_epoch=True, prog_bar=True)
    #     return loss
    
    def test_step(self, test_batch, batch_idx):
        X, y = test_batch
        y = torch.argmax(y, dim=1)
        outputs = self.forward(X)
        loss = self.loss(outputs, y)
        acc = self.metric(outputs, y)

        self.log('loss_test', loss, on_epoch=True, prog_bar=True)
        self.log('acc_test', acc, on_epoch=True, prog_bar=True)
        return loss
    
    def predict_step(self, test_batch, batch_idx, dataloader_idx=0):
        X, y = test_batch
        outputs = self.forward(X)
        return outputs
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        return optimizer