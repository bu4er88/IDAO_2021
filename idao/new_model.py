import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F

class ConvNN(pl.LightningModule):
    def __init__(self, mode: ["classification", "regression"] = "classification"):
        super().__init__()
        self.mode = mode
        self.layer1 = nn.Sequential(
                    nn.Conv2d(1, 32, kernel_size=3),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=3),
                )
        
        self.layer2 = nn.Sequential(
                    nn.Conv2d(32, 64, kernel_size=3),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=3),
                )
        self.layer3 = nn.Sequential(
                    nn.Conv2d(64, 128, kernel_size=3),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=3),
                    nn.Flatten(),
                )

        self.drop_out = nn.Dropout(0.5)

        self.fc1 = nn.Linear(51200, 128)
        self.fc2 = nn.Linear(128, 2)  # for classification
        self.fc3 = nn.Linear(128, 1)  # for regression

        self.stem = nn.Sequential(
            self.layer1, self.layer2, self.layer3, self.drop_out, self.fc1, nn.ReLU(),
            )
        if self.mode == "classification":
            self.classification = nn.Sequential(self.stem, self.fc2,)
        else:
            self.regression = nn.Sequential(self.stem, self.fc3)

        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()
    

    def training_step(self, batch, batch_idx):
        # --------------------------
        x_target, class_target, reg_target, _ = batch
        if self.mode == "classification":
            class_pred = self.classification(x_target.float())
            class_loss = F.binary_cross_entropy_with_logits(
                class_pred, class_target.float()
            )
            self.train_acc(torch.sigmoid(class_pred), class_target)
            self.log("train_acc", self.train_acc, on_step=True, on_epoch=False)
            self.log("classification_loss", class_loss)

            return class_loss

        else:
            reg_pred = self.regression(x_target.float())
            #             reg_loss = F.l1_loss(reg_pred, reg_target.float().view(-1, 1))
            reg_loss = F.mse_loss(reg_pred, reg_target.float().view(-1, 1))

            #             reg_loss = torch.sum(torch.abs(reg_pred - reg_target.float().view(-1, 1)) / reg_target.float().view(-1, 1))
            self.log("regression_loss", reg_loss)
            return reg_loss

    def validation_step(self, batch, batch_idx):
        x_target, class_target, reg_target, _ = batch
        if self.mode == "classification":
            class_pred = self.classification(x_target.float())
            class_loss = F.binary_cross_entropy_with_logits(
                class_pred, class_target.float()
            )
            self.valid_acc(torch.sigmoid(class_pred), class_target)
            self.log("valid_acc", self.train_acc, on_step=True, on_epoch=False)
            self.log("classification_loss", class_loss)
            return class_loss

        else:
            reg_pred = self.regression(x_target.float())
            #             reg_loss = F.l1_loss(reg_pred, reg_target.float().view(-1, 1))
            reg_loss = F.mse_loss(reg_pred, reg_target.float().view(-1, 1))

            #             reg_loss = torch.sum(torch.abs(reg_pred - reg_target.float().view(-1, 1)) / reg_target.float().view(-1, 1))
            self.log("regression_loss", reg_loss)
            return reg_loss
    

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
    
    def forward(self, x):
        if self.mode == "classification":
            class_pred = self.classification(x.float())
            return {"class": torch.sigmoid(class_pred)}
        else:
            reg_pred = self.regression(x.float())
            return {"energy": reg_pred}
