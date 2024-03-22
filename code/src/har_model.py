from torch import optim, nn
import torch as t
from torch.optim.lr_scheduler import ReduceLROnPlateau
import lightning as L

class ConvolutionalBlock(L.LightningModule):
    
    def __init__(self, in_channels, out_channels, device, dropout = None) -> None:
        super().__init__()

        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='circular', device=device, bias=False)
        self.b_norm1 = nn.BatchNorm2d(out_channels, device=device)
        self.act1 = nn.LeakyReLU()
        
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode='circular', device=device, bias=False)
        self.b_norm2 = nn.BatchNorm2d(out_channels, device=device)
        self.act2 = nn.LeakyReLU()

        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.drop = nn.Dropout(dropout) if dropout is not None else None

    def forward(self, x):
        x = self.act1(self.b_norm1(self.cnn1(x)))
        x = self.act2(self.b_norm2(self.cnn2(x)))
        x = self.maxpool(x)
        
        if self.drop is not None:
            x = self.drop(x)
        
        return x

# define the LightningModule
class HARModule(L.LightningModule):
    def __init__(self, lr = 1e-3, device = "auto"):
        super().__init__()

        self.lr = lr

        self.conv_block1 = ConvolutionalBlock(1, 20, device=device)
        self.conv_block2 = ConvolutionalBlock(20, 50, device=device)
        self.conv_block3 = ConvolutionalBlock(50, 100, device=device)

        # Global max pooling here
 
        self.fc = nn.Linear(102, 200, device=device)
        self.act = nn.LeakyReLU()
        self.fc2 = nn.Linear(200, 6, device=device)
        
        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, x):
        out = self.conv_block1(x[:, :, :-2])
        out = self.conv_block2(out)
        out = self.conv_block3(out)

        out = t.max(t.max(out, 2)[0], 2)[0] # Global max pooling

        delta_pos = x[:, 0, 29, -2:] - x[:, 0, 0, -2:]

        fc_in = t.cat((out, delta_pos), dim=1)

        out = self.act(self.fc(fc_in))
        out = self.fc2(out)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = self.forward(x)

        loss = self.loss(x, y)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = self.forward(x)

        loss = self.loss(x, y)
        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        return self(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, threshold=1e-3, min_lr=1e-4, patience=5, verbose=True)
        return {
           'optimizer': optimizer,
           'lr_scheduler': scheduler,
           'monitor': 'val_loss'
       }
