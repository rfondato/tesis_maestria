#%%
from har_model import HARModule
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from data_loader import get_loaders
import torch as t
import torch.nn.functional as F

t.set_float32_matmul_precision("high")

training_dl, testing_dl = get_loaders(seq_length=30, stride=5, batch_size=64, shuffle=True)

checkpoint_callback = ModelCheckpoint(dirpath='../models/', filename='CNN-Res100-Seq30-{epoch}-{val_loss:.2f}', monitor="val_loss", verbose=True)
early_stopping_callback = EarlyStopping("val_loss", patience=10, mode="min", verbose=True)

model = HARModule(device="cuda", lr=0.001)
trainer = L.Trainer(max_epochs=100, callbacks=[checkpoint_callback, early_stopping_callback])

trainer.fit(model, train_dataloaders=training_dl, val_dataloaders=testing_dl)
trainer.save_checkpoint(f'../models/final_model.ckpt')

pred = t.cat(trainer.predict(model, testing_dl), dim=0)
y = t.from_numpy(testing_dl.dataset.y)
loss = F.cross_entropy(pred, y)
print(loss)

# %%
